from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from utils.model_utils import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import math
from utils.model_utils import gen_grid_up, calc_emd, calc_cd
import dgl
from dgl import DGLGraph
import sys
from dgl.nn.pytorch import GraphConv, NNConv
from typing import Optional, Literal, Dict, Tuple, List
from nvidia.se3_transformer.model.basis import get_basis, update_basis_with_fused
from nvidia.se3_transformer.model.layers.attention import AttentionBlockSE3
from nvidia.se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from nvidia.se3_transformer.model.layers.norm import NormSE3
from nvidia.se3_transformer.model.layers.pooling import GPooling
from nvidia.se3_transformer.runtime.utils import str2bool
from nvidia.se3_transformer.model.fiber import Fiber


proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2

from .modules import (
    TransformerLayer,
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
)

from argparse import Namespace

class Sequential(nn.Sequential):
    """ Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features. """

    def forward(self, input, *args, **kwargs):

        for module in self:
            input = module(input, *args, **kwargs)
        return input

def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r], dim=1)
    else:
        edge_features['0'] = r[..., None]

    return edge_features

class SE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 node_feature_size: int,
                 num_channels: int,
                 n_heads: int,
                 div: int,
                 num_degrees: int=4,
                 edge_dim: int=4,
                 return_type: Optional[int] = 0,
                 pooling: Optional[Literal['avg', 'max']] = 'max',
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = True,
                 **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = n_heads
        self.channels_div = div
        self.return_type = return_type
        self.pooling = pooling
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.NODE_FEATURE_DIM = node_feature_size
        self.EDGE_FEATURE_DIM = edge_dim
        fiber_in=Fiber({0: node_feature_size})
        #fiber_out=Fiber({0: num_degrees * num_channels})
        fiber_out=Fiber({0: 1280})
        fiber_edge=Fiber({0: edge_dim})
        fiber_hidden=Fiber.create(num_degrees, num_channels)
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.fiber_edge = fiber_edge

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL


        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=n_heads,
                                                   channels_div=div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree,
                                     fuse_level=self.fuse_level,
                                     low_memory=low_memory))
        self.graph_modules = Sequential(*graph_modules)

        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(self, graph: DGLGraph, 
                basis: Optional[Dict[str, Tensor]] = None, get_features = False, need_head_weights=False):
        node_feats = {'0' : graph.ndata['f']}
        edge_feats = {'0': graph.edata['w']}
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['d'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        edge_feats = get_populated_edge_features(graph.edata['d'], edge_feats)
        
        node_feats= self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis, need_head_weights=need_head_weights)

        if need_head_weights:
            return node_feats[str(self.return_type)], node_feats["attentions"]
        #print(node_feats['0'].shape, edge_feats['0'].shape)
        if self.pooling is not None:
            return node_feats[str(self.return_type)], self.pooling_module(node_feats, graph=graph)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats


class ProteinBertModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--num_layers", default=36, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias", action="store_true", help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)
        if self.args.arch == "roberta_large":
            self.model_version = "ESM-1b"
            self._init_submodules_esm1b()
        else:
            self.model_version = "ESM-1"
            self._init_submodules_esm1()

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                )
                for _ in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions, self.args.embed_dim, self.padding_idx
        )
        self.emb_layer_norm_before = (
            ESM1bLayerNorm(self.args.embed_dim) if self.emb_layer_norm_before else None
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def _init_submodules_esm1(self):
        self._init_submodules_common()
        self.embed_scale = math.sqrt(self.args.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.args.embed_dim, self.padding_idx)
        self.embed_out = nn.Parameter(torch.zeros((self.alphabet_size, self.args.embed_dim)))
        self.embed_out_bias = None
        if self.args.final_bias:
            self.embed_out_bias = nn.Parameter(torch.zeros(self.alphabet_size))

    def forward(self, tokens, structure_emb, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x + self.embed_positions(tokens) #+ structure_emb

        if self.model_version == "ESM-1b":
            if self.emb_layer_norm_before:
                x = self.emb_layer_norm_before(x)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if layer_idx == 30:
                x = x + structure_emb.transpose(0, 1)
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        if self.model_version == "ESM-1b":
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
            #x = x + structure_emb
            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            x = self.lm_head(x)
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers

class PCN_decoder(nn.Module):
    def __init__(self, feature_size, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(feature_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 3)


    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse)
        return coarse

class Model(nn.Module):
    def __init__(self, args, bert_args, alphabet, num_coarse=256, size_z=256, feature_size = 128):
        super(Model, self).__init__()

        self.proteinBertEncoder = ProteinBertModel(Namespace(**bert_args), alphabet)
        for p in self.parameters():
            p.requires_grad = False

        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]

        self.size_z = size_z
        self.num_coarse = num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.scale = self.num_points // num_coarse
        self.cat_feature_num = 2 + 9 + feature_size
        self.num_degrees = 4
        assert(feature_size % self.num_degrees == 0)
        self.num_channels = int(feature_size / self.num_degrees)
        self.encoder = SE3Transformer(num_layers = 1,
            node_feature_size = 20,
            num_channels = self.num_channels,
            n_heads = 8,
            num_degrees=self.num_degrees,
            edge_dim = 4,
            div=1)
        self.decoder = PCN_decoder(1280, num_coarse, self.num_points, self.scale, self.cat_feature_num)
        self.mixModel = False


    def compute_kernel(self, x, y):
        x_size = x.size()[0]
        y_size = y.size()[0]
        dim = x.size()[1]

        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / float(dim))

    def mmd_loss(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def reshape(self, logits, length, maxlen):
        value = torch.zeros(length.shape[0], maxlen, logits.shape[2])
        length = length.cpu().numpy()
        start_idx = 0
        end_index = 0
        for i in range(length.shape[0]):
            end_index += int(length[i])
            value[i][1:int(length[i]+1),] = logits[0,start_idx:end_index,]
            start_idx += int(length[i])
        return value

    def cal_bert_loss(self, predict, pos, target):
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        predict = predict * pos.unsqueeze(2).repeat(1, 1, 33)
        ece_loss = ((criterion(predict.transpose(1,2), target) * pos).sum() / pos.sum())
        return ece_loss

    def forward(self, x, gt, crop, batch_tokens, masked_tokens, masked_pos, protein_len, is_training=True, mean_feature=None, alpha=None):
        all_feat, feat = self.encoder(x, get_features = True, need_head_weights=True)
        all_feat = all_feat.permute(2, 0, 1)
        pos_emb = self.reshape(all_feat, protein_len, masked_tokens.shape[1]).to(masked_tokens.device)
        bert_result = self.proteinBertEncoder(batch_tokens, pos_emb, repr_layers=[33], need_head_weights = False,return_contacts=False)
        logits = bert_result["logits"]
        return logits
