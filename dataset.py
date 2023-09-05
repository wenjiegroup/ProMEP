import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random
from typing import Sequence
import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import myesm

class TargetDataset(data.Dataset):
    def __init__(self,
                 root, task, pid):
        fname = ''
        if task == 'fold' or task == 'ec' or task == 'go':
            fname = os.path.join(root, 'Resolution-level-1', 'pdb_pointClouds.h5')
        elif task == 'ppi':
            fname = os.path.join(root,'pdb_pointClouds_'+str(pid)+'.h5')
        else:
            fname = os.path.join(root, 'Resolution-level-1', 'pdb_pointClouds.h5')
        hf = h5py.File(fname, 'r')
        self.root = root
        self.complete_pcd = []
        self.Sequence = []
        self.seq_len = []
        self.complete_pcd = np.array(hf.get('complete_pcd')[:,:,0,:])
        self.Sequence = np.array(hf.get('sequence'))
        self.seq_len = np.array(hf.get('seq_len'))
        self.Seq_Name= np.array(hf.get('Seq_Name'))
        self.npoints = self.complete_pcd.shape[1]
        self.centre_corpping = True
        self.aatype = False
        self.rotation = False
        self.edge_features = True
        self.corrupted_center_index = []
        self.crop_times = 3
        self.crop_point_num = 128
        self.standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.standard_toks)}
        self.neighbors = 30
        self.alphabet = myesm.Alphabet.default_alphabet()
        self.batch_converter = self.alphabet.get_batch_converter()

        random.seed(0)
        if not self.centre_corpping:
            for v in range(self.crop_times):
                  index = random.randint(0, int(self.complete_pcd.shape[1]) - 1)
                  self.corrupted_center_index.append(index)

    def __getitem__(self, index):
        pc = self.complete_pcd[index]
        protein_name = self.Seq_Name[index]
        protein_len = min(self.seq_len[index], 1022)

        #protein_len = 512
        if self.rotation:
            pc = self.data_augmentation(pc)
        complete = torch.from_numpy(pc.astype(np.float32))
        input_cropped = np.empty(pc.shape, dtype = float)
        for i in range(pc.shape[0]):
            input_cropped[i] = pc[i]

        distance_list = []
        if self.centre_corpping :
            p_center = np.mean(pc, axis=0)
        else:
            center_point = random.randint(0, self.crop_times - 1)
            p_center = pc[self.corrupted_center_index[center_point]]
        for n in range(int(pc.shape[0])):
            distance_list.append(self.distance_squre(pc[n], p_center))
        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
        for sp in range(int(self.crop_point_num)):
                input_cropped[distance_order[sp][0]] = np.array([0.0, 0.0, 0.0])

        sequence_data = [('seq', self.Sequence[index].decode()[:protein_len])]
        masked_tokens, masked_pos, batch_tokens = self.batch_converter(sequence_data)
        masked_tokens = [[0 for i in range(protein_len+2)]]
        masked_pos = [[0 for i in range(protein_len+2)]]
        #print(protein_len, batch_tokens[0].shape)
        node_feature = []
        if self.aatype:
            sequence = self.Sequence[index].decode()
            #print(sequence)
            node_feature = self.one_hot_encoding(sequence)
            #print(node_feature[0])
            node_feature= node_feature.unsqueeze(-1)
        else:
            node_feature = torch.ones(pc.shape[0], 20, 1)

        #Construct graph      
        knn_g = dgl.knn_graph(complete[:protein_len,:], self.neighbors)
        src, dst = knn_g.edges()

        # Add node features to graph
        knn_g.ndata['x'] = complete[:protein_len,:]#[num_atoms,3]
        #knn_g.ndata['x'] = complete[:protein_len]
        knn_g.ndata['f'] = node_feature[:protein_len,:,] #[num_atoms,20,1]

       # Add edge features to graph
        edge_feature = []
        if self.edge_features:
            edge_type = list(dst.numpy() - src.numpy())
            for e in edge_type:
                if e == 0:
                    edge_feature.append(np.array([1,0,0,0]))
                elif e == 1 or e == -1:
                    edge_feature.append(np.array([0,1,0,0]))
                elif e > 1:
                    edge_feature.append(np.array([0,0,1,0]))
                else:
                    edge_feature.append(np.array([0,0,0,1]))
            edge_feature = np.array(edge_feature)
        else:
            edge_feature = np.ones((src.shape[0],1))

        knn_g.edata['d'] = complete[:protein_len,:][dst] - complete[:protein_len,:][src] #[num_atoms,3]
        knn_g.edata['w'] = torch.from_numpy(edge_feature.astype(np.float32))  #[num_edges,4]

        return  knn_g, input_cropped, pc, protein_name, protein_len, np.array(batch_tokens[0]), np.array(masked_tokens[0]), np.array(masked_pos[0]), np.array([protein_len])


    def __len__(self):
        return self.complete_pcd.shape[0]

    def distance_squre(self, p1,p2):
        result = p1 - p2
        val = np.multiply(result, result)
        distance = np.sum(val)
        return distance

    def data_augmentation(self, point_set):
        theta = np.random.uniform(0,np.pi*2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        point_set[: , [0]] += 1 # translation
        point_set += np.random.rand(point_set.shape[0], point_set.shape[1]) * 5 #random jitter

        return point_set

    def one_hot_encoding(self, text):
        seq_encoded = [self.tok_to_idx[tok] for tok in list(text)]
        tokens = np.zeros(shape = (self.npoints, 20))
        for i in range(len(text)):
            tokens[i, seq_encoded[i]] = 1
        return torch.from_numpy(tokens.astype(np.int64))
