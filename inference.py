import torch.optim as optim
import torch
from utils.train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import TargetDataset
import shutil
import dgl
import numpy as np
import h5py
import myesm

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def collate(samples):
    graphs, crop, complete, p_name, p_len, batch_tokens, masked_tokens, masked_pos, protein_len= map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(np.array(crop), dtype=torch.float32), torch.tensor(np.array(complete), dtype=torch.float32), np.array(p_name), np.array(p_len), torch.tensor(np.array(batch_tokens)), torch.tensor(masked_tokens), torch.tensor(masked_pos), torch.tensor(protein_len)

def init():
    logging.info(str(args))
    device = torch.device("cuda:0" if args.cuda_available else "cpu")

    dataset_test = TargetDataset(root=args.testdata, task=args.task, pid=args.pid)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, collate_fn=collate, num_workers=int(args.workers))
    logging.info('Length of test dataset:%d', len(dataset_test))
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    protein_bert_args = np.load('myesm/defalult_args.npy', allow_pickle=True).item()
    alphabet = myesm.Alphabet.default_alphabet()
    net = torch.nn.DataParallel(model_module.Model(args, protein_bert_args, alphabet))
    net.to(device)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    #net_dict=net.state_dict()
    #print(net_dict)

    val(net, dataloader_test)

def val(net, dataloader_test,):
    logging.info('Testing...')
    device = torch.device("cuda:0" if args.cuda_available else "cpu")
    net.module.eval()
    n_count = 0
    total_emb = []
    file_name = []
    seq_len = []
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            inputs,  crop, gt, p_name, p_len, batch_tokens, masked_tokens, masked_pos, protein_len = data
            batch_size = gt.size()[0]
            inputs = inputs.to(device)
            crop = crop.to(device)
            gt = gt.to(device)
            batch_tokens = batch_tokens.to(device)
            masked_tokens = masked_tokens.to(device)
            masked_pos = masked_pos.to(device)
            protein_len = protein_len.to(device)
            embedding = net(inputs, gt, crop, batch_tokens, masked_tokens, masked_pos, protein_len, is_training=False)
            #print(embedding.shape)
            assert(batch_size == 1)
            for sample_id in range(batch_size):
                length = p_len[sample_id]
                name = p_name[sample_id]
                emb = embedding.cpu().detach().numpy()
                #write all data
                emb_norm = np.zeros((1024, 1280))
                emb_norm[:length,:] = emb[0,1:-1,:]
                total_emb.append(emb_norm)
                
                file_name.append(name)
                seq_len.append(length)
                n_count += 1
            if(n_count % 1000 == 0):
                print('processing: %d'%n_count)
                write_data(total_emb, file_name, seq_len)
                total_emb = []
                file_name = []
                seq_len = []

                

    #print(np.array(total_emb).shape)
    write_data(total_emb, file_name, seq_len)


def write_data(total_emb, p_name, p_len):
    if args.outfile != None:
        fname = args.outfile
    else:
        fname = 'human_test_0.h5'
    
    if not os.path.exists(fname):
        f = h5py.File(fname, 'w')
    #if len(np.array(total_emb).shape) == 2:
    #    f.create_dataset('embeddings', data=np.array(total_emb, dtype='float32'), compression="gzip", chunks=True, maxshape=(None, 65536))
    #else:
        f.create_dataset('embeddings', data=np.array(total_emb, dtype='float32'), compression="gzip", chunks=True, maxshape=(None,1024,1280)) 
        f.create_dataset('Seq_Name', data=np.array(p_name), compression="gzip", chunks=True, maxshape=(1000000)) 
        f.create_dataset('seq_len', data=np.array(p_len), compression="gzip", chunks=True, maxshape=(1000000)) 
        f.close()
    
    else:
        f = h5py.File(fname, 'a')
        f['embeddings'].resize((f['embeddings'].shape[0] + len(total_emb)), axis = 0)
        #f['embeddings'][-f['embeddings'].shape[0]:] = np.array(total_emb)
        f['embeddings'][-len(total_emb):] = np.array(total_emb, dtype='float32')
        f['Seq_Name'].resize((f['Seq_Name'].shape[0] + len(p_name)), axis = 0)
        f['seq_len'].resize((f['seq_len'].shape[0] + len(p_len)), axis = 0)
        f['Seq_Name'][-len(p_name):] = np.array(p_name)
        f['seq_len'][-len(p_len):] = np.array(p_len)
        f.close()
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-d', '--dataroot', help='path to pdb,h5 file', required=False)
    parser.add_argument('-g', '--gpuid', help='id of gpu', default = 0, required=False)
    parser.add_argument('-f', '--outfile', help='path to output file', required=False)
    parser.add_argument('-t', '--task', help='downstream task', required=False)
    parser.add_argument('-i', '--pid', help='protein id', default = 0, required=False)
    parser.add_argument('-m', '--load_model', help='path to model', required=False)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid
    config_path = 'cfgs/drlnet.yaml'
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
 
    if opt.dataroot != None:
        args.testdata = opt.dataroot

    if opt.outfile != None:
        args.outfile = opt.outfile

    if opt.task != None:
        args.task = opt.task

    if opt.pid != None:
        args.pid = opt.pid

    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    if opt.load_model != None:
        args.load_model = opt.load_model
    init()
