import os
import sys
import pandas as pd
import numpy as np
import h5py
from scipy.stats import pearsonr, spearmanr
import torch
sys.path.append('/home/robert/codes/esm-main')
import myesm


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    if len(row.split(':')) > 1:
        score = 0.0
        for row_s in row.split(':'):
            wt, idx, mt = row_s[0], int(row_s[1:-1]) - offset_idx, row_s[-1]
            assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
            wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
            m_score = token_probs[idx, mt_encoded] - token_probs[idx, wt_encoded]
            score += m_score.item()
    else:
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        try:
            assert sequence[idx] == wt
            wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
            score = token_probs[idx, mt_encoded] - token_probs[idx, wt_encoded]
            score = score.item()
        except:
            print('The listed wildtype does not match the provided sequence')
            exit()
    return score



def main():
    root_path = 'testdata/stability_dataset/Mut_csv_files'
    hf = h5py.File('testdata/stability_prediction.h5', 'r')
    all_data = np.array(hf.get('embeddings'))
    Seq_Name = list(hf.get('Seq_Name'))
    Seq_Name = [i.decode()[:-4] for i in Seq_Name]
    seq_len = np.array(hf.get('seq_len'))

    alphabet = myesm.Alphabet.default_alphabet()

    offset_idx = 1
    Seq_dict = {}

    name = ''
    seq = ''
    with open('testdata/stability_dataset/Total.fasta', 'r') as f:
        for l in f.readlines():
            if l.startswith('>'):
                if len(name) != 0 :
                    Seq_dict[name] = seq
                    seq = ''
                    name = ''
                name = l.strip()[1:]
            else:
                seq += l.strip()
    Seq_dict[name] = seq


    natural_list = []
    with open('testdata/stability_dataset/natural_protein_list', 'r') as inf:
        for l in inf.readlines():
            natural_list.append(l.strip()[:-4])

    denovo_list = []
    with open('testdata/stability_dataset/denovo_protein_list', 'r') as inf:
        for l in inf.readlines():
            denovo_list.append(l.strip()[:-4])

    all_metrics = []
    Total_metrics = 'Dataset, spearmanr, pvalue\n'
    for i, seq in enumerate(Seq_Name):

        #if seq not in denovo_list:
        #    continue
        if seq not in natural_list:
            continue
        #if seq != '1I6C':
        #    continue
        df = pd.read_csv(os.path.join(root_path, seq + '.csv'))
        token_probs = torch.log_softmax(torch.from_numpy(all_data[i]), dim=-1)
        df['predicted'] = df.apply(
            lambda row: label_row(
                row['mut_type'],
                Seq_dict[seq],
                token_probs,
                alphabet,
                offset_idx,
            ),
            axis=1,
        )
        metrics,p_value = spearmanr(df['score'], df['predicted'])
        print(seq + ',' + str(metrics) + ',' + str(p_value))
        #print(df['predicted'].shape[0])
        all_metrics.append(metrics)
        Total_metrics += seq + ',' + str(metrics) + ',' + str(p_value) + '\n'
    print('Mean metrics: ', str(np.mean(all_metrics)))

    with open('Total_metrics', 'w') as of:
        of.write(Total_metrics)



if __name__ == '__main__':
    main()
