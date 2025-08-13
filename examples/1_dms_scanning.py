import os
import sys
import random
import torch
import numpy as np 
import h5py
import pandas as pd
from typing import Sequence, Tuple, List, Union
from sklearn.preprocessing import normalize,scale,MinMaxScaler

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa


        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks
        self.standard_idx = [self.get_idx(i) for i in self.standard_toks]

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def default_alphabet(cls) -> "Alphabet":
        standard_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = True
        use_msa = False
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)


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
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
        score = token_probs[idx, mt_encoded] - token_probs[idx, wt_encoded]
        score = score.item()
    return score

def main():

    #Step 1, prepare dms data

    Source_seq = ''
    with open('cas9.fasta', 'r') as f:
        for l in f.readlines():
            if not l.startswith('>'):
                Source_seq += l.strip().upper()
    #print(Source_seq)

    mutList = ['R','A', 'G', 'V', 'L', 'I', 'S', 'T', 'C', 'M', 'D', 'E', 'N', 'Q', 'K', 'F', 'Y', 'W', 'P', 'H']
    outinfo = 'mutant\n'
    SequenceList = list(Source_seq)

    n_count = 0
    
    #single mutants
    for mut in mutList:
        print('Prepare mut : ' + mut)
        final_out = []
        n_count = 0
        cur_mut = ''
        cur_site = []
        for i in range(len(Source_seq)):
        
            if SequenceList[i] == mut:
                continue
            else:
                cur_mut = SequenceList[i] + str(i + 1) + mut + ':'
            outinfo += SequenceList[i] + str(i + 1) + mut + '\n'
            n_count += 1
    
    with open('dms_data/single-mutants.csv', 'w') as of:
        of.write(outinfo)
    
    #Step 2. calculates logits
    ensemble = []
    df = pd.read_csv('dms_data/single-mutants.csv')
    for j in range(1):
        print('Model: ', j)
        hf = h5py.File('promep-fitness_prediction.h5', 'r')
        all_data = np.array(hf.get('embeddings'))
        Seq_Name = list(hf.get('Seq_Name'))
        Seq_Name = [i.decode()[:-4] for i in Seq_Name]
        seq_len = np.array(hf.get('seq_len'))
        alphabet = Alphabet.default_alphabet()
        # we cut the raw cas9 sequence into 2 splits, each contains 1000 aa (split1: 1-1000; split2: 368-1368)
        cas9_score = np.concatenate((all_data[0][:1000,], all_data[1][633:1369,]), axis = 0)
        offset_idx = 1
        token_probs = torch.log_softmax(torch.from_numpy(cas9_score), dim=-1)
        df['predicted-'+str(j)] = df.apply(
            lambda row: label_row(
                row['mutant'],
                Source_seq,
                token_probs,
                alphabet,
                offset_idx,
            ),
            axis=1,
        )
        ensemble.append('predicted-'+str(j))

    #Step 3. sort results
    prediction = df[ensemble].mean(1).values
    ascending = np.argsort(prediction)
    descending = ascending[::-1]
    #print(descending)
    mutant = df['mutant'].values
    mutant = mutant[descending]
    
    with open('dms_data/scanning-cas9.csv', 'w') as of:
        of.write('\n'.join(list(mutant)))
    

    #Step 4. Calculate normalized scores
    write_col = ['mutant-sort', 'mean_score','nm_score']
    df['mutant-sort'] = mutant
    df['mean_score'] = prediction[descending]
    min_max_scaler = MinMaxScaler(feature_range = (0,1))
    promep_score_list = min_max_scaler.fit_transform(prediction.reshape(-1,1)).reshape(-1)
    df['nm_score'] = promep_score_list[descending]
    df[write_col].to_csv('score_data/cas9-score.csv', index = False)


if __name__ == '__main__':
    main()
