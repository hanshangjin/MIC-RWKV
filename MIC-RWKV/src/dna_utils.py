import os
import numpy as np
import re
import pandas as pd
import torch
from torch.utils.data import Dataset

# Function from eMIC-AntiKP: https://github.com/quangnhbk/mic_klebsiella_pneumoniae
def convertMIC(s):
    new_s = re.sub('\>([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))*2), s)
    new_s = re.sub('\<([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))/2), new_s)
    new_s = re.sub('\<=([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    new_s = re.sub('^([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    return float(new_s)

# Function from eMIC-AntiKP: https://github.com/quangnhbk/mic_klebsiella_pneumoniae
def getListData(config, ANTIBIOTIC):
    df = pd.read_csv(config.data_map_file)
    df_filtered = df[df['Antibiotic'].eq(ANTIBIOTIC)]

    df_filtered['MIC'] = df_filtered['Actual MIC'].apply(convertMIC)

    lst_data = []
    for index, row in df_filtered.iterrows():
       lst_data.append({'PATRIC ID':'{:0.5f}'.format(row['PATRIC ID']), 'MIC': row['MIC']})

    print("lst_data: ", len(lst_data))
    return lst_data

def split_dataset(NB_SAMPLES):
    NB_SAMPLES_TEST = int(NB_SAMPLES / 10)
    NB_SAMPLES_VAL = int(NB_SAMPLES / 10)
    
    lst_index = np.arange(NB_SAMPLES)
    np.random.shuffle(lst_index)
    
    test_index = lst_index[:NB_SAMPLES_TEST]
    val_index = lst_index[NB_SAMPLES_TEST:NB_SAMPLES_TEST + NB_SAMPLES_VAL]
    train_index = lst_index[NB_SAMPLES_TEST + NB_SAMPLES_VAL:]
    
    return train_index, val_index, test_index


class DataGenerator(Dataset):
    def __init__(self, config, lst_data, max_length=2048, shuffle=True):
        self.config = config
        self.lst_data = lst_data
        self.shuffle = shuffle
        self.max_length = max_length
        self.char_to_index = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'a': 1, 't': 2, 'c': 3, 'g': 4}
        self.keys = [i['PATRIC ID'] for i in self.lst_data]
        val2cls = config.val2cls[config.ANTIBIOTIC]
        val2cls = {int(k) if k.isdigit() else float(k) if k.replace('.', '', 1).isdigit() else k: v for k, v in val2cls.items()}

        self.X = {}
        self.y = {}
        self.keys = []
        for data in self.lst_data:
            if data['MIC'] not in val2cls.keys():
                continue
            self.keys.append(data['PATRIC ID'])
            file_name = data['PATRIC ID'] + '.fasta.txt'
            file_path = os.path.join(config.amr_path, file_name)
            df = pd.read_csv(file_path, sep = '\t')
            df = df[(df['Cut_Off'] == 'Perfect') | (df['Cut_Off'] == 'Strict')].values.tolist()
            amr_genes = []
            for i in range(len(df)):
                amr_genes.append(df[i][17])
            self.X[data['PATRIC ID']] = amr_genes
            self.y[data['PATRIC ID']] = val2cls[data['MIC']]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        seq_list = self.X[key]
        target = self.y[key]
        truncated_seqs = []
        for sequence in seq_list:
            if len(sequence) <= self.max_length:
                truncated_seqs.append(sequence)
            else:
                truncated_seqs.append(sequence[:self.max_length])
                truncated_seqs.append(sequence[self.max_length:])
        encoded_seqs = [torch.tensor([self.char_to_index[char] for char in seq], dtype=torch.long) for seq in truncated_seqs]

        return encoded_seqs, target


def collate_fn(batch, max_num_sequences, max_sequence_length):
    padded_batch = []
    y_batch = []
    masks = []
    num_sequence_masks = []
    for x, y in batch:
        num_sequences = len(x)
        pad_sequence_list = []
        mask_list = []
        num_sequence_mask = torch.ones(max_num_sequences)
        for seq in x:
            pad_length = max_sequence_length - len(seq)
            padded_seq = torch.nn.functional.pad(seq, (0, pad_length))
            pad_sequence_list.append(padded_seq)
            mask = torch.ones_like(padded_seq)
            mask[len(seq):] = 0
            mask_list.append(mask)

        pad_sequence_list.extend([torch.zeros_like(padded_seq) for _ in range(max_num_sequences - num_sequences)])
        mask_list.extend([torch.zeros_like(mask) for _ in range(max_num_sequences - num_sequences)])

        num_sequence_mask[num_sequences:] = 0

        padded_batch.append(torch.stack(pad_sequence_list))
        masks.append(torch.stack(mask_list))
        num_sequence_masks.append(num_sequence_mask)
        y_batch.append(y)

    return torch.stack(padded_batch), torch.tensor(y_batch), torch.stack(masks).unsqueeze(-1), torch.stack(num_sequence_masks).unsqueeze(-1)
