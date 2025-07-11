import numpy as np
import os
import pickle

import torch

from utils.data_utils.wrangle_data import decode_to_string

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_train_valid_data():
    train_data = np.fromfile(os.path.join('data', 'train_data_encoded_id.bin'), dtype=np.int64)
    valid_data = np.fromfile(os.path.join('data', 'valid_data_encoded_id.bin'), dtype=np.int64)

    train_data = torch.tensor(train_data, dtype=torch.int64)
    valid_data = torch.tensor(valid_data, dtype=torch.int64)

    return train_data, valid_data

def get_batch(split, train_data, valid_data, block_size=8, batch_size=32):
    data = train_data if split == 'train' else valid_data
    start_idxs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in start_idxs])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_idxs])
    x, y = x.to(device), y.to(device)
    return x, y

def get_vocab_size():
    vocab_size = pickle.load(open(os.path.join('data', 'vocab_size.pkl'), 'rb'))['vocab_size']
    vocab_size = torch.tensor(vocab_size, dtype=torch.long)
    return vocab_size

def generate_string(generated_ints):
    int_to_char = pickle.load(open(os.path.join('data', 'int_to_char.pkl'), 'rb'))
    return decode_to_string(int_to_char, generated_ints)
