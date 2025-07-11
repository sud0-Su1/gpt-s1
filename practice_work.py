import numpy as np
import os
import pickle

import torch

from neural_network.bigram import BigramLanguageModel

from utils.data_utils.wrangle_data import decode_to_string

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

train_data = np.fromfile(os.path.join('data', 'train_data_encoded_id.bin'), dtype=np.int64)
valid_data = np.fromfile(os.path.join('data', 'valid_data_encoded_id.bin'), dtype=np.int64)

train_data = torch.tensor(train_data, dtype=torch.int64)
valid_data = torch.tensor(valid_data, dtype=torch.int64)

block_size = 8  # 32
batch_size = 4  # 16

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    start_idxs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in start_idxs])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_idxs])
    x, y = x.to(device), y.to(device)
    return x, y


vocab_size = pickle.load(open(os.path.join('data', 'vocab_size.pkl'), 'rb'))['vocab_size']
vocab_size = torch.tensor(vocab_size, dtype=torch.long)
xb, yb = get_batch('train')


import torch
import torch.nn as nn
import torch.nn.functional as F





# print(xb)
# print(xb.shape)
# print(yb)
# print(yb.shape)
# m = BigramLanguageModel(vocab_size).to(device)
# logits, loss = m(xb, yb)
# print(logits.shape)
# int_to_char = pickle.load(open(os.path.join('data', 'int_to_char.pkl'), 'rb'))
# idx = torch.zeros((1, 1), dtype=torch.long).to(device) # (batch_size, time_step) holding zero
# generated_ints = m.generate(idx, max_new_tokens=100)[0].tolist()
# # print(decode_to_string(int_to_char, generated_ints))


# optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)

# batch_size = 32
# for step in range(10001):
#     # Sample a batch of data
#     xb, yb = get_batch('train')

#     # Evaluate loss
#     logits, loss = m(xb, yb)

#     # Backpropagate loss
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     if step % 1000 == 0:
#         print(f'Step: {step}, Loss: {loss.item()}')


# int_to_char = pickle.load(open(os.path.join('data', 'int_to_char.pkl'), 'rb'))
# idx = torch.zeros((1, 1), dtype=torch.long).to(device) # (batch_size, time_step) holding zero
# generated_ints = m.generate(idx, max_new_tokens=300)[0].tolist()
# print(decode_to_string(int_to_char, generated_ints))















####################################################################
# Simple BIGRAM MODEL TRAINING AND GENERATION
####################################################################

# class BigramLanguageModel(nn.Module):
    
#     def __init__(self, vocab_size):
#         super().__init__()
#         # Each token directly reads off the logits for the next token from a lookup table.
#         self.token_embed_table = nn.Embedding(vocab_size, vocab_size)
    
#     def forward(self, idx, targets=None):
#         # idx and targets are both (batch_size, time_step) of integers.
#         logits = self.token_embed_table(idx) # (batch_size, time_step, channel)
#         if targets is None:
#             return logits, None
#         # Channel has to be 2nd dimension for cross_entropy
#         B, T, C = logits.shape
#         logits = logits.view(B * T, C)
#         targets = targets.view(B * T)
#         loss = F.cross_entropy(logits, targets)
#         return logits, loss

#     def generate(self, idx, max_new_tokens):
#         # idx is (batch_size, time_step) of indices in current context
#         for _ in range(max_new_tokens):
#             # get the predictions
#             logits, loss = self(idx)
#             # focus only on the last time step
#             logits = logits[:, -1, :] # (batch_size, channel)
#             # apply softmax to get probabilities
#             probs = F.softmax(logits, dim=-1) # (batch_size, channel)
#             # sample from the distribution
#             # Each batch has single prediction
#             idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
#             # append sampled index to the running sequence
#             idx = torch.cat([idx, idx_next], dim=-1) # (batch_size, time_step + 1)
#         return idx

# model = BigramLanguageModel(vocab_size).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# eval_iters = 100

# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'valid']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


# batch_size = 32
# max_iters = 5001
# for iter in range(max_iters):

#     if iter % eval_iters == 0:
#         losses = estimate_loss()
#         print(f'Step: {iter}, Train Loss: {losses["train"]}, Valid Loss: {losses["valid"]}')
        
#     # Sample a batch of data
#     xb, yb = get_batch('train')

#     # Evaluate loss
#     logits, loss = model(xb, yb)

#     # Backpropagate loss
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# # Generate text from the model
# int_to_char = pickle.load(open(os.path.join('data', 'int_to_char.pkl'), 'rb'))
# context = torch.zeros((1, 1), dtype=torch.long).to(device) # (batch_size, time_step) holding zero
# generated_ints = model.generate(context, max_new_tokens=500)[0].tolist()
# print(decode_to_string(int_to_char, generated_ints))


# torch.manual_seed(1337)
# B, T, C = 4, 8, 2 # batch_size, time_step, channels
# x = torch.randn(B, T, C)

# # We want x[b, t] = mean_(i <= t) x[b, i]
# xbow = torch.zeros((B, T, C)) # x bag of words
# for b in range(B):
#     for t in range(T):
#         xprev = x[b, :t+1] # (t, C)
#         xbow[b, t] = torch.mean(xprev, dim=0) # (C,)


# Version 2

# wei = torch.tril(torch.ones(T, T)) # (T, T)
# wei = wei / torch.sum(wei, dim=1, keepdim=True) # (T, T)
# xbow2 = wei @ x # (pytorchB, T, T) @ (B, T, C) --> (B, T, C)

# print(xbow[0])
# print('-' * 10)
# print(xbow2[0])
# print('-' * 10)
# print(torch.allclose(xbow, xbow2))
# # # print(x[0])
# # # print(xbow[0])


# Version 1

# torch.manual_seed(42)
# a = torch.tril(torch.ones(3, 3))
# a = a / torch.sum(a, dim=1, keepdim=True)
# b = torch.randint(0, 10, (3, 2)).float()
# c = a @ b
# print("a")
# print(a)
# print('-' * 10)
# print("b")
# print(b)
# print('-' * 10)
# print("c")
# print(c)

# ------------------------------------------------------------------

# # NOTE: Version 3

# tril = torch.tril(torch.ones(T, T)) # (T, T)
# wei = torch.zeros((T, T)) # (T, T)
# wei = wei.masked_fill(tril == 0, float('-inf')) # (T, T)
# wei = torch.softmax(wei, dim=1) # (T, T)
# xbow3 = wei @ x # (pytorchB, T, T) @ (B, T, C) --> (B, T, C)
# print(torch.allclose(xbow, xbow3))



# ####################################################################
# # BIGRAM MODEL TRAINING AND GENERATION with positional embeddings
# ####################################################################
# # vocab_size, n_embed=32, block_size=8, device='cpu'
# model = BigramLanguageModel(vocab_size=vocab_size, n_embed=32, block_size=block_size, device=device).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# eval_iters = 1000

# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'valid']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


# batch_size = 32
# max_iters = 5001
# for iter in range(max_iters):

#     if iter % eval_iters == 0:
#         losses = estimate_loss()
#         print(f'Step: {iter}, Train Loss: {losses["train"]}, Valid Loss: {losses["valid"]}')
        
#     # Sample a batch of data
#     xb, yb = get_batch('train')

#     # Evaluate loss
#     logits, loss = model(xb, yb)

#     # Backpropagate loss
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# # Generate text from the model
# int_to_char = pickle.load(open(os.path.join('data', 'int_to_char.pkl'), 'rb'))
# context = torch.zeros((1, 1), dtype=torch.long).to(device) # (batch_size, time_step) holding zero
# generated_ints = model.generate(context, max_new_tokens=500)[0].tolist()
# # print(decode_to_string(int_to_char, generated_ints))



# # ------------------------------------------------------------------

# # # NOTE: Version 4: self-attention
# torch.manual_seed(1337)
# B, T, C = 4, 8, 32 # batch_size, time_step, channels
# x = torch.randn(B, T, C)

# head_size = 16 # hyperparameter
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)

# k = key(x) # (B, T, head_size)
# q = query(x) # (B, T, head_size)
# wei = q @ k.transpose(-2, -1) * (head_size ** -0.5) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)

# tril = torch.tril(torch.ones(T, T)) # (T, T)
# wei = wei.masked_fill(tril == 0, float('-inf')) # (T, T)
# wei = torch.softmax(wei, dim=-1) # (T, T)
# # xbow4 = wei @ x # (pytorchB, T, T) @ (B, T, C) --> (B, T, C)
# v = value(x) # (B, T, head_size)
# xbow4 = wei @ v # (pytorchB, T, T) @ (B, T, head_size) --> (B, T, head_size)

# print(xbow4[0])
# print(xbow4[0].shape)
# print(xbow4.shape)
# print('-' * 10)
# print(wei)



####################################################################
# BIGRAM MODEL TRAINING AND GENERATION with attention
####################################################################
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 64
n_head = 4
n_layer = 4
dropout = 0.0

#  vocab_size, n_embed=32, block_size=8, n_layer=4, n_head=4, dropout=0.1, device='cpu'
model = BigramLanguageModel(vocab_size=vocab_size, n_embed=n_embed, block_size=block_size, n_layer=n_layer, n_head=n_head, dropout=dropout, device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for iter in range(max_iters):

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'Step: {iter}, Train Loss: {losses["train"]}, Valid Loss: {losses["valid"]}')
        
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)

    # Backpropagate loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from the model
int_to_char = pickle.load(open(os.path.join('data', 'int_to_char.pkl'), 'rb'))
context = torch.zeros((1, 1), dtype=torch.long).to(device) # (batch_size, time_step) holding zero
generated_ints = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode_to_string(int_to_char, generated_ints))
