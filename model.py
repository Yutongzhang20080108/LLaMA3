#We are going to implement the simplest version of LLaMA 3.1 8B in this repository.
#We keep almost all the model details but not the training details.

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2Tokenizer
#---------------------------------------------------------

#Part1: Config
@dataclass
class LLaMAConfig:
    n_embd: int = 1024
    n_head: int = 8
    n_layer:int = 8
    n_vocab:int = 50257
    mlp_ratio: float = 4
    batch_size: int = 4
    max_seq_len: int = 512

#Part2: Tokenizer
#We are going to use the tokenizer from Huggingface.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#Part3: Model
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.fc = nn.Linear(n_embd, n_embd)
        self.head_dim = n_embd // n_head

    def forward(self, x):
        super().__init__()
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y

#We are going to implemet the activation function used in LLaMA model.
#The equation looks like this SwiGLU(x) = x * sigmoid(beta * x) + (1 - sigmoid(beta * x)) * (Wx + b)
class SwiGLU(nn.Module):
    def __init__(self, n_embd, beta):
        super().__init__()
        self.beta = beta
        self.ln = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        super().__init__()
        x = x * F.sigmoid(self.beta * x) + (1 - F.sigmoid(self.beta * x)) * self.ln(x)
        return x

class MLP(nn.Module):
    def __init__(self, n_embd, mlp_ratio):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd*mlp_ratio)
        self.GELU = nn.GELU(n_embd*mlp_ratio)
        self.fc2 = nn.Linear(n_embd*mlp_ratio, n_embd)

    def forward(self, x):
        super().__init__()
        x = self.fc1(x)
        x = self.GELU(x)
        x = self.fc2(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.RMSnorm = nn.RMSNorm(config.n_embd)
        self.SelfAttention = SelfAttention(config.n_embd, config.n_head)
        self.RMSnorm = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.mlp_ratio)

    def forward(self, x):
        x = self.RMSnorm(x)
        x = self.SelfAttention(x)
        x = self.RMSnorm(x)
        x = self.mlp(x)
        return x

class LLaMA3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.Decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln = nn.Linear(config.n_embd, config.n_vocab)
    
    def forward(self, idx, target=None):
        super().__init__()
        token_emb = torch.tensor(tokenizer.encode(idx)).to(device)
        pos_emb = self.pos_emb(idx)
        emb = token_emb + pos_emb
        logits = self.Decoder(emb)
        logits = F.softmax(self.ln(logits), dim=-1)
        return logits
    
#---------------------------------------------------------
#Part4: Test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

with open("input.txt", "r") as f:
    data = f.read()
test_data = data[:512]
model = LLaMA3(LLaMAConfig()).to(device)
logits = model(test_data)
print(logits)