#We are going to implement the simplest version of LLaMA 3.1 8B in this repository.
#We keep almost all the model details but not the training details.

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
#---------------------------------------------------------

#Part1: Config
@dataclass
class LLaMAConfig:
    n_embd: int = 768
    n_head: int = 8
    n_layer:int = 8
    n_vocab:int = 50340
    mlp_ratio: float = 4
    batch_size: int = 4
    max_seq_len: int = 64

#Part2: Tokenizer
#We are going to use the tokenizer from Huggingface.
tokenizer = tiktoken.get_encoding("gpt2")

#Part3: DataLoader
class DataLoader:
    def __init__(self, data, config):
        self.batch_size = config.batch_size
        self.max_seq_len = config.max_seq_len
        self.current_position = 0
        self.data = tokenizer.encode(data)
        print(f"Data length: {len(self.data)}")
        print(f"We can get {len(self.data) // (self.batch_size * self.max_seq_len)} batches.")
    
    def get_batch(self):
        batch = self.data[self.current_position: self.current_position + self.batch_size*self.max_seq_len]
        batch = torch.tensor(batch).view(self.batch_size, self.max_seq_len)
        target = self.data[self.current_position + 1: self.current_position + self.batch_size*self.max_seq_len+1]
        target = torch.tensor(target).view(self.batch_size, self.max_seq_len)
        self.current_position += self.batch_size*self.max_seq_len
        if self.current_position + self.batch_size*self.max_seq_len > len(self.data):
            self.current_position = 0
        return batch, target

#Part4: Model
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
        B, T, C = x.size()
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view([B, T, self.n_head, C // self.n_head]).transpose(1, 2)
        k = k.view([B, T, self.n_head, C // self.n_head]).transpose(1, 2)
        v = v.view([B, T, self.n_head, C // self.n_head]).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).view([B, T, C])
        return y

#We are going to implemet the activation function used in LLaMA model.
#The equation looks like this SwiGLU(x) = x * sigmoid(beta * x) + (1 - sigmoid(beta * x)) * (Wx + b)
class SwiGLU(nn.Module):
    def __init__(self, n_embd, beta):
        super().__init__()
        self.beta = beta
        self.ln = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        x = x * F.sigmoid(self.beta * x) + (1 - F.sigmoid(self.beta * x)) * self.ln(x)
        return x

class MLP(nn.Module):
    def __init__(self, n_embd, mlp_ratio):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd*mlp_ratio)
        self.GELU = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(n_embd*mlp_ratio, n_embd)

    def forward(self, x):
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
        self.emb = nn.Embedding(config.n_vocab, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.Decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln = nn.Linear(config.n_embd, config.n_vocab)
        self.loss_fn  = nn.CrossEntropyLoss()
    
    def forward(self, idx, target=None):
        model.train()
        B, T = idx.size()
        assert T <= self.config.max_seq_len
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        token_emb = self.emb(idx)
        pos_emb = self.pos_emb(pos)
        emb = token_emb + pos_emb
        for block in self.Decoder:
            emb = block(emb)
        logits = self.ln(emb)
        if target is not None:
            loss = self.loss_fn(logits.view(-1, self.config.n_vocab), target.view(-1))
            return logits, loss
        else:
            return logits
    
    def generation(self, idx):
        model.eval()
        B, T = idx.size()
        logits = self.forward(idx)
        logits = logits[:, -1, :]
        next_tokens_probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(next_tokens_probs, 1)
        idx = torch.cat([idx, next_token], dim=-1)
        generation = [tokenizer.decode(idx[i].tolist()) for i in range(B)]
        return generation
#---------------------------------------------------------
#Part4: Test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

#test training
with open("input.txt", "r") as f:
    data = f.read()
dataloader = DataLoader(data, LLaMAConfig())
model = LLaMA3(LLaMAConfig()).to(device)
#x, y = dataloader.get_batch()
#print(x.shape, y.shape)
#logits, loss = model(x, y)
#print(logits.shape, loss)

#test sampling
source_sentence = "Hello world"
source_sentence = torch.tensor(tokenizer.encode(source_sentence))
source_sentence = source_sentence.repeat(4, 1).to(device)
generation = model.generation(source_sentence)
print(generation)


