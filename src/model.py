import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Keep for PositionalEncoding if not fully PyTorch-ified yet
import pickle
import os
from src.config import N_BLOCKS # Import N_BLOCKS

class TransformerConfig:
    def __init__(self, vocab_size, seq_len, d_model=64, n_heads=2, d_ff=128):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

# LayerNorm replaced by torch.nn.LayerNorm
# WordEmbedding replaced by torch.nn.Embedding

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.register_buffer('P', torch.zeros(seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                self.P[pos, i] = torch.sin(torch.tensor(pos / (10000 ** ((2 * i)/d_model))))
                if i+1 < d_model:
                    self.P[pos, i+1] = torch.cos(torch.tensor(pos / (10000 ** ((2 * (i+1))/d_model))))

    def forward(self, x):
        return x + self.P[:x.shape[1]]

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_q(x) # (B, T, C)
        K = self.W_k(x) # (B, T, C)
        V = self.W_v(x) # (B, T, C)

        # Split heads
        Q_heads = Q.view(B, T, self.n_heads, self.d_k).transpose(1,2) # (B, H, T, D_k)
        K_heads = K.view(B, T, self.n_heads, self.d_k).transpose(1,2) # (B, H, T, D_k)
        V_heads = V.view(B, T, self.n_heads, self.d_k).transpose(1,2) # (B, H, T, D_k)

        # Scaled dot-product attention
        scores = (Q_heads @ K_heads.transpose(-2,-1)) / (self.d_k**0.5) # (B, H, T, T)

        # Causal mask
        if mask is None:
            # Create an upper triangular mask (everything above main diagonal is 0)
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ V_heads # (B, H, T, D_k)

        out = out.transpose(1,2).contiguous().view(B, T, C) # Concatenate heads

        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.d_model, config.n_heads)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config.d_model, config.d_ff)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        # Residual connection 1: Add & Norm
        x = x + self.attn(self.norm1(x)) # Pre-norm: norm is applied before attn/ff
        # x = self.norm1(x + self.attn(x)) # Post-norm: norm is applied after attn/ff

        # Residual connection 2: Add & Norm
        x = x + self.ff(self.norm2(x)) # Pre-norm
        # x = self.norm2(x + self.ff(x)) # Post-norm
        return x

class Narayana(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = PositionalEncoding(config.seq_len, config.d_model)
        # Create a list of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(N_BLOCKS)])
        self.proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x)
        logits = self.proj(x) # (B, T, Vocab_size)
        return logits

    # Save and load methods now use PyTorch's state_dict
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # Example usage
    config = TransformerConfig(vocab_size=100, seq_len=10)
    model = Narayana(config)
    x = torch.randint(0, 100, (2, 10))  # batch of 2, seq_len 10
    logits = model(x)
    print("Logits shape:", logits.shape)

    # Verify save/load
    test_path = "test_model.pth"
    model.save(test_path)
    loaded_model = Narayana(config)
    loaded_model.load(test_path)
    print("Model saved and loaded successfully.")
    os.remove(test_path) 