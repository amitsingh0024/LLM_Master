import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Keep for PositionalEncoding if not fully PyTorch-ified yet
import pickle
import os
from src.config import N_BLOCKS # Import N_BLOCKS

class TransformerConfig:
    """Configuration class for the Narayana Transformer model.

    Args:
        vocab_size (int): The size of the vocabulary.
        seq_len (int): The maximum sequence length.
        d_model (int, optional): The dimensionality of the model's embeddings. Defaults to 64.
        n_heads (int, optional): The number of attention heads. Defaults to 2.
        d_ff (int, optional): The dimensionality of the feed-forward network. Defaults to 128.
        dropout_rate (float, optional): The dropout rate to apply. Defaults to 0.1.
    """
    def __init__(self, vocab_size, seq_len, d_model=64, n_heads=2, d_ff=128, dropout_rate=0.1):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

# LayerNorm replaced by torch.nn.LayerNorm
# WordEmbedding replaced by torch.nn.Embedding

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings.

    This encoding scheme uses sine and cosine functions of different frequencies
    to provide the model with information about the relative or absolute position
    of tokens within a sequence.

    Args:
        seq_len (int): The maximum sequence length for which to generate positional encodings.
        d_model (int): The dimensionality of the model's embeddings.
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.register_buffer('P', torch.zeros(seq_len, d_model))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.P[:, 0::2] = torch.sin(position * div_term)
        self.P[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to which positional encoding is added.
                              Expected shape: (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        return x + self.P[:x.shape[1]]

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions.

    Args:
        d_model (int): The dimensionality of the model's embeddings.
        n_heads (int): The number of attention heads. `d_model` must be divisible by `n_heads`.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """Performs the forward pass of the self-attention mechanism.

        Args:
            x (torch.Tensor): The input tensor to the attention layer.
                              Expected shape: (batch_size, sequence_length, d_model).
            mask (torch.Tensor, optional): An optional mask to prevent attention to certain positions.
                                           Typically used for causal masking in decoders.
                                           Expected shape: (sequence_length, sequence_length).

        Returns:
            torch.Tensor: The output of the self-attention mechanism.
                          Shape: (batch_size, sequence_length, d_model).
        """
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
    """A simple two-layer feed-forward network with ReLU activation.

    Args:
        d_model (int): The dimensionality of the input and output.
        d_ff (int): The dimensionality of the inner layer.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Performs the forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): The input tensor.
                              Expected shape: (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The output tensor of the feed-forward network.
                          Shape: (batch_size, sequence_length, d_model).
        """
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """A single Transformer block, comprising self-attention and feed-forward layers.

    Each sub-layer (self-attention and feed-forward) includes residual connections,
    layer normalization (pre-norm), and dropout.

    Args:
        config (TransformerConfig): The configuration object for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.d_model, config.n_heads)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.ff = FeedForward(config.d_model, config.d_ff)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        """Performs the forward pass of the Transformer block.

        Args:
            x (torch.Tensor): The input tensor to the Transformer block.
                              Expected shape: (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The output tensor of the Transformer block.
                          Shape: (batch_size, sequence_length, d_model).
        """
        # Residual connection 1: Add & Norm
        x = x + self.dropout1(self.attn(self.norm1(x))) # Pre-norm: norm is applied before attn/ff

        # Residual connection 2: Add & Norm
        x = x + self.dropout2(self.ff(self.norm2(x))) # Pre-norm
        return x

class Narayana(nn.Module):
    """The main Narayana Transformer model.

    Comprises an embedding layer, positional encoding, a stack of Transformer blocks,
    and a final projection layer to the vocabulary size.

    Args:
        config (TransformerConfig): The configuration object for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = PositionalEncoding(config.seq_len, config.d_model)
        self.dropout_embed = nn.Dropout(config.dropout_rate) # Dropout after embedding and positional encoding
        # Create a list of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(N_BLOCKS)])
        self.proj = nn.Linear(config.d_model, config.vocab_size)
        self.dropout_final = nn.Dropout(config.dropout_rate) # Dropout before final projection

    def forward(self, x):
        """Performs the forward pass of the Narayana model.

        Args:
            x (torch.Tensor): The input tensor of token IDs.
                              Expected shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: The logits for the next token prediction.
                          Shape: (batch_size, sequence_length, vocab_size).
        """
        x = self.embed(x)
        x = self.pos(x)
        x = self.dropout_embed(x)
        # Pass through each Transformer block
        for block in self.blocks:
            x = block(x)
        x = self.dropout_final(x)
        logits = self.proj(x) # (B, T, Vocab_size)
        return logits

    def save(self, path):
        """Saves the model's state dictionary to a specified path.

        Args:
            path (str): The file path where the model's state dictionary will be saved.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Loads the model's state dictionary from a specified path.

        Args:
            path (str): The file path from which to load the model's state dictionary.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # If it's a full training state checkpoint, extract model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Otherwise, assume it's just the model's state_dict
            self.load_state_dict(checkpoint)

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