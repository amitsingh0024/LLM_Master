import os
import torch
import torch.nn.functional as F
import numpy as np # Keep for PositionalEncoding if not fully PyTorch-ified yet in model.py
import pickle
from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, DATA_DIR, d_model, n_heads, d_ff

# Load vocab
with open(os.path.join(DATA_DIR, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)
word2idx = vocab['word2idx']
idx2word = vocab['idx2word']
vocab_size = len(word2idx)

# Model
config = TransformerConfig(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=d_model, n_heads=n_heads, d_ff=d_ff)
model = Narayana(config)
model.eval() # Set model to evaluation mode

# Load trained weights
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
weights_path = os.path.join(MODELS_DIR, 'narayana_weights.pkl')
if os.path.exists(weights_path):
    try:
        model.load(weights_path)
        print("Trained model weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}. This might happen if model architecture changed.")
        print("Using random weights.")
else:
    print("Warning: Trained weights not found, using random weights.")

def encode(prompt, word2idx, seq_len):
    tokens = prompt.lower().split()
    idxs = [word2idx.get(w, 0) for w in tokens]
    # Pad or truncate
    if len(idxs) < seq_len:
        idxs = [0] * (seq_len - len(idxs)) + idxs
    else:
        idxs = idxs[-seq_len:]
    return torch.tensor([idxs], dtype=torch.long)

def generate(prompt, num_words=20):
    x = encode(prompt, word2idx, SEQ_LEN)
    generated = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for _ in range(num_words):
            # Forward pass through the model
            logits = model(x) # (B, T, Vocab_size)

            # Get predictions for the last token in the sequence
            next_token_logits = logits[:, -1, :]
            next_idx = torch.argmax(next_token_logits, dim=-1).item() # Get the index as a Python int

            generated.append(idx2word.get(next_idx, '<unk>'))
            
            # Append new token and slide window
            # x = torch.roll(x, -1, dims=1) # This would move the whole batch
            x = torch.cat((x[:, 1:], torch.tensor([[next_idx]], dtype=torch.long)), dim=1) # Efficiently slide window

    return ' '.join(generated)

if __name__ == "__main__":
    prompt = input(f"Enter prompt (up to {SEQ_LEN} words): ")
    out = generate(prompt, num_words=20)
    print(f"Generated: {out}") 