import os
import torch
import torch.nn.functional as F
# import numpy as np # Keep for PositionalEncoding if not fully PyTorch-ified yet in model.py - no longer needed with optimized PositionalEncoding
import pickle
import sys

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, DATA_DIR, d_model, n_heads, d_ff, DROPOUT_RATE # Import DROPOUT_RATE

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon Macs
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device for generation: {device}")

# Load vocab
with open(os.path.join(DATA_DIR, 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)
word2idx = vocab['word2idx']
idx2word = vocab['idx2word']
vocab_size = len(word2idx)

# Model
config = TransformerConfig(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=DROPOUT_RATE)
model = Narayana(config).to(device) # Move model to device
model.eval() # Set model to evaluation mode

# Load trained weights
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
# Updated to load .pt file, which can be a full training state checkpoint
weights_path = os.path.join(MODELS_DIR, 'narayana_weights.pt') # Changed extension to .pt

# Check for specific checkpoint path from arguments if any test script relies on it
import argparse
parser = argparse.ArgumentParser(description="Generate text from the Narayana Transformer model.")
parser.add_argument('--epoch', type=int, default=None, help='Epoch number of the checkpoint to load.')
args = parser.parse_args()

if args.epoch:
    checkpoint_name = f'narayana_epoch_{args.epoch}.pt'
    weights_path = os.path.join(MODELS_DIR, 'checkpoints', checkpoint_name)
    print(f"Loading model from checkpoint epoch {args.epoch}...")

if os.path.exists(weights_path):
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        # If it's a full training state checkpoint, extract model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state dictionary loaded from full training checkpoint successfully.")
        else:
            # Otherwise, assume it's just the model's state_dict
            model.load_state_dict(checkpoint)
            print("Model state dictionary loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}. This might happen if model architecture changed.")
        print("Using random weights.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {weights_path}. Using random weights.")
else:
    print(f"Warning: Trained weights not found at {weights_path}, using random weights.")

def encode(prompt, word2idx, seq_len):
    tokens = prompt.lower().split()
    idxs = [word2idx.get(w, 0) for w in tokens]
    # Pad or truncate
    if len(idxs) < seq_len:
        idxs = [0] * (seq_len - len(idxs)) + idxs
    else:
        idxs = idxs[-seq_len:]
    return torch.tensor([idxs], dtype=torch.long).to(device) # Move input tensor to device

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
            x = torch.cat((x[:, 1:], torch.tensor([[next_idx]], dtype=torch.long).to(device)), dim=1) # Efficiently slide window and move to device

    return ' '.join(generated)

if __name__ == "__main__":
    prompt = input(f"Enter prompt (up to {SEQ_LEN} words): ")
    out = generate(prompt, num_words=20)
    print(f"Generated: {out}") 