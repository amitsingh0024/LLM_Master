import os
import torch
import sys
import pickle

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, DATA_DIR, d_model, n_heads, d_ff, DROPOUT_RATE

def get_model_size(model):
    """Calculates the total size of the model in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / (1024**2)
    return size_all_mb

if __name__ == "__main__":
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon Macs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize a dummy config (values don't matter for size calculation)
    # We need vocab_size, so load it.
    with open(os.path.join(DATA_DIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['word2idx'])

    config = TransformerConfig(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=DROPOUT_RATE)
    model = Narayana(config).to(device)

    print(f"Model size: {get_model_size(model):.2f} MB") 