import os
import torch
import torch.nn.functional as F
import pickle
import sys
from collections import deque # For tracking recent tokens for repetition penalty

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, DATA_DIR, d_model, n_heads, d_ff, CHECKPOINTS_DIR, DROPOUT_RATE
import argparse

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon Macs
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device for generation test: {device}")

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

def encode(prompt, word2idx, seq_len):
    tokens = prompt.lower().split()
    idxs = [word2idx.get(w, 0) for w in tokens]
    # Pad or truncate
    if len(idxs) < seq_len:
        idxs = [0] * (seq_len - len(idxs)) + idxs
    else:
        idxs = idxs[-seq_len:]
    return torch.tensor([idxs], dtype=torch.long).to(device) # Move input tensor to device

def generate(model, prompt, num_words=20, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, no_repeat_ngram_size=0, stop_on_repetition=0):
    x = encode(prompt, word2idx, SEQ_LEN)
    generated = []
    recent_tokens = deque(maxlen=no_repeat_ngram_size) # For N-gram blocking
    
    with torch.no_grad(): # Disable gradient calculations for inference
        for _ in range(num_words):
            # Forward pass through the model
            logits = model(x) # (B, T, Vocab_size)

            # Get predictions for the last token in the sequence
            next_token_logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for i in set(recent_tokens):
                    next_token_logits[:, i] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_value = values[:, -1].unsqueeze(1)
                next_token_logits = torch.where(next_token_logits < min_value, torch.full_like(next_token_logits, float('-inf')), next_token_logits)
            
            # Apply top-p (nucleus) sampling
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep at least one token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item() # Use multinomial for sampling

            # Add to recent tokens for repetition penalty/n-gram blocking
            recent_tokens.append(next_idx)

            # Stop generation on repetition
            if stop_on_repetition > 0 and len(generated) >= stop_on_repetition:
                if all(generated[i] == generated[i - stop_on_repetition] for i in range(len(generated) - stop_on_repetition, len(generated))):
                    print(f"Stopping generation due to {stop_on_repetition}-token repetition.")
                    break

            generated.append(idx2word.get(next_idx, '<unk>'))
            
            # Append new token and slide window
            x = torch.cat((x[:, 1:], torch.tensor([[next_idx]], dtype=torch.long).to(device)), dim=1) # Efficiently slide window and move to device

    return ' '.join(generated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using the Narayana LLM.")
    parser.add_argument('--epoch', type=int, help='Epoch number of the checkpoint to load. If not specified, loads the final trained model.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature. Lower values make the model more confident.')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling: sample from the top k most likely tokens.')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top-p (nucleus) sampling: sample from the smallest set of tokens whose cumulative probability exceeds p.')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Penalty for repeating tokens. Higher values discourage repetition.')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='If set to int > 0, all ngrams of that size can only occur once.')
    parser.add_argument('--stop_on_repetition', type=int, default=0, help='Stop generation if the same token sequence repeats N times.')

    args = parser.parse_args()

    if args.epoch:
        weights_path = os.path.join(CHECKPOINTS_DIR, f'narayana_epoch_{args.epoch}.pt')
        if not os.path.exists(weights_path):
            print(f"Error: Checkpoint for epoch {args.epoch} not found at {weights_path}")
            exit()
        print(f"Loading model from checkpoint epoch {args.epoch}...")
    else:
        MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
        weights_path = os.path.join(MODELS_DIR, 'narayana_weights.pt')
        print("Loading final trained model...")

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

    prompt = input(f"Enter prompt (up to {SEQ_LEN} words): ")
    out = generate(model, prompt, num_words=100, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size, stop_on_repetition=args.stop_on_repetition)
    print(f"Generated: {out}") 