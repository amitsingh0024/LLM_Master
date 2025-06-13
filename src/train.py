import os
import sys

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
from dataset.dataloader import DataLoader
from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, BATCH_SIZE, EPOCHS, LR, DATA_DIR, d_model, n_heads, d_ff, MAX_GRAD_NORM, CHECKPOINT_INTERVAL, CHECKPOINTS_DIR, TENSORBOARD_LOG_DIR, DROPOUT_RATE
import argparse

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon Macs
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Argument parsing
parser = argparse.ArgumentParser(description="Train the Narayana Transformer model.")
parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a model checkpoint to load.')
args = parser.parse_args()

# Prepare model directories and TensorBoard writer
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists
os.makedirs(CHECKPOINTS_DIR, exist_ok=True) # Ensure checkpoints directory exists
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True) # Ensure TensorBoard log directory exists

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

# Log hyperparameters to TensorBoard
hparams = {
    "SEQ_LEN": SEQ_LEN,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "MAX_GRAD_NORM": MAX_GRAD_NORM,
    "CHECKPOINT_INTERVAL": CHECKPOINT_INTERVAL,
    "d_model": d_model,
    "n_heads": n_heads,
    "d_ff": d_ff,
    "DROPOUT_RATE": DROPOUT_RATE,
    "LR": LR # Explicitly log learning rate
}
writer.add_hparams(hparam_dict=hparams, metric_dict={'hparam/accuracy': 0, 'hparam/loss': 0}) # Placeholder metrics

# Load data
loader = DataLoader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
vocab_size = len(loader.word2idx)

# Model
config = TransformerConfig(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout_rate=DROPOUT_RATE)
model = Narayana(config).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
model.train() # Set model to training mode
global_step = 0
start_epoch = 0 # Initialize start_epoch

# Load checkpoint if specified (after model and optimizer are initialized)
if args.load_checkpoint:
    if os.path.exists(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Resume from the next epoch
        # You might also want to load global_step if tracking it across runs
        # global_step = checkpoint.get('global_step', 0)
        print(f"Model and optimizer loaded from checkpoint: {args.load_checkpoint}. Resuming from epoch {start_epoch}.")
    else:
        print(f"Warning: Checkpoint path {args.load_checkpoint} not found. Starting training from scratch.")

for epoch in range(start_epoch, EPOCHS): # Use start_epoch here
    total_loss = 0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    for batch_idx, (X_np, y_np) in enumerate(loader):
        # Convert numpy arrays to PyTorch tensors and move to device
        X = torch.from_numpy(X_np).long().to(device)
        y = torch.from_numpy(y_np).long().to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(X) # (B, T, Vocab_size)

        # We predict the next token at *each* position in the sequence.
        # Reshape logits to (batch_size * seq_len, vocab_size)
        # Reshape targets to (batch_size * seq_len)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(logits.view(-1, vocab_size), 1)
        total_correct += (predicted == y.view(-1)).sum().item()
        total_samples += y.numel()

        n_batches += 1

        # Backward pass
        loss.backward()
        
        # Apply gradient clipping (now using PyTorch's built-in)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # Update weights
        optimizer.step()

        # Log step loss to TensorBoard
        writer.add_scalar('Loss/train_step', loss.item(), global_step)
        global_step += 1

    avg_loss = total_loss / n_batches
    avg_accuracy = total_correct / total_samples

    # Log epoch average loss and accuracy to TensorBoard
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/train_epoch', avg_accuracy, epoch)

    log_message = f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f} - Average Accuracy: {avg_accuracy:.4f}"
    print(log_message)

    # Save checkpoint every CHECKPOINT_INTERVAL epochs
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'narayana_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training complete.")

# Close the TensorBoard writer
writer.close()

# Save final model weights to 'models' directory
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'narayana_weights.pt'))
print("Final model saved.") 