import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
from dataset.dataloader import DataLoader
from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, BATCH_SIZE, EPOCHS, LR, DATA_DIR, d_model, n_heads, d_ff, MAX_GRAD_NORM, CHECKPOINT_INTERVAL, CHECKPOINTS_DIR, TENSORBOARD_LOG_DIR

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
    "d_ff": d_ff
}
writer.add_hparams(hparam_dict=hparams, metric_dict={'hparam/accuracy': 0, 'hparam/loss': 0}) # Placeholder metrics

# Load data
loader = DataLoader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
vocab_size = len(loader.word2idx)

# Model
config = TransformerConfig(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=d_model, n_heads=n_heads, d_ff=d_ff)
model = Narayana(config)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
model.train() # Set model to training mode
global_step = 0
for epoch in range(EPOCHS):
    total_loss = 0
    n_batches = 0
    for batch_idx, (X_np, y_np) in enumerate(loader):
        # Convert numpy arrays to PyTorch tensors and move to device (CPU for now)
        X = torch.from_numpy(X_np).long()
        y = torch.from_numpy(y_np).long()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(X) # (B, T, Vocab_size)

        # We predict the next token at *each* position in the sequence.
        # Reshape logits to (batch_size * seq_len, vocab_size)
        # Reshape targets to (batch_size * seq_len)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        total_loss += loss.item()
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
    # Log epoch average loss to TensorBoard
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

    log_message = f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}"
    print(log_message)

    # Save checkpoint every CHECKPOINT_INTERVAL epochs
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'narayana_epoch_{epoch+1}.pkl')
        model.save(checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training complete.")

# Close the TensorBoard writer
writer.close()

# Save final model weights to 'models' directory
model.save(os.path.join(MODELS_DIR, 'narayana_weights.pkl'))
print("Final model saved.") 