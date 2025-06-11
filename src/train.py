import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset.dataloader import DataLoader
from src.model import TransformerConfig, Narayana
from src.config import SEQ_LEN, BATCH_SIZE, EPOCHS, LR, DATA_DIR, d_model, n_heads, d_ff, MAX_GRAD_NORM

# Prepare logging file
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists
log_file_path = os.path.join(MODELS_DIR, 'training_log.txt')

with open(log_file_path, 'w') as log_f:
    log_f.write("Training Log\n")
    log_f.write(f"Hyperparameters: SEQ_LEN={SEQ_LEN}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, MAX_GRAD_NORM={MAX_GRAD_NORM}\n")
    log_f.write(f"Model Config: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}\n")
    log_f.write("-----------------------------------------------------\n")

    # Load data
    loader = DataLoader(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    vocab_size = len(loader.word2idx)

    # Model
    config = TransformerConfig(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    model = Narayana(config)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_losses = [] # To store average loss per epoch for plotting

    # Training loop
    model.train() # Set model to training mode
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

            # Log every step to file
            log_message = f"Epoch {epoch+1}/{EPOCHS}, Step {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}"
            log_f.write(log_message + "\n")

        avg_loss = total_loss / n_batches
        epoch_losses.append(avg_loss) # Store epoch average loss

        log_message = f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}"
        log_f.write(log_message + "\n")

        # Print to terminal every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(log_message)

    print("Training complete.")

    # Plotting the loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', markersize=4, linestyle='-')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory
    print(f"Loss plot saved to {plot_path}")

# Save model weights to 'models' directory (outside with block)
model.save(os.path.join(MODELS_DIR, 'narayana_weights.pkl'))
print("Model saved.") 