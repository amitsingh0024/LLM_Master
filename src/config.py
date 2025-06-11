import os

# Data parameters
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-4
MAX_GRAD_NORM = 1.0 # Added for gradient clipping

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'models', 'checkpoints') # New checkpoints directory
TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, 'runs') # New TensorBoard log directory

# Model parameters
d_model = 64
n_heads = 2
d_ff = 128
N_BLOCKS = 2 # Number of Transformer blocks
CHECKPOINT_INTERVAL = 10 # Save model every 10 epochs

# Model parameters
d_model = 64
n_heads = 2
d_ff = 128 