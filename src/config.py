import os

# Data parameters
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
MAX_GRAD_NORM = 1.0 # Added for gradient clipping

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

# Model parameters
d_model = 64
n_heads = 2
d_ff = 128
N_BLOCKS = 2 # Number of Transformer blocks

# Model parameters
d_model = 64
n_heads = 2
d_ff = 128 