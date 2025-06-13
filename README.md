# Narayana LLM

## Project Overview

Narayana is a custom-built, word-level Transformer-based Large Language Model (LLM) implemented in PyTorch. This project covers the entire pipeline from data collection and preprocessing to model training and text generation. The goal is to develop a foundational understanding of LLM architecture and training dynamics.

## Features

-   **Data Collection & Cleaning:** Python script to download and clean text data from Project Gutenberg.
-   **Preprocessing:** Tokenization, vocabulary creation (word2idx, idx2word), and sequence generation for training.
-   **Custom DataLoader:** Efficient batching of preprocessed data for training.
-   **Transformer Architecture:** A self-contained implementation of a mini-GPT style Transformer, including:
    -   Word Embeddings
    -   Positional Encoding
    -   Self-Attention Mechanism
    -   Feed-Forward Networks
    -   Layer Normalization (Pre-normalization)
    -   Multiple Transformer Blocks (configurable via `N_BLOCKS` in `config.py`)
-   **Dropout Regularization:** Added dropout layers for better model generalization.
-   **Device Handling:** Automatic detection and utilization of CUDA, MPS (Apple Silicon), or CPU for training.
-   **Improved Logging:** Detailed logging of training loss, accuracy, and learning rate to TensorBoard, along with console output.
-   **Full Training State Persistence:** Ability to save and load the complete training state (model weights, optimizer state, current epoch) for seamless training resumption.
-   **Text Generation:** Script for generating new text based on a trained model and a given prompt.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amitsingh0024/LLM_Master.git
    cd LLM_Master
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Follow these steps to run the LLM pipeline:

1.  **Download and Clean Data:**
    This script downloads raw text from Project Gutenberg and cleans it.
    ```bash
    python -m dataset.download_and_clean
    ```
    This will create `dataset/raw.txt` and `dataset/corpus.txt`.

2.  **Preprocess Data:**
    This script tokenizes the corpus, builds a vocabulary, and generates sequences for training.
    ```bash
    python -m dataset.preprocess
    ```
    This will create `dataset/vocab.pkl` and `dataset/sequences.pkl`.

3.  **Train the Model:**
    This script trains the Narayana LLM. Training progress is logged to TensorBoard.
    ```bash
    python -m src.train
    ```
    To resume training from a checkpoint:
    ```bash
    python -m src.train --load_checkpoint models/checkpoints/narayana_epoch_X.pt
    ```
    (Replace `X` with the desired epoch number. Checkpoints are saved every `CHECKPOINT_INTERVAL` epochs.)
    Trained model weights and training states will be saved as `.pt` files in `models/checkpoints/` and the final model as `models/narayana_weights.pt`.

4.  **Generate Text:**
    After training, use this script to generate text based on a prompt.
    ```bash
    python -m src.generate
    ```
    The script will prompt you to enter a starting sequence of words.

## Project Structure

```
LLM_Master/
├── dataset/                  # Contains data download, preprocessing, and loading scripts
│   ├── __init__.py           # Makes 'dataset' a Python package
│   ├── corpus.txt            # Cleaned text data
│   ├── dataloader.py         # DataLoader for batching preprocessed data
│   ├── download_and_clean.py # Script for downloading and cleaning raw text
│   ├── preprocess.py         # Script for tokenization, vocabulary, and sequence generation
│   └── sequences.pkl         # Preprocessed sequences of token IDs
│   └── vocab.pkl             # Word-to-index and index-to-word mappings
├── models/                   # Directory for saving trained model weights and logs
│   ├── checkpoints/          # Saved training checkpoints (*.pt)
│   ├── narayana_weights.pt   # Final trained model weights
│   └── runs/                 # TensorBoard log directories
├── src/                      # Source code for the Narayana LLM
│   ├── __init__.py           # Makes 'src' a Python package
│   ├── config.py             # Centralized hyperparameters and configurations
│   ├── generate.py           # Script for generating text
│   ├── model.py              # Narayana LLM architecture definition
│   └── train.py              # Model training script
├── venv/                     # Python virtual environment (ignored by Git)
├── .gitignore                # Specifies files and directories to ignore in Git
└── requirements.txt          # Python dependencies
```

## Model Details

**Narayana** is a word-level Transformer model. Key components include:

-   **Word Embedding:** Converts input token IDs into dense vector representations.
-   **Positional Encoding:** Adds positional information to word embeddings to account for word order.
-   **Transformer Block:** The core building block, consisting of a Multi-Head Self-Attention layer and a Feed-Forward Network, each with residual connections and layer normalization.
-   **Multi-Head Self-Attention:** Allows the model to jointly attend to information from different representation subspaces at different positions.
-   **Pre-normalization:** Layer normalization is applied *before* the self-attention and feed-forward sub-layers.
-   **Configurable Depth:** The number of Transformer blocks (`N_BLOCKS`) can be adjusted in `src/config.py` to control model capacity.

## Training Progress and Results

The model has been successfully migrated to PyTorch, and initial training shows a consistently decreasing loss, indicating stable learning. Further training with increased model capacity (more Transformer blocks) is planned to improve text generation quality and coherence.

## Future Enhancements

-   **Longer Training:** Train for more epochs to achieve better convergence and generation quality.
-   **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and model dimensions.
-   **Larger Dataset:** Utilize a more extensive and diverse dataset for training.
-   **GPU Acceleration:** Further optimization for GPU usage to speed up computation.
-   **Beam Search/Sampling:** Implement more advanced text generation strategies.
-   **Evaluation Metrics:** Incorporate quantitative evaluation metrics (e.g., perplexity).
-   **Deployment:** Explore options for deploying the trained model. 