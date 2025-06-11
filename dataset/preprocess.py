import os
import re
import pickle

# Parameters
SEQ_LEN = 10  # You can adjust this

# File paths
DATA_DIR = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.txt')
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.pkl')
SEQS_PATH = os.path.join(DATA_DIR, 'sequences.pkl')

# Read text
def read_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    text = ' '.join([line.strip() for line in lines if line.strip()])
    return text

# Tokenize into words (simple whitespace + punctuation split)
def tokenize(text):
    # Lowercase and split on non-word chars
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens

# Create vocabulary dicts
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

# Generate sequences and targets
def create_sequences(tokens, seq_len):
    sequences = []
    targets = []
    for i in range(len(tokens) - seq_len):
        seq = tokens[i:i+seq_len]
        # Target sequence: the next token for each position in the input sequence
        target_seq = tokens[i+1:i+seq_len+1]
        
        # Ensure target_seq has the same length as seq_len
        if len(target_seq) == seq_len:
            sequences.append(seq)
            targets.append(target_seq)
    return sequences, targets

def main():
    text = read_corpus(CORPUS_PATH)
    tokens = tokenize(text)
    word2idx, idx2word = build_vocab(tokens)
    sequences, targets = create_sequences(tokens, SEQ_LEN)
    # Save vocab and sequences
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    with open(SEQS_PATH, 'wb') as f:
        pickle.dump({'sequences': sequences, 'targets': targets}, f)
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Saved vocab to {VOCAB_PATH} and sequences to {SEQS_PATH}")

if __name__ == "__main__":
    main() 