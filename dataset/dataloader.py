import os
import pickle
import numpy as np

class DataLoader:
    def __init__(self, data_dir, batch_size=32, seq_len=10):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._load_data()
        self._prepare_batches()

    def _load_data(self):
        # Load vocab
        vocab_path = os.path.join(self.data_dir, 'vocab.pkl')
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        self.word2idx = vocab['word2idx']
        self.idx2word = vocab['idx2word']
        # Load sequences
        seqs_path = os.path.join(self.data_dir, 'sequences.pkl')
        with open(seqs_path, 'rb') as f:
            data = pickle.load(f)
        self.sequences = data['sequences']
        self.targets = data['targets']

    def _prepare_batches(self):
        # Convert sequences and targets to indices
        X = np.array([[self.word2idx[w] for w in seq] for seq in self.sequences], dtype=np.int32)
        # y should now be a 2D array: (num_sequences, seq_len)
        y = np.array([[self.word2idx[w] for w in target_seq] for target_seq in self.targets], dtype=np.int32)
        # Shuffle
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        # Split into batches
        self.num_batches = int(np.ceil(len(X) / self.batch_size))
        self.X_batches = np.array_split(X, self.num_batches)
        self.y_batches = np.array_split(y, self.num_batches)
        self.batch_idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration
        X_batch = self.X_batches[self.batch_idx]
        y_batch = self.y_batches[self.batch_idx]
        self.batch_idx += 1
        return X_batch, y_batch

if __name__ == "__main__":
    # Example usage
    loader = DataLoader(os.path.dirname(__file__), batch_size=4, seq_len=10)
    print(f"Number of batches: {len(loader)}")
    for X, y in loader:
        print("X:", X)
        print("y:", y)
        break  # Show only the first batch 