# Dataset_Loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, data_dir, vocab=None, max_len=100):
        """
        data_dir: path to train or test folder containing pos/neg subfolders
        vocab: dict token -> index; if None, build vocab from this dataset
        max_len: max sequence length for padding/truncation
        """
        self.texts = []
        self.labels = []
        self.max_len = max_len

        # Load texts and labels
        for label_dir, label in [('pos',1), ('neg',0)]:
            folder = os.path.join(data_dir, label_dir)
            for fname in os.listdir(folder):
                if fname.endswith('.txt'):
                    with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        self.texts.append(text)
                        self.labels.append(label)

        # Tokenize texts (simple whitespace split)
        self.tokenized_texts = [text.split() for text in self.texts]

        # Build vocab if not provided
        if vocab is None:
            self.vocab = self.build_vocab(self.tokenized_texts)
        else:
            self.vocab = vocab

        # Encode texts to indices and pad
        self.encoded_texts = [self.encode_and_pad(tokens) for tokens in self.tokenized_texts]

    def build_vocab(self, tokenized_texts, min_freq=2):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        # Keep tokens with freq >= min_freq
        vocab_tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
        # Add special tokens
        vocab = {'<PAD>':0, '<UNK>':1}
        for i, token in enumerate(vocab_tokens, start=2):
            vocab[token] = i
        return vocab

    def encode_and_pad(self, tokens):
        encoded = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
        # Pad or truncate
        if len(encoded) < self.max_len:
            encoded += [self.vocab['<PAD>']] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return encoded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx]), torch.tensor(self.labels[idx], dtype=torch.float)


def load_datasets(train_dir, test_dir, batch_size=64, max_len=100, cache_dir=None):
    """
    Loads or builds datasets with caching.
    Returns: train_loader, test_loader, vocab
    """

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = os.path.join(cache_dir, 'train_dataset.pt')
        test_cache_path = os.path.join(cache_dir, 'test_dataset.pt')
    else:
        train_cache_path = None
        test_cache_path = None

    if train_cache_path and os.path.exists(train_cache_path) and test_cache_path and os.path.exists(test_cache_path):
        print("🔁 Loading cached datasets...")
        train_dataset = torch.load(train_cache_path)
        test_dataset = torch.load(test_cache_path)
        vocab = train_dataset.vocab
    else:
        print("📦 Building datasets from scratch...")
        train_dataset = TextDataset(train_dir, vocab=None, max_len=max_len)
        test_dataset = TextDataset(test_dir, vocab=train_dataset.vocab, max_len=max_len)
        vocab = train_dataset.vocab
        if train_cache_path and test_cache_path:
            torch.save(train_dataset, train_cache_path)
            torch.save(test_dataset, test_cache_path)
            print("✅ Cached datasets saved.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab
