import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

# Basic tokenization: lowercase, remove punctuation, split by whitespace
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_jokes = [tokenize(joke) for joke in jokes]
all_words = [word for joke in tokenized_jokes for word in joke]


# Build vocab
special_tokens = ["<pad>", "<unk>"]
vocab = special_tokens + sorted(set(all_words))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)

print(f"Vocab size: {vocab_size}")
print(tokenized_jokes[0])

def tokens_to_indices(tokens, word2idx):
    return [word2idx.get(token, word2idx["<unk>"]) for token in tokens]

sequences = []

for tokens in tokenized_jokes:
    if len(tokens) > 5:
        for i in range(len(tokens) - 5):
            input_seq = tokens[i:i+5]
            target = tokens[i+5]
            input_tensor = torch.tensor(tokens_to_indices(input_seq, word2idx), dtype=torch.long)
            target_tensor = torch.tensor(word2idx.get(target, word2idx["<unk>"]), dtype=torch.long)
            sequences.append((input_tensor, target_tensor))


print("Words:", [idx2word[idx.item()] for idx in sequences[0][0]])
print("Target word:", idx2word[sequences[0][1].item()])



class JokeDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

dataset = JokeDataset(sequences)

from torch.utils.data import random_split, DataLoader

# Define lengths for the split
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)