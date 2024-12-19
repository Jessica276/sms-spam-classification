import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def build_dict(texts):
    word_count = {}
    for text in texts:
        if not isinstance(text, str): 
            text = str(text)
        for word in text.split():
            word_count[word] = word_count.get(word, 0) + 1

    return {word: idx + 1 for idx, word in enumerate(word_count.keys())}


def tokenize_and_pad(texts, dictionary, max_seq_len):
    sequences = []
    for text in texts:
        seq = [dictionary.get(word, 0) for word in text.split()]
        sequences.append(seq + [0] * (max_seq_len - len(seq)))
    return np.array(sequences)

def prepare_datasets(X_train, X_test, y_train, y_test):
    dictionary = build_dict(X_train)
    max_seq_len = max(max(len(x.split()) for x in X_train), max(len(x.split()) for x in X_test))

    X_train_padded = tokenize_and_pad(X_train, dictionary, max_seq_len)
    X_test_padded = tokenize_and_pad(X_test, dictionary, max_seq_len)

    train_data = TensorDataset(
        torch.tensor(X_train_padded, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32)
    )
    test_data = TensorDataset(
        torch.tensor(X_test_padded, dtype=torch.long),
        torch.tensor(y_test.values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader, dictionary, max_seq_len
