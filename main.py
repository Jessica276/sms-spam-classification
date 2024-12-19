import torch
from load_data import load_data
from preprocessing import preprocessing
from tokenizer import build_dict, max_sequence, tokenize_and_pad, pad_data
from train_lstm import train_lstm
from lstm import LSTMClassifier
from bert import train_bert

def main():
    load_data()
    y_train, y_test, X_train, X_test = preprocessing()
    dictionary = build_dict(X_train)
    max_seq_len = max_sequence(X_train, X_test)
    train_loader, test_loader = pad_data(X_train, X_test, y_train, y_test, dictionary, max_seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_lstm(train_loader, test_loader, device, dictionary, max_seq_len)
    train_bert(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
