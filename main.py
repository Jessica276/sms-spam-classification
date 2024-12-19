import torch
from preprocessing import preprocessing
from train_lstm import train_lstm
from lstm import LSTMClassifier
from bert import train_bert
from tokenizer import build_dict, tokenize_and_pad, prepare_datasets

def main():
    y_train, y_test, X_train, X_test = preprocessing()
    dictionary = build_dict(X_train)
    max_seq_len = prepare_datasets(X_train, X_test, y_train, y_test)
    train_loader, test_loader = tokenize_and_pad(X_train, dictionary, max_seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_lstm(train_loader, test_loader, device, dictionary)
    train_bert(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
