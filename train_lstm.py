import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def train_lstm(train_loader, test_loader, dictionary, device):
    model = LSTMClassifier(len(dictionary) + 1, 128, 1, 1, 0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(5):
        model.train()
        train_predictions, train_true_labels = [], []

        for x_batch, y_batch in train_loader:
            x, y = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_predictions += (torch.sigmoid(y_pred) >= 0.5).cpu().numpy().tolist()
            train_true_labels += y.cpu().numpy().tolist()

        test_predictions, test_true_labels = [], []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x, y = x_batch.to(device), y_batch.to(device)
                y_pred = model(x)
                test_predictions += (torch.sigmoid(y_pred) >= 0.5).cpu().numpy().tolist()
                test_true_labels += y.cpu().numpy().tolist()

        train_f1 = f1_score(train_true_labels, train_predictions)
        test_f1 = f1_score(test_true_labels, test_predictions)
        
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
