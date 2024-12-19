from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import torch.optim as optim

def encode_texts(texts, tokenizer, max_len):
    return tokenizer(
        list(texts), max_length=max_len, padding="max_length", truncation=True, return_tensors="pt"
    )

def train_bert(X_train, X_test, y_train, y_test):
    tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
    train_encodings = encode_texts(X_train, tokenizer, 64)
    test_encodings = encode_texts(X_test, tokenizer, 64)

    train_data = TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(y_train.values, dtype=torch.float32)
    )
    test_data = TensorDataset(
        test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(y_test.values, dtype=torch.float32)
    )

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-small-uncased", num_labels=1).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    y_pred_bert = []
    y_true_bert = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)

            predictions = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
            y_pred_bert.extend(predictions > 0.5)
            y_true_bert.extend(labels.cpu().numpy())

    test_f1 = f1_score(y_true_bert, y_pred_bert)
    test_accuracy = accuracy_score(y_true_bert, y_pred_bert)

    print("BERT Accuracy:", test_accuracy)
    print("BERT F1-Score:", test_f1)