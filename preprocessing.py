import gdown
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocessing():
    url = "https://drive.google.com/uc?id=1gg-OD4pQeyShfUt4w7QqxYQDOeS3trF0"
    output = "spam.csv"
    gdown.download(url, output, quiet=False)

    with open(output, "rb") as file:
        result = chardet.detect(file.read())
    encoding = result["encoding"]

    # Read file
    data = pd.read_csv(output, encoding=encoding)

    # Drop unnecessary columns
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data.columns = ["label", "text"]

    # Separate features and labels
    X = data["text"]
    y = data["label"]

    # Map labels to numeric values
    label_mapping = {"ham": 0, "spam": 1}
    y = y.map(label_mapping)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
