import gdown
import chardet
import pandas as pd

def load_data():
    url = "https://drive.google.com/uc?id=1gg-OD4pQeyShfUt4w7QqxYQDOeS3trF0"
    output = "spam.csv"
    gdown.download(url, output, quiet=False)

    with open(output, "rb") as file:
        result = chardet.detect(file.read())
    encoding = result["encoding"]

    # Read file
    data = pd.read_csv(output, encoding=encoding)
    return data
