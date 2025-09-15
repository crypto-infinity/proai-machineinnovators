
import os
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

DATA_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"


def download_and_extract_sentiment140(
        data_dir="./sentiment140_data",
        data_url=DATA_URL):
    """
    Download and extracts Sentiment140.

    Args:
        data_dir: the path in which the dataset will be extracted.
        data_url: the URL from which the dataset will be downloaded.

    Returns:
        None
    """

    print("Download Sentiment140 starting.")

    zip_path = os.path.join(data_dir, "trainingandtestdata.zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        r = requests.get(data_url)
        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        print("Download finished.")
    else:
        print("Dataset already downloaded, skipping.")


def load_sentiment140_as_hf_dataset() -> DatasetDict:
    """
    Loads Sentiment140 as HuggingFace Dataset() class.

    Args:
        None

    Returns:
        DatasetDict: the dataset in a dictionary format.
    """

    train_path = os.path.join(
        os.getcwd(),
        "sentiment140_data",
        "training.1600000.processed.noemoticon.csv"
    )
    test_path = os.path.join(
        os.getcwd(),
        "sentiment140_data",
        "testdata.manual.2009.06.14.csv"
    )

    col_names = ["target", "id", "date", "flag", "user", "text"]
    train_df = pd.read_csv(train_path, encoding="latin-1", names=col_names)
    test_df = pd.read_csv(test_path, encoding="latin-1", names=col_names)

    # Targets: 0=neg, 2=neu, 4=pos
    label_map = {0: 0, 2: 1, 4: 2}
    train_df = train_df[train_df["target"].isin(label_map.keys())]
    test_df = test_df[test_df["target"].isin(label_map.keys())]
    train_df["label"] = train_df["target"].map(label_map)
    test_df["label"] = test_df["target"].map(label_map)

    train_df = train_df.iloc[:5000]
    test_df = test_df.iloc[:5000]

    train_ds = Dataset.from_pandas(
        train_df[["text", "label"]],
        preserve_index=False
    )
    test_ds = Dataset.from_pandas(
        test_df[["text", "label"]],
        preserve_index=False
    )
    train_valid = train_ds.train_test_split(test_size=0.1)

    train_split = train_valid["train"].select(
        range(min(5000, len(train_valid["train"])))
    )
    valid_split = train_valid["test"].select(
        range(min(5000, len(train_valid["test"])))
    )
    test_ds = test_ds.select(range(min(5000, len(test_ds))))

    return DatasetDict({
        "train": train_split,
        "validation": valid_split,
        "test": test_ds
    })


def load_sentiment140_as_pandas_dataset() -> dict:
    """
    Loads Sentiment140 as Pandas dataframe.

    Args:
        None

    Returns:
        DatasetDict: the dataset in a dictionary format.
    """

    train_path = os.path.join(
        os.getcwd(),
        "sentiment140_data",
        "training.1600000.processed.noemoticon.csv"
    )
    test_path = os.path.join(
        os.getcwd(),
        "sentiment140_data",
        "testdata.manual.2009.06.14.csv"
    )

    col_names = ["target", "id", "date", "flag", "user", "text"]
    train_df = pd.read_csv(train_path, encoding="latin-1", names=col_names)
    test_df = pd.read_csv(test_path, encoding="latin-1", names=col_names)

    # Target: 0=neg, 2=neu, 4=pos
    label_map = {0: 0, 2: 1, 4: 2}
    train_df = train_df[train_df["target"].isin(label_map.keys())]
    test_df = test_df[test_df["target"].isin(label_map.keys())]
    train_df["label"] = train_df["target"].map(label_map)
    test_df["label"] = test_df["target"].map(label_map)

    # Limita a 5000 elementi per velocizzare il training
    train_df = train_df.iloc[:5000]
    test_df = test_df.iloc[:5000]

    train_df_split, valid_df_split = train_test_split(
        train_df,
        test_size=0.1,
        shuffle=True
    )
    return {
        "train": train_df_split.reset_index(drop=True),
        "validation": valid_df_split.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }
