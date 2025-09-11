
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "./results/hf_model"
DATA_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
DATA_DIR = "./sentiment140_data"

def download_and_extract_sentiment140():
    import zipfile
    import requests
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "trainingandtestdata.zip")
    if not os.path.exists(zip_path):
        r = requests.get(DATA_URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

def load_sentiment140_as_hf_dataset():
    train_path = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
    test_path = os.path.join(DATA_DIR, "testdata.manual.2009.06.14.csv")
    col_names = ["target", "id", "date", "flag", "user", "text"]
    train_df = pd.read_csv(train_path, encoding="latin-1", names=col_names)
    test_df = pd.read_csv(test_path, encoding="latin-1", names=col_names)

    # Mappa i target: 0=neg, 2=neu, 4=pos
    label_map = {0: 0, 2: 1, 4: 2}
    train_df = train_df[train_df["target"].isin(label_map.keys())]
    test_df = test_df[test_df["target"].isin(label_map.keys())]
    train_df["label"] = train_df["target"].map(label_map)
    test_df["label"] = test_df["target"].map(label_map)

    # Limita a 5000 elementi per velocizzare il training
    train_df = train_df.iloc[:5000]
    test_df = test_df.iloc[:5000]

    # Crea Dataset HuggingFace
    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False)
    
    # Split validation dal train
    train_valid = train_ds.train_test_split(test_size=0.1, seed=42)

    # Limita anche validation e train split a max 5000 elementi (in realtà saranno 4500 train, 500 valid)
    train_split = train_valid["train"].select(range(min(5000, len(train_valid["train"]))))
    valid_split = train_valid["test"].select(range(min(5000, len(train_valid["test"]))))
    test_ds = test_ds.select(range(min(5000, len(test_ds))))

    return DatasetDict({
        "train": train_split,
        "validation": valid_split,
        "test": test_ds
    })

print("Importing model and tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

print("Downloading and preparing Sentiment140 dataset.")
download_and_extract_sentiment140()
dataset = load_sentiment140_as_hf_dataset()

print("Preprocessing dataset.")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

print("Setting arguments for training.")
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

print("Training the model.")
trainer.train()

print("Saving the model.")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)