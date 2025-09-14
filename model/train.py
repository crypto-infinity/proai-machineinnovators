
import sys
import os

# Pipeline execution fix from ./ or ./model
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.dataset import (
    download_and_extract_sentiment140,
    load_sentiment140_as_hf_dataset,
)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "./results/hf_model"


print("Importing model and tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
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
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
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
