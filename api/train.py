from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


class ModelTrainer():
    """
    """

    def __init__(self, model_name):
        """
        """
        # 2. Carica tokenizer e modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
    )
  
    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )

    def retrain(self) -> dict:
        """
        """
        # 1. Carica il dataset (esempio: un dataset custom su HuggingFace Hub)
        dataset = load_dataset("nome_utente/nome_dataset")  # restituisce un DatasetDict
        tokenized_datasets = dataset.map(self.preprocess_function, batched=True)

        # 4. Imposta Trainer
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation"),
        )

        trainer.train()

        return {}

