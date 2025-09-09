from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

class Inference():

    def __init__(self, model):
        self.pipe = pipeline(
            "text-classification",
            model=model
        )

    def infer_wrap(self, record):
        label, score = self.single_inference(record["text"])
        return {"label": label, "score": score}
    
    def single_inference(self, string):
        result = self.pipe(string)[0]
        return result['label'], result['score']
    
    def batch_inference(self, dataset):
        y_true = list(dataset['label'])

        results = dataset.map(self.infer_wrap)
        y_pred = results["label"]

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, 
            average="weighted", 
            zero_division=0
        )

        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return {"results": results, "metrics": metrics}

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

