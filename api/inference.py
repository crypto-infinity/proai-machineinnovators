from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    
    def batch_inference(self, hf_dataset):
        dataset = load_dataset(hf_dataset)
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



