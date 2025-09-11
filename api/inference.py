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
    
    def batch_inference(self, hf_dataset, split):
        pass



