
from transformers import pipeline


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
