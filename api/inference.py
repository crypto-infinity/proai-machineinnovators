
from transformers import pipeline


class Inference():

    def __init__(self, model):
        self.pipe = pipeline(
            "text-classification",
            model=model
        )

    def single_inference(self, string, return_as_strings=False) -> tuple:
        result = self.pipe(string)[0]
        if not return_as_strings:
            return result['label'], result['score']
        else:
            label_map = {"positive": 2, "neutral": 1, "negative": 0}
            return label_map[result["label"]], result['score']
