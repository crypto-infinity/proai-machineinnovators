
from transformers import pipeline


class Inference():
    """
    Provides inference methods for the backend API.
    """

    def __init__(self, model):
        self.pipe = pipeline(
            "text-classification",
            model=model
        )

    def single_inference(self, string, return_as_strings=False) -> tuple:
        """
        Uses self.pipe to make predictions on string.

        Args:
            string: the user input.
            return_as_strings: specifies if the output will be mapped to
                targets instead of strings.

        Returns:
            tuple: (label, score) of predictions, either str or numbers.
        """

        result = self.pipe(string)[0]
        if not return_as_strings:
            return result['label'], result['score']
        else:
            label_map = {"positive": 2, "neutral": 1, "negative": 0}
            return label_map[result["label"]], result['score']
