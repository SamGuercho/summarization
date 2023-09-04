from transformers import AutoTokenizer

class ItemPreprocessor:

    def __init__(self, checkpoint, **kwargs):
        self.tokenizer = self._get_tokenizer(checkpoint, **kwargs)

    def _get_tokenizer(self, checkpoint, **kwargs):
        return AutoTokenizer.from_pretrained(checkpoint, **kwargs)

    def tokenize(self, raw_inputs, return_tensors='pt', **kwargs):
        """
        Tokenizes raw text via the AutoTokenizer object.
        :param args:
        - padding
        - truncation
        - return_tensors
        :return:
        """
        return self.tokenizer(raw_inputs, return_tensors=return_tensors, **kwargs)