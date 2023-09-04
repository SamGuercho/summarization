from abc import ABC, abstractmethod, abstractproperty
from item_preprocessor import ItemPreprocessor
class _BasePromptClass(ABC):

    def __init__(self, task):
        self.start_prompt = ""
        self.end_prompt = ""
        self._prompt = None

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    def tokenize_prompt(self, text, item_preprocessor:ItemPreprocessor):
        example = {}
        prompt = self.start_prompt + '\n' + text + '\n' + self.end_prompt
        example['input_ids'] = item_preprocessor.tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        # example['labels'] = item_preprocessor.tokenizer(example["summary"], padding="max_length", truncation=True,
        #                               return_tensors="pt").input_ids

        return example


