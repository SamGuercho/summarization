from item_preprocessor import ItemPreprocessor
from llm_task import LLMTask
from project_object import ProjectObject
from transformers import GenerationConfig

class LLMTaskHandler(ProjectObject):

    def __init__(self, checkpoint, task='summarize', **kwargs):
        super().__init__()
        self.input_processor = ItemPreprocessor(checkpoint, **kwargs)
        self.llm_task = LLMTask(checkpoint, task, **kwargs)
        self.llm = self.llm_task.llm

    def get_tokenized_prompt(self, text):
        return self.llm_task.prompt.tokenize_prompt(text, self.input_processor)

    def predict(self, text:str, **kwargs):
        if not kwargs:
            config = GenerationConfig(max_new_tokens=200,
                         num_beams=1)
        else:
            config = GenerationConfig(**kwargs)
        full_prompt_text = self.llm_task.prompt.get_full_prompt(text)
        input_ids = self.input_processor.tokenizer(full_prompt_text, return_tensors="pt").input_ids
        original_model_outputs = self.llm.model.generate(input_ids=input_ids,  generation_config=config)
        original_model_text_output = self.input_processor.tokenizer.decode(original_model_outputs[0],
                                                                           skip_special_tokens=True)
        return original_model_text_output