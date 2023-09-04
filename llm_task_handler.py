from item_preprocessor import ItemPreprocessor
from llm_task import LLMTask
from project_object import ProjectObject


class LLMTaskHandler(ProjectObject):

    def __init__(self, checkpoint, task='summarize', **kwargs):
        super().__init__()
        self.input_processor = ItemPreprocessor(checkpoint, **kwargs)
        self.llm_task = LLMTask(checkpoint, task, **kwargs)
        self.llm = self.llm_task.llm