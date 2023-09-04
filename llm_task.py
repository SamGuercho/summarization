from ll_model import LLM
from project_object import ProjectObject
from prompts import PromptSummarize

class LLMTask(ProjectObject):

    def __init__(self, checkpoint, task, **kwargs):
        super().__init__()
        self.task = task
        self.llm = LLM(checkpoint, task, **kwargs)
        if task.upper() in self._config.get('PROMPT_TASK').keys():
            self.prompt = self._get_prompt()

    def _get_prompt(self):
        if self.task.upper() == "SUMMARIZE":
            return PromptSummarize(2)

    def predict(self, text):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids