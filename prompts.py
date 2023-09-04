from base_prompt import _BasePromptClass

class PromptSummarize(_BasePromptClass):

    def __init__(self, max_sentences=5):
        self.start_prompt = f"Summarize the following conversation in max {max_sentences} sentences:"
        self.end_prompt = "Summary:"
        self.prompt_seperator = "\n\n"
