class LLM:

    def __init__(self, checkpoint, task, **kwargs):
        self.model = self._get_model(checkpoint, task, **kwargs)

    def _get_model(self, checkpoint, task, **kwargs):
        if task == "summarize":
            from transformers import AutoModelForSeq2SeqLM
            return AutoModelForSeq2SeqLM.from_pretrained(checkpoint, **kwargs)
        elif task == "classify":
            from transformers import AutoModelForSequenceClassification
            return AutoModelForSequenceClassification.from_pretrained(checkpoint, **kwargs)

    def predict(self, inputs):
        return self.model(**inputs)

    def print_number_of_trainable_model_parameters(self):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in self.model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
