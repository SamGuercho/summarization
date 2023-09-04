import os
import torch

from llm_task_handler import LLMTaskHandler
from project_object import ProjectObject
from summarizer import Summarizer

if __name__ == "__main__":

    llm_manager = LLMTaskHandler("google/flan-t5-xxl", torch_dtype=torch.bfloat16)
    parameters = llm_manager.llm.print_number_of_trainable_model_parameters()
    llm_manager.llm_task.prompt.tokenize_prompt()
    # dataset_name = "knkarthick/dialogsum"
    # model_name = "google/flan-t5-xl"
    # # model_name = "gpt-3.5-turbo-16k"
    # dataset = load_dataset(dataset_name)
    # projectObj = ProjectObject()
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = projectObj._secrets.get('HUGGING_FACE').get('API_KEY')
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1e-10})
    # # llm = ChatOpenAI(temperature=0, model_name=model_name)
    # summarizer = Summarizer(llm)
    # print()

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

    from datasets import load_dataset

    dataset = load_dataset("JulesBelveze/tldr_news")

    input_text = "translate English to German: How old are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

    import os

    projectObj = ProjectObject()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = projectObj._secrets.get('HUGGING_FACE').get('API_KEY')
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxxxxxxxxxxxxxxxxxx"
    from langchain import PromptTemplate, HuggingFaceHub, LLMChain

    template = """Question: {question}

    Answer: Let's think step by step."""
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 1e-10})
    prompt = """Question: Can Barack Obama have a conversation with George Washington?

    Let's think step by step.

    Answer: """
    answer = llm(prompt)
    question = "Can you tell me when was Google founded?"
    prompt_template = PromptTemplate.from_template(template=template)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )
    print(llm_chain.run(question))
    print()