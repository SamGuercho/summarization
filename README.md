# summarization

Welcome to the summarization project, once again just a project to hand on the fine tuning of a summarizer model.

I inspired myself from ybagoury/flan-t5-base-tldr_news from the HuggingFace Community.

In this project we will:
1) Create an LLM architecture which can
   1) Preprocess and Embed text
   2) Call different tasks (we will implement only summarization)
   3) Fine Tune itself, calling a PEFT method (LORA, Soft Prompt)
2) Fine tune a model to summarize financial news: JulesBelveze/tldr_news
3) Evaluate performance of the model before and after fine-tuning