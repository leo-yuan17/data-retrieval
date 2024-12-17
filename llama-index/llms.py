from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
def deepseek_llm(**kwargs):
    llm = HuggingFaceLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct",tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct", **kwargs)
    return llm