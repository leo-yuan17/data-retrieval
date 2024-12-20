from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.nvidia import NVIDIA
def nvidia_llm(**kwargs):
    llm = NVIDIA(model="meta/llama-3.1-405b-instruct",
                 api_key="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",
                 **kwargs)
    return llm