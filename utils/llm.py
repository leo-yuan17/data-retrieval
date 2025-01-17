from openai import OpenAI
import logging


class llm:

    def __init__(self,
                 model_name: str,
                 api_key: str = "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",
                 base_url: str = "https://integrate.api.nvidia.com/v1",
                 model_type: str = "OpenAI",
                 **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=self.api_key, **kwargs)
        if self.client is not None:
            logging.info(f"LLM model has been loaded successfully.")
        

    def generate(self, query: str):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": "基于这个{query}问题，用中文回答我"
            }],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True)
        return completion

    def print_response(self, completion):
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

if __name__ == "__main__":
    import requests

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    payload = {
        "model": "meta/llama-3.1-405b-instruct",
        "temperature": 0.2,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1024,
        "stream": False,
        "messages": [{"role": "user", "content": "what is the article about?"}]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer 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"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
