from openai import OpenAI
import logging
import requests


class load_llm:

    def __init__(
            self,
            model: str = "meta/llama-3.1-405b-instruct",
            url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
            api_key:
        str = "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",
            **kwargs):  #kwargs: dict = {"temperature": 0.2, "top_p": 0.7, "frequency_penalty": 0, "presence_penalty": 0, "max_tokens": 1024, "stream": False}
        self.model = model
        self.url = url
        self.api_key = api_key
        self.kwargs = kwargs
        
    def get_response(self, prompt: str,question: str):
        self.payload = {
            "model": self.model,
            "temperature": self.kwargs.get("temperature", 0.2),
            "top_p": self.kwargs.get("top_p", 0.7),
            "frequency_penalty": self.kwargs.get("frequency_penalty", 0),
            "presence_penalty": self.kwargs.get("presence_penalty", 0),
            "max_tokens": self.kwargs.get("max_tokens", 4096),
            "stream": self.kwargs.get("stream", False),
            "messages": [{
                "role": "user",
                "content": f"You are a doctor,now you should answer the question:{question}based on the following information:{prompt}"
            }]
        }
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(self.url, json=self.payload, headers=self.headers)
        result = response.text#解析错误
        print(result)
        return result


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
        "messages": [{
            "role": "user",
            "content": "what is the article about?"
        }]
    }

    headers = {
        "accept":
        "application/json",
        "content-type":
        "application/json",
        "Authorization":
        "Bearer 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"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
