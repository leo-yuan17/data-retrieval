from openai import OpenAI
import logging
import requests


class load_llm:

    def __init__(
        self,
        model: str = "deepseek-chat",  # 模型名称，默认为"deepseek-chat"
        url:
        str = "https://api.deepseek.com",  # API地址，默认为"https://api.deepseek.com"
        api_key:
        str = "sk-16824413873f4defa607185b05663278",  # API密钥，默认为"sk-16824413873f4defa607185b05663278"
        **kwargs
    ):  #kwargs: dict = {"temperature": 0.2, "top_p": 0.7, "frequency_penalty": 0, "presence_penalty": 0, "max_tokens": 1024, "stream": False}
        self.model = model  # 模型名称
        self.url = url  # API地址
        self.api_key = api_key  # API密钥
        self.kwargs = kwargs  # 其他参数
        self.history = []  # 历史记录
        self.reset = False  # 是否重置
        self.messages = []  # 消息列表

    def get_response(self, prompt: str, question: str):
        # 如果模型是meta/llama-3.1-405b-instruct
        if self.model == "meta/llama-3.1-405b-instruct":
            # 设置payload
            self.payload = {
                "model":
                self.model,
                "temperature":
                self.kwargs.get("temperature", 0.2),
                "top_p":
                self.kwargs.get("top_p", 0.7),
                "frequency_penalty":
                self.kwargs.get("frequency_penalty", 0),
                "presence_penalty":
                self.kwargs.get("presence_penalty", 0),
                "max_tokens":
                self.kwargs.get("max_tokens", 4096),
                "stream":
                self.kwargs.get("stream", False),
                "messages": [{
                    "role":
                    "user",
                    "content":
                    f"You are a doctor,now you should answer the question:{question}based on the following information:{prompt}"
                }]
            }
            # 设置headers
            self.headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            # 发送post请求
            response = requests.post(self.url,
                                     json=self.payload,
                                     headers=self.headers)
            # 解析错误
            result = response.text  #解析错误
            print(response.text)
            result = response.json()
            print(result)
        # 如果模型是deepseek-chat
        elif self.model == "deepseek-chat":
            # 添加系统消息
            self.messages.append({
                "role":
                "system",
                "content":
                f"You are a doctor, now you should answer the question: {question} based on the following information: {prompt},you should answer the question in Chinese."
            })

            # 添加用户消息
            self.messages.append({"role": "user", "content": question})
            from openai import OpenAI

            # 创建OpenAI客户端
            client = OpenAI(api_key=self.api_key, base_url=self.url)

            # 发送请求
            response = client.chat.completions.create(model="deepseek-chat",
                                                      messages=self.messages,
                                                      stream=False,
                                                      max_tokens=8000)

            result = response.choices[0].message.content
            # 添加助手消息
            self.messages.append({"role": "assistant", "content": result})
        return result

    def _generate_response(self):
        """
        分段处理对话，确保 token 数量不超过限制
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.url)

        # 确保对话历史不超过最大 token 限制
        if self._count_tokens(self.messages) > self.token_limit:
            # 取最新的对话历史，不超过最大 token 数量
            trimmed_messages = self._trim_to_token_limit(self.messages)
        else:
            trimmed_messages = self.messages

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=trimmed_messages,  # 传递处理过的对话历史
            stream=False,
            max_tokens=self.max_tokens)

        return response

    def _trim_to_token_limit(self, messages):
        """
        截取对话历史，确保不超过最大 token 数量
        """
        total_tokens = self._count_tokens(messages)
        while total_tokens > self.token_limit:
            messages.pop(0)  # 丢弃最早的对话
            total_tokens = self._count_tokens(messages)
        return messages

    def _count_tokens(self, messages):
        """
        计算对话历史的 token 数量
        """
        # 这里可以根据你的实际情况使用 OpenAI 的 `tiktoken` 库来计算 token 数量
        total_tokens = 0
        for message in messages:
            total_tokens += len(message["content"].split())  # 估算 token 数量
        return total_tokens
