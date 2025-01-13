import logging
from transformers import AutoTokenizer, AutoModel
class load_embeddingmodel:

    def __init__(self, model_name: str = "maidalun1020/bce-embedding-base_v1", model_type="huggingface", **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.kwargs = kwargs
    
    def get_embedding_model(self):
        if self.model_type == "huggingface":
            return AutoModel.from_pretrained(self.model_name, **self.kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def get_tokenizer(self):
        if self.model_type == "huggingface":
            return AutoTokenizer.from_pretrained(self.model_name, **self.kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")