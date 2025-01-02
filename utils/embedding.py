from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging


class Embedding:

    def __init__(self, model_name: str, model_type="huggingface", **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.kwargs = kwargs
        if model_type == "huggingface":
            self.embedding = HuggingFaceEmbedding(model_name=model_name,
                                                  **kwargs)
            logging.info(f"Embedding model {model_name} loaded")
            return self.embedding
        else:
            raise ValueError("Model type not supported")
