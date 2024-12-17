from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_embedding_model(**kwargs):
    # Load the HuggingFace embedding model
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5",cache_folder="./cache", **kwargs)
    
    return embedding_model