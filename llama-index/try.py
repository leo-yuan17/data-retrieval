from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./processed_data")

