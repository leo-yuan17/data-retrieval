import sys
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.storage import StorageContext
from llms import nvidia_llm

sys.path.append("./utils")
from embedding_2 import Embedding # type: ignore
from llama_index.core import Settings
import os
import json

Settings.llm = nvidia_llm()
embed_model = Embedding()
Settings.embed_model = embed_model.get_embedding_model()
Settings.chunk_size = 1024


def creat_vector_query_engine():
    datas = SimpleDirectoryReader(input_dir="./data").load_data()
    vector_query_engine_tools = []
    for data in datas:
        vector_index = VectorStoreIndex.from_documents([data],
                                                       show_progress=True)
        vector_index.storage_context.persist(persist_dir="./processed_data")
        vector_query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=vector_index.as_query_engine())
        vector_query_engine_tools.append(vector_query_engine_tool)
    return vector_query_engine_tools


def create_route_query_engine():
    vector_query_engine_tool = creat_vector_query_engine()

    query_engine = RouterQueryEngine.from_defaults(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=vector_query_engine_tool)
    return query_engine


def load_vector_query_tools():
    vector_query_engine_tools = []
    vector_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./processed_data"))
    vector_query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine())
    vector_query_engine_tools.append(vector_query_engine_tool)
    return vector_query_engine_tools


if __name__ == "__main__":
    query_engine = create_route_query_engine()
    response = query_engine.query(f"这篇文章讲了什么?用中文回答我")
    print(response)
