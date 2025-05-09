# pip install langchain-chroma
import chromadb
from langchain_chroma import Chroma
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

persistent_client = chromadb.PersistentClient()
# 创建开源嵌入函数
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
collection = persistent_client.get_or_create_collection("collection_1")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])
langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="collection_1",
    embedding_function=embedding_function,
)
print("在集合中有", langchain_chroma._collection.count(), "个文档")
