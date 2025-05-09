# pip install --upgrade --quiet  pymilvus
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Zilliz
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

vector_db = Zilliz.from_documents(  # or Milvus.from_documents
    docs,
    embeddings,
    #存储到collection_1中
    collection_name="collection_2",
    connection_args={"uri": os.getenv("MILVUS_API_URL"), "token": os.getenv("MILVUS_API_KEY")},
    #drop_old=True,  # Drop the old Milvus collection if it exists
    auto_id=True,
)

query = "Pixar公司是做什么的?"
docs = vector_db.similarity_search(query)

print(docs[0].page_content)
