# pip install --upgrade --quiet langchain-pinecone langchain pinecone-notebooks langchain-openai
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
#pip install pinecone-client
from pinecone import Pinecone, ServerlessSpec
import os
import time
from langchain_community.document_loaders import TextLoader


loader = TextLoader("../../resource/knowledge.txt",encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-index"  # 如果需要，可以更改
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
query = "Pixar"

docs = docsearch.similarity_search(query)
print(docs[0].page_content)
