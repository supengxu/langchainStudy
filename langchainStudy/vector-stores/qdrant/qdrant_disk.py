# pip install --upgrade --quiet  langchain-qdrant langchain-openai langchain
# pip install langchain-qdrant
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    path="./qdrant_db",
    collection_name="my_documents",
    #重新创建集合，如果集合已经存在，则会重用该集合。将force_recreate设置为True允许删除旧集合并从头开始。
    #force_recreate=True,
)

query = "Pixar公司是做什么的?"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)
