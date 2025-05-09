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
    location=":memory:",  # 本地模式，仅内存存储
    collection_name="my_documents",
)

query = "Pixar公司是做什么的?"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)
