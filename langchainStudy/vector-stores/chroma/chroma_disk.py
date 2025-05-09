# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# 创建开源嵌入函数
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

query = "Pixar公司是做什么的?"

# 保存到磁盘
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# 从磁盘加载
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)
