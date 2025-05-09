# pip install -U langchain-community faiss-cpu langchain-openai tiktoken
# pip install faiss-cpu
# 如果您需要使用没有 AVX2 优化的 FAISS 进行初始化，请取消下面一行的注释
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
query = "Pixar公司是做什么的?"
#保存索引
db.save_local("faiss_index")
#读取索引
new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
docs = new_db.similarity_search(query)

print(docs[0])
