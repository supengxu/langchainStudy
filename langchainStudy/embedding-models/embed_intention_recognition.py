from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from llm_basic.llm_client import llmClient, embeddingClient

# 数据加载
loader = TextLoader(".././resource/qa.txt", encoding="UTF-8")
docs = loader.load()

# 文本切分
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 创建 embedding 并构建向量数据库
embeddings = embeddingClient()
vectorstore = FAISS.from_documents(documents, embeddings)  # 更清晰的变量名

# 构建 prompt
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题：:
<context>
{context}
</context>
Question: {input}""")

# 初始化 LLM 和 document chain
llm = llmClient()
document_chain = create_stuff_documents_chain(llm, prompt)

# 设置检索器
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 执行基本检索
response = retrieval_chain.invoke({"input": "学习任务"})
print(response["answer"])
