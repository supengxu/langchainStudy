# 安装依赖
# pip install -U langchain-community faiss-cpu langchain-openai tiktoken

import os

from langchain import hub
# 如果您需要使用没有 AVX2 优化的 FAISS 进行初始化，请取消下面一行的注释
# os.environ['FAISS_NO_AVX2'] = '1'
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from llm_basic.llm_client import embeddingClient, llmClient

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题你可以做润色总结,:
<context>
{context}
</context>
Question: {input}""")
# 初始化 LLM 和 document chain
llm = llmClient()
document_chain = create_stuff_documents_chain(llm, prompt)

# 加载文档并进行分割
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 创建嵌入并构建向量数据库
embeddings = embeddingClient()
db = FAISS.from_documents(docs, embeddings)

# 设置检索器

# 创建关键词检索器
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 2  # 返回 top-2 的关键词匹配文档

# 创建向量检索器
vector_retriever = db.as_retriever(search_kwargs={"k": 2})  # 使用正确的变量名


# 创建混合检索器
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 权重各占一半
)
retrieval_chain = create_retrieval_chain(hybrid_retriever, document_chain)
# 执行混合查询
query = "Pixar公司是做什么的?"
results = hybrid_retriever.get_relevant_documents(query)
print("Hybrid search results:", results)

# 执行查询
query = "Pixar是一家什么公司"
result = retrieval_chain.invoke({"input": query})
print("Retrieved context:", result.get('context', 'No context found'))  # 打印检索到的上下文
print(result.get('answer', 'No answer found'))