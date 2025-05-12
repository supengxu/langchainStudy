# 安装依赖
# pip install -U langchain-community faiss-cpu langchain-openai tiktoken

import os

from langchain import hub
# 如果您需要使用没有 AVX2 优化的 FAISS 进行初始化，请取消下面一行的注释
# os.environ['FAISS_NO_AVX2'] = '1'
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from llm_basic.llm_client import embeddingClient, llmClient

# 创建提示模板
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# 初始化 LLM 和 document chain
llm = llmClient()
document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# 加载文档并进行分割
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 创建嵌入并构建向量数据库
embeddings = embeddingClient()
db = FAISS.from_documents(docs, embeddings)

# 设置检索器
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度评分过滤
    search_kwargs={"score_threshold": 0.5}     # 降低阈值以提高召回率
)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 执行查询
query = "Pixar是一家什么公司"
result = retrieval_chain.invoke({"input": query})
print("Retrieved context:", result.get('context', 'No context found'))  # 打印检索到的上下文
print(result.get('answer', 'No answer found'))