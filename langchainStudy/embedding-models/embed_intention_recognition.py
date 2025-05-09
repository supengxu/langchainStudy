from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# 数据导入
loader = TextLoader(".././resource/qa.txt", encoding="UTF-8")
docs = loader.load()
# 数据切分
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 创建embedding
embeddings = OpenAIEmbeddings()
# 通过向量数据库存储
vector = FAISS.from_documents(documents, embeddings)
# 查询检索
# 创建 prompt
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题：:
<context>
{context}
</context>
Question: {input}""")
# 创建模型
llm = ChatOpenAI()
# 创建 document 的chain 查询
document_chain = create_stuff_documents_chain(llm, prompt)
from langchain.chains import create_retrieval_chain

# 创建搜索chain 返回值为 VectorStoreRetriever
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# 执行请求
response = retrieval_chain.invoke({"input": "天气"})
print(response["answer"])
