import os
import  asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import llm_basic
from llm_basic.llm_client import llmClient

# 初始化向量数据库并添加文本
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("EMBEDDING_API_KEY")
    ),
)

retriever = vectorstore.as_retriever()

# 定义模板并创建 prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 使用自定义 LLM 模型
model =llmClient()
# 创建链
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
)

# 使用 stream_events 监听链的执行过程
config = {"configurable": {"session_id": "abc123"}}
input_data = {"question": "where did harrison work?"}
#
# for event in model.stream("hello"):
#     print(f"Event Type: {event['event']}")
#     print(f"Name: {event['name']}")
#     print(f"Data: {event['data']}\n")
