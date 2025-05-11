# pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain.globals import set_debug
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain

from llm_basic.llm_client import embeddingClient, llmClient

# 打印调试日志
set_debug(False)

# 创建一个 WebBaseLoader 对象，用于从指定网址加载文档
loader = WebBaseLoader(
    web_paths=("https://www.beisen.com",)
)
# 加载文档
docs = loader.load()
# 创建一个 RecursiveCharacterTextSplitter 对象，用于将文档拆分成较小的文本块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 将文档拆分成文本块
splits = text_splitter.split_documents(docs)
# 创建一个 Chroma 对象，用于存储文本块的向量表示
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddingClient())
# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()

# 定义系统提示词模板
system_prompt = (
    "您是一个用于问答任务的助手。"
    "使用以下检索到的上下文片段来回答问题。"
    "如果您不知道答案，请说您不知道。"
    "最多使用三句话，保持回答简洁。"
    "\n\n"
    "{context}"
)
# 创建一个 ChatPromptTemplate 对象，用于生成提示词
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 创建一个带有聊天历史记录的提示词模板
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 创建一个 ChatOpenAI 对象，表示聊天模型
llm = llmClient()
# 创建一个问答链
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# 创建一个检索链，将检索器和问答链结合
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 定义上下文化问题的系统提示词
contextualize_q_system_prompt = (
    "给定聊天历史和最新的用户问题，"
    "该问题可能引用聊天历史中的上下文，"
    "重新构造一个可以在没有聊天历史的情况下理解的独立问题。"
    "如果需要，不要回答问题，只需重新构造问题并返回。"
)
# 创建一个上下文化问题提示词模板
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
    # 创建一个带有历史记录感知的检索器
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 创建一个带有聊天历史记录的问答链
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# 创建一个带有历史记录感知的检索链
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 创建一个字典，用于存储聊天历史记录
store = {}


# 定义一个函数，用于获取指定会话的聊天历史记录
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建一个 RunnableWithMessageHistory 对象，用于管理有状态的聊天历史记录
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 调用有状态的检索链，获取回答
response = conversational_rag_chain.invoke(
    {"input": "什么是任务分解?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # 在 `store` 中构建一个键为 "abc123" 的键。
)["answer"]
print(response)

# 再次调用有状态的检索链，获取另一个回答
response = conversational_rag_chain.invoke(
    {"input": "我刚刚问了什么?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]
print(response)

# 再次调用有状态的检索链，换一个session_id
response = conversational_rag_chain.invoke(
    {"input": "我刚刚问了什么?"},
    config={"configurable": {"session_id": "abc456"}},
)["answer"]
print(response)

#打印存储在会话 "abc123" 中的所有消息
for message in store["abc123"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"
    print(f"{prefix}: {message.content}\n")
