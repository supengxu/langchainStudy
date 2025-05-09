from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from llm_basic.llm_client import llmClient


def chat_basic():
    model = llmClient()
    messages = [
        SystemMessage(content="将以下内容从英语翻译成中文"),
        HumanMessage(content="It's a nice day today"),
    ]
    parser = StrOutputParser()
    chain = model|parser
    # 使用parser处理model返回的结果
    response = chain.invoke(messages)
    print(response)

chat_basic()

