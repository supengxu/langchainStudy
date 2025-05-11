from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from llm_basic.llm_client import llmClient


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")



prompt = ChatPromptTemplate.from_messages(["给我讲个关于{animal}的笑话，限制100个字"])
# 为启用流式处理，我们在ChatModel构造函数中传入`streaming=True`
# 另外，我们将自定义处理程序作为回调参数的列表传入
model = llmClient()
chain = prompt | model

# 使用 stream() 方法获取流式输出
for chunk in chain.stream({"animal": "薄熙来"}):
    print(chunk.content, end="", flush=True)
