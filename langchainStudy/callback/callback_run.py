# 导入所需的类型提示和类
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate

from llm_basic.llm_client import llmClient


# 定义一个日志处理器类，继承自BaseCallbackHandler
class LoggingHandler(BaseCallbackHandler):
    # 当聊天模型开始时调用的方法
    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")  # 打印“Chat model started”

    # 当LLM结束时调用的方法
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")  # 打印“Chat model ended, response: {response}”

    # 当链开始时调用的方法
    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain started, inputs:{inputs}")  # 打印“Chain {serialized.get('name')} started”

    # 当链结束时调用的方法
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")  # 打印“Chain ended, outputs: {outputs}”

class LoggingHandler2(BaseCallbackHandler):
    # 当聊天模型开始时调用的方法
    def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat2 model started")  # 打印“Chat model started”

    # 当LLM结束时调用的方法
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat2 model ended, response: {response}")  # 打印“Chat model ended, response: {response}”

    # 当链开始时调用的方法
    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain2 started, inputs:{inputs}")  # 打印“Chain {serialized.get('name')} started”

    # 当链结束时调用的方法
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain2 ended, outputs: {outputs}")  # 打印“Chain ended, outputs: {outputs}”

# 创建一个包含LoggingHandler实例的回调列表
callbacks = [LoggingHandler(),LoggingHandler2()]

# 实例化一个ChatOpenAI对象，使用gpt-4模型
llm = llmClient()

# 创建一个聊天提示模板，模板内容为“What is 1 + {number}?”
prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

# 将提示模板和LLM组合成一个链
chain = prompt | llm

# 调用链的invoke方法，传入参数number为"2"，并配置回调
result = chain.invoke({"number": "2"}, config={"callbacks": callbacks})
print(result)
