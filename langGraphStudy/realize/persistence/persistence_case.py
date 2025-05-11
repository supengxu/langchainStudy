# 导入所需的类型注解和模块
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from llm_basic.llm_client import llmClient


# 定义一个状态类，包含一个消息列表，消息列表带有 add_messages 注解
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 从 langchain_core.tools 导入工具装饰器
from langchain_core.tools import tool


# 定义一个名为 search 的工具函数，用于模拟网络搜索
@tool
def search(query: str):
    """Call to surf the web."""
    # 这是实际实现的占位符
    return ["The answer to your question lies within."]


# 将工具函数存入列表
tools = [search]

from langgraph.prebuilt import ToolNode

# 创建一个 ToolNode 实例，传入工具列表
tool_node = ToolNode(tools)

# 从 langchain_openai 导入 ChatOpenAI 模型
from langchain_openai import ChatOpenAI

# 创建一个 ChatOpenAI 模型实例，设置 streaming=True 以便可以流式传输 tokens
model = llmClient()

# 将工具绑定到模型上
bound_model = model.bind_tools(tools)

# 导入 Literal 类型
from typing import Literal


# 定义一个函数，根据状态决定是否继续执行
def should_continue(state: State) -> Literal["action", "__end__"]:
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # 如果没有函数调用，则结束
    if not last_message.tool_calls:
        return "__end__"
    # 否则继续执行
    return "action"


# 定义一个函数调用模型
def call_model(state: State):
    response = model.invoke(state["messages"])
    # 返回一个列表，因为这将被添加到现有列表中
    return {"messages": response}


# 从 langgraph.graph 导入 StateGraph 和 START
from langgraph.graph import StateGraph, START

# 定义一个新的图形工作流
workflow = StateGraph(State)

# 添加两个节点，分别是 agent 和 action
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 设置入口点为 agent
workflow.add_edge(START, "agent")

# 添加条件边，根据 should_continue 函数决定下一个节点
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 添加从 action 到 agent 的普通边
workflow.add_edge("action", "agent")

# 从 langgraph.checkpoint.memory 导入 MemorySaver
from langgraph.checkpoint.memory import MemorySaver

# 创建一个 MemorySaver 实例
memory = MemorySaver()

# 编译工作流，生成一个 LangChain Runnable
app = workflow.compile(checkpointer=memory)

# 将生成的图片保存到文件
graph_png = app.get_graph().draw_mermaid_png()
with open("persistence_case.png", "wb") as f:
    f.write(graph_png)

# 从 langchain_core.messages 导入 HumanMessage
from langchain_core.messages import HumanMessage

# 设置配置参数
config = {"configurable": {"thread_id": "2"}}

# 创建一个 HumanMessage 实例，内容为 "hi! I'm bob"
input_message = HumanMessage(content="hi! I'm bob")

# 在流模式下运行应用程序，传入消息和配置，逐个打印每个事件的最后一条消息
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 创建一个 HumanMessage 实例，内容为 "what is my name?"
input_message = HumanMessage(content="what is my name?")

# 在流模式下运行应用程序，传入消息和配置，逐个打印每个事件的最后一条消息
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 创建一个 HumanMessage 实例，内容为 "what is my name?"
input_message = HumanMessage(content="what is my name?")

# 在流模式下运行应用程序，传入消息和新的配置，逐个打印每个事件的最后一条消息
for event in app.stream(
        {"messages": [input_message]},
        {"configurable": {"thread_id": "3"}},
        stream_mode="values",
):
    event["messages"][-1].pretty_print()

# 创建一个 HumanMessage 实例，内容为 "You forgot?"
input_message = HumanMessage(content="You forgot??")

# 在流模式下运行应用程序，传入消息和原来的配置，逐个打印每个事件的最后一条消息
for event in app.stream(
        {"messages": [input_message]},
        {"configurable": {"thread_id": "2"}},
        stream_mode="values",
):
    event["messages"][-1].pretty_print()
