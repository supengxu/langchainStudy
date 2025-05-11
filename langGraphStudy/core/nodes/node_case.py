from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.graph import END

# 初始化 StateGraph，状态类型为字典
graph = StateGraph(dict)

# 定义节点
def my_node(state: dict, config: RunnableConfig):
    print("In node: ", config["configurable"]["user_id"])
    return {"results": f"Hello, {state['input']}!"}

def my_other_node(state: dict):
    return state

# 将节点添加到图中
graph.add_node("my_node", my_node)
graph.add_node("other_node", my_other_node)

# 连接节点以确保它们是可达的
graph.add_edge(START, "my_node")
graph.add_edge("my_node", "other_node")

graph.add_edge("other_node", END)

# 编译图
print(graph.compile())

app = graph.compile();
# 将生成的图片保存到文件
graph_png = app.get_graph().draw_mermaid_png()
with open("node_case.png", "wb") as f:
    f.write(graph_png)