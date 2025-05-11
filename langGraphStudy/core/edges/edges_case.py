from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# 初始化 StateGraph，状态类型为字典
graph = StateGraph(dict)


# 定义节点
def my_node(state: dict, config: RunnableConfig):
    print("In node: ", config["configurable"]["user_id"])
    return {"results": f"Hello, {state['input']}!"}


def other_node(state: dict):
    return state


def node_a(state: dict):
    return {"result": "This is node B"}


def node_b(state: dict):
    return {"result": "This is node B"}


def node_c(state: dict):
    return {"result": "This is node C"}


# 将节点添加到图中
graph.add_node("my_node", my_node)
graph.add_node("other_node", other_node)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

# 普通边
graph.add_edge("my_node", "other_node")
graph.add_edge("other_node", "node_a")


# 条件边和条件路由函数
def routing_function(state: dict):
    # 假设我们根据 state 中的某个键值来决定路由
    # 如果 state 中有 'route_to_b' 且其值为 True，则路由到 node_b，否则路由到 node_c
    return state.get('route_to_b', False)

graph.add_edge("node_b", END)
graph.add_edge("node_c", END)

#条件边
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
graph.add_edge(START, "my_node")
#条件入口点
#graph.add_conditional_edges(START,  routing_function, {True: "node_b", False: "node_c"})


#条件入口点

def routing_my(state: dict):
    # 假设我们根据 state 中的某个键值来决定路由
    # 如果 state 中有 'route_to_b' 且其值为 True，则路由到 node_b，否则路由到 node_c
    return state.get('route_to_my', False)


# 条件入口点
# graph.add_conditional_edges(START, routing_my,{True: "my_node", False: "other_node"})

# 编译图
app = graph.compile()
# 将生成的图片保存到文件
graph_png = app.get_graph().draw_mermaid_png()
with open("edges_case.png", "wb") as f:
    f.write(graph_png)
