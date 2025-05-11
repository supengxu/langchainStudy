# 从langgraph.graph模块导入START和StateGraph
from langgraph.graph import START, StateGraph

# 定义一个节点函数my_node，接收状态和配置，返回新的状态
def my_node(state, config):
    return {"x": state["x"] + 1,"y": state["y"] + 2}

# 创建一个状态图构建器builder，使用字典类型作为状态类型
builder = StateGraph(dict)
# 向构建器中添加节点my_node，节点名称将自动设置为'my_node'
builder.add_node(my_node)  # node name will be 'my_node'
# 添加一条边，从START到'my_node'节点
builder.add_edge(START, "my_node")
# 编译状态图，生成可执行的图
graph = builder.compile()
print(graph)
# 调用编译后的图，传入初始状态{"x": 1}
print(graph.invoke({"x": 1,"y":2}))
