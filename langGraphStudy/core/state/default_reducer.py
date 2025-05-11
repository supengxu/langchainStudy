from typing import TypedDict, List, Dict, Any


class State(TypedDict):
    foo: int
    bar: List[str]


def update_state(current_state: State, updates: Dict[str, Any]) -> State:
    # 创建一个新的状态字典
    new_state = current_state.copy()
    # 更新状态字典中的值
    new_state.update(updates)
    return new_state


# 初始状态
state: State = {"foo": 1, "bar": ["hi"]}

# 第一个节点返回的更新
node1_update = {"foo": 2}
state = update_state(state, node1_update)
print(state)  # 输出: {'foo': 2, 'bar': ['hi']}

# 第二个节点返回的更新
node2_update = {"bar": ["bye"]}
state = update_state(state, node2_update)
print(state)  # 输出: {'foo': 2, 'bar': ['bye']}
