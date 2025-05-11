from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langgraph.utils import runnable


def get_message_history(session_id: str, REDIS_URL=None) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)
with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)
with_message_history.invoke({"input": "What is the capital of France?"})