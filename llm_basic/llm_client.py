import os

from httpx import Client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def llmClient():
    custom_client = Client()

    return ChatOpenAI(
        model=os.environ["MODEL"],
        base_url=os.environ["BASE_URL"],
        http_client=custom_client
    )

def embeddingClient():
    return OpenAIEmbeddings(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("EMBEDDING_API_KEY"))