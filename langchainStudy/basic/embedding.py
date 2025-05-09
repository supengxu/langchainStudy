import os

from dotenv import load_dotenv
# 加载.env文件中的环境变量到os.environ
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
def testembedding():
    embeddings = OpenAIEmbeddings(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("EMBEDDING_API_KEY"))

    result = embeddings.embed_documents([
        "你好",
        "世界",
        " LangChain 是一个开源项目。",
        " LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。",
        " LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain",
        " LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个开源项目。 LangChain 是一个"
    ])
    return result
res = testembedding()
print(len(res),len(res[0]))

