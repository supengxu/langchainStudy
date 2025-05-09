#pip install langchain-openai

from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        "嗨！",
        "哦，你好！",
        "你叫什么名字？",
        "我的朋友们叫我World",
        "Hello World！"
    ]
)
embedded_query = embeddings_model.embed_query("对话中提到的名字是什么？")
#查询嵌入的前五个值
print(embedded_query[:5])