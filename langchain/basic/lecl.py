# Requires:
import os

# pip install langchain docarray tiktoken
os.environ['LANGCHAIN_TRACING_V2']="true"

from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_openai.chat_models import ChatOpenAI

from langchain_openai.embeddings import OpenAIEmbeddings

from llm_basic.llm_client import llmClient

# 初始化一个向量存储


vectorstore = DocArrayInMemorySearch.from_texts(

    ["harrison worked at kensho", "bears like to eat honey"],

    embedding=OpenAIEmbeddings(
        model="embedding-3",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("EMBEDDING_API_KEY")),

)

# 生成检索索引器


retriever = vectorstore.as_retriever()

# 一个对话模板，内含2个变量context和question


template = """Answer the question based only on the following context:



{context}







Question: {question}



"""

# 基于模板生成提示


prompt = ChatPromptTemplate.from_template(template)

# 基于对话openai生成模型


model = llmClient()

# 生成输出解析器


output_parser = StrOutputParser()

# 将检索索引器和输入内容（问题）生成检索


setup_and_retrieval = RunnableParallel(

    {"context": retriever, "question": RunnablePassthrough()}

)

# 打印 Embedding 结果示例
sample_text = "harrison worked at kensho"
embedding_vector = vectorstore.embedding.embed_query(sample_text)
print(f"'{sample_text}' 的 Embedding 向量为:")
print(embedding_vector)

# 建立增强链
chain = setup_and_retrieval | prompt | model | output_parser

print(chain.invoke("where did harrison work?"))
