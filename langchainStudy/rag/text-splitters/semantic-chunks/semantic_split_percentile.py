# pip install --quiet langchain_experimental langchain_openai
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# 这是一个长文档，我们可以将其拆分。
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90
)
docs = text_splitter.create_documents([knowledge])
print(docs[0].page_content)
print(docs[1].page_content)
print(len(docs))

