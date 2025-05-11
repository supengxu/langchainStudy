# pip install --quiet langchain_experimental langchain_openai
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# 这是一个长文档，我们可以将其拆分。
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())

#拆分的默认方式是基于百分位数。在此方法中，计算所有句子之间的差异，然后任何大于50%的差异都会被拆分。
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)
docs = text_splitter.create_documents([knowledge])
print(docs[0].page_content)
