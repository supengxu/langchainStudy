# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from llm_basic.llm_client import llmClient, embeddingClient

# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# 创建开源嵌入函数
embedding_function = embeddingClient()
query = "Pixar公司是做什么的?"
# 创建简单的 ids
ids = [str(i) for i in range(1, len(docs) + 1)]
# 添加数据
example_db = Chroma.from_documents(docs, embedding_function, ids=ids)
docs = example_db.similarity_search(query,  k=3,)
# 更新文档的元数据
docs[0].metadata = {
    "source": "../../resource/knowledge.txt",
    "new_value": "hello world",
}
print("更新前内容：")
print(example_db._collection.get(ids=[ids[0]]))
example_db.update_document(ids[0], docs[0])
print("更新后内容：")
print(example_db._collection.get(ids=[ids[0]]))
# 删除最后一个文档
print("删除前计数", example_db._collection.count())
print(example_db._collection.get(ids=[ids[-1]]))
example_db._collection.delete(ids=[ids[-1]])
print("删除后计数", example_db._collection.count())
print(example_db._collection.get(ids=[ids[-1]]))
