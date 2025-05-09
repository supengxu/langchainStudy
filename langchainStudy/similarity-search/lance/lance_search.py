# pip install -U langchain-openai
# pip install lancedb
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

docsearch = LanceDB.from_documents(documents, embeddings)
query = "Pixar公司是做什么的?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
print(docs[0].metadata)

tbl = docsearch.get_table()
print("tbl:", tbl)
pd_df = tbl.to_pandas()
pd_df.to_csv("docsearch.csv", index=False)