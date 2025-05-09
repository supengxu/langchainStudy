# pip install -Uqq langchain-weaviate
import os
import weaviate
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth

embeddings = OpenAIEmbeddings()
# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Best practice: store your credentials in environment variables
wcd_url = os.environ["WCD_DEMO_URL"]
wcd_api_key = os.environ["WCD_DEMO_RO_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': openai_api_key},  # Replace with your OpenAI API key
    skip_init_checks=True
)
db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)
query = "Pixar"
docs = db.similarity_search(query, alpha=0)
print(docs[0])

weaviate_client.close()
