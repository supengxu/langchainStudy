# pip install bs4
from langchain_community.document_loaders import BSHTMLLoader

file_path = "../../resource/content.html"
loader = BSHTMLLoader(file_path, open_encoding="UTF-8")
data = loader.load()
print(data)
