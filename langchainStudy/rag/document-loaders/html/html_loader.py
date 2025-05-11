# pip install "unstructured[html]"
from langchain_community.document_loaders import UnstructuredHTMLLoader

file_path = "../../resource/content.html"
loader = UnstructuredHTMLLoader(file_path, encodings="UTF-8")
data = loader.load()
print(data)
