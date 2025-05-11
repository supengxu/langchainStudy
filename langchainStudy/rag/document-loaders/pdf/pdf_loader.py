#pip install pypdf

from langchain_community.document_loaders import PyPDFLoader
file_path = ("../../resource/pytorch.pdf")
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
print(pages[0])