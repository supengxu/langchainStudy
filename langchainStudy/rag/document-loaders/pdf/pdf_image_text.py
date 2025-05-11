# pip install rapidocr-onnxruntime

from langchain_community.document_loaders import PyPDFLoader

file_path = ("../../resource/pytorch.pdf")
loader = PyPDFLoader(file_path, extract_images=True)
pages = loader.load()
#识别第9页图片文字
print(pages[8].page_content)