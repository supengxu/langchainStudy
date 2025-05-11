from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = (
    "../../resource/doc_search.csv"
)
loader = CSVLoader(file_path=file_path,encoding="UTF-8")
data = loader.load()
for record in data[:2]:
    print(record)
