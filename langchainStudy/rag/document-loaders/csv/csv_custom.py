from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = (
    "../../resource/doc_search.csv"
)
loader = CSVLoader(
    file_path=file_path,
    encoding="UTF-8",
    csv_args={
        "delimiter": ",",
        # quotechar: 表示在解析CSV文件时，使用双引号 " 作为字段值的引用字符
        # 比如"狮子,哺乳动物" 都被双引号包围，解析器会将它们识别为单个字段，而不是多个字段。
        # page_content='Name: 狮子,哺乳动物
        #"quotechar": '"',
        "fieldnames": ["Name", "Species", "Age", "Habitat"],
    },
)
data = loader.load()
for record in data[:2]:
    print(record)
