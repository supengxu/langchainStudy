from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document
from typing import AsyncIterator, Iterator


class MyParser(BaseBlobParser):
    """一个简单的解析器，每行创建一个文档。"""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """逐行将 blob 解析为文档。"""
        line_number = 0
        with blob.as_bytes_io() as f:
            for line in f:
                line_number += 1
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": blob.source},
                )


blob = Blob.from_path("./meow.txt", metadata={"foo": "bar"})


#blob API
print(blob.encoding)
print(blob.as_bytes())
print(blob.as_string())
print(blob.as_bytes_io())

#Blob 元数据
print(blob.metadata)
print(blob.source)



