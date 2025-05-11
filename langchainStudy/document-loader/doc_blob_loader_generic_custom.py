from typing import Iterator

from langchain_community.document_loaders.generic import GenericLoader
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document
from typing import Any


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


class MyCustomLoader(GenericLoader):
    @staticmethod
    def get_parser(**kwargs: Any) -> BaseBlobParser:
        """Override this method to associate a default parser with the class."""
        return MyParser()


loader = MyCustomLoader.from_filesystem(path="./../resource", glob="*.mdx", show_progress=True)
for idx, doc in enumerate(loader.lazy_load()):
    if idx < 5:
        print(doc)
print("... output truncated for demo purposes")
