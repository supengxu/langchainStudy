from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import asyncio


class AnimalRetriever(BaseRetriever):
    """包含用户查询的前k个文档的动物检索器。k从0开始
    该检索器实现了同步方法`_get_relevant_documents`。
    如果检索器涉及文件访问或网络访问，它可以受益于`_aget_relevant_documents`的本机异步实现。
    与可运行对象一样，提供了默认的异步实现，该实现委托给在另一个线程上运行的同步实现。
    """
    documents: List[Document]
    """要检索的文档列表。"""
    k: int
    """要返回的前k个结果的数量"""

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """检索器的同步实现。"""
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) >= self.k:
                break
            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """异步获取与查询相关的文档。
        Args:
            query: 要查找相关文档的字符串
            run_manager: 要使用的回调处理程序
        Returns:
            相关文档列表
        """
        matching_documents = []
        for document in self.documents:
            if len(matching_documents) >= self.k:
                break
            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents


documents = [
    Document(
        page_content="狗是很好的伴侣，以其忠诚和友好著称。",
        metadata={"type": "狗", "trait": "忠诚"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"type": "猫", "trait": "独立"},
    ),
    Document(
        page_content="金鱼是初学者的热门宠物，护理相对简单。",
        metadata={"type": "鱼", "trait": "低维护"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"type": "鸟", "trait": "聪明"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"type": "兔子", "trait": "社交"},
    ),
]

retriever = AnimalRetriever(documents=documents, k=1)

# 测试同步方法
# print("同步方法测试:")
# print(retriever.invoke("宠物"))
# print(retriever.batch(["猫", "兔子"]))


async def ainvoke():
    print("异步方法测试:")
    print(await retriever.ainvoke("宠物"))
    print(await retriever.abatch(["猫", "兔子"]))

    async for event in retriever.astream_events("猫", version="v2"):
        print(event)

# 测试异步方法
asyncio.run(ainvoke())
