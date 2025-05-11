# pip install -qU langchain-text-splitters

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载示例文档
with open("../../resource/knowledge.txt", encoding="utf-8") as f:
    knowledge = f.read()
text_splitter = RecursiveCharacterTextSplitter(
    # 设置一个非常小的块大小，只是为了展示。
    chunk_size=100,
    # 块之间的目标重叠。重叠的块有助于在上下文分割时减少信息丢失。
    chunk_overlap=20,
    # 确定块大小的函数。
    length_function=len,
    # 分隔符列表（默认为 ["\n\n", "\n", " ", ""]）是否应被解释为正则表达式。
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # 零宽空格
        "\uff0c",  # 全角逗号
        "\u3001",  # 表意逗号
        "\uff0e",  # 全角句号
        "\u3002",  # 表意句号
        "",
    ],
)
texts = text_splitter.create_documents([knowledge])
print(texts[0])
print(texts[1])

print(text_splitter.split_text(knowledge)[:2])
