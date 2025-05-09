import os
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import llm_basic.llm_client
from llm_basic import llm_client
from llm_basic.llm_client import embeddingClient

os.environ['LANGCHAIN_TRACING_V2'] = "true"
examples = [
    {
        "question": "谁活得更长，穆罕默德·阿里还是艾伦·图灵？",
        "answer": """
   是否需要后续问题：是的。
   后续问题：穆罕默德·阿里去世时多大年纪？
   中间答案：穆罕默德·阿里去世时74岁。
   后续问题：艾伦·图灵去世时多大年纪？
   中间答案：艾伦·图灵去世时41岁。
   所以最终答案是：穆罕默德·阿里
   """,
    },
    {
        "question": "克雷格斯列表的创始人是什么时候出生的？",
        "answer": """
   是否需要后续问题：是的。
   后续问题：克雷格斯列表的创始人是谁？
   中间答案：克雷格斯列表的创始人是克雷格·纽马克。
   后续问题：克雷格·纽马克是什么时候出生的？
   中间答案：克雷格·纽马克于1952年12月6日出生。
   所以最终答案是：1952年12月6日
   """,
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
   是否需要后续问题：是的。
   后续问题：乔治·华盛顿的母亲是谁？
   中间答案：乔治·华盛顿的母亲是玛丽·波尔·华盛顿。
   后续问题：玛丽·波尔·华盛顿的父亲是谁？
   中间答案：玛丽·波尔·华盛顿的父亲是约瑟夫·波尔。
   所以最终答案是：约瑟夫·波尔
   """,
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
        "answer": """
   是否需要后续问题：是的。
   后续问题：《大白鲨》的导演是谁？
   中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
   后续问题：史蒂文·斯皮尔伯格来自哪个国家？
   中间答案：美国。
   后续问题：《皇家赌场》的导演是谁？
   中间答案：《皇家赌场》的导演是马丁·坎贝尔。
   后续问题：马丁·坎贝尔来自哪个国家？
   中间答案：新西兰。
   所以最终答案是：不是
   """,
    },
]
def create_prompt_template():
    """使用 PromptTemplate 构建基础文本模板"""
    prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
    return prompt_template.invoke({"topic": "cats"})

def create_chat_prompt_template():
    """使用 ChatPromptTemplate 构建多角色对话模板"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("user", "Tell me a joke about {topic}")
    ])
    return prompt_template.invoke({"topic": "cats"})

def create_message_with_placeholder_template():
    """使用 ChatPromptTemplate + MessagesPlaceholder 构建可扩展消息模板"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("msgs")
    ])
    return prompt_template.invoke({"msgs": [HumanMessage(content="hi!"),AIMessage(content="what's up?")]})

def few_shot_prompt():
    """使用 few-shot 示例构建 PromptTemplate"""
    example_prompt = PromptTemplate.from_template("问题：{question}\n{answer}")

    from langchain_core.prompts import FewShotPromptTemplate
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="问题：{input}",
        input_variables=["input","2"],
    )
    print(
        prompt.invoke({"input": "乔治·华盛顿的父亲是谁？"}).to_string()
    )

def example_selectors():
    # 创建一个LLM实例
    llm = llm_client.llmClient()

    # 将示例选择器和提示模板组合成LCEL形式
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddingClient(),
        Chroma,
        k=1,
    )

    prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template("Question: {question}\nAnswer: {answer}"),
        prefix="请回答以下问题：",
        suffix="Question: {question}\nAnswer:",
        input_variables=["question"],
    )

    # 创建LCEL链
    chain = prompt_template | RunnableLambda(lambda prompt: llm.invoke(prompt))

    # 调用链
    question = "玛丽·波尔·华盛顿的父亲是谁？"
    response = chain.invoke({"question": question})
    print(f"\n问题: {question}")
    print(f"回答: {response.content}")


example_selectors()
