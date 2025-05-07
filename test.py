import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量到os.environ
load_dotenv()


from httpx import Client
from langchain.agents import initialize_agent, AgentType, Tool, AgentExecutor
from langchain.globals import set_debug
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import tool
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.tracers import langchain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests  # 确保正确导入requests模块

load_dotenv()

def llmClient():
    custom_client = Client()

    return ChatOpenAI(
        model=os.environ["MODEL"],
        base_url=os.environ["BASE_URL"],
        http_client=custom_client
    )
class EchoTool(BaseTool):
    name:str = "echo"
    description:str = "Echoes back the input text."

    def _run(self, text: str) -> str:
        return text

    async def _arun(self, text: str) -> str:
        return text

langchain.verbose = False
def agentType():
    llm = llmClient()  # 获取ChatOpenAI实例
    output_parser = StrOutputParser()  # 创建输出解析器

    # 使用 DuckDuckGoSearchAPIWrapper 替代 SerpAPIWrapper
    search = DuckDuckGoSearchAPIWrapper()

    lookup_tool = LookupTool()  # 使用新的 Lookup 工具

    tools = [
        Tool(
            name="Search",
            description="用于搜索实时信息，例如天气、新闻等。",
            func=lambda query: search.run(str(query)) if search.run(str(query)) else "No results found"
        ),
        Tool(
            name="get_weather",
            description="根据城市名获取当前天气数据。",
            func=get_weather
        ),
        Tool(
            name="Lookup",
            description="用于查找特定信息，如百科知识、数据库记录等。",
            func=lookup_tool._run
        )
    ]
    # 加载工具
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # 创建代理
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 创建执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke({"input": "我想要查询今天上海的天气"})

    # 初始化代理
    # agent = initialize_agent(
    #     tools,
    #     llm,  # 直接使用llmClient返回的ChatOpenAI实例
    #     agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    #     verbose=True,
    #     handle_parsing_errors=True
    # )
    # # 运行代理来回答问题
    # question = "今天上海天气怎么样"
    # set_debug(True)
    # result = agent.run(question)
    # cleaned_result = result.encode('utf-8', 'ignore').decode('utf-8')  # 去除不可见字符和乱码
    # print(cleaned_result)  # 打印清理后的结果

class LookupTool(BaseTool):
    name: str = "Lookup"
    description: str = "不执行这个工具"

    def _run(self, text: str) -> str:
        # 这里可以实现具体的查找逻辑
        return f"Lookup result for: {text}"

    async def _arun(self, text: str) -> str:
        # 异步版本的查找逻辑
        return f"Async lookup result for: {text}"


@tool
def get_weather(location):
    """根据城市获取天气数据"""
    api_key = "SKcA5FGgmLvN7faJi"
    url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
    response = requests.get(url)
    # print(location)
    if response.status_code == 200:
        data = response.json()
        # print(data)
        weather = {
            'description': data['results'][0]["now"]["text"],
            'temperature': data['results'][0]["now"]["temperature"]
        }
        return weather
    else:
        raise Exception(f"失败接收天气信息：{response.status_code}")


if __name__ == '__main__':
    agentType()


