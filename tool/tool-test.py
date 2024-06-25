from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_react_agent, create_structured_chat_agent
from dotenv import load_dotenv, find_dotenv
import openai, os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import hub

class SearchInput(BaseModel):
    query: str = Field(description="计算字符串长度")


@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search(query: str) -> str:
    """Look up things online."""
    return len(query+"winjean")


class CalculatorInput(BaseModel):
    a: int = Field(description="输入字符串")
    b: int = Field(description="输入字符串")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,  # 工具具体逻辑
    name="Calculator",  # 工具名
    description="计算两个数字的乘积",  # 工具信息
    args_schema=CalculatorInput,  # 工具接受参数信息
    return_direct=True,  # 直接作为工具的输出返回给调用者
    handle_tool_error=True,  # 报错了继续执行，不会吧那些报错行抛出，也可以自定义函数处理，handle_tool_error=函数名
)


load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
openai.base_url = os.environ['OPENAI_API_BASE']
openai.temperature = 0.7

tools = [search, calculator]
model=ChatOpenAI()
# agent = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# val = agent.run("计算 'aaaaa'的长度, 计算字符串长度的平方 ")
# prompt = PromptTemplate.from_template("hwchase17/react{question}")
# agent = create_react_agent(model, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor.invoke({"input": "langchian是什么东西？`阿萨德防守打法`有多少个字符？"})
# print(val)


class CalculatorInput(BaseModel):
    a: str = Field(description="第一个字符串")
    b: str = Field(description="第二个字符串")


def multiply(a: str, b: str) -> int:
    """Multiply two numbers."""
    return len(a) * len(b)


calculator = StructuredTool.from_function(
    func=multiply,  # 工具具体逻辑
    name="Calculator",  # 工具名
    description="计算两个字符长度的乘积",  # 工具信息
    args_schema=CalculatorInput,  # 工具接受参数信息
    return_direct=False,  # 直接作为工具的输出返回给调用者
    handle_tool_error=True,  # 报错了继续执行，不会吧那些报错行抛出，也可以自定义函数处理，handle_tool_error=函数名
)


class SearchInput(BaseModel):
    query: str = Field(description="参数")


@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search(query: str) -> str:
    """Look up things online."""
    return "你好啊" + query


class SortList(BaseModel):
    num: str = Field(description="待排序列表")


def sort_fun(num):
    """Multiply two numbers."""
    return sorted(eval(num))


sorter = StructuredTool.from_function(
    func=sort_fun,  # 工具具体逻辑
    name="sort_num",  # 工具名
    description="排序字符串列表中的数字",  # 工具信息
    args_schema=SortList,  # 工具接受参数信息
    return_direct=True,  # 直接作为工具的输出返回给调用者
    handle_tool_error=True,  # 报错了继续执行，不会吧那些报错行抛出，也可以自定义函数处理，handle_tool_error=函数名
)


tools = [search, calculator, sorter]
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
var = agent_executor.invoke({"input": "`adddd`的字符串长度乘以`sssss`的字符串长度是多少？ 然后对字符串`[10,4,7]`中的数字排序"})
print(var)


if __name__ == '__main__':
    pass
