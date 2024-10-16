from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.llms.moonshot import Moonshot
from langchain.tools import StructuredTool, tool,BaseTool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, initialize_agent,create_structured_chat_agent,AgentType
from langchain.prompts import PromptTemplate
from langchain import hub
import os

os.environ["moonshot_api_key"]="sk-eJ55Ld7u8mu42HK1gqF6C8VRIknLjIGH4ZWXOLUMG14ip0Yl10"
os.environ["model_name"]="moonshot-v1-8k"
os.environ["api_base_url"]="https://api.moonshot.cn/v1"
model=Moonshot()

messages=[
    SystemMessage(content="你是一个数学专家。"),
    HumanMessage(content="请回答我1加3等于多少.")
]

class SearchInput(BaseModel):
    query: str = Field(description="计算字符串长度")

@tool("search-tool", args_schema=SearchInput, return_direct=True)
def search(query: str) -> int:
    """Look up things online."""
    return len(query+"winjean")

class CalculatorInput(BaseModel):
    a: int = Field(description="输入字符串的长度")
    b: int = Field(description="输入字符串的长度")

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

calculator = StructuredTool.from_function(
    func=multiply,  # 工具具体逻辑
    name="Calculator",  # 工具名
    description="计算两个数字的乘积",  # 工具信息
    args_schema=CalculatorInput,  # 工具接受参数信息
    return_direct=False,  # 直接作为工具的输出返回给调用者
    handle_tool_error=True,  # 报错了继续执行，不会吧那些报错行抛出，也可以自定义函数处理，handle_tool_error=函数名
)

class SortList(BaseModel):
    num: str = Field(description="待排序列表")

def sort_fun(num):
    """sort number list."""
    return sorted(eval(num))

sorter = StructuredTool.from_function(
    func=sort_fun,  # 工具具体逻辑
    name="sort_num",  # 工具名
    description="排序字符串列表中的数字",  # 工具信息
    args_schema=SortList,  # 工具接受参数信息
    return_direct=False,  # 直接作为工具的输出返回给调用者
    handle_tool_error=True,  # 报错了继续执行，不会吧那些报错行抛出，也可以自定义函数处理，handle_tool_error=函数名
)

tools = [calculator,sorter]

prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
response = agent_executor.invoke({"input": "`adddd`的字符串长度乘以`sssss`的字符串长度是多少？ 然后对`[10,4,7]`中的数字排序"})
# response = agent_executor.invoke({"input": "对`[10,4,7]`中的数字排序"})
# response = agent_executor.invoke({"input": "`adddd`的字符串长度乘以`sssss`的字符串长度是多少？"})

print(response)

if __name__ == '__main__':
    pass