from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from datetime import datetime
from langchain.tools.base import BaseTool
from langchain import hub
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


load_dotenv(find_dotenv())


@tool
# 获取当前日期和时间
def query_current_date_time(query: str) -> str:
    """
    查询当前年份
    """
    print("query:", query)
    return datetime.now().year


# 定义一个简单的“当前时间”工具
def current_time_tool():
    from datetime import datetime
    return str(datetime.now())


# 将这个函数包装成LangChain可以识别的Tool形式
class CurrentTimeTool(BaseTool):
    name = "Current Time"
    description = "Get the current time."

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return current_time_tool()

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async.")


prompt = hub.pull("hwchase17/react-chat-json")

# 初始化LLM模型，这里使用的是OpenAI模型，设置temperature为0以获得更确定的输出
llm = ChatOpenAI(temperature=0)

# 加载工具，例如维基百科查询工具
# tools = load_tools([query_current_date_time, CurrentTimeTool()])
tools = [query_current_date_time, CurrentTimeTool()]

# 使用初始化的LLM和工具创建Agent，这里使用的是ZeroShotAgent类型

agent = create_json_chat_agent(llm, tools, prompt)

# 使用Agent执行任务，比如查询当前时间信息

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
result = agent_executor.invoke({"input": "当前哪一年?"})

print(result)


if __name__ == '__main__':
    pass
