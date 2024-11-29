from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_TRACING_V2"] = "true"
os.environ["TAVILY_API_KEY"] = "395314a6eb747c8c8f86b62303fb0e44.Xfl0gVyAyO7CpDIJ"
llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

word_phrase = """
你是日志精灵，你专注于处理用户提供的日志数据，通过智能解析生成固定格式的解码器。
你的能力有:
    - 自动识别日志格式
    - 快速生成解码器
    - 支持多种日志类型解析

提供如下格式化输出:
decoder:
    parent: useradd
    name: useradd-newusr
    conditions:
        - regex:
            field: message
            pattern:'new user'
    processors:
        - regex:
            field: message
            offset: whole
            pattern: 'new user:\s+name=(\S+)\..*'
            targets:['host.user.name']
"""

tools = [TavilySearchResults(max_results=2)]
prompt = PromptTemplate.from_template(
    word_phrase +
"""
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question,假如tools不能正确回答，可不参照工具

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": word_phrase + """
www.pipixia.org---100.122.17.191 - - [06/Oct/2024:03:46:21 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white,green,blue,red&sort=_default HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=black%2Cwhite%2Cgreen%2Cblue%2Cred" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.130.32"
www.pipixia.org---100.122.17.142 - - [06/Oct/2024:03:48:25 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=blue%2Cblack%2Cwhite" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.151.174"
"""})

print(response["input"])
print(response["output"])
