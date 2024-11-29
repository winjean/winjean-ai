import requests
# from bs4 import BeautifulSoup
from langchain.tools.base import BaseTool
from langchain.agents import (
    AgentExecutor,
    create_react_agent
)
from langchain_openai import ChatOpenAI
from dotenv import (
    load_dotenv,
    find_dotenv
)
from langchain import hub
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


load_dotenv(find_dotenv())


class BaiduSearchTool(BaseTool):
    name = "Baidu Search"
    description = "Useful for when you need to answer questions about current events or the latest information."

    def _run(self, query: str):
        # url = f"https://www.baidu.com/s?wd={query}"
        # response = requests.get(url)
        # soup = BeautifulSoup(response.text, 'html.parser')
        #
        # # 解析搜索结果
        # results = []
        # for result in soup.find_all('div', class_='result c-container'):
        #     title = result.find('h3').text
        #     link = result.find('a')['href']
        #     snippet = result.find('div', class_='c-abstract').text if result.find('div', class_='c-abstract') else ''
        #     results.append({"title": title, "link": link, "snippet": snippet})

        return None

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


# 创建工具实例
baidu_search_tool = BaiduSearchTool()


# 初始化大模型
llm = ChatOpenAI(temperature=0)
prompt = hub.pull("hwchase17/react")
tools = [baidu_search_tool]

# 初始化代理
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
# 执行调用
response = agent.invoke({"input": "查询关于 LangChain 的最新信息"})
print(response)

if __name__ == '__main__':
    pass
