from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_community.llms.moonshot import Moonshot
from langchain.tools import StructuredTool, tool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor,create_structured_chat_agent
from langchain import hub
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

import os

os.environ["moonshot_api_key"]="your api key"
os.environ["model_name"]="moonshot-v1-8k"
os.environ["api_base_url"]="https://api.moonshot.cn/v1"

model=Moonshot()

class SearchInput(BaseModel):
    query: str = Field(description="计算字符串长度")

@tool("StringLength", args_schema=SearchInput, return_direct=False)
def string_length(query: str) -> int:
    """计算字符串长度."""
    return len(query)

class CalculatorInput(BaseModel):
    a: int = Field(description="输入第一个数字")
    b: int = Field(description="输入第二个数字")

@tool("Multiply", args_schema=CalculatorInput, return_direct=False)
def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积."""
    return a * b

class SortList(BaseModel):
    num: str = Field(description="待排序序列")

@tool("SortList", args_schema=SortList, return_direct=False)
def sorter(num):
    """排序字符串中的数字序列."""
    return sorted(eval(num))

tools = [multiply,sorter,string_length]

# prompt = hub.pull("hwchase17/structured-chat-agent")

system = '''
    Respond to the human as helpfully and accurately as possible. You have access to the following tools:
    {tools}
    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
    Valid "action" values: "Final Answer" or {tool_names}
    Provide only ONE action per $JSON_BLOB, as shown:
    ```
    {{
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }}
    ```

    Follow this format:
    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {{
      "action": "Final Answer",
      "action_input": "Final response to human"
    }}
    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
'''

human = '''
    {input}
    {agent_scratchpad}
    (reminder to respond in a JSON blob no matter what)
'''
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", human),
])

agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

response = agent_executor.invoke({
    "input": "告诉我`adddd`的字符串长度乘以`sssss`的字符串长度是多少？ 然后对`[10,4,7]`中的数字排序,最后我的名字叫什么？再用中文简单介绍一下南京,",
    "chat_history": [
        HumanMessage(content="hi! my name is bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
    ]
})

print(response)
print(response['input'])
print(response['chat_history'])
print(response['output'])

if __name__ == '__main__':
    pass