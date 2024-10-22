from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_community.llms.moonshot import Moonshot
from langchain.tools import (
    StructuredTool,
    tool
)
from pydantic import BaseModel, Field
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
    create_react_agent,
    create_json_chat_agent,
    create_self_ask_with_search_agent,
    create_tool_calling_agent,
    create_openai_tools_agent,
    create_xml_agent
)
from langchain import hub
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.prompts import PromptTemplate

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os

os.environ["moonshot_api_key"]="sk-"
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

@tool("Intermediate Answer", args_schema=SortList, return_direct=False)
def intermediate_Answer(num):
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

# agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
# agent = create_json_chat_agent(llm=model, tools=tools, prompt=prompt)
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt)

agent_callbacks = None

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    max_execution_time=1000,
    handle_parsing_errors=True,
    callbacks=agent_callbacks
)


response = agent_executor.invoke({
    "input": "用中文简单介绍一下南京, 告诉我`adddd`的字符串长度乘以`sssss`的字符串长度是多少？ 然后对`[10,4,7]`中的数字排序,最后我的名字叫什么？",
    "chat_history": [
        HumanMessage(content="hi! my name is bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
    ]
})

print(response)
print(response['input'])
print(response['chat_history'])
print(response['output'])


template = '''Question: Who lived longer, Muhammad Ali or Alan Turing?
    Are follow up questions needed here: Yes.
    Follow up: How old was Muhammad Ali when he died?
    Intermediate answer: Muhammad Ali was 74 years old when he died.
    Follow up: How old was Alan Turing when he died?
    Intermediate answer: Alan Turing was 41 years old when he died.
    So the final answer is: Muhammad Ali

    Question: When was the founder of craigslist born?
    Are follow up questions needed here: Yes.
    Follow up: Who was the founder of craigslist?
    Intermediate answer: Craigslist was founded by Craig Newmark.
    Follow up: When was Craig Newmark born?
    Intermediate answer: Craig Newmark was born on December 6, 1952.
    So the final answer is: December 6, 1952

    Question: Who was the maternal grandfather of George Washington?
    Are follow up questions needed here: Yes.
    Follow up: Who was the mother of George Washington?
    Intermediate answer: The mother of George Washington was Mary Ball Washington.
    Follow up: Who was the father of Mary Ball Washington?
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
    So the final answer is: Joseph Ball

    Question: Are both the directors of Jaws and Casino Royale from the same country?
    Are follow up questions needed here: Yes.
    Follow up: Who is the director of Jaws?
    Intermediate answer: The director of Jaws is Steven Spielberg.
    Follow up: Where is Steven Spielberg from?
    Intermediate answer: The United States.
    Follow up: Who is the director of Casino Royale?
    Intermediate answer: The director of Casino Royale is Martin Campbell.
    Follow up: Where is Martin Campbell from?
    Intermediate answer: New Zealand.
    So the final answer is: No

    Question: {input}
    Are followup questions needed here:{agent_scratchpad}
'''

# prompt_Intermediate_Answer = PromptTemplate.from_template(template)
# tools_Intermediate_Answer = [intermediate_Answer]
# agent = create_self_ask_with_search_agent(llm=model, tools=tools_Intermediate_Answer, prompt=prompt_Intermediate_Answer)






# message_history = ChatMessageHistory()
#
# agent_with_chat_history = RunnableWithMessageHistory(
#     agent_executor,
#     lambda session_id: message_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )
# agent_with_chat_history.invoke(
#     {"input": "请用中文回答美国现任总统是谁？"},
#     config={"configurable": {"session_id": "session-10086"}},
# )
#
# agent_with_chat_history.invoke(
#     {"input": "他在2024年做了哪些特别的事情？"},
#     config={"configurable": {"session_id": "session-10086"}},
# )

if __name__ == '__main__':
    pass