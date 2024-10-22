from langgraph.graph import END, StateGraph
from langgraph.graph import StateGraph
from typing import TypedDict, List, Annotated, Union
import operator
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolNode
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain_core.messages import HumanMessage
import random




#############定义一些自定义的tool##############################
@tool("lower_case", return_direct=True)
def to_lower_case(input: str) -> str:
    """返回全部小写的输入"""
    return input.lower()


@tool("random_number", return_direct=True)
def random_number_maker(input: str) -> str:
    """返回0-100之间的随机数"""
    return random.randint(0, 100)


tools = [to_lower_case, random_number_maker]

tool_executor = ToolNode(tools)

####################################定义另一个stategraph################################
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage


# 不需要中间步骤，全部都在message里面
class AgentState_chatmodel(TypedDict):
    message: Annotated[Sequence[BaseMessage], operator.add]


###################################初始化节点############################################

import os
from langchain_community.llms.moonshot import Moonshot

os.environ["moonshot_api_key"]="sk-"
os.environ["model_name"]="moonshot-v1-8k"
os.environ["api_base_url"]="https://api.moonshot.cn/v1"

model=Moonshot()

# print(model.invoke("1+1等于"))

# 定义agent
def run_agent(data):
    # 调用agent可执行对象，并传入数据
    agent_outcome = model.invoke(data["input"])
    # agent_outcome = {"tool":"返回0-100之间的随机数"}
    # 返回代理的结果
    return {"agent_outcome": AgentFinish({"结果":agent_outcome},agent_outcome)}


from langgraph.prebuilt import ToolInvocation
# 定义执行工具的函数
def execute_tools(data):
    action = ToolInvocation(
        tool="random_number",
        tool_input="",
    )
    # 获取最近的代理结果 - 这是在上面的agent中添加的关键字
    # agent_action = data['agent_outcome']
    agent_action = action
    # 执行工具
    output = tool_executor.invoke(agent_action)
    # 打印代理操作
    print(f"The agent action is {agent_action}")
    # 打印工具结果
    print(f"The tool result is {output}")
    # 返回输出
    return {"intermediate_steps": [(agent_action, str(output))]}


# 定义用于确定哪条conditional edge该走的逻辑
def should_continue(data):
    # 如果代理结果是AgentFinish，那么返回“end”字符串
    # 在设置图的流程时将使用这个
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    # 否则，返回一个AgentAction
    # 这里我们返回‘continue’字符串
    # 在设置图的流程时也将使用这个
    else:
        return "continue"

####################################初始化一个stategraph##############################
class AgentState(TypedDict):
    # 定义一些初始化的功能
    # 用户输入字符串
    input: str
    # 对话中之前的消息列表
    chat_history: list[BaseMessage]
    # agent执行过后的结果，是要继续执行action还是完成finish，还是没有任何的状态None
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # 动作列表和相应的观察结果
    # operator.add表明对这个状态的操作是添加到现有值上而不是覆盖掉
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

#######################################################定义一个新的图（工作流）##################################################
workflow = StateGraph(AgentState)  # agentstate是上面初始化过的一个stategraph

# 定义两个节点，我们将在他们之间循环
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)  # agent输出不满足我们的要求，那就继续调用它的action

# 设置初始节点entrypoint
# 将入口节点设为agent，表示这个节点是第一个被调用的
workflow.set_entry_point("agent")

# 添加一个conditional egde
workflow.add_conditional_edges(
    # 首先定义边的起始节点，我们用‘agent’，表示在调用agent节点之后将采取这些边
    "agent",
    # 接下来我们加载将决定下一个调用哪个节点的函数
    should_continue,
    # 最后我们传入一个映射，key是字符串，value是其他节点
    # 将会发生的是，我们调用“should_continue”，然后其输出将与此映射中的key匹配，根据匹配情况调用相应的节点
    {
        "continue": "action",  # 如果是“continue”，则调用工具节点
        "end": END  # END是一个特殊节点，表示图的结束
    }
)

# agent到action是需要conditional egde，但action到agent是不用的，因为action之后肯定要回到agent节点，所以加一个normal edge就行
workflow.add_edge('action', 'agent')

# 最后进行编译compile，将其编译成一个LangChain可运行对象
app = workflow.compile()
inputs = {"input":"3乘以5等于多少,输出最终的结果"}
response=app.invoke(inputs)

print(response)
