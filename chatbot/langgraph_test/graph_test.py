from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, MessageGraph

import os
from langchain_community.llms.moonshot import Moonshot
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Literal, Any
from langchain_core.messages import AnyMessage


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


os.environ["moonshot_api_key"] = "sk-"
os.environ["model_name"] = "moonshot-v1-8k"
os.environ["api_base_url"] = "https://api.moonshot.cn/v1"

model = Moonshot()
# model.bind(multiply)

graph = MessageGraph()


def show_img(path) -> None:
    img = Image.open(path)
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(img)  # 将数据显示为图像，即在二维常规光栅上。
    plt.show()  # 显示图片


def lsmsp_tools_condition(
        state: Union[list[AnyMessage],
        dict[str, Any]],
) -> Literal["model2", "model3", "end"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "model2"
    elif hasattr(ai_message, "content"):
        return "end"
    else:
        return "model3"


def invoke_model(state: List[BaseMessage]):
    return model.invoke(state)


graph.add_node("model1", model)
graph.add_node("model2", model)
graph.add_node("model3", model)
# graph.add_node("结束", END)

graph.set_entry_point("model1")

graph.add_conditional_edges('model1', lsmsp_tools_condition, {
    "1 -> 2": "model2",
    "1 -> 3": "model3",
    "end": END
})
graph.add_edge("model2", "model3")
graph.add_edge("model3", "model1")
graph.set_finish_point("model2")
graph.set_finish_point("model3")

runnable = graph.compile()

show_img(BytesIO(runnable.get_graph(xray=True).draw_mermaid_png()))
# response=runnable.invoke("简单介绍一下南京")
# print(response[-1].content)
