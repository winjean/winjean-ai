from langgraph.graph import END, MessageGraph
from langchain.tools import tool
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Literal, Any
from langchain_core.messages import AnyMessage


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


graph = MessageGraph()


def show_img(path) -> None:
    img = Image.open(path)
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(img)  # 将数据显示为图像，即在二维常规光栅上。
    plt.show()  # 显示图片


def check_telephone_condition(state: Union[list[AnyMessage], dict[str, Any]]) -> Literal["check_business", "end"]:
    if state:
        return "check_business"
    else:
        return "end"


def check_business_condition(state: Union[list[AnyMessage], dict[str, Any]]) -> Literal["check_card", "end"]:
    if state:
        return "check_card"
    else:
        return "end"


def check_card_condition(state: Union[list[AnyMessage], dict[str, Any]]) -> Literal["check_card_business", "end"]:
    if state:
        return "check_card_business"
    else:
        return "end"


graph.add_node("greeting", multiply)
graph.add_node("check_telephone", multiply)
graph.add_node("check_business", multiply)
graph.add_node("check_card", multiply)
graph.add_node("check_card_business", multiply)

graph.set_entry_point("greeting")
graph.add_edge("greeting", "check_telephone")

graph.add_conditional_edges('check_telephone', check_telephone_condition, {
    "check_business": "check_business",
    "end": END
})

graph.add_conditional_edges('check_business', check_business_condition, {
    "check_card": "check_card",
    "end": END
})

graph.add_conditional_edges('check_card', check_card_condition, {
    "check_card_business": "check_card_business",
    "end": END
})

graph.set_finish_point("check_card_business")
runnable = graph.compile()
show_img(BytesIO(runnable.get_graph(xray=True).draw_mermaid_png()))
