from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
store = {}

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="您是一个与人类对话的聊天机器人。"
        ),  # 持续系统提示
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # 存储Memory的位置
        HumanMessagePromptTemplate.from_template(
            "{message}"
        ),  # 将人类输入注入的位置
    ]
)


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory(memory_key="chat_history", return_messages=True)
    return store[session_id]

memory = ChatMessageHistory(memory_key="chat_history", return_messages=True)
memory.add_ai_message("aaaa")
memory.add_user_message("bbbb")
store["123456"] = memory

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4-flash",
)

parser = StrOutputParser()

chat_llm_chain = prompt | llm | parser

# response = chat_llm_chain.invoke({"chat_history": memory.messages, "message": "嗨，朋友,刚才你说了什么"})
# print(response)


with_message_history = RunnableWithMessageHistory(
    chat_llm_chain,
    get_session_history,
    input_messages_key="message",
    history_messages_key="chat_history",
)

response = with_message_history.invoke(
    {"message": "嗨，朋友,刚才你说了什么"},
    {"configurable": {"session_id": "123456"}},
)
print(response)


if __name__ == '__main__':
    ...

