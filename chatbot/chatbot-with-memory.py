from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="您是一个与人类对话的聊天机器人。"
        ),  # 持续系统提示
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # 存储Memory的位置
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # 将人类输入注入的位置
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

chat_llm_chain.predict(human_input="嗨，朋友")

if __name__ == '__main__':
    pass
