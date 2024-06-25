from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

chat = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "您是一个有用的助手。尽力回答所有问题。",
        ),

        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat
#
# answer = chain.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="将这个句子从中文翻译成英语：我喜欢编程。"
#             ),
#             AIMessage(content="I love programmation."),
#             HumanMessage(content="你刚才回答了什么？"),
#         ],
#     }
# )
# demo_ephemeral_chat_history = ChatMessageHistory()

# demo_ephemeral_chat_history.add_user_message(
#     "将这个句子从英语翻译成英语语：我喜欢编程。"
# )
#
# demo_ephemeral_chat_history.add_ai_message("I love programmation.")
#
# demo_ephemeral_chat_history.messages
# input1 = "将这个句子从中文翻译成英语：我喜欢编程。"
#
# demo_ephemeral_chat_history.add_user_message(input1)

# response = chain.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#     }
# )

# demo_ephemeral_chat_history.add_ai_message(response)
#
# input2 = "我刚才问了你什么？"
#
# demo_ephemeral_chat_history.add_user_message(input2)
#
# answer = chain.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#     }
# )

demo_ephemeral_chat_history_for_chain = ChatMessageHistory()
demo_ephemeral_chat_history_for_chain.add_user_message("嗨！我是win。")
demo_ephemeral_chat_history_for_chain.add_ai_message("你好！")
demo_ephemeral_chat_history_for_chain.add_user_message("你今天好吗？")
demo_ephemeral_chat_history_for_chain.add_ai_message("很好，谢谢！")


def trim_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history_for_chain.messages
    if len(stored_messages) <= 2:
        return False

    demo_ephemeral_chat_history_for_chain.clear()

    for message in stored_messages[-2:]:
        demo_ephemeral_chat_history_for_chain.add_message(message)

    return True


def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history_for_chain.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "将以上聊天消息精简成一条摘要消息。请尽可能包含具体细节。",
            ),
        ]
    )
    summarization_chain = summarization_prompt | chat
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    demo_ephemeral_chat_history_for_chain.clear()
    demo_ephemeral_chat_history_for_chain.add_message(summary_message)
    return True


chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# answer = chain_with_message_history.invoke(
#     {"input": "我是谁。"},
#     {"configurable": {"session_id": "unused"}},
# )
# print(f"答案: {answer}")
#
# answer = chain_with_message_history.invoke(
#     {"input": "你是谁？"},
#     {"configurable": {"session_id": "unused"}}
# )

# print(f"答案: {answer}")

# chain_with_trim = (
#         RunnablePassthrough.assign(messages_trimmed=trim_messages)
#         | chain_with_message_history
# )

chain_with_summarization = (
        RunnablePassthrough.assign(messages_summarized=summarize_messages)
        | chain_with_message_history
)


answer = chain_with_summarization.invoke(
    {"input": "我说过我的名字是什么吗？"},
    {"configurable": {"session_id": "unused"}}
)

print(f"答案: {answer}")

demo_ephemeral_chat_history_for_chain.messages

if __name__ == '__main__':
    pass

