from autogen import Agent, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
#
# # 定义第一个智能体 AgentA
# agent_a = Agent(
#     name="agent_a",
#     system_message="你是AgentA，负责询问AgentB的问题。",
#     llm_config={
#         "config_list": [
#             {
#                 "llm_type": "openai",
#                 "model": "gpt-3.5-turbo",
#                 "api_key": "test",
#             }
#         ]
#     },
# )
#
# # 定义第二个智能体 AgentB
# agent_b = Agent(
#     name="agent_b",
#     system_message="你是AgentB，负责回答AgentA的问题。",
#     llm_config={
#         "config_list": [
#             {
#                 "llm_type": "openai",
#                 "model": "gpt-3.5-turbo",
#                 "api_key": "<YOUR_OPENAI_API_KEY>",
#             }
#         ]
#     },
# )
#
# # 创建代理代理（UserProxyAgent）
# user_proxy = UserProxyAgent(
#     name="user_proxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
#     code_execution_config={"work_dir": "workspace"},
# )
#
# # 创建助手代理（AssistantAgent）
# assistant = AssistantAgent(
#     name="assistant",
#     llm_config={
#         "config_list": [
#             {
#                 "llm_type": "openai",
#                 "model": "gpt-3.5-turbo",
#                 "api_key": "<YOUR_OPENAI_API_KEY>",
#             }
#         ]
#     },
# )
#
# # 创建一个群聊
# group_chat = GroupChat(
#     agents=[agent_a, agent_b],
#     messages=[],
#     max_round=5,
# )
#
# # 开始对话
# group_chat.append_message(
#     message="请告诉我今天的天气怎么样？",
#     sender="user_proxy",
#     receiver="agent_a",
# )
#
# # 运行对话
# group_chat.run()

import requests

class WeatherAssistantAgent(AssistantAgent):
    def respond_to_message(self, message, *args, **kwargs):
        content = message["content"]
        if "你好" in content:
            return {"content": "你好！有什么我可以帮助你的吗？"}
        elif "天气" in content:
            # 调用天气API获取天气信息
            weather_info = self.get_weather_info()
            return {"content": f"今天的天气是 {weather_info}"}
        else:
            return {"content": "我不太明白你的意思，请再说一遍。"}

    def get_weather_info(self):
        # 假设这是一个简单的天气API调用
        # response = requests.get("https://api.example.com/weather")
        # if response.status_code == 200:
        #     return response.json()["description"]
        # else:
        return "无法获取天气信息"

# 初始化助理代理
weather_assistant_agent= WeatherAssistantAgent(name="weather_assistant")

# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 初始化用户代理
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False
    },
    is_termination_msg=lambda x: x.get("content", "").strip().lower() == "exit"
)

# 创建一个简单的助理代理
# class SimpleAssistantAgent(AssistantAgent):
#     def respond_to_message(self, message, *args, **kwargs):
#         content = message["content"]
#         if "你好" in content:
#             return {"content": "你好！有什么我可以帮助你的吗？"}
#         elif "天气" in content:
#             return {"content": "今天的天气非常好，适合外出活动。"}
#         else:
#             return {"content": "我不太明白你的意思，请再说一遍。"}

# 初始化助理代理
# assistant_agent = SimpleAssistantAgent(name="assistant1")

# 创建另一个助理代理
# class AnotherAssistantAgent(AssistantAgent):
#     def respond_to_message(self, message, *args, **kwargs):
#         content = message["content"]
#         if "你好" in content:
#             return {"content": "你好！我是另一个助理。"}
#         elif "天气" in content:
#             return {"content": "今天的天气有点阴，记得带伞。"}
#         else:
#             return {"content": "我不太明白你的意思，请再说一遍。"}
#
# # 初始化另一个助理代理
# assistant_agent2 = AnotherAssistantAgent(name="assistant2")

# 创建一个群聊对象
# group_chat = GroupChat(
#     # name="group_chat",
#     max_round=5,
#     agents=[user_proxy, assistant_agent, assistant_agent2],
#     messages=["你好，助理们！"]
# )

# 创建群聊管理器
# group_chat_manager = GroupChatManager(groupchat=group_chat)
#
# # 开始对话
# # group_chat_manager.run_chat(user_proxy, messages="你好，助理们！")
# group_chat_manager.run_chat(
#     # max_round=5,
#     # messages=["你好，助理们！"],
#     config=group_chat
# )


# user_proxy = UserProxyAgent(
#     name="user_proxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
#     code_execution_config={
#         # "last_n_messages": 2,
#         # "work_dir": "groupchat",
#         "use_docker": False
#     },
#     is_termination_msg=lambda x: x.get("content", "").strip().lower() == "exit"
# )

# 将助理代理添加到用户代理的消息处理链
# user_proxy.add_assistant(weather_assistant_agent)

# 开始对话
user_proxy.initiate_chat(weather_assistant_agent, message="你好，助理！今天天气怎么样？")

print(user_proxy.chat_messages)

# group_chat = GroupChat(
#     agents=[user_proxy, weather_assistant_agent],
#     messages=[],
#     max_round=5,
# )
#
# # 开始对话
# group_chat.append_message(
#     message="你好，助理！今天天气怎么样？",
#     sender="user_proxy",
#     receiver="agent_a",
# )

# 运行对话
# group_chat.run()

if __name__ == '__main__':
    pass
