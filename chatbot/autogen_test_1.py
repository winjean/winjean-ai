from autogen import Agent, UserProxyAgent

# 定义一个简单的智能体
class SimpleAgent(Agent):
    def __init__(
            self,
            name: str,
            messages=None
    ):
        super().__init__()
        self.messages = messages or []
        self.name = name

    @property
    def name(self) -> str:
        """The name of the agent."""
        return self.name

    @name.setter
    def name(self, value):
        self.name = value.lower()

    def receive(self, sender, message):
        print(f"Received message from {sender}: {message}")
        if self.messages:
            next_message = self.messages.pop(0)
            self.send_message(sender, next_message)
        else:
            print("No more messages to send.")

# 初始化两个智能体
agent_a = SimpleAgent(name="agent_a", messages=["你好，我是A。"])
# agent_a.name = "agent_a"

agent_b = SimpleAgent(name="agent_b", messages=["很高兴见到你，B。"])

# 初始化用户代理
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 不允许人类输入
    max_consecutive_auto_reply=2,  # 允许的最大连续自动回复次数
    code_execution_config={
        "last_n_messages": 2,
        "use_docker": False,
        "work_dir": "./",
    },  # 代码执行的工作目录
)

print(agent_a.name)

# 让 A 发送消息给 B
agent_a.send(recipient=agent_b, message="这是A的第一条消息。")

# 让 B 发送消息给 A
agent_b.send(recipient=agent_a, message="这是B的第一条消息。")


if __name__ == '__main__':
    pass
