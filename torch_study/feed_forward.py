import torch
import torch.nn as nn


# 定义一个简单的前馈神经网络
class SimpleFeedforwardNN(nn.Module):
    def __init__(self):
        super(SimpleFeedforwardNN, self).__init__()
        self.layer1 = nn.Linear(10, 5)  # 输入层到隐藏层
        self.activation1 = nn.ReLU()  # 激活函数
        self.layer2 = nn.Linear(5, 2)  # 隐藏层到输出层
        print(self.layer1.weight)
        print(self.layer2.weight)

    def forward(self, x):
        print("aa", x)
        x = self.layer1(x)
        print("bb", x)
        x = self.activation1(x)
        print("cc", x)
        x = self.layer2(x)
        return x


# 创建模型实例
model = SimpleFeedforwardNN()
print(model)

# 假设输入数据
input_data = torch.randn(2, 10)
print("input_data ===== ", input_data)

# 前向传播
output = model(input_data)
print(output)
