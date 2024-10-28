import torch
import torch.nn as nn
import torch.optim as optim

# x = torch.tensor([1.0, 2.0])
# y = torch.tensor([3.0, 4.0])
# z = x + y
# print(z)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.input_layer = nn.Linear(784, 128)  # 输入层到隐藏层
        self.hidden_layer1 = nn.Linear(128, 64)  # 隐藏层到隐藏层
        self.hidden_layer2 = nn.Linear(64, 32)  # 第二个隐藏层
        self.output_layer = nn.Linear(32, 10)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入展平
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x


# 实例化模型
model = SimpleNet()
# model = SimpleNet().cuda()   # 将模型移动到 GPU 上

# 打印模型结构
print(model)


# 定义损失函数和优化器
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 假设我们有一些训练数据
train_data = torch.randn(100, 784)
train_labels = torch.randint(0, 10, (100,))

# 训练循环
for epoch in range(10):  # 迭代次数
    optimizer.zero_grad()  # 清零梯度
    outputs = model(train_data)  # 前向传播
    loss = criterion(outputs, train_labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')


# 保存模型
# torch.save(model.state_dict(), 'simple_net.pth')

# 加载模型
# model = SimpleNet()
# model.load_state_dict(torch.load('simple_net.pth'))
# model.eval()  # 设置为评估模式

if __name__ == '__main__':
    pass
