import torch

x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])
z = x + y
print(z)

import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

net = Net()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(100):
    # 前向传播
    outputs = net(x)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print(x)
print(y)
z = x + y
print(z)

if __name__ == '__main__':
    pass
