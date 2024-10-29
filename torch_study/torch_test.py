import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# 线性层：nn.Linear(in_features, out_features)
# 卷积层：nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
# 池化层：nn.MaxPool2d(kernel_size, stride=None, padding=0)
# 批量归一化层：nn.BatchNorm2d(num_features)
# 激活函数：nn.ReLU(), nn.Sigmoid(), nn.Tanh()
# 损失函数：nn.MSELoss(), nn.CrossEntropyLoss(), nn.BCELoss()
# 优化器：optim.SGD(), optim.Adam(), optim.RMSprop()


def download_data():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 下载并加载训练数据
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    # 下载并加载测试数据
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    return trainloader, testloader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)       # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 10)        # 隐藏层到输出层

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建模型实例
# model = MLP()
# model = MLP().cuda()   # 将模型移动到 GPU 上

# 打印模型结构
# print(model)


# 定义损失函数和优化器
# criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# print("start train ......")


# 模型评估
def evaluate(model, testLoader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# 绘制训练过程中的损失曲线
# 记录训练过程中的损失
# losses = []

# 训练模型并记录损失
def train(model, criterion, optimizer, trainloader, losses, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()   # 清零梯度
            outputs = model(inputs)     # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()     # 反向传播和优化
            optimizer.step()    # 更新参数
            running_loss += loss.item()
        losses.append(running_loss / len(trainloader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')
    print('Finished Training')
    return losses

# 绘制损失曲线
# 创建线图
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

# 使用模型进行预测
# 选择一些测试样本进行预测
def predict(model, testloader):
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # 前向传播
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 显示预测结果
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Predicted: {predicted[i].item()}')
        plt.axis('off')

    plt.show()


# 保存模型
# torch.save(model.state_dict(), 'mnist_model.pth')

# 加载模型
# model = MLP()
# model.load_state_dict(torch.load('mnist_model.pth'))

# 使用加载的模型进行预测

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    trainloader, testloader = download_data()
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # train(model, criterion, optimizer, trainloader, losses)
    # evaluate(model, testloader)
    losses = []
    # train(model, criterion, optimizer, trainloader, losses)
    # # 保存模型
    # torch.save(model.state_dict(), 'mnist_model.pth')
    # print("save model finish")

    # 加载模型
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    print("load model finish")

    train(model, criterion, optimizer, trainloader, losses)
    print("train finish")
    plot_losses(losses)
    predict(model, testloader)
