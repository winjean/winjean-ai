import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 防止数值溢出
    return exp_x / np.sum(exp_x)


# 生成数据
x = np.linspace(-5, 5, 100)

# 计算激活函数值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)

# 绘制图形
plt.figure(figsize=(8, 12))

plt.subplot(3, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Tanh')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(x, y_relu, label='ReLU')
plt.title('ReLU')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.title('Leaky ReLU')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(x, y_elu, label='ELU')
plt.title('ELU')
plt.legend()

plt.tight_layout()
plt.show()
