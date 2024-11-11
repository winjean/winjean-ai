import numpy as np


# print(np.random.rand(10))
# print(np.random.randint(0, 10, 5))
# print(np.random.randn(10))
# print(np.arange(10))
# print(np.linspace(1, 10, 4))
# print(np.zeros(4))
# print(np.ones(4))
# print(np.full(4, 5))
# print(np.eye(5))

# print(np.stack((np.zeros(4), np.ones(4), np.full(4,8))))
# print(np.hstack((np.zeros(4), np.ones(3), np.full(4,8))))
# print(np.dstack((np.zeros(4), np.ones(4), np.full(4,8))))
# print(np.vstack((np.zeros(4), np.ones(4), np.full(4,8))))
# arr = np.arange(24).reshape(2, 3, 4)
# print(arr)
# print(np.unstack(arr, axis=1))




"""
# 向量加减乘除时，两个向量的维度必须相同
# 创建两个向量
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a)
print(b)

# 向量相加
print(a + b)
# 输出: [5 7 9]

a = np.array([1, 2, 3])
print(a)
print(a + 3)
# 输出: [4 5 6]

# 向量相减
print(a - b)
# 输出: [-3 -3 -3]

# 逐元素乘法
print(a * b)
# 输出: [4 10 18]

# 逐元素除法
print(a / b)
# 输出: [0.25 0.4  0.5 ]

# 计算点积 点积是两个矢量对应元素的乘积之和
print(np.dot(a, b))
# 输出: 32

# 计算叉积 叉积只适用于三维矢量，结果是一个新的矢量，垂直于原来的两个矢量
print(np.cross(a, b))
# 输出: [-3  6 -3]

# 计算模长  模长是指矢量的长度，通常用 L2 范数表示
print(np.linalg.norm(a))
# 输出: 3.7416573867739413
"""