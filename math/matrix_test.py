import numpy as np

# 创建两个矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A)
print(B)

# 矩阵加法
print(A + B)
# 输出:
# [[ 6  8]
#  [10 12]]

# 矩阵减法
print(A - B)
# 输出:
# [[-4 -4]
#  [-4 -4]]

# 矩阵乘法
print(np.dot(A, B))
# 输出:
# [[19 22]
#  [43 50]]

# 矩阵转置 矩阵转置是将矩阵的行变为列，列变为行
print(A.T)
# 输出:
# [[1 3]
#  [2 4]]

