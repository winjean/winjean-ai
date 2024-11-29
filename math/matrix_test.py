import numpy as np
import cv2
"""
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
"""
a=np.random.randint(0, 255, 600*800*3)
b=a.reshape(600, 800, 3)
cv2.imwrite(r"image0.jpg", b)

c=np.full((20, 600, 800, 3), 0)
print(c.shape)
c[1, :] = b
cv2.imwrite(r"image1.jpg", c[1])
c[2, :] = np.full((600, 800, 3), 255);
cv2.imwrite(r"image2.jpg", c[2])
