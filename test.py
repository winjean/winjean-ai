import numpy as np
full_3d = np.arange(24).reshape(2, 3, 4)
# print(full_3d)

print("shape===")
# print(full_3d.shape)

c=full_3d[:,:,3]

# print(c)
alpha=c/ 24.0
# print(alpha)

a=alpha[:,:,np.newaxis]
print(full_3d[:,:,:3])
print(a)
# print("c * a==")
print(full_3d[:,:,:3] * a)
# print(1-a)



