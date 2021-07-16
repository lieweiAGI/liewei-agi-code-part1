import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4],[8,6]])
#矩阵求逆
print(np.linalg.inv(a))
print(np.linalg.inv(b.dot(b.T)))