import numpy as np


a = np.array([[1,2],[3,4]])
b = np.linalg.inv(a)
print(b)
print(np.dot(a,b))