import numpy as np

a = np.array([[1,3],[3,2],[5,1]])

print(a[(-a[:,1]).argsort()])


