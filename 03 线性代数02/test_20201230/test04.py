import numpy as np
import torch
from torch.nn import functional

a = np.zeros((3,2))
print(a)
b = np.ones((3,2))
print(b)

a = torch.zeros(3,2,2)
print(a)
b = torch.ones(3,2)
print(b)

#单位矩阵
print(np.eye(5))
print(torch.eye(5))
print(np.eye(2,3))

#下三角矩阵
print(np.tri(3,3))
print(torch.tril(torch.ones(3,3)))

#对角矩阵
a = np.diag([1,2,3,4])
print(a)
print(torch.diag(torch.Tensor([1,2,3,4])))

#稀疏向量
print(functional.one_hot(torch.tensor(3),10))