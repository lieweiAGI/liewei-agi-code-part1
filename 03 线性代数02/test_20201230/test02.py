import numpy as np
import torch

a = np.array([[1,2],[3,4]])
print(np.linalg.det(a))

b = torch.tensor([[1.,2.],[3.,4.]])
print(b.det())