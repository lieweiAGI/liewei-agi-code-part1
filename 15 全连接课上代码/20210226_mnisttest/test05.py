import torch

a = torch.Tensor([[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0]])
print(a)
b = torch.argmax(a, dim=1)
print(b)