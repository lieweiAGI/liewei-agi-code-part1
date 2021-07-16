import torch

a = torch.Tensor([1,2,3,4,5,6])
b = torch.Tensor([2,2,4,4,6,6])
c = torch.eq(a,b)
e = torch.Tensor([0,1,0,1,0,0,0,0])
d = torch.sum(e)
print(c)
print(d)