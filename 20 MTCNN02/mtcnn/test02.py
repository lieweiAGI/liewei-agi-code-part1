import torch

a = torch.tensor([1,2,3,4,5])

print(torch.lt(a,4))
print(a<4)
print(torch.gt(a,4))
print(torch.le(a,4))
print(torch.masked_select(a,torch.lt(a,4)))
print(a[torch.lt(a,4)])