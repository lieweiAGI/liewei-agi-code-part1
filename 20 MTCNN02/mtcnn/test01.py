import torch

a = torch.rand(3,6)
print(a)
print(torch.gt(a, 0.5))
idxs = torch.nonzero(torch.gt(a, 0.5))
print(idxs)
print(idxs[:, 0])
# for idx in idxs:
#     print(a[idx[0],idx[1]])