import torch

a = torch.Tensor([[1,2],[3,4]])

print(torch.eig(a,eigenvectors=True))