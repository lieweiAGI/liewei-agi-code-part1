import torch

x = torch.tensor([3.0],requires_grad=True)

y = x*3+2

y.backward()
print(x.grad)