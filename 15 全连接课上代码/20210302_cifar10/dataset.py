import torch
import net
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import one_hot

train_data = datasets.CIFAR10("data",True,transforms.ToTensor(),download=True)
test_data = datasets.CIFAR10("data",False,transforms.ToTensor())

train_dataloader = DataLoader(train_data, 500, True)
test_dataloader = DataLoader(test_data, 100, True)

DEVICE = "cuda:0"

# for img, tag in train_dataloader:
#     img = transforms.ToPILImage()(img[50])
#     img.show()

#实例化网络
net1 = net.Net().to(DEVICE)

#创建优化器
opt = optim.Adam(net1.parameters())

#损失函数,均方差
loss = torch.nn.MSELoss()

for epoch in range(10000):
    sum_loss = 0
    for i,(imgs, tags) in enumerate(train_dataloader):
        img, tag = imgs.to(DEVICE), tags.to(DEVICE)
        img = img.reshape(-1,3072)
        # print(img.shape)
        y = net1(img)
        # print(y.dtype)

        tag = one_hot(tag,10)
        # print(tag.dtype)
        loss1 = loss(y,tag.float())

        opt.zero_grad()
        loss1.backward()
        opt.step()

        print(loss1.item())
