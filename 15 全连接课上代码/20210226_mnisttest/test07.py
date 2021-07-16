import torch
import data,net
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import optim
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


train_dataset = datasets.MNIST("data",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST("data",train=False,transform=transforms.ToTensor(),download=False)
print(type(train_dataset))
# print(test_dataset[0])
# print(train_dataset.targets[0])
# print(test_dataset.targets[0])
# print(train_dataset.data[0].dtype)

train_dataloader = DataLoader(train_dataset,100,True)
for img, tag in train_dataloader:
    print(img.dtype)
    # print(tag.shape)
    print(one_hot(tag[5]))
    img1 = transforms.ToPILImage()(img[5])
    print(type(img1))
    img1.show()

