from torch import nn
import torch
class Net(nn.Module):
    #初始化网络
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100,64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64,10)
        self.softmax1 = nn.Softmax(dim=1)

    #前向计算
    def forward(self, x):
        h = self.fc1(x)
        h = self.relu1(h)
        h = self.fc2(h)
        h = self.relu2(h)
        h = self.fc3(h)
        h = self.softmax1(h)
        return h

if __name__ == '__main__':
    net = Net()
    x = torch.randn((1,784))
    print(net(x).shape)