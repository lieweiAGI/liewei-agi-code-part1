from torch import nn
import torch

class Net(nn.Module):
    #初始化网络
    def __init__(self):
        super(Net, self).__init__()
        #输入层
        self.fc1 = nn.Linear(784,100)
        self.relu1 = nn.ReLU()
        #中间层
        self.fc2 = nn.Linear(100,64)
        self.relu2 = nn.ReLU()
        #输出层
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

class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    net = NetV2()
    x = torch.randn((5,784))
    print(net(x).shape)
    print(net.layer[0])
    print(net.layer[0].weight)