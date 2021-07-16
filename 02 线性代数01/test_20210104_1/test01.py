import random
import matplotlib.pyplot as plt
import torch
from torch import nn

xs = torch.unsqueeze(torch.range(-10,10),dim=1)
ys = [e.pow(3) for e in xs]
ys = torch.stack(ys)
# plt.plot(xs,ys,".")
# plt.show()

#创建模型
class line(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #模型初始化
        # self.w1 = torch.nn.Parameter(torch.randn(1,2))#构建系统参数，参数值是随机的
        # self.b1 = torch.nn.Parameter(torch.randn(2))
        # self.w2 = torch.nn.Parameter(torch.randn(2,3))
        # self.b2 = torch.nn.Parameter(torch.randn(3))
        # self.w3 = torch.nn.Parameter(torch.randn(3,1))
        # self.b3 = torch.nn.Parameter(torch.randn(1))

        # self.layer1 = nn.Linear(1,20)
        # self.layer2 = nn.Linear(20, 64)
        # self.layer3 = nn.Linear(64,128)
        # self.layer4 = nn.Linear(128,64)
        # self.layer5 = nn.Linear(64,1)
        # self.sigmoid1 = nn.ReLU()
        # self.sigmoid2 = nn.ReLU()
        # self.sigmoid3 = nn.ReLU()
        # self.sigmoid4 = nn.ReLU()
        self.layer = nn.Sequential(
            nn.Linear(1,20),
            nn.ReLU(),
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)

        )

    #前向计算
    def forward(self,x):
        #x:1,1  w:1,2
        # fc1 = torch.matmul(x,self.w1)+self.b1
        # fc2 = torch.matmul(fc1,self.w2)+self.b2
        # fc3 = torch.matmul(fc2,self.w3)+self.b3

        # fc1 = self.layer1(x)
        # fc1 = self.sigmoid1(fc1)
        # fc2 = self.layer2(fc1)
        # fc2 = self.sigmoid2(fc2)
        # fc3 = self.layer3(fc2)
        # fc3 = self.sigmoid3(fc3)
        # fc4 = self.layer4(fc3)
        # fc4 = self.sigmoid4(fc4)
        # fc5 = self.layer5(fc4)
        # return fc5
        out = self.layer(x)
        return out
if __name__ == '__main__':
    #创建网络对象
    net = line()
    #创建优化器
    # opt = torch.optim.SGD(net.parameters(),lr=0.9,momentum=0.1)
    opt = torch.optim.Adam(net.parameters(),lr=0.01)
    plt.ion()
    #开始训练

    for epoch in range(100000):
        # 将数据输入到模型中，得到输出
        out = net.forward(xs)

        # 定义损失函数(均方差损失)
        loss = torch.mean((out - ys) ** 2)

        # 清空梯度
        opt.zero_grad()
        # 自动求导
        loss.backward()
        # 更新参数
        opt.step()
        if epoch % 5 == 0:
            print(loss.item())

            # 可视化操作
            plt.clf()
            plt.plot(xs, ys, ".")
            plt.plot(xs.detach().numpy(),out.detach().numpy())
            plt.title("loss=%.4f" % loss.item(), fontdict={'size': 20, "color": 'red'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
