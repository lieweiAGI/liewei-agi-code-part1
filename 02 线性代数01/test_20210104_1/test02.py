import random
import matplotlib.pyplot as plt
import torch
from torch import nn

xs = torch.unsqueeze(torch.range(-20,20),dim=1)
ys = [e.pow(3)*random.randint(1,6) for e in xs]
ys = torch.stack(ys)
# plt.plot(xs,ys,".")
# plt.show()

#创建模型
class line(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #模型初始化
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
        out = self.layer(x)
        return out
if __name__ == '__main__':
    #创建网络对象
    net = line()
    #创建优化器
    # opt = torch.optim.SGD(net.parameters(),lr=0.9,momentum=0.1)
    opt = torch.optim.Adam(net.parameters(),lr=0.01)
    #定义损失函数(均方差)
    loss_fun = nn.MSELoss()
    plt.ion()
    #开始训练

    for epoch in range(100000):
        # 将数据输入到模型中，得到输出
        out = net.forward(xs)

        # 计算损失
        loss = loss_fun(out,ys)

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
