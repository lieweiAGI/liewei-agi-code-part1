import random
import matplotlib.pyplot as plt
import torch

xs = [i/100 for i in range(100)]
ys = [3*e+4+random.random()/10 for e in xs]

#创建模型
class line(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #模型初始化
        self.w = torch.nn.Parameter(torch.randn(1))#构建系统参数，参数值是随机的
        self.b = torch.nn.Parameter(torch.randn(1))
    #前向计算
    def forward(self,x):
        return self.w * x +self.b
if __name__ == '__main__':
    #创建网络对象
    net = line()
    #创建优化器
    # opt = torch.optim.SGD(net.parameters(),lr=0.9,momentum=0.1)
    opt = torch.optim.Adam(net.parameters(),lr=0.1)
    plt.ion()
    #开始训练
    for epoch in range(30):
        for _x,_y in zip(xs,ys):
            #将数据输入到模型中，得到输出
            out = net.forward(_x)
            #定义损失函数
            loss = (out-_y)**2

            #清空梯度
            opt.zero_grad()
            #自动求导
            loss.backward()
            #更新参数
            opt.step()

            print(net.w.item(),net.b.item(),loss.item())
            # #可视化操作
            # plt.clf()
            # plt.plot(xs,ys,".")
            # v = [net.w * e + net.b for e in xs]
            # plt.plot(xs,v)
            # plt.pause(0.001)
    plt.ioff()
    plt.show()
