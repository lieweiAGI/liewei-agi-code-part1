import random
import matplotlib.pyplot as plt
import torch

xs = torch.range(1,100)/100
xs = xs.reshape(-1,1)

ys = 3*xs+4+random.random()/10


#创建模型
class line(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #模型初始化
        self.w1 = torch.nn.Parameter(torch.randn(1,2))#构建系统参数，参数值是随机的
        self.b1 = torch.nn.Parameter(torch.randn(2))
        self.w2 = torch.nn.Parameter(torch.randn(2,3))
        self.b2 = torch.nn.Parameter(torch.randn(3))
        self.w3 = torch.nn.Parameter(torch.randn(3,1))
        self.b3 = torch.nn.Parameter(torch.randn(1))
    #前向计算
    def forward(self,x):
        #x:1,1  w:1,2
        fc1 = torch.matmul(x,self.w1)+self.b1
        fc2 = torch.matmul(fc1,self.w2)+self.b2
        fc3 = torch.matmul(fc2,self.w3)+self.b3
        return fc3
if __name__ == '__main__':
    #创建网络对象
    net = line()
    #创建优化器
    # opt = torch.optim.SGD(net.parameters(),lr=0.9,momentum=0.1)
    opt = torch.optim.Adam(net.parameters())
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

            print(loss.item())

            #可视化操作
            # plt.clf()
            plt.plot(xs,ys,".")
            out = net.forward(xs)
            plt.plot(xs.detach().numpy(),out.detach().numpy())
            plt.pause(0.001)
    plt.ioff()
    plt.show()
