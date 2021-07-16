import torch
import data,net
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import optim

DEVICE = "cuda:0"

class Train():
    def __init__(self, root):
        #加载数据
        self.train_data = data.MNIST_dataset(root)
        # self.data = datasets.MNIST(root,True,transforms.ToTensor(),download=True)
        self.train_dataloader = DataLoader(self.train_data, 100, True)
        self.test_data = data.MNIST_dataset(root, False)
        self.test_dataloader = DataLoader(self.test_data, 100, True)

        #创建模型
        self.net = net.Net()

        #将模型加载到cuda上
        self.net = self.net.to(DEVICE)

        #创建优化器来优化网络参数
        # self.opt = optim.SGD(self.net.parameters(),0.1) #梯度下降法
        self.opt = optim.Adam(self.net.parameters()) #Adam优化器，自动调整学习率

    def __call__(self):
        for epoch in range(10000):
            sum_loss = 0
            for i,(img, tag) in enumerate(self.train_dataloader):
                img = img.to(DEVICE)
                tag = tag.to(DEVICE)

                # #开启训练
                # self.net.train()
                y = self.net(img)
                #定义损失函数
                loss = torch.mean((tag-y)**2)

                #梯度清零
                self.opt.zero_grad()
                #反向求导
                loss.backward()
                #梯度更新
                self.opt.step()

                sum_loss = sum_loss+loss.item()
            avg_loss = sum_loss/100
            print(avg_loss)

            torch.save(self.net.state_dict(), f"param/{epoch}.t") #保留模型参数
            torch.save(self.net, f"param/{epoch}.t") #保留整个模型

if __name__ == '__main__':
    train = Train("MNIST_IMG")
    train()
