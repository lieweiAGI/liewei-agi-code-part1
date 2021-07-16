import torch
import data,net
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
import os

DEVICE = "cuda:0"




class Train():
    def __init__(self, root):
        self.summarywriter = SummaryWriter("logs")

        #加载数据
        # self.train_data = data.MNIST_dataset(root)
        self.train_data = datasets.MNIST(root,True,transforms.ToTensor(),download=True)
        self.train_dataloader = DataLoader(self.train_data, 100, True)
        # self.test_data = data.MNIST_dataset(root, False)
        self.test_data = datasets.MNIST(root, False, transforms.ToTensor())
        self.test_dataloader = DataLoader(self.test_data, 100, True)

        #创建模型
        self.net = net.NetV2()

        #将模型加载到cuda上
        self.net = self.net.to(DEVICE)

        #创建优化器来优化网络参数
        # self.opt = optim.SGD(self.net.parameters(),0.1) #梯度下降法
        self.opt = optim.Adam(self.net.parameters()) #Adam优化器，自动调整学习率

    def __call__(self):
        k = 0
        for epoch in range(10000):
            sum_loss = 0
            for i,(img, tag) in enumerate(self.train_dataloader):
                # print(i)
                img = img.to(DEVICE)
                img = img.reshape(-1,784)
                # print(img.shape)
                tag = tag.to(DEVICE)
                # print(tag.shape)
                tag = one_hot(tag,10)
                # print(tag.shape)
                #开启训练
                self.net.train()

                y = self.net(img)
                #定义损失函数
                loss = torch.mean((tag-y)**2)

                #梯度清零
                self.opt.zero_grad()
                #反向求导
                loss.backward()
                #梯度更新
                self.opt.step()

                # print(loss.item())
                sum_loss = sum_loss+loss.item()
                self.summarywriter.add_scalar("loss", loss.item(), k)
                k += 1
            avg_loss = sum_loss/len(self.train_dataloader)

            print(epoch,'=======',avg_loss)

            #验证模型
            test_sum_loss = 0 #总损失
            sum_score = 0 #总分数
            for i,(img,tag) in enumerate(self.test_dataloader):
                img, tag = img.to(DEVICE), tag.to(DEVICE)
                #开启测试
                self.net.eval()
                img = img.reshape(-1,784)
                test_y = self.net(img)
                tag = one_hot(tag)
                test_loss = torch.mean((tag-test_y)**2)
                test_sum_loss += test_loss

                # 计算分数，计算正确率
                predict_tags = torch.argmax(test_y, dim=1)
                # print(predict_tags)
                label = torch.argmax(tag, dim=1)
                # print(label)
                sum_score += torch.sum(torch.eq(predict_tags,label))
                # print(a)
                # print(a.shape)

            #平均损失
            test_avg_loss = test_sum_loss/len(self.test_dataloader)
            #平均分数
            score = sum_score/len(self.test_data)
            print("平均损失",test_avg_loss.item(),"得分",score.item())
            # self.summarywriter.add_scalars("avgloss", {"avgloss":avg_loss,"testavgloss":test_avg_loss}, epoch)
            # self.summarywriter.add_scalar("score", score,epoch)

            self.summarywriter.add_histogram("weight", self.net.layer[0].weight, epoch)
            # self.summarywriter.add_histogram("weight", self.net.layer[2].weight, epoch)
            # self.summarywriter.add_histogram("weight", self.net.layer[4].weight, epoch)
            torch.save(self.net.state_dict(), f"param/{epoch}.t") #保留模型参数
            # torch.save(self.net, f"param/{epoch}.t") #保留整个模型
        self.summarywriter.close()
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    train = Train("data")
    train()
