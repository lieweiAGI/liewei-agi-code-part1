from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import net
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
import torch
DEVICE = "cuda:0"
import os
#tensorboard记录数据
writer = SummaryWriter("logs")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#建立数据集
train_data = datasets.CIFAR10("data", True, transforms.ToTensor(), download=True)
test_data = datasets.CIFAR10("data", False, transforms.ToTensor())
#加载数据集
train_loader = DataLoader(train_data, 200, True)
test_loader = DataLoader(test_data, 200, True)

# 创建网络
conv_net = net.Net().to(DEVICE)
# 创建优化器
opt = optim.Adam(conv_net.parameters())
# 创建损失函数
mseloss = MSELoss()

k = 0 #记录每一批loss
for epoch in range(10000):
    #开始训练
    conv_net.train()
    sum_loss = 0 #总损失
    for i, (imgs, tags) in enumerate(train_loader):
        img, tag = imgs.to(DEVICE), tags.to(DEVICE)
        out = conv_net(img)
        tag_one_hot = one_hot(tag).float()

        loss = mseloss(out, tag_one_hot)

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), k)
        k += 1
        sum_loss += loss.item()

    avg_loss = sum_loss / len(train_loader)

    test_sumloss = 0 #测试总损失
    sum_score = 0 #总正确数
    # 开始验证
    conv_net.eval()
    for i,(imgs,tags) in enumerate(test_loader):
        img, tag = imgs.to(DEVICE), tags.to(DEVICE)
        out = conv_net(img)
        tag_one_hot = one_hot(tag).float()
        #预测结果
        predict = torch.argmax(out, dim=1)
        #每批次正确个数
        score = torch.sum(torch.eq(predict, tag))

        test_loss = mseloss(out, tag_one_hot)
        test_sumloss += test_loss.item()
        sum_score += score.item()

    test_avg_loss = test_sumloss/len(test_loader)#验证平均损失
    score = sum_score/len(test_data) #正确率

    writer.add_scalars("avg_loss",{"train":avg_loss, "test":test_avg_loss},epoch)
    writer.add_scalar("score",score,epoch)
    writer.add_histogram("weight", conv_net.layer[0].weight,epoch)
    print(epoch,"=============",avg_loss)
    print(test_avg_loss, "======" ,score)
writer.close()


