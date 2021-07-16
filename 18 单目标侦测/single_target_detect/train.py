import conv_net
import dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    data_set = dataset.yellow_dataset("data")
    net = conv_net.conv_Net().cuda()

    mseloss = MSELoss()
    opt = optim.Adam(net.parameters())

    train_loader = DataLoader(data_set,100,shuffle=True)

    writer = SummaryWriter("logs")

    for epoch in range(10000):
        sum_loss = 0
        net.train()
        for i,(imgs, labels) in enumerate(train_loader):
            img = imgs.cuda()
            #换轴hwc->>chw
            img = img.permute(0,3,1,2)
            label = labels.cuda()
            out = net(img)
            loss = mseloss(out,label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item()
        avg_loss = sum_loss/len(train_loader)
        print(epoch, avg_loss)
        writer.add_scalar("avg_loss",avg_loss,epoch)
        torch.save(net.state_dict(),f"points/{epoch}.t")

