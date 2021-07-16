# 训练P网络

import nets
import train

if __name__ == '__main__':
    net = nets.PNet()

    trainer = train.Trainer(net, './param/pnet.pt', r"D:\celeba_3\12") # 网络、保存参数、训练数据
    trainer.train()                                                    # 调用训练方法
