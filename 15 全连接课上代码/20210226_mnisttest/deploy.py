from torch import jit
import net
import torch

if __name__ == '__main__':
    model = net.NetV2()
    model.load_state_dict(torch.load("param/14.t"))

    #定义一个虚拟输入  占位
    input = torch.randn(1,784)

    #打包模型，保留一个输入(可以被C++,JAVA掉用了)
    packle = torch.jit.trace(model,input)
    packle.save("mnsit.pt")