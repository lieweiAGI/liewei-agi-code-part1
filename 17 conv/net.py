from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.Conv2d(20, 40, 3),
            nn.ReLU(),
            nn.Conv2d(40, 80, 3),
            nn.ReLU(),
            nn.Conv2d(80, 160, 3),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(3, 20, 3)
        print(self.conv.weight)
        self.output_layer = nn.Sequential(
            nn.Linear(160 * 22 * 22, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        conv_out = self.layer(x)
        # NCHW-->NV
        conv_out = conv_out.reshape(-1, 160 * 22 * 22)
        line_out = self.output_layer(conv_out)
        return line_out


if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    y = Net()
    out = y(x)
    # out1 = y.forward(x)
    print(out.shape)
