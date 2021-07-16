from torch import nn
import torch
class conv_Net(nn.Module):
    def __init__(self):
        super(conv_Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 20, 3, 1), #298
            nn.MaxPool2d(2), #149
            nn.ReLU(),

            nn.Conv2d(20,40,3,1,1),#149
            nn.MaxPool2d(2),#74
            nn.ReLU(),
            #
            nn.Conv2d(40,80,3,1),#72
            nn.MaxPool2d(2),#36
            nn.ReLU(),
            #
            nn.Conv2d(80,160,3,2,1), #18
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layer = nn.Linear(160*18*18, 4)

    def forward(self, x):
        conv_out = self.layer(x)
        line_out = self.fc_layer(conv_out)
        return line_out

if __name__ == '__main__':
    x = torch.randn(1,3,300,300)
    y = conv_Net()
    out =y(x)
    print(out.shape)