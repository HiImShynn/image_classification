import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def conv_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class AlexNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AlexNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            conv_block(self.in_channels, 96, kernel_size= 11, stride= 4, padding= 1),
            nn.MaxPool2d(3, 2),
            conv_block(96, 256, kernel_size= 5, stride= 1, padding= 2),
            nn.MaxPool2d(3, 2),
            conv_block(256, 384, kernel_size= 3, stride= 1, padding= 1),
            conv_block(384, 384, kernel_size= 3, stride= 1, padding= 1),
            conv_block(384, 256, kernel_size= 3, stride= 1, padding= 1),
            nn.MaxPool2d(3, 2),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(5 * 5 * 256, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, self.out_channels),
            nn.Softmax(dim= 1)
        )

    def forward(self, inp):
        x = self.encoder(inp)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        return x

if __name__ == "__main__":

    model = AlexNet(3, 10)
    x = torch.randn(32, 3, 224, 224)
    # print(model(ins1))
    # print(label1)

    print(model(x).shape)
