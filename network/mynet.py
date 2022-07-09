import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class MyNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(self, MyNet).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d()