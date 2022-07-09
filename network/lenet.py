import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels= 6, kernel_size= 5, stride= 1, padding= 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size= 2, stride= 2)
        self.conv2 = nn.Conv2d(in_channels= 6, out_channels= 16, kernel_size= 5, stride= 1, padding= 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size= 2, stride= 2)
        self.conv3 = nn.Conv2d(in_channels= 16, out_channels= 90, kernel_size= 5, stride= 1, padding= 1)
        self.relu3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features= 160 * 3 * 3, out_features= 84)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features= 84, out_features= out_channels)
        self.softmax = nn.Softmax(dim= 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = x.reshape(x.shape[0], -1) #flatten step
        x = self.relu4(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return x

if __name__ == "__main__":
    
    x = torch.randn(64, 1, 28, 28)
    model = LeNet(1, 32)
    print(model(x).shape)