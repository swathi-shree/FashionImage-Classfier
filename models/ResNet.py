import torch
import torch.nn as nn
import torch.nn.functional as F




def conv(in_size, out_size, pad=1):
    return nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=pad)


class ResBlock(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, pad):
        super().__init__()
        self.conv1 = conv(in_size, hidden_size, pad)
        self.conv2 = conv(hidden_size, out_size, pad)

        # batch normalisaton used to improve speed,performance and stability of neural networks
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x):

        # skips convblock and adds input that was not passed through convblock with the output of the convblock
        return x + self.convblock(x)


class ResNet(nn.Module):

    # n_classes - fashionmnist contains 10 classes
    def __init__(self, n_classes=10):
        super().__init__()
        self.res1 = ResBlock(1, 8, 16, 15)
        self.res2 = ResBlock(16, 32, 16, 15)
        self.conv = conv(16, n_classes)
        # to improve speed and performance
        self.batchnorm = nn.BatchNorm2d(n_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.res1(x)
        x = self.res2(x)
        x = self.maxpool(self.batchnorm(self.conv(x)))
        return x.view(x.size(0), -1)
