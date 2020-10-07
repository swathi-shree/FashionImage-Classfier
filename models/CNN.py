import torch
import torch.nn as nn
import torch.nn.functional as F


# # The network should inherit from the nn.Module
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # 1: input channels 6: output channels, 3: kernel size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 16, 3),
            nn.ReLU()
        )

        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(16*2*2, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)


    # it is inherit from nn.Module, nn.Module have both forward() and backward()
    # In this case, forward() link all layers together,
    # backward is already implemented to compute the gradient descents.

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # print(x.size())
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.dropout(out, 0.25)
        out = self.conv3(out)
        out = F.dropout(out, 0.2)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out
