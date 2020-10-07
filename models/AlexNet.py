import torch
import torch.nn as nn
import torch.nn.functional as F


# # The network should inherit from the nn.Module
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 1: input channels  64: output channels, 3: kernel size, 1: stride 2:padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 120, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(120, 240, 3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(240, 240, 3, 1, 2),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(240, 120, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(120 * 12 * 12, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        # it is inherit from nn.Module, nn.Module have both forward() and backward()
        # In this case, forward() link all layers together,
        # backward is already implemented to compute the gradient descents.

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        # It will 'filter' out some of the input by the probability(assign zero)
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out
