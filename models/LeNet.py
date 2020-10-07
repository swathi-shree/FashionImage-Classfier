import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # input of 1 - grayscale image, #filter size - 5
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        # dropout used to prevent overfitting
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=128, out_features=50)
        # out features - 10 classes
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 128)
        # converts data into 1d array
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
