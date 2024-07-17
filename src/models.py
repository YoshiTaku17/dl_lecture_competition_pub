import torch
import torch.nn as nn

class SimpleEEGNet(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, dropout_rate: float = 0.5):
        super(SimpleEEGNet, self).__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=(in_channels, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(4 * seq_len // 4, num_classes, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_classes, False)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.batchnorm3(x)
        return x
