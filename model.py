import torch.nn as nn


class Network(nn.Module):
    def __init__(self, hidden=128, hidden1=256, hidden2=256):
        super(Network, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.seq3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(1152, hidden)
        self.fc2 = nn.Linear(hidden, hidden1)
        self.fc3 = nn.Linear(hidden1, hidden2)
        self.fc4 = nn.Linear(hidden2, 2)

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
