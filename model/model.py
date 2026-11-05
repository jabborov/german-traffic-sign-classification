import torch.nn as nn


class TrafficSignModel(nn.Module):
    """Small CNN for 3x32x32 GTSRB images: 2 x (Conv-BN-ReLU-MaxPool) + 2 FC layers."""

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 32x32 input -> 8x8 after two poolings
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)  # keeps the batch dimension intact (unlike view(-1, ...))
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
