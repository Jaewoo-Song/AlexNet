from torch import nn
from torch.nn import functional as F


class FashionMnistAlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            # 1x224x224
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=3),
            # 96x55x55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 96x27x27
        )

        self.conv2 = nn.Sequential(
            # 96x27x27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 256x27x27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 256x13x13
        )

        self.conv3 = nn.Sequential(
            # 256x13x13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            # 384x13x13
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            # 384x13x13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            # 384x13x13
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            # 384x13x13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 256x13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 256x6x6
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)  # 64x4096x1x1
        out = out.view(out.size(0), -1)  # 64x4096

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out

