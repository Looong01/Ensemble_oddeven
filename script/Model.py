import torch.nn as nn

class ConvBN2d(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=False, bias=True):
        super(ConvBN2d, self).__init__()
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        layers = []
        if relu:
            layers.append(nn.ReLU(inplace=True))
        if bn:
            layers.append(nn.BatchNorm2d(out_dim))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.features(x)
        return x
    

class CNN(nn.Module):
    def __init__(self, classifier=True):
        super(CNN, self).__init__()

        self.features = nn.ModuleList([
            ConvBN2d(1, 16, kernel_size=7, stride=1, bn=True, relu=True),
            ConvBN2d(16, 16, kernel_size=5, stride=1, bn=True, relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBN2d(16, 32, kernel_size=3, stride=1, bn=True, relu=True),
            ConvBN2d(32, 32, kernel_size=3, stride=1, bn=True, relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBN2d(32, 64, kernel_size=3, stride=1, bn=True, relu=True),
            ConvBN2d(64, 64, kernel_size=3, stride=1, bn=True, relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.classifier = None
        if classifier:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256),
                nn.Linear(256, 1),
            )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBN2d(in_channels, out_channels, kernel_size=3, stride=stride, bn=True, relu=True)
        self.conv2 = ConvBN2d(out_channels, out_channels, kernel_size=3, stride=1, bn=True, relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)  # 添加 Dropout 层
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)  # 在残差连接前应用 Dropout
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out
    


class MyResNet(nn.Module):
    def __init__(self, classifier=True, dropout_prob=0.0):
        super(MyResNet, self).__init__()

        self.features = nn.Sequential(
            ConvBN2d(1, 16, kernel_size=7, stride=1, bn=True, relu=True),
            ConvBN2d(16, 16, kernel_size=5, stride=1, bn=True, relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(16, 16, dropout_prob=dropout_prob),
            ResidualBlock(16, 32, stride=2, dropout_prob=dropout_prob),
            ResidualBlock(32, 64, stride=2, dropout_prob=dropout_prob),
        )

        self.classifier = None
        if classifier:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),  # 在全连接层前应用 Dropout
                nn.Linear(256, 1),
            )

    def forward(self, x):
        x = self.features(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x