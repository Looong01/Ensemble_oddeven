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
    

class hidden_layer(nn.Module):
    def __init__(self, classifier=True):
        super(hidden_layer, self).__init__()
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
            nn.AdaptiveAvgPool2d((1,1))
        ])

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return x
    

class DecisionModel(nn.Module):
    def __init__(self):
        super(DecisionModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 256),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
    

class ConfidenceModel(nn.Module):
    def __init__(self):
        super(ConfidenceModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 1 * 1, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x