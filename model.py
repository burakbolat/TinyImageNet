import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, ds=False):
        super().__init__()

        s = 2 if ds else 1
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, 1, stride=2) if ds else None

        self.conv1 = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, stride=1, padding=p)

    def forward(self, x):
        # Taken from ResNet paper (https://arxiv.org/pdf/1512.03385.pdf). See Fig. 2
        f_x = F.relu(self.conv1(x))
        f_x = self.conv2(f_x)

        if self.conv_1x1:
            x = self.conv_1x1(x)

        return f_x + x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_res = nn.Conv2d(3, 64, 3)  # Original paper's kernel size is 7.
                                            # This version has kernel size 5 as images are smaller (256 --> 64).
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        channels = [64, 128, 256, 512]

        # ResNet-18 like architecture with 10 layers, i.e. ResNet-10
        # See Table 1. from the paper
        self.layers = []
        self.layers.append(ResNetBlock(64, channels[0]))
        self.layers.append(ResNetBlock(64, channels[0]))
        self.layers.append(ResNetBlock(channels[0], channels[1], ds=True))
        self.layers.append(ResNetBlock(channels[1], channels[1]))
        self.layers.append(ResNetBlock(channels[1], channels[2], ds=True))
        self.layers.append(ResNetBlock(channels[2], channels[2]))
        self.layers.append(ResNetBlock(channels[2], channels[3], ds=True))
        self.layers.append(ResNetBlock(channels[3], channels[3]))
        self.layers = nn.Sequential(*self.layers)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        x = self.pre_res(x)
        # x = self.max_pool(x) Decreases dimensions too much
        x = self.layers(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x)
        return x

if __name__=="__main__":
    model = ResNet()

    input_tensor = torch.ones((32, 3, 64, 64))
    output = model(input_tensor)