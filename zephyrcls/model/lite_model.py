import torch
import torch.nn as nn
import torch.nn.functional as f
from .backbone import MobileNetV2
from .backbone import act_layers


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class LiteModel(nn.Module):
    def __init__(self, class_num, width_mult=1., last_channel=1280, pool_pad=3):
        super().__init__()
        self.backbone = MobileNetV2(width_mult=width_mult, last_channel=last_channel)
        # self.pool1 = nn.AvgPool2d(pool_pad, stride=1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 128)
        self.prelu1 = nn.PReLU()
        self.cls_rec = nn.Linear(128, class_num)
        self.softmax = act_layers("Softmax")

    def forward(self, x):
        x = self.backbone(x)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc(x)
        x = self.prelu1(x)
        cls = self.cls_rec(x)
        cls = self.softmax(cls)

        return cls


if __name__ == '__main__':
    net = LiteModel(class_num=3, width_mult=0.35, )
    x = torch.rand(1, 3, 96, 96)
    y = net(x)
    print(y)
