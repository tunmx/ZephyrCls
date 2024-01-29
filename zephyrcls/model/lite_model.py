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
        # 假设 MobileNetV2 的最终特征图尺寸是 7x7，需要根据实际情况调整
        self.pool1 = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(last_channel, 128)  # 保持 last_channel 作为输入尺寸
        self.prelu1 = nn.PReLU()
        self.cls_rec = nn.Linear(128, class_num)
        self.softmax = act_layers("Softmax")

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool1(x)
        print('pool1:', x.shape)  # 应该是 [batch_size, 1280, 1, 1]
        x = x.view(x.size(0), -1)
        print('x.view:', x.shape)  # 应该是 [batch_size, 1280]
        x = self.fc(x)
        x = self.prelu1(x)
        cls = self.cls_rec(x)
        cls = self.softmax(cls)
        return cls


if __name__ == '__main__':
    net = LiteModel(class_num=3, width_mult=1.0, )
    x = torch.rand(1, 3, 224, 224)
    y = net(x)
    print(y)
