import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34Classifier(nn.Module):
    def __init__(self, class_num, use_weights=True, dropout_rate=0.3):
        super(ResNet34Classifier, self).__init__()
        # 加载预训练的ResNet34模型
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if use_weights else None)

        # 替换最后一个全连接层
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, class_num),
        )

    def forward(self, x):
        return self.resnet34(x)

# 示例用法
if __name__ == "__main__":
    # 定义分类数目
    class_num = 10

    # 创建分类器实例
    model = ResNet34Classifier(class_num=class_num, use_weights=True)

    # 打印模型架构
    print(model)

    # 创建一个虚拟输入张量
    inputs = torch.randn(1, 3, 224, 224)

    # 前向传播
    outputs = model(inputs)

    # 打印输出形状
    print(outputs.shape)

    # 打印输出内容
    print(outputs)
