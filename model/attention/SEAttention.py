import numpy as np
import torch
from torch import nn
from torch.nn import init

# reduction：控制第一个全连接层神经元个数。直接影响SE Block的参数量和计算量，r越大，参数越少；r=16时，在精度和参数上得到好的平衡

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()

        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   # 自适应平均池化，不需要自己设置kernelsize stride等，只需要设置输出的size为1即可
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()
        )

# 下面这段代码是一个用于初始化神经网络权重的函数。它遍历了神经网络的所有模块，并根据模块的类型进行不同的权重初始化操作。具体来说：
# 对于卷积层（nn.Conv2d），使用 kaiming_normal_ 函数初始化权重，使用 fan_out 模式。如果存在偏置项（bias），则将其初始化为常数 0。
# 对于批标准化层（nn.BatchNorm2d），将权重初始化为常数 1，将偏置项初始化为常数 0。
# 对于全连接层（nn.Linear），使用 normal_ 函数初始化权重，标准差为 0.001。如果存在偏置项（bias），则将其初始化为常数 0。
# 这样的初始化操作有助于在训练神经网络时避免梯度消失或梯度爆炸的问题，以及加速训练过程。

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 通过平均池化和 Squeeze 操作得到注意力权重 y
        y = self.fc(y).view(b, c, 1, 1)  # Excitation，通过全连接层和 Excitation 操作得到注意力权重 y
        attention_weights = y  # 保存注意力权重
        out = x * y.expand_as(x)  # Scale，通过 Scale 操作得到最终的特征图
        return out, attention_weights


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)  # 2D image
    se = SEAttention(channel=512,reduction=8)
    output, atten_weights =se(input)
    print(output.shape)
    print(atten_weights.shape)  # attention_weights 的形状为 (batch_size, channels, 1, 1)，其中每个通道上都有一个相应的注意力权重

    