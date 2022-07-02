"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from mobilenet_v2.cbam import CBAM, ChannelGate
from mobilenet_v2.spp_layer import spatial_pyramid_pool

__all__ = ['mobilenetv2']

model_urls = {

    'moblienetv2': 'https://download.pytorch.org/models/mobilenetv2_0.75-dace9791.pth',
}

def mobilenet_v2(pretrained=True, **kwargs):
    if pretrained:
        # # model_path = 'mobilenetv2_0.75-dace9791.pth'
        # # model_path = 'mobilenetv2-c5e733a8.pth'
        # model_path = 'mobilenetv2_0.5-eaa6f9ad.pth'
        # # 预训练参数的位置
        # model = MobileNetV2()
        # model_dict = model.state_dict()  # 网络层的参数
        # # print(model_dict)
        # # print(torch.load(model_path))
        # # 需要加载的预训练参数
        # pretrained_dict = torch.load(model_path)  # torch.load得到是字典，我们需要的是state_dict下的参数
        # # pretrained_dict = {k.replace('module.', ''): v for k, v in
        # #                    pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。
        # # print(pretrained_dict)
        # # 删除pretrained_dict.items()中model所没有的东西
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        # model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        # model_dict.pop('classifier.weight')
        # model_dict.pop('classifier.bias')
        # model.load_state_dict(model_dict, strict=False)  # model加载dict中的数据，更新网络的初始值
        # print("使用预训练")

        # pretrained_dict = model_zoo.load_url(model_urls['moblienetv2'])
        model = MobileNetV2()
        pretrained_dict = torch.load('mobilenetv2_0.75-dace9791.pth')
        model_dict = model.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'classifier' not in k)}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        model.load_state_dict(model_dict)
        return model
    return MobileNetV2(**kwargs)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3, width_mult=0.75):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.output_num = [1]
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cbam = ChannelGate(1280)
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        # print(x.size())
        spp = spatial_pyramid_pool(x, int(x.size()[0]), [int(x.size(2)), int(x.size(3))], self.output_num)
        x = spp.view(spp.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

if __name__ == '__main__':
    import torch
    from thop import profile
    from thop import clever_format

    model = mobilenet_v2()
    x = torch.ones(8, 3, 512, 512)
    y = model(x)
    print(y.size())
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)