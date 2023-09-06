import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=True, model_root=None, **kwargs):
    # model = AlexNet(**kwargs)
    if pretrained:
        model = models.alexnet(pretrained=True)
        # model1 = model_zoo.load_url(model_urls['alexnet'], model_root)
        # model.load_state_dict(model1, strict=False)

        fc_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(fc_features, 4)
        nn.Softmax(dim=1)
        print("使用预训练")
        return model
    return AlexNet(**kwargs)

if __name__ == '__main__':
    import torch
    from thop import profile
    from thop import clever_format

    model = alexnet()
    x = torch.ones(4, 3, 256, 256)
    y = model(x)
    print(y.size())
    # flops, params = profile(model, inputs=(x,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)