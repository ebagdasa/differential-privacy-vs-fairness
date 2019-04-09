import torch.nn as nn
import torchvision.models as models

from models.simple import SimpleNet


class Res(SimpleNet):
    def __init__(self, cifar10=True):
        super(Res, self).__init__()
        if cifar10:
            self.res = models.resnet18(num_classes=10)
        else:
            self.res = models.resnet18(num_classes=100)


    def forward(self, x):
        x = self.res(x)
        return x


class PretrainedRes(SimpleNet):
    def __init__(self, cifar10=True):
        super(PretrainedRes, self).__init__()
        if cifar10:
            self.res = models.resnet18(pretrained=True)
            self.fc = nn.Linear(1000, 10)
        else:
            self.res = models.resnet18(pretrained=True)
            self.fc = nn.Linear(1000, 100)


    def forward(self, x):
        x = self.res(x)
        x = self.fc(x)
        return x