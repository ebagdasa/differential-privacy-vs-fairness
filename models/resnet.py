import torch.nn as nn
import torchvision.models as models

from models.simple import SimpleNet


def get_resnet_extractor(num_classes, freeze_pretrained_weights:bool):
    """Fetch a pretrained resnet feature extractor, with a FC layer.

    This model uses the feature output of the conv layer (the correct way to use the
    pretrained weights), not the output of the ImageNet classification layer (which is
    how the Bagdasaryan et al. models work).
    """
    net = models.resnet18(pretrained=True)
    feature_extractor = nn.Sequential(
        *list(net.children())[:-2] + [nn.Flatten(), nn.Linear(2048, num_classes)])
    if freeze_pretrained_weights:
        for parameter in feature_extractor.parameters():
            parameter.requires_grad = False
        # Set the last layer to be trainable
        for parameter in feature_extractor[-1].parameters():
            parameter.requires_grad = True
    return feature_extractor


def get_pretrained_resnet(num_classes, freeze_pretrained_weights:bool):
    """Fetch a pretrained resnet with a new FC layer with num_classes.

    This is the original pretrained model implemented in (Bagdasaryan et al.), but it
    uses the output of the ResNet classification layer, not the feature output of the
    conv layer. (To use the feature output of the conv layer, see get_resnet_extractor().)
    """
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, num_classes)
    if freeze_pretrained_weights:
        for parameter in net.parameters():
            parameter.requires_grad = False
        # Set the last layer to be trainable
        for parameter in net.fc.parameters():
            parameter.requires_grad = True
    return net


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
    def __init__(self, no_classes):
        super(PretrainedRes, self).__init__()
        self.res = models.resnet101(pretrained=True)
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, no_classes)


    def forward(self, x):
        x = self.res(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x