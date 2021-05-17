from torch import nn as nn
from torchvision import models as models

from dpdi.models.densenet import DenseNet
from dpdi.models.inception import inception_v3
from dpdi.models.mobilenet import MobileNetV2
from dpdi.models.resnet import get_pretrained_resnet, get_resnet_extractor
from dpdi.models.simple import FlexiNet, RegressionNet, Net
from dpdi.models.word_model import RNNModel


def get_net(helper, num_classes):
    model_type = helper.params['model']
    if model_type == 'densenet':
        net = DenseNet(num_classes=num_classes, depth=helper.params['densenet_depth'])
    elif model_type == 'resnet':
        net = models.resnet18(num_classes=num_classes)
    elif model_type == 'PretrainedRes':
        net = get_pretrained_resnet(num_classes,
                                    helper.params['freeze_pretrained_weights'])
        net = net.cuda()
    elif model_type == 'PretrainedRegressionRes':
        net = get_pretrained_resnet(1, helper.params['freeze_pretrained_weights'])

    elif model_type == 'PretrainedResExtractor':
        net = get_resnet_extractor(num_classes,
                                   helper.params['freeze_pretrained_weights'])
    elif model_type == 'FlexiNet':
        net = FlexiNet(3, num_classes)
    elif model_type == 'inception':
        net = inception_v3(pretrained=True)
        net.fc = nn.Linear(2048, num_classes)
        net.aux_logits = False
        # model = torch.nn.DataParallel(model).cuda()
    elif model_type == 'mobilenet':
        net = MobileNetV2(n_class=num_classes, input_size=64)
    elif model_type == 'word':
        net = RNNModel(rnn_type='LSTM', ntoken=helper.n_tokens,
                       ninp=helper.params['emsize'], nhid=helper.params['nhid'],
                       nlayers=helper.params['nlayers'],
                       dropout=helper.params['dropout'],
                       tie_weights=helper.params['tied'])
    elif model_type == 'regressionnet':
        net = RegressionNet(output_dim=1)
    else:
        net = Net(output_dim=num_classes)
    return net