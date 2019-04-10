import json
from datetime import datetime
import argparse

import torch
import torchvision
import os
import torchvision.transforms as transforms
from collections import defaultdict
from tensorboardX import SummaryWriter
import torchvision.models as models


from helper import Helper
from image_helper import ImageHelper
from models.densenet import DenseNet
from models.simple import Net
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm
import visdom
import time
import random
import yaml

from models.resnet import Res, PretrainedRes
from utils.utils import dict_html

writer = SummaryWriter()
layout = {'accuracy_per_class': {
    'accuracy_per_class': ['Multiline', ['accuracy_per_class/accuracy_var',
                                         'accuracy_per_class/accuracy_min',
                                         'accuracy_per_class/accuracy_max']]}}
writer.add_custom_scalars(layout)

torch.cuda.is_available()
torch.cuda.device_count()

import logging

logger = logging.getLogger("logger")




def plot(x, y, name):
    writer.add_scalar(tag=name, scalar_value=y, global_step=x)



def compute_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def test(net, epoch, name, testloader, vis=True):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(f'Name: {name}. Epoch {epoch}. acc: {100 * correct / total}')
    if vis:
        plot(epoch, 100 * correct / total, name)
    return 100 * correct / total


def train_dp(trainloader, model, optimizer, epoch):
    norm_type = 2
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += torch.mean(loss).item()

        losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)
        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)

        for j in losses:
            j.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), S)
            for tensor_name, tensor in model.named_parameters():
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name].add_(torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, sigma))
            tensor.grad = saved_var[tensor_name] / num_microbatches
        optimizer.step()

        if i > 0 and i % 20 == 0:
            #             logger.info('[%d, %5d] loss: %.3f' %
            #                   (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0


def clip_grad(parameters, max_norm, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type


def train(trainloader, model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i > 0 and i % 20 == 0:
            #             logger.info('[%d, %5d] loss: %.3f' %
            #                   (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f)
    helper = ImageHelper(current_time=datetime.now().strftime('%b.%d_%H.%M.%S'), params=params, name='utk')
    batch_size = int(helper.params['batch_size'])
    num_microbatches = int(helper.params['num_microbatches'])
    lr = float(helper.params['lr'])
    momentum = float(helper.params['momentum'])
    decay = float(helper.params['decay'])
    epochs = int(helper.params['epochs'])
    S = float(helper.params['S'])
    z = float(helper.params['z'])
    sigma = z * S
    dp = helper.params['dp']
    mu = helper.params['mu']
    logger.info(f'DP: {dp}')

    logger.info(batch_size)
    logger.info(lr)
    logger.info(momentum)
    helper.load_cifar_data(dataset=params['dataset'])
    helper.create_loaders()
    helper.sampler_per_class()
    helper.sampler_exponential_class(mu=mu)
    num_classes = 10 if helper.params['dataset'] == 'cifar10' else 100
    if helper.params['model'] == 'densenet':
        net = DenseNet(num_classes=num_classes, depth=helper.params['densenet_depth'])
    elif helper.params['model'] == 'resnet':
        net = models.resnet18(num_classes=num_classes)
    else:
        net = Net()

    net.cuda()
    if dp:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * epochs,
                                                                 0.75 * epochs],
                                                     gamma=0.1)


    writer.add_text('Model Params', json.dumps(helper.params))
    name = "accuracy"

    acc = test(net, 0, name, helper.test_loader, vis=True)
    for epoch in range(1, epochs):  # loop over the dataset multiple times
        if dp:
            train_dp(helper.train_loader, net, optimizer, epoch)
        else:
            train(helper.train_loader, net, optimizer, epoch)
        scheduler.step()
        acc = test(net, epoch, name, helper.test_loader, vis=True)
        acc_list = list()
        for class_no, loader in helper.per_class_loader.items():
            acc_list.append(test(net, epoch, class_no, loader, vis=False))
        plot(epoch, np.var(acc_list), name='accuracy_per_class/accuracy_var')
        plot(epoch, np.max(acc_list), name='accuracy_per_class/accuracy_max')
        plot(epoch, np.min(acc_list), name='accuracy_per_class/accuracy_min')

        helper.save_model(net, epoch, acc)
