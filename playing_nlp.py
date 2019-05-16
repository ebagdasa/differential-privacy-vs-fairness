import logging

from models.word_model import RNNModel
from text_helper import TextHelper

logger = logging.getLogger('logger')

import json
from datetime import datetime
import argparse

import torch
import torchvision
import os
import torchvision.transforms as transforms
from collections import defaultdict, OrderedDict
from tensorboardX import SummaryWriter
import torchvision.models as models
from models.mobilenet import MobileNetV2
from helper import Helper
from image_helper import ImageHelper
from models.densenet import DenseNet
from models.simple import Net, FlexiNet, reseed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm
import time
import random
import yaml
from utils.text_load import *
from models.resnet import Res, PretrainedRes
from utils.utils import dict_html, create_table, plot_confusion_matrix
from inception import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layout = {'cosine': {
    'cosine': ['Multiline', ['cosine/0',
                                         'cosine/1',
                                         'cosine/2',
                                         'cosine/3',
                                         'cosine/4',
                                         'cosine/5',
                                         'cosine/6',
                                         'cosine/7',
                                         'cosine/8',
                                         'cosine/9']]}}


def plot(x, y, name):
    writer.add_scalar(tag=name, scalar_value=y, global_step=x)


def compute_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm



def test(model, epoch, name, data_source, vis=True):
    model.eval()
    total_loss = 0.0
    ntokens = len(helper.corpus.dictionary)
    hidden = model.init_hidden(helper.params['test_batch_size'])
    correct = 0.0
    total_test_words = 0.0
    dataset_size = len(data_source)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, helper.params['bptt']):
            data, targets = helper.get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float).item()
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = helper.repackage_hidden(hidden)
            total_test_words += targets.data.shape[0]

    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss / (dataset_size - 1)
    logger.info(f'Name: {name}. Epoch {epoch}. acc: {acc}: {correct}/{total_test_words}')
    if vis:
        plot(epoch, acc, name)

    return acc


def train_dp(train_data, model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    hidden = model.init_hidden(helper.params['batch_size'])
    logger.info(f'Training: Epoch {epoch}. ds size: {len(train_data)} ')
    for batch, i in tqdm(enumerate(range(0, train_data.size(0) - 1, helper.params['bptt']))):
        # get the inputs

        inputs, labels = helper.get_batch(train_data, i)

        inputs = inputs.to(device)
        labels = labels.to(device)

        hidden = helper.repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output, labels)
        losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)
        saved_var = dict()
        for key, tensor in model.named_parameters():
            if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                continue
            saved_var[key] = torch.zeros_like(tensor)

        for pos, j in enumerate(losses):
            j.backward(retain_graph=True)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), S)

            for key, tensor in model.named_parameters():
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                    continue
                if  tensor.grad is not None:
                     new_grad = tensor.grad
                     saved_var[key].add_(new_grad)
            model.zero_grad()

        for key, tensor in model.named_parameters():
            if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                continue
            if tensor.grad is not None:
                if device.type == 'cuda':
                    saved_var[key].add_(torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, sigma))
                else:
                    saved_var[key].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, sigma))
                tensor.grad = saved_var[key] / num_microbatches

        loss.backward()

        for key, p in model.named_parameters():
            if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                continue
            p.data.add_(-lr, p.grad.data)

        # logger.info statistics
        running_loss += loss.item()
        if batch > 0 and batch % 200 == 0:
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # plot(epoch * len(train_data) + batch, running_loss, 'Train Loss')
            running_loss = 0.0



def train(train_data, model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    hidden = model.init_hidden(helper.params['batch_size'])
    logger.info(f'Training: Epoch {epoch}. ds size: {len(train_data)} ')
    for batch, i in tqdm(enumerate(range(0, train_data.size(0) - 1, helper.params['bptt']))):
        # get the inputs

        inputs, labels = helper.get_batch(train_data, i)

        inputs = inputs.to(device)
        labels = labels.to(device)

        hidden = helper.repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, helper.n_tokens), labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
        for key, p in model.named_parameters():
            if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                continue
            p.data.add_(-lr, p.grad.data)

        # logger.info statistics
        running_loss += loss.item()
        if batch > 0 and batch % 200 == 0:
            logger.info('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            # plot(epoch * len(train_data) + batch, running_loss, 'Train Loss')
            running_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')
    writer = SummaryWriter(log_dir=f'runs/{args.name}')
    writer.add_custom_scalars(layout)

    with open(args.params) as f:
        params = yaml.load(f)
    if params.get('model', False) == 'word':
        helper = TextHelper(current_time=d, params=params, name='text')

        helper.corpus = torch.load(helper.params['corpus'])
        logger.info(helper.corpus.train.shape)
    else:
        helper = ImageHelper(current_time=d, params=params, name='utk')
    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'current path: {helper.folder_path}')
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
    reseed(5)
    if helper.params['dataset'] == 'word':
        helper.load_data()
    else:
        raise Exception('aaa')

    # helper.compute_rdp()

    reseed(5)
    if helper.params['model'] == 'word':
        net = RNNModel(rnn_type='LSTM', ntoken=helper.n_tokens,
                 ninp=helper.params['emsize'], nhid=helper.params['nhid'],
                 nlayers=helper.params['nlayers'],
                 dropout=helper.params['dropout'], tie_weights=helper.params['tied'])
    else:
        raise Exception('aaa')


    net.to(device)


    if helper.params.get('resumed_model', False):
        logger.info('Resuming training...')
        loaded_params = torch.load(f"saved_models/{helper.params['resumed_model']}")
        net.load_state_dict(loaded_params['state_dict'])
        helper.start_epoch = loaded_params['epoch']
        # helper.params['lr'] = loaded_params.get('lr', helper.params['lr'])
        logger.info(f"Loaded parameters from saved model: LR is"
                    f" {helper.params['lr']} and current epoch is {helper.start_epoch}")
    else:
        helper.start_epoch = 1

    logger.info(f'Total number of params for model {helper.params["model"]}: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    if dp:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    table = create_table(helper.params)
    writer.add_text('Model Params', table)
    epoch =0
    for epoch in range(helper.start_epoch, epochs):  # loop over the dataset multiple times
        if dp:
            train_dp(helper.train_data, net, optimizer, epoch)
        else:
            train(helper.train_data, net, optimizer, epoch)
        wh_acc = test(net, epoch, "whaccuracy", helper.wh_test_tweets, vis=True)
        aa_acc = test(net, epoch, "aaaccuracy", helper.aa_test_tweets, vis=True)

        unb_acc_dict = dict()


        helper.save_model(net, epoch, wh_acc)
    logger.info(f"Finished training for model: {helper.current_time}. Folder: {helper.folder_path}")
