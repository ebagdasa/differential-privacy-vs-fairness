import logging

from models.word_model import RNNModel
from text_helper import TextHelper

logger = logging.getLogger('logger')

import json
from datetime import datetime
import argparse
from scipy import ndimage
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
from models.simple import Net, FlexiNet, reseed, RegressionNet
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


def check_tensor_finite(x: torch.Tensor):
    if torch.isnan(x).any():
        logger.warning("nan values detected in tensor.")
    if torch.isinf(x).any():
        logger.warning("inf values detected in tensor.")
    return


def make_uid(params, number_of_entries_train:int=None):
    # If number_of_entries_train is provided, it overrides the params file. Otherwise,
    # fetch the value from the params file.
    if number_of_entries_train is None:
        number_of_entries_train = params.get('number_of_entries')
    uid = "{dataset}-sigma{sigma}-alpha-{alpha}-ada{adaptive_sigma}-n{n}".format(
        dataset=params['dataset'],
        sigma=params.get('sigma'), alpha=params.get('alpha'),
        adaptive_sigma=params.get('adaptive_sigma', False),
        n=number_of_entries_train)
    if params.get('positive_class_keys') and params.get('negative_class_keys'):
        pos_keys = [str(i) for i in params['positive_class_keys']]
        neg_keys = [str(i) for i in params['negative_class_keys']]
        pos_keys_str = '-'.join(pos_keys)
        neg_keys_str = '-'.join(neg_keys)
        keys_str = pos_keys_str + '-vs-' + neg_keys_str
        uid += '-' + keys_str
    if params.get('target_colname'):
        uid += '-' + params['target_colname']
    return uid


def plot(x, y, name):
    writer.add_scalar(tag=name, scalar_value=y, global_step=x)


def compute_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def compute_mse(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert outputs.shape == labels.shape, \
        "Expected outputs and labels same shape, got shapes {} and {}".format(
            outputs.shape, labels.shape
        )
    mse = (outputs - labels) ** 2
    return torch.mean(mse)


def per_class_mse(outputs, labels, target_class, grouped_label=None) -> torch.Tensor:
    per_class_idx = labels == target_class
    per_class_outputs = outputs[per_class_idx]
    if grouped_label is not None:
        # Create a new labels tensor, with all values equal to grouped_label
        per_class_labels = torch.full_like(per_class_outputs,
                                           fill_value=grouped_label, dtype=torch.float32)
    else:
        # Use the existing labels tensor, with all values equal to target_class
        per_class_labels = labels[per_class_idx]
    mse_per_class = compute_mse(per_class_outputs, per_class_labels)
    return mse_per_class


def test(net, epoch, name, testloader, vis=True, mse: bool = False,
         labels_mapping: dict = None):
    net.eval()
    running_metric_total = 0
    n_test = 0
    i = 0
    correct_labels = []
    predict_labels = []
    metric_name = 'accuracy' if not mse else 'mse'
    with torch.no_grad():
        for data in tqdm(testloader):
            if helper.params['dataset'] == 'dif':
                inputs, idxs, labels = data
            else:
                inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            check_tensor_finite(labels)
            check_tensor_finite(outputs)

            n_test += labels.size(0)
            if not mse:
                _, predicted = torch.max(outputs.data, 1)
                predict_labels.extend([x.item() for x in predicted])
                correct_labels.extend([x.item() for x in labels])
                running_metric_total += (predicted == labels).sum().item()
                main_test_metric = 100 * running_metric_total / n_test
            else:
                assert labels_mapping, "provide labels_mapping to use mse."
                pos_labels = [k for k, v in labels_mapping.items() if v == 1]
                binarized_labels_tensor = binarize_labels_tensor(labels, pos_labels)

                running_metric_total += compute_mse(outputs, binarized_labels_tensor)
                main_test_metric = running_metric_total / n_test
            # logger.info(f'Name: {name}. Epoch {epoch}. {metric_name}: {
            # main_test_metric}')

    if vis:
        plot(epoch, main_test_metric, metric_name)
        metric_list = list()
        metric_dict = dict()
        if not mse:
            fig, cm = plot_confusion_matrix(correct_labels, predict_labels,
                                            labels=helper.labels, normalize=True)
            writer.add_figure(figure=fig, global_step=epoch, tag='tag/normalized_cm')
        for i, class_name in enumerate(helper.labels):
            if not mse:
                metric_value = cm[i][i] / cm[i].sum() * 100
                fig, cm = plot_confusion_matrix(correct_labels, predict_labels,
                                                labels=helper.labels, normalize=False)
                cm_name = f'{helper.params["folder_path"]}/cm_{epoch}.pt'
                torch.save(cm, cm_name)
                writer.add_figure(figure=fig, global_step=epoch,
                                  tag='tag/unnormalized_cm')
            else:
                metric_value = per_class_mse(
                    outputs, labels, class_name, grouped_label=labels_mapping[class_name]
                ).cpu().numpy()
            metric_dict[class_name] = metric_value
            logger.info(f'Class: {i}, {class_name}: {metric_value}')
            plot(epoch, metric_value, name=f'{metric_name}_per_class/class_{class_name}')
            metric_list.append(metric_value)

        fig2 = helper.plot_acc_list(metric_dict, epoch, name='per_class',
                                    accuracy=main_test_metric)
        writer.add_figure(figure=fig2, global_step=epoch, tag='tag/per_class')
        torch.save(metric_dict,
                   f"{helper.folder_path}/test_{metric_name}_class_{epoch}.pt")

        plot(epoch, np.var(metric_list),
             name=f'{metric_name}_per_class/{metric_name}_var')
        plot(epoch, np.max(metric_list),
             name=f'{metric_name}_per_class/{metric_name}_max')
        plot(epoch, np.min(metric_list),
             name=f'{metric_name}_per_class/{metric_name}_min')
        plot(epoch, np.max(metric_list) - np.min(metric_list),
             name=f'{metric_name}_intra_class_max_diff/'
                  f'{metric_name}_intra_class_max_diff')

    return main_test_metric


def binarize_labels_tensor(labels: torch.Tensor, pos_labels: list):
    binary_labels = torch.zeros_like(labels, dtype=torch.float32)
    for l in pos_labels:
        is_l = (labels == l)
        binary_labels += is_l.type(torch.float32)
    assert torch.max(binary_labels) <= 1., "Sanity check on binarized grouped labels."
    return binary_labels


def train_dp(trainloader, model, optimizer, epoch, sigma, alpha, labels_mapping=None,
             adaptive_sigma=False):
    norm_type = 2
    model.train()
    running_loss = 0.0
    label_norms = defaultdict(list)
    ssum = 0
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        if helper.params['dataset'] == 'dif':
            inputs, idxs, labels = data
        elif helper.params['dataset'] == 'celeba':
            inputs, anno, labels = data
        else:
            inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()

        outputs = model(inputs)

        if labels_mapping:
            pos_labels = [k for k, v in labels_mapping.items() if v == 1]
            binarized_labels_tensor = binarize_labels_tensor(labels, pos_labels)
            loss = criterion(outputs, binarized_labels_tensor)
        else:
            loss = criterion(outputs, labels)

        running_loss += torch.mean(loss).item()

        losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)
        grad_vecs_sum_by_label = dict()
        grad_vecs = list()
        count_vecs = defaultdict(int)
        for pos, j in enumerate(losses):
            j.backward(retain_graph=True)

            grad_vec = helper.get_grad_vec(model, device)
            grad_vecs.append(grad_vec)
            # Note: by default, count_norm_cosine_per_batch is set to false in our params.
            if helper.params.get('count_norm_cosine_per_batch', False):

                label = labels[pos].item()
                count_vecs[label] += 1
                if grad_vecs_sum_by_label.get(label, False) is not False:
                    grad_vecs_sum_by_label[label].add_(grad_vec)
                else:
                    grad_vecs_sum_by_label[label] = grad_vec

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), S)
            if helper.params['dataset'] == 'dif':
                label_norms[f'{labels[pos]}_{helper.label_skin_list[idxs[pos]]}'].append(
                    total_norm)
            else:
                label_norms[int(labels[pos])].append(total_norm)

            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    check_tensor_finite(new_grad)
                    check_tensor_finite(tensor)
                    # logger.info('new grad: ', new_grad)
                    saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        # Compute average norm and the sigma value (if adaptive)
        grad_norms = [torch.norm(x, p=2) for x in grad_vecs]
        avg_grad_norm = torch.mean(torch.stack(grad_norms))
        if adaptive_sigma:
            # Case: Use adaptive noise
            sigma_dp = alpha * avg_grad_norm
        else:
            # Case: Do not use adaptive noise
            sigma_dp = sigma

        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                # Sometimes we use dp training even when sigma is set to zero (to get
                #  gradient magnitudes); we do not add noise when sigma==0.
                if sigma_dp > 0:
                    if device.type == 'cuda':
                        saved_var[tensor_name].add_(
                            torch.cuda.FloatTensor(tensor.grad.shape).normal_(0,
                                                                              sigma_dp))
                    else:
                        saved_var[tensor_name].add_(
                            torch.FloatTensor(tensor.grad.shape).normal_(0, sigma_dp))
                tensor.grad = saved_var[tensor_name] / num_microbatches
                check_tensor_finite(tensor.grad)

        optimizer.step()

        if i > 0 and i % 20 == 0:
            logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0
    print(ssum)
    plot(epoch, avg_grad_norm, "norms/avg_grad_norm")
    for pos, norms in sorted(label_norms.items(), key=lambda x: x[0]):
        logger.info(f"{pos}: {torch.mean(torch.stack(norms))}")
        if helper.params['dataset'] == 'dif':
            plot(epoch, torch.mean(torch.stack(norms)), f'dif_norms_class/{pos}')
        else:
            plot(epoch, torch.mean(torch.stack(norms)), f'norms/class_{pos}')


def train(trainloader, model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        # get the inputs
        if helper.params['dataset'] == 'dif':
            inputs, idxs, labels = data
        else:
            inputs, labels = data

        keys_input = labels == helper.params['key_to_drop']
        inputs_keys = inputs[keys_input]

        inputs[keys_input] = torch.tensor(
            ndimage.filters.gaussian_filter(inputs[keys_input].numpy(),
                                            sigma=helper.params['csigma']))
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float32)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # logger.info statistics
        running_loss += loss.item()
        if i > 0 and i % 20 == 0:
            #             logger.info('[%d, %5d] loss: %.3f' %
            #                   (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    # parser.add_argument('--name', dest='name', required=True)
    parser.add_argument("--majority_key", default=None, type=int,
                        help="Optionally specify the majority group key (e.g. '1').")
    parser.add_argument("--number_of_entries_train", default=None, type=int,
                        help="Optional number of minority class entries/size to "
                             "downsample to; if provided, this value overrides value in "
                             ".yaml parameters.")
    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params = yaml.load(f)
    name = make_uid(params, number_of_entries_train=args.number_of_entries_train)

    writer = SummaryWriter(log_dir=f'runs/{name}')
    writer.add_custom_scalars(layout)

    if params.get('model', False) == 'word':
        helper = TextHelper(current_time=d, params=params, name='text')

        helper.corpus = torch.load(helper.params['corpus'])
        logger.info(helper.corpus.train.shape)
    else:
        helper = ImageHelper(current_time=d, params=params, name=name)
    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'experiment uid: {name}')
    logger.info(f'current path: {helper.folder_path}')
    batch_size = int(helper.params['batch_size'])
    num_microbatches = int(helper.params['num_microbatches'])
    lr = float(helper.params['lr'])
    momentum = float(helper.params['momentum'])
    decay = float(helper.params['decay'])
    epochs = int(helper.params['epochs'])
    z = float(helper.params['z'])
    # If clipping bound S is not specified, it is set to inf.
    S = float(helper.params['S'])
    if helper.params.get('S') != 'inf':
        # Case: clipping bound S is specified; use this to compute sigma.
        sigma = z * S
    else:
        # Case: clipping bound S is not specified (no clipping);
        # sigma must be set explicitly in the params.
        sigma = helper.params.get('sigma')
    alpha = helper.params.get('alpha')
    adaptive_sigma = helper.params.get('adaptive_sigma', False)
    logger.debug("sigma = %s" % sigma)
    logger.debug("alpha = %s" % alpha)
    logger.debug("adaptive_sigma = %s" % adaptive_sigma)
    dp = helper.params['dp']
    mu = helper.params['mu']
    logger.info(f'DP: {dp}')

    logger.info(batch_size)
    logger.info(lr)
    logger.info(momentum)
    reseed(5)
    classes_to_keep = None
    true_labels_to_binary_labels = None
    if helper.params['dataset'] == 'inat':
        helper.load_inat_data()
        helper.balance_loaders()
    elif helper.params['dataset'] == 'word':
        helper.load_data()
    elif helper.params['dataset'] == 'dif':
        helper.load_dif_data()
        helper.get_unbalanced_faces()
    elif helper.params['dataset'] == 'celeba':
        helper.load_celeba_data()
    else:
        if helper.params.get('binary_mnist_task'):
            # Labels are assigned in order of index in this array; so minority_key has
            # label 0, majority_key has label 1.
            classes_to_keep = (args.majority_key, helper.params['key_to_drop'])
            true_labels_to_binary_labels = {
                label: i for i, label in enumerate(classes_to_keep)}
        elif helper.params.get('grouped_mnist_task'):
            classes_to_keep = helper.params['positive_class_keys'] + \
                              helper.params['negative_class_keys']
            true_labels_to_binary_labels = {
                label: int(label in helper.params['positive_class_keys'])
                for label in classes_to_keep}
        else:
            raise ValueError
        helper.load_cifar_data(dataset=params['dataset'], classes_to_keep=classes_to_keep)
        logger.info('before loader')
        helper.create_loaders()
        logger.info('after loader')

        keys_to_drop = params.get('key_to_drop')
        if not isinstance(keys_to_drop, list):
            keys_to_drop = list(keys_to_drop)
        # Create a unique DataLoader for each class
        helper.sampler_per_class()
        logger.info('after sampler')
        if args.number_of_entries_train:
            number_of_entries_train = args.number_of_entries_train
            print("[INFO] overriding number of entries in parameters file; "
                  "using %s entries" % number_of_entries_train)
        else:
            number_of_entries_train = params['number_of_entries']
        helper.sampler_exponential_class(mu=mu, total_number=params['ds_size'],
                                         keys_to_drop=keys_to_drop,
                                         number_of_entries=number_of_entries_train)
        logger.info('after sampler expo')
        helper.sampler_exponential_class_test(mu=mu, keys_to_drop=keys_to_drop,
                                              number_of_entries_test=params[
                                                  'number_of_entries_test'])
        logger.info('after sampler test')
    if dp and sigma != 0:
        helper.compute_rdp()
    num_classes = helper.get_num_classes(classes_to_keep)
    print('[DEBUG] num_classes is %s' % num_classes)
    reseed(5)
    if helper.params['model'] == 'densenet':
        net = DenseNet(num_classes=num_classes, depth=helper.params['densenet_depth'])
    elif helper.params['model'] == 'resnet':
        logger.info(f'Model size: {num_classes}')
        net = models.resnet18(num_classes=num_classes)
    elif helper.params['model'] == 'PretrainedRes':
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(512, num_classes)
        net = net.cuda()
    elif helper.params['model'] == 'FlexiNet':
        net = FlexiNet(3, num_classes)
    elif helper.params['model'] == 'dif_inception':
        net = inception_v3(pretrained=True, dif=True)
        net.fc = nn.Linear(768, num_classes)
        net.aux_logits = False
    elif helper.params['model'] == 'inception':
        net = inception_v3(pretrained=True)
        net.fc = nn.Linear(2048, num_classes)
        net.aux_logits = False
        # model = torch.nn.DataParallel(model).cuda()
    elif helper.params['model'] == 'mobilenet':
        net = MobileNetV2(n_class=num_classes, input_size=64)
    elif helper.params['model'] == 'word':
        net = RNNModel(rnn_type='LSTM', ntoken=helper.n_tokens,
                       ninp=helper.params['emsize'], nhid=helper.params['nhid'],
                       nlayers=helper.params['nlayers'],
                       dropout=helper.params['dropout'],
                       tie_weights=helper.params['tied'])
    elif helper.params['model'] == 'regressionnet':
        net = RegressionNet(output_dim=1)
    else:
        net = Net(output_dim=num_classes)
    if helper.params.get('multi_gpu', False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

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

    logger.info(
        'Total number of params for model {}: {}'.format(
            helper.params["model"],
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        ))

    # For DP training, no loss reduction is used; otherwise, use default (mean) reduction.
    if helper.params.get('criterion') == 'mse':  # Case: MSE objective.
        print('[DEBUG] using MSE loss')
        if dp:
            criterion = nn.MSELoss(reduction='none')
        else:
            criterion = nn.MSELoss()
    else:  # Case: not MSE; use cross-entropy objective.
        if dp:
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = nn.CrossEntropyLoss()

    if helper.params['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                              weight_decay=decay)
    elif helper.params['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
    else:
        raise Exception('Specify `optimizer` in params.yaml.')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * epochs,
                                                                 0.75 * epochs],
                                                     gamma=0.1)

    table = create_table(helper.params)
    writer.add_text('Model Params', table)
    logger.info(table)
    logger.info(helper.labels)
    epoch = 0
    metric_name = 'mse' if helper.params.get('criterion') == 'mse' else 'accuracy'
    for epoch in range(helper.start_epoch,
                       epochs):  # loop over the dataset multiple times
        if dp:
            train_dp(helper.train_loader, net, optimizer, epoch,
                     labels_mapping=true_labels_to_binary_labels,
                     sigma=sigma, alpha=alpha, adaptive_sigma=adaptive_sigma)
        else:
            raise NotImplementedError(
                "Label binarization is not implemented for non-DP training.")
            train(helper.train_loader, net, optimizer, epoch)
        if helper.params['scheduler']:
            scheduler.step()
        test_loss = test(net, epoch, name, helper.test_loader,
                         mse=metric_name == 'mse',
                         labels_mapping=true_labels_to_binary_labels)
        unb_acc_dict = dict()
        if helper.params['dataset'] == 'dif':
            for name, value in sorted(helper.unbalanced_loaders.items(),
                                      key=lambda x: x[0]):
                unb_acc = test(net, epoch, name, value, vis=False)
                plot(epoch, unb_acc, name=f'dif_unbalanced/{metric_name}')
                unb_acc_dict[name] = unb_acc

            unb_acc_list = list(unb_acc_dict.values())
            logger.info(f'Accuracy on unbalanced set: {sorted(unb_acc_list)}')

            plot(epoch, np.mean(unb_acc_list), f'accuracy_detailed/mean')
            plot(epoch, np.min(unb_acc_list), f'accuracy_detailed/min')
            plot(epoch, np.max(unb_acc_list), f'accuracy_detailed/max')
            plot(epoch, np.var(unb_acc_list), f'accuracy_detailed/var')

            fig = helper.plot_acc_list(unb_acc_dict, epoch, name='per_subgroup',
                                       accuracy=test_loss)

            torch.save(unb_acc_dict, f"{helper.folder_path}/acc_subgroup_{epoch}.pt")
            writer.add_figure(figure=fig, global_step=epoch, tag='tag/subgroup')

        helper.save_model(net, epoch, test_loss)
    logger.info(
        f"Finished training for model: {helper.current_time}. Folder: {helper.folder_path}")
