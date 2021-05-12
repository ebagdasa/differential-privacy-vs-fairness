import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import re
import itertools
import matplotlib
matplotlib.use('AGG')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output



def poison_random(batch, target, poisoned_number, poisoning, test=False):

    batch = batch.clone()
    target = target.clone()
    for iterator in range(0,len(batch)-1,2):

        if random.random()<poisoning:
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator + 1] = batch[iterator]
            batch[iterator+1][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)

            target[iterator+1] = poisoned_number
    return (batch, target)

def poison_test_random(batch, target, poisoned_number, poisoning, test=False):
    for iterator in range(0,len(batch)):
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator] = batch[iterator]
            batch[iterator][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)


            target[iterator] = poisoned_number
    return (batch, target)


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def clip_grad_norm_dp(named_parameters, target_params, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p[1]-target_params[p[0]], named_parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def create_table(params: dict):
    header = f"| {' | '.join([x[:12] for x in params.keys() if x != 'folder_path'])} |"
    line = f"|{'|:'.join([3*'-' for x in range(len(params.keys())-1)])}|"
    values = f"| {' | '.join([str(params[x]) for x in params.keys() if x != 'folder_path'])} |"
    return '\n'.join([header, line, values])




def plot_confusion_matrix(correct_labels, predict_labels,
                          labels,  title='Confusion matrix',
                          tensor_name = 'Confusion', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)




    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', str(x)) for x in labels]
    classes = ['\n'.join(l) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=8, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]:.2f}" if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=10,
                verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    return fig, cm
