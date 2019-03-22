import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from image_helper import ImageHelper
from torchvision import transforms
from utils.utils import *

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import visdom
import numpy as np

vis = visdom.Visdom()
import random

criterion = torch.nn.CrossEntropyLoss()


def train(helper, epoch, model):
    return


def test(helper, epoch, model):
    return 1, 1


if __name__ == '__main__':
    print('Start training')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='DPF')
    parser.add_argument('--params', dest='params', default='params_words.yaml')
    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                         name=params_loaded.get('name', 'image'))

    helper.load_data()
    model = helper.create_model()

    for epoch in range(0, helper.params['epochs'] + 1):
        start_time = time.time()

        train(helper=helper, epoch=epoch, model=model)
        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, model=model)

        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time() - start_time} sec.')

    vis.save([helper.params['environment_name']])
