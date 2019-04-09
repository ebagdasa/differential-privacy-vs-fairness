import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import datetime
import random


def reseed(seed=5):
    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name
        reseed()



    def visualize(self, vis, epoch, acc, loss=None, eid='main', is_dp=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_dp else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='vacc_{0}'.format(self.created_time), env=eid,
                                update='append' if vis.win_exists('vacc_{0}'.format(self.created_time), env=eid) else None,
                                opts=dict(showlegend=True, title='Accuracy_{0}'.format(self.created_time),
                                          width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                                     win='vloss_{0}'.format(self.created_time),
                                     update='append' if vis.win_exists('vloss_{0}'.format(self.created_time), env=eid) else None,
                                     opts=dict(showlegend=True, title='Loss_{0}'.format(self.created_time), width=700, height=400))

        return



    def train_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='vtrain'):

        vis.line(X=np.array([(epoch-1)*data_len+batch]), Y=np.array([loss]),
                                 env=eid,
                                 name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                                 update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                                 opts=dict(showlegend=True, width=700, height=400, title='Train loss_{0}'.format(self.created_time)))



    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)


    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                #
                random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(
                    torch.cuda.FloatTensor)
                negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())




class Net(SimpleNet):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(1)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 47 * 47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        #         print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #         print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #         print(x.shape)
        x = x.view(-1, 16 * 47 * 47)
        #         print(x.shape)
        x = F.relu(self.fc1(x))
        #         print(x.shape)
        x = F.relu(self.fc2(x))
        #         print(x.shape)
        x = self.fc3(x)
        #         print(x.shape)
        return x
