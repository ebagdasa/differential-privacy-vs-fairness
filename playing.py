import torch
import torchvision
import shutil
import os
import torchvision.transforms as transforms
from collections import defaultdict
from models.simple import Net
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm
import visdom
import time
vis = visdom.Visdom()
torch.cuda.is_available()
torch.cuda.device_count()

import torchvision.models as models



class Res(nn.Module):
    def __init__(self):
        torch.manual_seed(5)
        super(Res, self).__init__()
        self.res = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.res(x)
        x = self.fc(x)
        return x





normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize])

if os.path.exists('data/utk/train_ds.pt') and os.path.exists('data/utk/test_ds.pt'):
    print('DS already exists. Loading.')
    train_ds = torch.load('data/utk/train_ds.pt')
    test_ds = torch.load('data/utk/test_ds.pt')
else:
    train_ds = torchvision.datasets.ImageFolder('data/utk/clustered/gender/', transform=transform)
    torch.save(train_ds, 'data/utk/train_ds.pt')
    test_ds = torchvision.datasets.ImageFolder('data/utk/test/gender/', transform=transform)
    torch.save(train_ds, 'data/utk/test_ds.pt')

testloader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=True, num_workers=2)



def plot(x, y, name, win):
    vis.line(Y=np.array([y]), X=np.array([x]),
             win=win,
             name=f'Model_{name}',
             update='append' if vis.win_exists(win) else None,
             opts=dict(showlegend=True, title=win, width=700, height=400)
             )

def compute_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def test(net, epoch, name, testloader, vis=True, win='Test'):
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

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total}')
    if vis:
        plot(epoch, 100*correct/total, name, win=win)
    return 100 * correct / total


def train_dp(trainloader, model, optimizer, scheduler, epoch, name):
    norm_type = 2
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

        losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)
        saved_var = dict()
        for name, tensor in model.named_parameters():
            saved_var[name] = torch.zeros_like(tensor)

        for i in losses:
            i.backward(retain_graph=True)
            # total_norm=compute_norm(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), S)
            for name, tensor in model.named_parameters():
                new_grad = tensor.grad
                saved_var[name].add_(new_grad)
            model.zero_grad()

        for name, tensor in net.named_parameters():
            saved_var[name].add_(torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, sigma))
            tensor.grad = saved_var[name] / num_microbatches
        optimizer.step()
        running_loss += torch.sum(loss).item()
        if i > 0 and i % 20 == 0:  # print every 2000 mini-batches
            #             print('[%d, %5d] loss: %.3f' %
            #                   (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, name, win='Train')
            running_loss = 0.0

def train(trainloader, model, optimizer, scheduler, epoch, name):

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
        # print(f'inputs shape: {inputs.shape}')
        # print(f'outputs shape: {outputs.shape}')
        # print(f'labels shape: {labels.shape}')
        # print(f'Loss shape: {loss.shape}')
        # for key, value in model.named_parameters():
        #     print(f'{key}: {value.shape}')
        optimizer.step()
        # scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i > 0 and i % 20 == 0:  # print every 2000 mini-batches
            #             print('[%d, %5d] loss: %.3f' %
            #                   (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, name, win='Train')
            running_loss = 0.0



if __name__ == '__main__':

    batch_size = 64
    num_microbatches = 16
    lr = 0.01
    momentum = 0.99
    decay = 1e-5
    S = 100
    z = 0.00001
    sigma = z*S
    dp = True

    races = {'white': 0, 'black': 1, 'asian': 2, 'indian': 3, 'other': 4}
    inverted_races = dict([[v, k] for k, v in races.items()])

    race_ds = dict()
    race_loaders = dict()
    for name, i in races.items():
        race_ds[i] = torchvision.datasets.ImageFolder(f'data/utk/test_gender/race/{i}/', transform=transform)
        race_loaders[i] = torch.utils.data.DataLoader(race_ds[i], batch_size=8, shuffle=True, num_workers=2)


    print(batch_size)
    print(lr)
    print(momentum)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    net = Net()
    net.cuda()
    if dp:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 16], gamma=0.1)
    name = f'{batch_size} {momentum} {lr} {decay} {num_microbatches}'
    acc = test(net, 0, name, testloader, vis=True, win=f'Momentum {momentum}')
    for epoch in range(50):  # loop over the dataset multiple times
        if dp:
            train_dp(trainloader, net, optimizer, None, epoch, name)
        else:
            train(trainloader, net, optimizer, None, epoch, name)
        acc = test(net, epoch, name, testloader, vis=True, win=f'Momentum {momentum}')

        if acc <= 60 and epoch>4:
            break
        # for i, loader in race_loaders.items():
        #     print(inverted_races[i])
        #     test(net, epoch, inverted_races[i], loader, vis=True, win=name)



