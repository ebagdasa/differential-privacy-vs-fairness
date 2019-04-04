from collections import defaultdict

import torch
import torchvision
import os
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.simple import SimpleNet

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0


class ImageHelper(Helper):

    def poison(self):
        return

    def sampler_per_class(self):
        self.per_class_loader = dict()
        per_class_list = defaultdict(list)
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            per_class_list[label].append(ind)
        for key, indices in per_class_list.items():
            self.per_class_loader[key] = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params[
                'test_batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))


    def sampler_exponential_class(self, mu=1):
        per_class_list = defaultdict(list)
        for ind, x in enumerate(self.train_dataset):
            _, label = x

            per_class_list[label].append(ind)

        ds_indices = list()
        subset_lengths = list()
        for key, indices in per_class_list.items():
            random.shuffle(indices)
            subset_len = len(indices) * (mu ** key)
            subset_lengths.append(subset_len)
            logger.info(f'Key: {key}, len: {subset_len}')
            ds_indices.extend(indices[:subset_len])
        logger.info(f'Imbalance: {max(subset_lengths)/min(subset_lengths)}')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params[
            'batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices), drop_last=True)


    def load_cifar_data(self):
        logger.info('Loading data')

        ### data load
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                              transform=transform_train)

        self.test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

        return

    def create_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.params['batch_size'],
                                                        shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.params['test_batch_size'],
                                                       shuffle=True)

    def load_faces_data(self):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [transforms.ToTensor(),
             normalize])

        if os.path.exists('data/utk/train_ds.pt') and os.path.exists('data/utk/test_ds.pt'):
            logger.info('DS already exists. Loading.')
            self.train_dataset = torch.load('data/utk/train_ds.pt')
            self.test_dataset = torch.load('data/utk/test_ds.pt')
        else:
            self.train_dataset = torchvision.datasets.ImageFolder('data/utk/clustered/gender/', transform=transform)
            torch.save(self.train_dataset, 'data/utk/train_ds.pt')
            self.test_dataset = torchvision.datasets.ImageFolder('data/utk/test/gender/', transform=transform)
            torch.save(self.test_dataset, 'data/utk/test_ds.pt')

        self.races = {'white': 0, 'black': 1, 'asian': 2, 'indian': 3, 'other': 4}
        self.inverted_races = dict([[v, k] for k, v in self.races.items()])

        race_ds = dict()
        race_loaders = dict()
        for name, i in self.races.items():
            race_ds[i] = torchvision.datasets.ImageFolder(f'data/utk/test_gender/race/{i}/', transform=transform)
            race_loaders[i] = torch.utils.data.DataLoader(race_ds[i], batch_size=8, shuffle=True, num_workers=2)
        self.race_datasets = race_ds
        self.race_loaders = race_loaders

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=True, num_workers=2)
        self.train_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params['batch_size'],
                                                        shuffle=True, num_workers=2, drop_last=True)
        return True

    def create_model(self):

        return