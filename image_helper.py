import logging

logger = logging.getLogger('logger')

from collections import defaultdict

import torch
import torchvision
import os
import torch.utils.data

from helper import Helper
import random

from torchvision import datasets, transforms
import numpy as np

from models.simple import SimpleNet
from collections import OrderedDict

POISONED_PARTICIPANT_POS = 0


class ImageHelper(Helper):

    def poison(self):
        return

    def sampler_per_class(self):
        self.per_class_loader = OrderedDict()
        per_class_list = defaultdict(list)
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
        for key, indices in per_class_list.items():
            self.per_class_loader[int(key)] = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params[
                'test_batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def sampler_exponential_class(self, mu=1, total_number=40000, key_to_drop=False, number_of_entries=False):
        per_class_list = defaultdict(list)
        sum = 0
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            sum += 1
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
        unbalanced_sum = 0
        for key, indices in per_class_list.items():
            if key and key != key_to_drop:
                unbalanced_sum += len(indices)
            elif key and key == key_to_drop:
                unbalanced_sum += number_of_entries
            else:
                unbalanced_sum += int(len(indices) * (mu ** key))

        if key_to_drop:
            proportion = 1
        else:
            if total_number / unbalanced_sum > 1:
                raise ValueError(
                    f"Expected at least {total_number} elements, after sampling left only: {unbalanced_sum}.")
            proportion = total_number / unbalanced_sum
        logger.info(sum)
        ds_indices = list()
        subset_lengths = list()
        sum = 0
        for key, indices in per_class_list.items():
            random.shuffle(indices)
            if key and key != key_to_drop:
                subset_len = len(indices)
            elif key and key != key_to_drop:
                subset_len = number_of_entries
            else:
                subset_len = int(len(indices) * (mu ** key) * proportion)
            sum += subset_len
            subset_lengths.append(subset_len)
            logger.info(f'Key: {key}, len: {subset_len} class_len {len(indices)}')
            ds_indices.extend(indices[:subset_len])
        print(sum)
        self.dataset_size = sum
        logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params[
            'batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices), drop_last=True)

    def sampler_exponential_class_test(self, mu=1, key_to_drop=False, number_of_entries_test=False):
        per_class_list = defaultdict(list)
        sum = 0
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            sum += 1
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
        unbalanced_sum = 0
        for key, indices in per_class_list.items():
            unbalanced_sum += int(len(indices) * (mu ** key))

        logger.info(sum)
        ds_indices = list()
        subset_lengths = list()
        sum = 0
        for key, indices in per_class_list.items():
            random.shuffle(indices)
            if key and key != key_to_drop:
                subset_len = len(indices)
            elif key and key == key_to_drop:
                subset_len = number_of_entries_test
            else:
                subset_len = int(len(indices) * (mu ** key))
            sum += subset_len
            subset_lengths.append(subset_len)
            logger.info(f'Key: {key}, len: {subset_len} class_len {len(indices)}')
            ds_indices.extend(indices[:subset_len])
        logger.info(sum)
        logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
        self.test_loader_unbalanced = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params[
            'batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices), drop_last=True)

    def load_cifar_data(self, dataset):
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
        if dataset == 'cifar10':
            self.train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

        elif dataset == 'cifar100':
            self.train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                                   transform=transform_train)
            self.test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)
        elif dataset == 'mnist':
            self.train_dataset = datasets.MNIST('../data', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))
                                                ]))
            self.test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.dataset_size = len(self.train_dataset)
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
