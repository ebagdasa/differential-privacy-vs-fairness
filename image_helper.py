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
from utils.dif_dataset import DiFDataset
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

    def sampler_exponential_class(self, mu=1, total_number=40000,
                                  keys_to_drop:list =False, number_of_entries=False):
        per_class_list = defaultdict(list)
        sum = 0
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            sum += 1
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
        unbalanced_sum = 0
        for key, indices in per_class_list.items():
            if key and key not in keys_to_drop:
                unbalanced_sum += len(indices)
            elif key and key in keys_to_drop:
                unbalanced_sum += number_of_entries
            else:
                unbalanced_sum += int(len(indices) * (mu ** key))

        if keys_to_drop:
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
            if key and key not in keys_to_drop:
                subset_len = len(indices)
            elif key and key in keys_to_drop:
                subset_len = number_of_entries
            else:
                subset_len = int(len(indices) * (mu ** key) * proportion)
            sum += subset_len
            subset_lengths.append(subset_len)
            logger.info(f'Key: {key}, subset len: {subset_len} original class len: {len(indices)}')
            ds_indices.extend(indices[:subset_len])
        logger.info(sum)
        self.dataset_size = sum
        logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params[
            'batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices), drop_last=True)

    def sampler_exponential_class_test(self, mu=1, keys_to_drop:list=False,
                                       number_of_entries_test=False):
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
            if key and key not in keys_to_drop:
                subset_len = len(indices)
            elif key and key in keys_to_drop:
                subset_len = number_of_entries_test
            else:
                subset_len = int(len(indices) * (mu ** key))
            sum += subset_len
            subset_lengths.append(subset_len)
            logger.info(f'Key: {key}, len: {subset_len} class_len {len(indices)}')
            ds_indices.extend(indices[:subset_len])
        logger.info(sum)
        logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
        self.test_loader_unbalanced = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.params['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices),
            drop_last=True)

    def load_cifar_data(self, dataset, classes_to_keep=None):
        """Loads cifar10, cifar100, or MNIST datasets."""
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
            if classes_to_keep:
                # Filter the training data to only contain the specified classes.
                print("[DEBUG] train data start size: %s" % len(self.train_dataset))
                train_idx = np.isin(self.train_dataset.targets.numpy(), classes_to_keep)
                self.train_dataset.targets = self.train_dataset.targets[train_idx].to(dtype=torch.float32)
                self.train_dataset.data = self.train_dataset.data[train_idx]
                print("[DEBUG] train data after filtering size: %s" % len(self.train_dataset))
                print("[DEBUG] test data start size: %s" % len(self.test_dataset))
                test_idx = np.isin(self.test_dataset.targets.numpy(), classes_to_keep)
                self.test_dataset.targets = self.test_dataset.targets[test_idx].to(dtype=torch.float32)
                self.test_dataset.data = self.test_dataset.data[test_idx]
                print("[DEBUG] test data after filtering size: %s" % len(self.test_dataset))
                print("[DEBUG] unique train labels: {}".format(self.train_dataset.targets.unique()))
                print("[DEBUG] unique test labels: {}".format(self.test_dataset.targets.unique()))

        self.dataset_size = len(self.train_dataset)
        if classes_to_keep:
            self.labels = classes_to_keep
        else:
            self.labels = list(range(10))
        return

    # def recode_labels_to_binary(self, classes=None):
    #     if not classes:
    #         classes = self.test_dataset.targets.unique()
    #     # Recode the labels, starting at zero
    #     for idx, c in enumerate(classes):
    #         self.train_dataset.targets[self.train_dataset.targets == c] = idx
    #         self.test_dataset.targets[self.test_dataset.targets == c] = idx
    #     self.labels = list(range(len(classes)))
    #     return

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
        self.dataset_size = len(self.train_dataset)
        logger.info(self.dataset_size)
        return True

    def load_inat_data(self):
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.im_size = [299, 299]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)
        normalize = transforms.Normalize(mean=self.mu_data, std=self.std_data)

        transform_train = transforms.Compose(
            [self.scale_aug, self.flip_aug, self.color_aug, transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([self.center_crop, transforms.ToTensor(), normalize])

        self.train_dataset = torchvision.datasets.ImageFolder(
            '/media/classes', transform=transform_train)
        logger.info('len train before : ', len(self.train_dataset))
        # if self.params['ds_size']:
        #     indices = list(range(0, len(self.train_dataset)))
        #     random.shuffle(indices)
        #     random_split = indices[:self.params['ds_size']]
        #     self.train_dataset = torch.utils.data.Subset(self.train_dataset, random_split)
        #     logger.info('len train: ', len(self.train_dataset))
        self.test_dataset = torchvision.datasets.ImageFolder(
            '/media/classes_test', transform=transform_test)
        logger.info('len test: ', len(self.test_dataset))
        self.labels = list(range(len(os.listdir('/media/classes_test/'))))
        logger.info(self.labels)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=8, shuffle=True, num_workers=2)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params['batch_size'],
                                                        shuffle=True, num_workers=2, drop_last=True)
        self.dataset_size = len(self.train_dataset)
        return True

    def balance_loaders(self):
        per_class_index = defaultdict(list)
        for i in range(len(self.train_dataset)):
            _, label = self.train_dataset.samples[i]
            per_class_index[label].append(i)
        total_indices = list()
        if self.params['inat_drop_proportional']:
            for key, value in per_class_index.items():
                random.shuffle(value)
                per_class_no = int(len(value) * (self.params['ds_size'] / len(self.train_dataset)))
                logger.info(f'class: {key}, len: {len(value)}. new length: {per_class_no}')
                total_indices.extend(value[:per_class_no])
        else:
            per_class_no = self.params['ds_size'] / len(per_class_index)
            for key, value in per_class_index.items():
                logger.info(f'class: {key}, len: {len(value)}. new length: {per_class_no}')
                random.shuffle(value)
                total_indices.extend(value[:per_class_no])
        logger.info(f'total length: {len(total_indices)}')
        self.dataset_size = len(total_indices)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=total_indices)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.params['batch_size'],
                                                        sampler=train_sampler,
                                                        num_workers=2, drop_last=True)

    def get_unbalanced_faces(self):
        self.unbalanced_loaders = dict()
        files = os.listdir(self.params['folder_per_class'])
        # logger.info(files)
        for x in sorted(files):
            indices = torch.load(f"{self.params['folder_per_class']}/{x}")
            # logger.info(f'unbalanced: {x}, {len(indices)}')
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=indices)
            self.unbalanced_loaders[x] = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=self.params['test_batch_size'],
                                                        sampler=sampler,
                                                        num_workers=2, drop_last=True)



        return True


    def load_dif_data(self):

        mu_data = [0.485, 0.456, 0.406]
        std_data = [0.229, 0.224, 0.225]
        im_size = [80, 80]
        crop_size = [64, 64]
        brightness = 0.4
        contrast = 0.4
        saturation = 0.4
        hue = 0.25

        resize = transforms.Resize(im_size)
        rotate = transforms.RandomRotation(degrees=30)
        crop = transforms.RandomCrop(crop_size)
        flip_aug = transforms.RandomHorizontalFlip()

        normalize = transforms.Normalize(mean=mu_data, std=std_data)

        center_crop = transforms.CenterCrop(crop_size)
        transform_train = transforms.Compose([resize, rotate, crop,
                                              flip_aug, transforms.ToTensor(),
                                              normalize])

        transform_test = transforms.Compose([resize, center_crop, transforms.ToTensor(), normalize])

        self.train_dataset = DiFDataset(class_list=self.params['class_list'],
                                        root_dir=self.params['root_dir'],
                                        crop_list=self.params['crop_list'],
                                        transform=transform_train)

        self.test_dataset = DiFDataset(class_list=self.params['class_list'],
                                       root_dir=self.params['root_dir'],
                                       crop_list=self.params['crop_list'],
                                       transform=transform_test)

        indices_train = torch.load(self.params['indices_train'])
        indices_test = torch.load(self.params['indices_test'])
        self.labels = list()
        combined_train = list()
        combined_test = list()
        for key, value in indices_train.items():
            combined_train.extend(value)
            self.labels.append(key)

        self.labels = sorted(self.labels)
        t_l = list()
        for key, value in indices_test.items():
            combined_test.extend(value)
            t_l.append(key)
        logger.info(f"Loaded dataset: labels: {self.labels}, len_train: {len(combined_train)}, len_test: {len(combined_test)} labels: {t_l}")

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=combined_train)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=combined_test)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.params['batch_size'],
                                                        sampler=train_sampler,
                                                        num_workers=2, drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.params['test_batch_size'],
                                                       sampler=test_sampler,
                                                       num_workers=2)
        self.dataset_size = len(combined_train)
        self.label_skin_list = torch.load(self.params['label_skin_list'])

        return True


    def load_jigsaw(self):
        import pickle
        import pandas as pd

        max_features = 50000

        train = pd.read_csv('data/jigsaw/processed_train.csv')
        test = pd.read_csv('data/jigsaw/processed_test.csv')
        # after processing some of the texts are emply
        train['comment_text'] = train['comment_text'].fillna('')
        test['comment_text'] = test['comment_text'].fillna('')
        with open(f'data/jigsaw/tokenizer_{max_features}.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

        X_train = tokenizer.texts_to_sequences(train['comment_text'])
        X_test = tokenizer.texts_to_sequences(test['comment_text'])
        x_train_lens = [len(i) for i in X_train]
        x_test_lens = [len(i) for i in X_test]

    def create_model(self):
        return

    def plot_acc_list(self, acc_dict, epoch, name, accuracy):
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt

        acc_list = sorted(acc_dict.items(), key=lambda t: t[1])
        sub_lists = list()
        names = list()
        for x, y in acc_list:
            sub_lists.append(y)
            names.append(str(x))
        fig, ax = plt.subplots(1, figsize=(40, 10))
        ax.plot(names, sub_lists)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Accuracy')
        fig.autofmt_xdate()
        plt.title(f'Accuracy plots. Epoch {epoch}. Main accuracy: {accuracy}')
        plt.savefig(f'{self.folder_path}/figure__{name}_{epoch}.pdf', format='pdf')

        return fig
