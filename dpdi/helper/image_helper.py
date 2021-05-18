import logging

logger = logging.getLogger('logger')

from collections import defaultdict

import torch
import torchvision
import os
import torch.utils.data

from dpdi.helper import Helper
import random

from torchvision import datasets, transforms
import numpy as np
from dpdi.datasets.dif_dataset import DiFDataset
from dpdi.datasets.celeba_dataset import CelebADataset, get_celeba_transforms
from dpdi.datasets.lfw_dataset import LFWDataset, get_lfw_transforms
from dpdi.datasets.mnist_dataset import MNISTWithAttributesDataset
from dpdi.datasets.mc10_dataset import CIFAR10WithAttributesDataset
from dpdi.datasets.zillow_dataset import ZillowDataset
from dpdi.datasets.dsprites import DspritesDataset
from collections import OrderedDict


POISONED_PARTICIPANT_POS = 0



def apply_alpha_to_dataset(dataset, alpha:float=None,
                           minority_keys=None, majority_keys=None,
                           n_train:int=None):
    """

    :param dataset: torch dataset.
    :param alpha: float; proportion of samples to keep in the majority group. Majority
        group is defined as the group with label 1.
    :param labels_mapping: dict mapping true labels to binary labels.
    :return:
    """
    if alpha is not None:
        majority_idxs = np.argwhere(np.isin(dataset.targets, majority_keys)).flatten()
        minority_idxs = np.argwhere(np.isin(dataset.targets, minority_keys)).flatten()
        if n_train:
            print("[DEBUG] applying n_train %s" % n_train)
            # Check that fixed training set size is less than or equal to full data size.
            assert n_train <= len(majority_idxs) + len(minority_idxs)
            n_maj = int(alpha * n_train)
            n_min = n_train - n_maj
        else:
            n_maj = len(majority_idxs)
            n_min = int((1 - alpha) * float(n_maj) / alpha)
        print("[DEBUG] sampling {} elements from minority group {}".format(n_min, minority_keys))
        print("[DEBUG] sampling {} elements from majority_group {}".format(n_maj, majority_keys))

        # Sample alpha * n_sub from the majority, and (1-alpha)*n_sub from the minority.
        majority_idx_sample = np.random.choice(majority_idxs, size=n_maj, replace=False)
        minority_idx_sample = np.random.choice(minority_idxs, size=n_min, replace=False)
        idx_sample = np.concatenate((majority_idx_sample, minority_idx_sample))
        dataset.data = dataset.data[idx_sample]
        dataset.targets = dataset.targets[idx_sample]
        assert len(dataset) == (n_min + n_maj), "Sanity check for dataset subsetting."
        assert abs(
            float(len(minority_idx_sample)) / len(dataset)
            - (1 - alpha)) < 0.001, \
            "Sanity check for minority size within 0.001 of (1-alpha)."
    return dataset



class ImageHelper(Helper):

    def poison(self):
        return

    def sampler_per_class(self):
        self.per_class_loader = OrderedDict()
        per_class_list = defaultdict(list)
        for ind, x in enumerate(self.test_dataset):
            label = x[-1]
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
        for key, indices in per_class_list.items():
            self.per_class_loader[int(key)] = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.params[
                'test_batch_size'], sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def sampler_exponential_class(self, mu=1, total_number=40000,
                                  keys_to_drop: list = None, number_of_entries=None):
        per_class_list = defaultdict(list)
        sum = 0
        for ind, x in enumerate(self.train_dataset):
            label = x[-1]
            sum += 1
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))
        unbalanced_sum = 0
        for key, indices in per_class_list.items():

            # Case: add all instances of the class to indices.
            if (keys_to_drop is None) or (key and key not in keys_to_drop):

                unbalanced_sum += len(indices)
            # Case: add only number_of_entries of the class to indices.
            elif key and key in keys_to_drop:
                unbalanced_sum += number_of_entries
            # This is a special case, keep (mu ** key) * proportion instances.
            else:
                unbalanced_sum += int(len(indices) * (mu ** key))

        if keys_to_drop:
            proportion = 1
        else:
            if total_number / unbalanced_sum > 1:
                raise ValueError("Expected at least "
                                 "{} elements, after sampling left only: {}.".format(
                    total_number, unbalanced_sum))
            proportion = total_number / unbalanced_sum
        logger.info(sum)
        ds_indices = list()
        subset_lengths = list()
        sum = 0
        # Build the list of indices for the dataset
        for key, indices in per_class_list.items():
            random.shuffle(indices)

            if (keys_to_drop is None) or (key and key not in keys_to_drop):
                # Case: add all instances of the class to indices.

                subset_len = len(indices)
            elif key and key in keys_to_drop:
                # Case: add only number_of_entries of the class to indices.
                subset_len = number_of_entries
            else:
                # This is a special case, keep (mu ** key) * proportion instances.
                subset_len = int(len(indices) * (mu ** key) * proportion)
            sum += subset_len
            subset_lengths.append(subset_len)
            logger.info(f'Key: {key}, subset len: {subset_len} original class len: {len(indices)}')
            ds_indices.extend(indices[:subset_len])
        self.dataset_size = sum
        logger.info(f'Imbalance: {max(subset_lengths) / min(subset_lengths)}')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(ds_indices),
            drop_last=True)

    def sampler_exponential_class_test(self, mu=1, keys_to_drop: list = None,
                                       number_of_entries_test=None):
        per_class_list = defaultdict(list)
        sum = 0
        for ind, x in enumerate(self.test_dataset):
            label = x[-1]
            sum += 1
            per_class_list[int(label)].append(ind)
        per_class_list = OrderedDict(sorted(per_class_list.items(), key=lambda t: t[0]))

        logger.info(sum)
        ds_indices = list()
        subset_lengths = list()
        sum = 0
        for key, indices in per_class_list.items():
            random.shuffle(indices)

            if (keys_to_drop is None) or (key and key not in keys_to_drop):

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
    
    def load_dsprites_data(self):
        self.train_dataset = DspritesDataset(self.params['root_dir'], True, True)
        self.test_dataset = DspritesDataset(self.params['root_dir'], False, True)
        self.unnormalized_test_dataset = DspritesDataset(self.params['root_dir'], 
                                                         False, False)
        self.create_loaders()

    def load_cifar_or_mnist_data(self, dataset, classes_to_keep=None,
                                 labels_mapping:dict=None,
                                 alpha:float=None):
        """Loads cifar10, cifar100, or MNIST datasets."""
        logger.info('Loading data')

        ### data load
        # Note: these are the actual statistics for grouped CIFAR with class 0, 3, 5, 8:
        # Channel 0 mean: 0.436337
        # Channel 1 mean: 0.433747
        # Channel 2 mean: 0.426294
        # Channel 0 sd: 0.289641
        # Channel 1 sd: 0.286119
        # Channel 2 sd: 0.299103
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

        minority_keys = self.params['minority_group_keys']
        majority_keys = list(set(labels_mapping.keys()) - set(minority_keys))

        if dataset == 'cifar10':
            self.train_dataset = CIFAR10WithAttributesDataset(
                minority_keys=minority_keys, majority_keys=majority_keys,
                root='../data', train=True,  download=True, transform=transform_train)
            self.test_dataset = CIFAR10WithAttributesDataset(
                minority_keys=minority_keys, majority_keys=majority_keys,
                root='../data', train=False, transform=transform_test)
            self.unnormalized_test_dataset = CIFAR10WithAttributesDataset(
                minority_keys=minority_keys, majority_keys=majority_keys,
                root='../data', train=False, transform=transforms.ToTensor())

        elif dataset == 'cifar100':
            self.train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                                   transform=transform_train)

            self.test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)
        elif dataset == 'mnist':
            self.train_dataset = MNISTWithAttributesDataset(
                minority_keys=minority_keys, majority_keys=majority_keys,

                root='../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]))

            self.test_dataset = MNISTWithAttributesDataset(
                minority_keys=minority_keys, majority_keys=majority_keys,
                root='../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]))
            self.unnormalized_test_dataset = MNISTWithAttributesDataset(
                minority_keys=minority_keys, majority_keys=majority_keys,
                root='../data', train=False, transform=transforms.ToTensor())

        if classes_to_keep:
            # Filter the training data to only contain the specified classes.
            print("[DEBUG] data start size: {} train / {} test".format(
                len(self.train_dataset), len(self.test_dataset)))
            self.train_dataset.apply_classes_to_keep(classes_to_keep)
            self.test_dataset.apply_classes_to_keep(classes_to_keep)
            self.unnormalized_test_dataset.apply_classes_to_keep(classes_to_keep)

            # Apply alpha-balancing to the training data only.
            fixed_n_train = self.params.get('fixed_n_train')
            self.train_dataset = apply_alpha_to_dataset(
                self.train_dataset, alpha, minority_keys=minority_keys,
                majority_keys=majority_keys, n_train=fixed_n_train)

            print("[DEBUG] data after filtering/alpha-balancing size: "
                  "{} train / {} test".format(len(self.train_dataset),
                                              len(self.test_dataset)))
            print("[DEBUG] unique train labels: {}".format(
                self.train_dataset.targets.unique()))
            print("[DEBUG] unique test labels: {}".format(
                self.test_dataset.targets.unique()))


        self.dataset_size = len(self.train_dataset)
        if labels_mapping:
            self.labels = [0, 1]
        else:
            self.labels = list(range(10))
        return

    def create_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.params['batch_size'],
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.params['test_batch_size'],
                                                       pin_memory=True,
                                                       num_workers=8)
        if hasattr(self, 'unnormalized_test_dataset'):
            self.unnormalized_test_loader = torch.utils.data.DataLoader(
                self.unnormalized_test_dataset, batch_size=self.params['test_batch_size'],
                num_workers=8, pin_memory=True)

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


    def load_zillow_data(self):
        self.train_dataset = ZillowDataset(
            self.params['root_dir'], is_train=True, normalize=True)
        self.unnormalized_test_dataset = ZillowDataset(
            self.params['root_dir'], is_train=False, normalize=False
        )
        self.test_dataset = ZillowDataset(
            self.params['root_dir'], is_train=False, normalize=True
        )

        self.create_loaders()


    def load_celeba_data(self):
        transform_train = get_celeba_transforms('train')
        transform_test = get_celeba_transforms('test')
        transform_test_unnormalized = get_celeba_transforms('test', normalize=False)

        self.train_dataset = CelebADataset(
            self.params['attr_file'],
            self.params['eval_partition_file'],
            self.params['root_dir'],
            self.params['target_colname'],
            self.params['attribute_colname'],
            transform_train,
            partition='train',
            train_attribute_subset=self.params.get('train_attribute_subset')
        )

        self.unnormalized_test_dataset = CelebADataset(
            self.params['attr_file'],
            self.params['eval_partition_file'],
            self.params['root_dir'],
            self.params['target_colname'],
            self.params['attribute_colname'],
            transform_test_unnormalized,
            partition='test')

        self.test_dataset = CelebADataset(
            self.params['attr_file'],
            self.params['eval_partition_file'],
            self.params['root_dir'],
            self.params['target_colname'],
            self.params['attribute_colname'],
            transform_test,
            partition='test')

        self.labels = [0,1]
        self.dataset_size = len(self.train_dataset)

        logger.info(f"Loaded dataset: labels: {self.labels}, "
                    f"len_train: {len(self.train_dataset)}, "
                    f"len_test: {len(self.test_dataset)}")

        self.create_loaders()

    def load_lfw_data(self):
        transform_train = get_lfw_transforms('train')
        transform_test = get_lfw_transforms('test')
        transform_test_unnormalized = get_lfw_transforms('test', normalize=False)

        self.train_dataset = LFWDataset(
            self.params['root_dir'],
            self.params['target_colname'],
            self.params['attribute_colname'],
            self.params.get('label_threshold'),
            transform_train,
            partition='train')

        self.test_dataset = LFWDataset(
            self.params['root_dir'],
            self.params['target_colname'],
            self.params['attribute_colname'],
            self.params.get('label_threshold'),
            transform_test,
            partition='test')

        self.unnormalized_test_dataset = LFWDataset(
            self.params['root_dir'],
            self.params['target_colname'],
            self.params['attribute_colname'],
            self.params.get('label_threshold'),
            transform_test_unnormalized,
            partition='test'
        )


        self.labels = [0, 1]
        self.dataset_size = len(self.train_dataset)

        logger.info(f"Loaded dataset: labels: {self.labels}, "
                    f"len_train: {len(self.train_dataset)}, "
                    f"len_test: {len(self.test_dataset)}")

        self.create_loaders()

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

    def get_num_classes(self, classes_to_keep):
        if self.params['dataset'] == 'cifar10':
            num_classes = len(classes_to_keep)
        elif self.params['dataset'] == 'cifar100':
            num_classes = 100
        elif ('mnist' in self.params['dataset']) and classes_to_keep:
            num_classes = len(classes_to_keep)
        elif self.params['dataset'] == 'inat':
            num_classes = len(self.labels)
            logger.info('num class: ', num_classes)
        elif self.params['dataset'] == 'dif':
            num_classes = len(self.labels)
        elif self.params['dataset'] == 'celeba':
            num_classes = len(self.labels)
        elif self.params['dataset'] == 'lfw':
            num_classes = len(self.labels)
        elif self.params['dataset'] in ('zillow', 'dsprites'):
            num_classes = None
        else:
            num_classes = 10
        return num_classes
