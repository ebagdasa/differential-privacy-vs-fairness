from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader
import random
class DiFDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, class_list, root_dir, crop_list, transform=None):
        """
        Args:
            storage_list (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.class_label_id = torch.load(class_list)
        self.crop_list = torch.load(crop_list)

        self.race_train_list = os.listdir('data/utk/clustered/race/1')
        self.race_test_list = os.listdir('data/utk/test/race/1')
        random.seed(5)
        random.shuffle(self.race_train_list)
        random.shuffle(self.race_test_list)

        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.class_label_id )

    def __getitem__(self, idx):
        if idx<1000000:
            sample = self.loader(f'{self.root_dir}/{idx}.jpg')
            target = self.class_label_id[idx]
            crop = self.crop_list[idx]

            sample = sample.crop(crop)
        elif idx<2000000:
            idx = idx - 1000000
            file_name = self.race_train_list[idx]

            sample = self.loader(f'data/utk/clustered/race/1/{file_name}')
            age, gender, race, _ = file_name.split('_')
            target = int(gender)

        else:
            idx = idx - 2000000
            file_name = self.race_test_list[idx]
            age, gender, race, _ = file_name.split('_')
            sample = self.loader(f'data/utk/test/race/1/{file_name}')
            target = int(gender)



        if self.transform is not None:
            sample = self.transform(sample)


        return sample, idx, target

