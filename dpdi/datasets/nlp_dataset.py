from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets.folder import default_loader
import random
class NLPDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, ds_file):
        """
        Args:
            storage_list (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ds_file = torch.load(ds_file)

    def __len__(self):
        return len(self.ds_file )

    def __getitem__(self, idx):
        sample, target = self.ds_file[idx]

        return sample, target
