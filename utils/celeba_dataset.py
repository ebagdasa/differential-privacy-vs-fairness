import torch
import torchvision
import pandas as pd
import os
import numpy as np
from skimage import io


class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataset."""

    def __init__(self, attr_file, root_dir, transform=None):
        """
        Args:
            attr_file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.anno = pd.read_csv(attr_file, delim_whitespace=True, skiprows=0,
                                header=1).sort_index().replace(-1, 0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.anno.index[idx])
        image = io.imread(img_name)
        anno = self.anno.iloc[idx, :].astype(float).to_dict()

        if self.transform:
            image = self.transform(image)
        sample = (image, anno)
        return sample
