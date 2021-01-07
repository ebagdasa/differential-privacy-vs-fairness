import torch
import torchvision
import pandas as pd
import os
import numpy as np
from skimage import io
from torchvision.datasets.folder import default_loader


def get_anno_df(attr_file, partition_file, partition):
    # Note: The partition sizes are:
    # 0: 162769
    # 2: 19962
    # 1: 19867
    assert partition in ('train', 'eval', 'test')
    anno = pd.read_csv(attr_file, delim_whitespace=True, skiprows=0,
                       header=1).sort_index().replace(-1, 0)
    partitions = pd.read_csv(partition_file, delim_whitespace=True, header=None,
                       index_col=0).sort_index()
    partition_codes = {'train': 0, 'eval': 1, 'test': 2}
    p = partition_codes[partition]
    ix = (partitions.iloc[:, 0] == p).values
    return anno[ix]


class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataset."""

    def __init__(self, attr_file, partition_file, root_dir, target_colname, transform=None,
                 partition='train'):
        """
        Args:
            attr_file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.anno = get_anno_df(attr_file, partition_file, partition)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader
        self.target_colname = target_colname

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.anno.index[idx])
        image = self.loader(img_name)
        anno = self.anno.iloc[idx, :].astype(float).to_dict()
        label_numpy = self.anno.iloc[idx, :][self.target_colname].astype(float).values
        label = torch.from_numpy(label_numpy)

        if self.transform:
            image = self.transform(image)
        sample = (image, anno, label)
        return sample
