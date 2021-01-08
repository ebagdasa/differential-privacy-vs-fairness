import torch
import pandas as pd
import os
import numpy as np
from torchvision.datasets.folder import default_loader


def get_anno_df(attr_file, partition_file, partition, attribute_colname,
                train_attribute_subset):
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
    anno_subset = anno[ix]
    if partition == 'train' and train_attribute_subset is not None:
        # Subset to only include the specified subset
        attr_ix = anno_subset[attribute_colname] == train_attribute_subset
        anno_subset = anno_subset[attr_ix]
        print("[DEBUG] attribute values in {} dataset:".format(partition))
        print(anno_subset[attribute_colname].value_counts())
    return anno_subset


class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataset."""

    def __init__(self, attr_file, partition_file, root_dir, target_colname,
                 attribute_colname,
                 transform=None,
                 partition='train',
                 train_attribute_subset=None):
        """
        Args:
            attr_file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.anno = get_anno_df(attr_file, partition_file, partition,
                                attribute_colname, train_attribute_subset)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader
        self.target_colname = target_colname
        self.attribute_colname = attribute_colname

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.anno.index[idx])
        image = self.loader(img_name)
        label_numpy = np.array(self.anno.iloc[idx, :][self.target_colname])
        label = torch.from_numpy(label_numpy)

        if self.transform:
            image = self.transform(image)
        sample = (image, idx, label)
        return sample

    def get_attribute_annotations(self, idxs):
        idx_annos = self.anno.iloc[idxs, :][self.attribute_colname]
        return idx_annos
