import torch
import os
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

TRAIN_TEST_SPLIT_SEED = 42904


def get_anno_df(root_dir, train):
    df = pd.read_csv(os.path.join(root_dir, "property-data.csv"))
    census_df = pd.read_csv(os.path.join(root_dir, "census-data.csv"))
    census_df.rename({"zip_code": "ZIP"}, axis=1, inplace=True)
    df["img_fp"] = df["zpid"].apply(
        lambda x: os.path.join(root_dir, "processed_images", str(x) + ".png"))
    df = df.join(census_df, on="ZIP", how="left", rsuffix="census")
    df_train, df_test = train_test_split(df, train_size=0.8,
                                         random_state=TRAIN_TEST_SPLIT_SEED)
    if train:
        return df_train
    else:
        return df_test


def get_zillow_transforms(is_train, normalize: bool = True):
    im_size = (224, 224)
    mu_data = [0, 0, 0]
    std_data = [1., 1., 1.]
    center_crop = transforms.CenterCrop([200, 200])
    resize = transforms.Resize(im_size)
    rotate = transforms.RandomRotation(degrees=5)
    flip = transforms.RandomHorizontalFlip()
    normalize_transf = transforms.Normalize(mu_data, std_data)

    if is_train:  # Training dataset
        assert normalize, "Unnormalized train transform not implemented."
        return transforms.Compose([
            center_crop, resize, rotate, flip, transforms.ToTensor(),
            normalize_transf
        ])

    else:
        if normalize:  # Normalized test dataset
            return transforms.Compose([transforms.ToTensor(), normalize_transf])
        else:  # Unnormalized test dataset
            return transforms.Compose([transforms.ToTensor()])


class ZillowDataset(Dataset):
    def __init__(self, root_dir, is_train: bool, normalize: bool):
        self.anno = get_anno_df(root_dir, is_train)
        self.loader = default_loader
        self.transform = get_zillow_transforms(is_train, normalize)

    @property
    def targets(self):
        return self.anno["price"].values

    @property
    def attributes(self):
        return self.anno["majority"].values

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fp = self.anno["img_fp"].values[idx]
        image = self.loader(fp)
        label = torch.from_numpy(
            np.ndarray(self.targets[idx])).float()
        if self.transform:
            image = self.transform(image)
        sample = (image, idx, label)
        return sample

    def get_attribute_annotations(self, idxs):
        return self.attributes[idxs]
