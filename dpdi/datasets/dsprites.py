import torch
import numpy as np
import torchvision
import os
from sklearn.model_selection import train_test_split

LATENTS_TO_IDX = {
    "color": 0,
    "shape": 1,  # (0:square, 1:ellipse, 2:heart)
    "scale": 2,
    "orientation": 3,
    "posx": 4,
    "posy": 5
}

# Not used, but may be useful
mu_dsprites = 0.042494423521889584
std_dsprites = 0.20171427190806362


def get_idxs(n, is_train, seed=23985, train_frac=0.9):
    train_idxs, test_idxs = train_test_split(np.arange(n), train_size=train_frac,
                                             random_state=seed)
    if is_train:
        return train_idxs
    else:
        return test_idxs


class DspritesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, target_colname="scale",
                 attribute_colname="shape", majority_group_keys=(2,)):
        self.target_colname = target_colname
        self.attribute_colname = attribute_colname
        self.majority_group_keys = majority_group_keys
        dsprites = np.load(
            os.path.join(root_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
        idxs = get_idxs(len(dsprites["imgs"]), is_train)
        self.data = dsprites["imgs"][idxs]

        # The order of latents is (Color, Shape, Scale, Orientation, PosX, PosY);
        #  see https://github.com/deepmind/dsprites-dataset for more info.
        self.latents_values = dsprites["latents_values"][idxs]
        self.latents_classes = dsprites["latents_classes"][idxs]

    @property
    def targets(self):
        latent_idx = LATENTS_TO_IDX[self.target_colname]
        return self.latents_values[:, latent_idx]

    @property
    def attribute_annotations(self):
        latent_idx = LATENTS_TO_IDX[self.attribute_colname]
        anno = np.isin(self.latents_values[:, latent_idx],
                       self.majority_group_keys).flatten()
        return anno

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Cast shape of data from [64, 64] -> [1, 64, 64], and change to float.
        img = torch.unsqueeze(self.data[idx, ...], 0).to(dtype=torch.float32)
        label = self.targets[idx]
        sample = (img, idx, label)
        return sample

    def get_attribute_annotations(self, idxs):
        return self.attribute_annotations[idxs]
