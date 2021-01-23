import torch
import numpy as np


class AlphaMNISTDataset(torch.utils.data.datasets.MNIST):
    def __init__(self, alpha, classes_to_keep, fixed_n_train,
                 minority_group_keys, labels_mapping, **kwargs):
        super(AlphaMNISTDataset, self).__init__(**kwargs)
        self.minority_group_keys = minority_group_keys
        self.majority_group_keys = list(
            set(labels_mapping.keys()) - set(minority_group_keys))
        if classes_to_keep:
            # Filter the dataset to only contain the specified classes.
            print("[DEBUG] dataset start size: %s" % len(self))
            idx = np.isin(self.targets.numpy(), classes_to_keep)
            self.targets = self.targets[idx].to(dtype=torch.float32)
            self.data = self.data[idx]
            self.apply_alpha_to_dataset(minority_group_keys,
                                        alpha,
                                        labels_mapping,
                                        fixed_n_train)

            print("[DEBUG] dataset size after filtering"
                  "/alpha-balancing size: %s" % len(self.train_dataset))
            print("[DEBUG] unique labels: {}".format(
                self.train_dataset.targets.unique()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.data[idx, ...]
        label = self.targets[idx]
        sample = (img, label, idx)
        return sample

    def get_attribute_annotations(self, idxs):
        labels = self.targets[idxs].cpu().detach().numpy()
        anno = np.isin(labels, self.majority_group_keys).flatten()
        return anno

    def apply_alpha_to_dataset(self, minority_group_keys: list = None,
                               alpha: float = None,
                               n_train: int = None):
        """

        :param dataset: torch dataset.
        :param alpha: float; proportion of samples to keep in the majority group. Majority
            group is defined as the group with label 1.
        :param labels_mapping: dict mapping true labels to binary labels.
        :return:
        """
        if alpha is not None:

            majority_idxs = np.argwhere(
                np.isin(self.targets, self.majority_group_keys)).flatten()
            minority_idxs = np.argwhere(
                np.isin(self.targets, minority_group_keys)).flatten()
            if n_train:
                # Check that fixed training set size is less than or equal to full data
                # size.
                assert n_train <= len(majority_idxs) + len(minority_idxs)
                n_maj = int(alpha * n_train)
                n_min = n_train - n_maj
            else:
                n_maj = len(majority_idxs)
                n_min = int((1 - alpha) * float(n_maj) / alpha)
            # Sample alpha * n_sub from the majority, and (1-alpha)*n_sub from the 
            # minority.
            print("[DEBUG] sampling n_maj={} elements from {} majority items {}".format(
                n_maj, len(majority_idxs), majority_group_keys))
            print("[DEBUG] sampling n_min={} elements from {} minority items {}".format(
                n_min, len(minority_idxs), minority_group_keys))
            majority_idx_sample = np.random.choice(majority_idxs, size=n_maj,
                                                   replace=False)
            minority_idx_sample = np.random.choice(minority_idxs, size=n_min,
                                                   replace=False)
            idx_sample = np.concatenate((majority_idx_sample, minority_idx_sample))
            self.data = self.data[idx_sample]
            self.targets = self.targets[idx_sample]
            assert len(self) == (n_min + n_maj), "Sanity check for self subsetting."
            assert abs(
                float(len(minority_idx_sample)) / len(self)
                - (1 - alpha)) < 0.001, \
                "Sanity check for minority size within 0.001 of (1-alpha)."
        return
