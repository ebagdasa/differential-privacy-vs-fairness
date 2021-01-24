from torchvision import datasets

class MNISTWithAttributesDataset(datasets.MNIST):
    """A clone of the MNIST dataset, but with
    minority/majority attribute annotatoins."""
    def __init__(self, minority_keys, majority_keys, **kwargs):
        super(MNISTWithAttributesDataset, self).__init__(**kwargs)
        self.minority_keys = minority_keys
        self.majority_keys = majority_keys

    def get_attribute_annotations(self, idxs):
        pass
