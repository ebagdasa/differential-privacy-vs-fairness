import numpy as np


def normalize_columns(data, exclude_cols=('sensitive', 'target')):
    for x in data.columns:
        if (x != 'sensitive' and x != 'target'):
            data[x] = (data[x] - np.mean(data[x])) / np.std(data[x])
    return data
