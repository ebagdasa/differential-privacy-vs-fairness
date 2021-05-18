import pandas as pd
import os
from dpdi.datasets.utils import normalize_columns


def load_insurance_dataset(root_dir="../data/insurance", normalize=False):
    data = pd.read_csv(os.path.join(root_dir, "insurance.csv"))
    data = pd.get_dummies(data)
    del data['region_northeast']
    del data['smoker_no']
    del data['sex_female']
    data.rename(columns={'sex_male': 'sensitive', 'charges': 'target'}, inplace=True)
    if normalize:
        data = normalize_columns(data)
    return data
