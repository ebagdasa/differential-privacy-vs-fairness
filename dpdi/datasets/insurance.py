import pandas as pd
import os


def load_insurance_dataset(root_dir="../data/insurance"):
    data = pd.read_csv(os.path.join(root_dir, "insurance.csv"))
    data = pd.get_dummies(data)
    del data['region_northeast']
    del data['smoker_no']
    del data['sex_female']
    data.rename(columns={'sex_male': 'sensitive', 'charges': 'target'}, inplace=True)
    return data
