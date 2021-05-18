import pandas as pd
import os
from dpdi.datasets.utils import normalize_columns


def load_life_expectancy_dataset(root_dir="../data/life-expectancy", year=2015,
                                 normalize=False, minority_label="Developing"):
    data = pd.read_csv(os.path.join(root_dir, 'life-expectancy-data.csv'))
    data.columns = [c.strip().lower() for c in data.columns]
    data = data[data.year == year]  # Keep only this year
    del data['country']
    del data['year']
    del data['alcohol']  # missing for 158 countries per year
    del data['total expenditure']  # missing for 162 countries per year
    data = data.dropna()
    # Create an integer indicator for minority group' note that the groupings are
    # "Developing" and "Developed"
    data['status'] = (data['status'] == minority_label).astype(int)
    data.rename(columns={'status': 'sensitive', 'life expectancy': 'target'}, inplace=True)
    if normalize:
        data = normalize_columns(data)
    return data
