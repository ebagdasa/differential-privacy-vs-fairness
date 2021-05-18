import pandas as pd
import os


def load_life_expectancy_dataset(root_dir="../data/life-expectancy", year=2015):
    data = pd.read_csv(os.path.join(root_dir, 'life-expectancy-data.csv'))
    data = data[data.Year == year]  # Keep only this year
    del data['Country']
    del data['Year']
    data.rename(columns={'Status': 'sensitive', 'Life expectancy': 'target'})
    return data
