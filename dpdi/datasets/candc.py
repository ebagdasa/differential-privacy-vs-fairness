import os
import pandas as pd
import numpy as np


def load_candc_dataset(root_dir="../data/communities-and-crime"):
    """Loads the communities and crime dataset.

    The data files can be accessed at
    https://worksheets.codalab.org/bundles/0xa8201e470d404a7aa03746291161d9e1
    """
    attrib = pd.read_csv(os.path.join(root_dir, 'attributes.csv'), delim_whitespace=True)
    data = pd.read_csv(os.path.join(root_dir, 'communities.data'),
                       names=attrib['attributes'])
    # remove non predicitive features
    data = data.drop(columns=['state', 'county',
                              'community', 'communityname',
                              'fold'], axis=1)

    data = data.replace('?', np.nan)
    data = data.dropna(axis=1)
    print(data.columns)

    # remove race related stuff
    data.loc[data.racePctWhite >= 0.85, 'racePctWhite'] = 1
    data.loc[data.racePctWhite < 0.85, 'racePctWhite'] = 0
    data = data.drop(columns=['racePctAsian', 'racePctHisp',
                              'racepctblack', 'whitePerCap',
                              'blackPerCap', 'indianPerCap',
                              'AsianPerCap',  # 'OtherPerCap',
                              'HispPerCap',
                              ], axis=1)

    data.rename(columns={'racePctWhite': 'sensitive'}, inplace=True)
    data.rename(columns={'ViolentCrimesPerPop': 'target'}, inplace=True)

    return data
