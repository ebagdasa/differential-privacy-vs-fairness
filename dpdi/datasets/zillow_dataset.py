import torch
import pandas as pd
import os
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import time

import pandas as pd


def get_anno_df(root_dir):
    df = pd.read_csv(os.path.join(root_dir, "property-data.csv"))
    census_df = pd.read_csv(os.path.join(root_dir, "census-data.csv"))
    census_df.rename({"zip_code": "ZIP"}, axis=1, inplace=True)

    df["image_fp"] = df["zpid"].apply(
        lambda x: os.path.join(root_dir, "processed_images", x, ".png"))

    df = df.join(census_df, on="ZIP", how="left", rsuffix="census")
    return df

