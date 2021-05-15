import torch
import pandas as pd
import os
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision import transforms


def get_anno_df(root_dir):
    df = pd.read_pickle(os.path.join(root_dir, "df_pickle4.pkl"))
    df["image_location"] = df["zpid"].apply(
        lambda x:os.path.join(root_dir, "processed_images", x, ".png"))
    return df
