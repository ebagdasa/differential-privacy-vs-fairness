import os
import time

import geopy
import pandas as pd
import numpy as np

with open("mapquest_api_key.txt", "r") as file:
    MAPQUEST_API_KEY = file.readline()

mapquest_coder = geopy.geocoders.OpenMapQuest(MAPQUEST_API_KEY)


def get_zips(ary):
    zips = list()
    for i, row in enumerate(ary):
        time.sleep(1)
        try:
            lat, long = row
            location = mapquest_coder.reverse((lat, long), exactly_one=True)
            zip = location.raw['address']['postcode']
            print("{}: ({}, {}) ZIP {}".format(i, lat, long, zip))
            zips.append(zip)
        except Exception as e:
            print(e)
            zips.append(np.nan)
            continue
    return zips


def get_anno_df(root_dir):
    df = pd.read_pickle(os.path.join(root_dir, "df_pickle4.pkl"))
    # df["image_fp"] = df["zpid"].apply(
    #     lambda x: os.path.join(root_dir, "processed_images", x, ".png"))
    return df


df = get_anno_df("/Users/jpgard/Documents/research/zillow-houses")

zips = get_zips(df[["latitude", "longitude"]].values)
df["ZIP"] = zips
df.to_csv("/Users/jpgard/Documents/research/zillow-houses/df_zip.csv", index=False)
