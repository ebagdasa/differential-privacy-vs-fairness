import requests
import pandas as pd


def get_census_df(threshold_ratio=0.5):
    # based on https://atcoordinates.info/2019/09/24/examples-of-using-the-census-bureaus
    # -api-with-python/
    response = requests.get(
        "https://api.census.gov/data/2017/acs/acs5?get=NAME,group("
        "B02001)&for=zip%20code%20tabulation%20area:*")
    data = response.json()
    census_df = pd.DataFrame(data[1:], columns=data[0])
    census_df.rename(columns={"B02001_001E": "all", "B02001_002E": "white",
                              'zip code tabulation area': 'zip_code'}, inplace=True)
    census_df["szip"] = census_df['zip_code'].apply(lambda x: x[:3])
    census_df["frac_white"] = (
            census_df["white"].astype(float) / census_df["all"].astype(float)
    )

    census_df["majority"] = (census_df["frac_white"] > threshold_ratio).astype(int)
    census_df.set_index('zip_code', inplace=True)
    all_count = {}
    white_count = {}
    for i in census_df.index:
        szip = i[:3]  # only first 3 digits appear in lendingclub data
        all_count[szip] = all_count.get(szip, 0) + int(census_df.loc[i]['all'])
        white_count[szip] = white_count.get(szip, 0) + int(census_df.loc[i]['white'])
    # note that a few rare zip codes are not in the census data (military bases)
    # so we have to be careful here. also some N/A
    # 1 = majority dataset = greater than 50% white counties
    census_df['szip_majority'] = \
        [1 if not pd.isna(zip)
              and zip[:3] in white_count
              and white_count[zip[:3]] > threshold_ratio * all_count[zip[:3]]
         else 0 for zip in census_df.index]
    return census_df[["szip", "majority", "szip_majority", "all", "white", "frac_white", "state"]]
