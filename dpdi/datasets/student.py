import os
import pandas as pd


def load_student_dataset(root_dir="../data/student"):
    """Reads the student dataset for the 'por' class only, as in (Khani et al. 2020).

    Adapted from
    https://worksheets.codalab.org/rest/bundles/0xff62528c13b84510b7f10562f21be280
    /contents/blob/data_preprocess/student.py
    """
    #     mat_df = pd.read_csv(os.path.join(root_dir, "student", "student-mat.csv"),
    #     delimiter=";")
    #     mat_df["course"] = "mat"
    #     por_df = pd.read_csv(os.path.join(root_dir, "student", "student-por.csv"),
    #     delimiter=";")
    #     por_df["course"] = "por"
    #     data = pd.concat((mat_df, por_df))
    data = pd.read_csv(os.path.join(root_dir, "student", "student-por.csv"),
                       delimiter=";")
    print(data.shape)

    data = pd.get_dummies(data)
    # For each categorical column, 'no/other' is the baseline, if this is a value.
    for x in data.columns:
        if ('_no' in x) or ('_other' in x):
            del data[x]
    # Set baselines for other categorical columns.
    del data['sex_M']
    del data['Pstatus_A']
    del data['famsize_LE3']
    del data['address_U']
    del data['school_GP']
    data.rename(columns={'sex_F': 'sensitive'}, inplace=True)

    print(data.columns)

    data.rename(columns={'G3': 'target'}, inplace=True)
    data = data.sample(frac=1)
    return data
