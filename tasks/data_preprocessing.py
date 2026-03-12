import pandas as pd

def handle_missing(ti):

    path = ti.xcom_pull(key="dataset_path")

    df = pd.read_csv(path)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    df.to_csv("/mnt/ml-data/data/processed.csv", index=False)


def feature_engineering():

    df = pd.read_csv("/mnt/ml-data/data/processed.csv")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df.to_csv("/mnt/ml-data/data/processed.csv", index=False)