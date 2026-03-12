import pandas as pd

def ingest_data(ti):

    path = "/mnt/ml-data/data/titanic.csv"

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)

    missing = df.isnull().sum()
    print("Missing values:\n", missing)

    ti.xcom_push(key="dataset_path", value=path)