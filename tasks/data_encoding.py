import pandas as pd

def encode_features():

    df = pd.read_csv("/mnt/ml-data/data/processed.csv")

    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    df = pd.get_dummies(df, columns=["Embarked"])

    df.drop(["Name","Ticket","Cabin"], axis=1, errors="ignore", inplace=True)

    df.to_csv("/mnt/ml-data/data/final.csv", index=False)