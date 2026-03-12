import pandas as pd

def validate_data(ti):

    path = ti.xcom_pull(key="dataset_path")

    df = pd.read_csv(path)

    age_missing = df["Age"].isnull().mean()
    embarked_missing = df["Embarked"].isnull().mean()

    print("Age missing %:", age_missing)
    print("Embarked missing %:", embarked_missing)

    if age_missing > 0.3 or embarked_missing > 0.3:
        raise Exception("Too many missing values")