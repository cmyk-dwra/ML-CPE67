import pandas as pd

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67"

def load_data():
    df = pd.read_csv(
        f"{Path}/cow_milk_mastitis_dataset.csv"
    )
    return df

def prepare_features(df):
    X = df.drop(columns=["Cow_ID", "class1", "Day" , "Milk_Yield"])

    return X