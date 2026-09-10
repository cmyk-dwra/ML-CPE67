import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "cow_milk_mastitis_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
raw_path = os.path.normpath(DATA_PATH)

def load_data():
    df = pd.read_csv(f"{DATA_PATH}")
    return df

def prepare_features(df):
    X = df.drop(columns=["Cow_ID", "class1", "Day" , "Milk_Yield"])

    return X