import pandas as pd

#Dataset Exploration
df = pd.read_csv(r'c:\Users\Ansel\Documents\Uni\Class work\cow_milk_mastitis_dataset.csv')
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
print("Duplicated columns", df.columns.duplicated())

#Dataset Cleaning
df = df.drop_duplicates()
df = df.drop(columns=['class1'])
df["Clotting"] = df["Clotting"].astype(bool)
print(df.Milk_Conductivity.describe())
#print(df.to_string())
