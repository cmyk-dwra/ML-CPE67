import pandas as pd
df = pd.read_csv(r'c:\Users\Ansel\Documents\Uni\Class work\cow_milk_mastitis_dataset.csv')
df = df.drop_duplicates()
df = df.drop(columns=['class1'])
df["Clotting"] = df["Clotting"].astype(bool)
print(df.to_string())