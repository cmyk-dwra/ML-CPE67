import pandas as pd
import seaborn as sns

#Dataset Exploration
df = pd.read_csv("cow_milk_mastitis_dataset.csv")
#sns.jointplot(data=df, x = "Milk_Conductivity", y = "Clotting", hue = "class1")
#print(df)
print("Data types\n", df.dtypes, "\n")
print("Summary \n", df.describe(), "\n")
print("Is null\n", df.isnull().sum(), "\n")
print("Duplicated columns\n", df.columns.duplicated(), "\n")

#Dataset Cleaning
df = df.drop_duplicates()
df = df.drop(columns=['class1'])
df["Clotting"] = df["Clotting"].astype(bool)
print(df.to_string())