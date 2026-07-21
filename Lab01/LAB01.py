import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67/"
#Dataset Exploration
df = pd.read_csv(f"{Path}/Lab01/CowMilkMastitisDataset.csv")
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

#Data Visualization
sns.histplot(data=df, x = "Milk_Yield", multiple = "stack", color = 'm')
plt.show() #Histogram of Milk Yield

sns.heatmap(df.select_dtypes(exclude=['string']).corr(), annot=True, cmap="coolwarm")
plt.show() #Correlation Heatmap