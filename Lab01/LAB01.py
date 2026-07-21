import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67/Lab01/"
#Dataset Exploration
df = pd.read_csv(f"{Path}CowMilkMastitisDataset.csv")
#sns.jointplot(data=df, x = "Milk_Conductivity", y = "Clotting", hue = "class1")
#print(df)
print("Data Types vvvvvvvvvvvvvvvvvvv \n", df.dtypes, "\n")
print("Data Summary vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv \n", df.describe(), "\n")
print("Rows with null values: ", df.isnull().all(axis=1).sum(), "\n")
print("Duplicated data: ", df.duplicated().sum(),"\n")

#Dataset Cleaning
df = df.drop_duplicates()
df = df.drop(columns=['class1'])
df["Clotting"] = df["Clotting"].astype(bool)
print("Data vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv \n", df.to_string())

#Data Visualization
sns.histplot(data=df, x = "Milk_Yield", multiple = "stack", color = 'm')
plt.show() #Histogram of Milk Yield

sns.heatmap(df.select_dtypes(exclude=['string']).corr(), annot=True, cmap="coolwarm")
plt.show() #Correlation Heatmap