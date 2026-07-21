import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67/Lab01/"
df = pd.read_csv(f"{Path}CowMilkMastitisDataset.csv")

#Dataset Cleaning
df = df.drop_duplicates()
df = df.drop(columns=['class1'])
df["Clotting"] = df["Clotting"].astype(bool)

#Dataset Exploration
Summary = df.describe()
#sns.jointplot(data=df, x = "Milk_Conductivity", y = "Clotting", hue = "class1")
#print(df)
print("Data Types", "-"*100, "\n", df.dtypes, "\n")
print("Data Summary", "-"*100, "\n", Summary, "\n")
print("Mean:Median ratio", "-"*100, "\n",Summary.loc['mean']/df.select_dtypes(include=['int64', 'float64']).median(), "\n", "-"*120, "\n")
print("Rows with null values: ", df.isnull().all(axis=1).sum(), "\n")
print("Duplicated data: ", df.duplicated().sum(),"\n")

print("Data", "-"*100, "\n", df.to_string(), "\n", "-"*120)

#Data Visualization
sns.histplot(data=df, x = "Milk_Yield", multiple = "stack", color = 'm')
plt.show() #Histogram of Milk Yield

sns.heatmap(df.select_dtypes(exclude=['string']).corr(), annot=True, cmap="coolwarm")
plt.show() #Correlation Heatmap