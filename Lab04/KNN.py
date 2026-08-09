import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67"
df = pd.read_csv(f"{Path}/Lab02/CowMilkMastitisDataset.csv")

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

#Classification
X = df.drop(columns=["class1"])
Y = df["class1"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)