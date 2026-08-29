from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Path = "C:/Users/Ansel/Desktop/Ansel/Python/ML-CPE67/"
df = pd.read_csv(f"{Path}/Lab01/CowMilkMastitisDataset.csv")

print(df["Milk_Conductivity"].corr(df["Milk_Yield"]))