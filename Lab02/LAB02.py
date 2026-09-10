# import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Dataset Exploration
# - Load Dataset

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "cow_milk_mastitis_dataset.csv")
df = pd.read_csv(DATA_PATH)

print("-" * 100)
print("Shape of the dataset:", df.shape)
print("-" * 100)

# - Display Data Types
print("Data Types of the dataset:\n", df.dtypes)
print("-" * 100)

# - Display Summary Statistics
print("Summary Statistics of the dataset:\n", df.describe())
print("-" * 100)

# - Display Missing Values
print("Missing Values in the dataset:\n", df.isnull().sum())
print("-" * 100)

# - Display Duplicate Records
print("Duplicate Records in the dataset:\n", df.duplicated().sum())
print("-" * 100)

# - Display Class Distribution
print("Class Distribution in the dataset:\n")
print(df["class1"].value_counts())
print("-" * 100)

#Data Visualization

# - Histogram of Milk Yield
sns.histplot(data=df,x="Milk_Yield",kde=True,bins=30)
plt.title("Milk Yield Distribution")
plt.xlabel("Milk Yield")
plt.ylabel("Count")
plt.show()

# - Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Data Cleaning

# - Missing Value Handling
print("\n[ 1. Missing Value Handling ]")
df_cleaned = df.copy()
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        col_median = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(col_median)
        print(f"-> Column '{col}' missing count: "f"{missing_count} "f"Filled with median value ({col_median:.2f})")
categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        col_mode = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(col_mode)
        print(f"-> Column '{col}' missing count: "f"{missing_count} "f"Filled with mode value ('{col_mode}')")
print("-" * 100)


# - Duplicate Removal
rows_before = df_cleaned.shape[0]
df_cleaned = df_cleaned.drop_duplicates()
rows_after = df_cleaned.shape[0]
print(f"Before delete: {rows_before} rows | "f"After delete: {rows_after} rows")
print(f"Deleted data: "f"{rows_before - rows_after} rows")
print("-" * 100)


# - Incorrect Data Correction
for col in numeric_cols:
    incorrect_mask = df_cleaned[col] < 0
    incorrect_count = incorrect_mask.sum()
    if incorrect_count > 0:
        col_median = df_cleaned[col].median()
        df_cleaned.loc[incorrect_mask, col] = col_median
        print(f"-> Detected {incorrect_count} "f"negative values in column '{col}': "f"corrected to median value")
    else:
        print(f"-> Column '{col}': "f"No negative values found")
print("-" * 100)

# - Data Type Conversion
df_cleaned["Clotting"] = df_cleaned["Clotting"].astype(bool)
df_cleaned["class1"] = df_cleaned["class1"].astype(bool)
print("-> Converted 'Clotting' to Boolean")
print("-> Converted 'class1' to Boolean")
print("-" * 100)

# - Compare Mean, Median
print("# - Compare Mean, Median")
for col in numeric_cols:
    mean_value = df_cleaned[col].mean()
    median_value = df_cleaned[col].median()
    print(f"{col} - "f"Mean: {mean_value:.2f}, "f"Median: {median_value:.2f}")
print("-" * 100)
# Feature Engineering

# - Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_cleaned["class1_Encoded"] = le.fit_transform(df_cleaned["class1"])
print("-> Label Encoding Completed for 'class1'")
print("Mapping:",dict(zip(le.classes_,le.transform(le.classes_))))
print("-" * 100)

# - Prepare Final Features
df_final = df_cleaned.drop(columns=["Cow_ID","Day"])
print("Final DataFrame shape:", df_final.shape)
print("-" * 100)
print("Final Data:")
print(df_final)