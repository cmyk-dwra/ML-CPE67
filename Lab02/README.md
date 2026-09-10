# Cow Milk Mastitis Dataset - Data Preprocessing

## Overview

This project performs data preprocessing and exploratory analysis on the Cow Milk Mastitis Dataset. The preprocessing workflow prepares the dataset for subsequent machine learning tasks involving regression and classification.

The dataset contains milk-related measurements and a binary classification attribute associated with mastitis detection.

## Dataset
[DOWNLOAD DATASET HERE](https://www.kaggle.com/datasets/amithadityacp/cow-mastitisfrom-milk)

The dataset contains the following attributes:

| Column | Description |
|---|---|
| `Cow_ID` | Unique identifier for each cow |
| `Day` | Recorded day of the measurement |
| `Milk_Temperature` | Temperature of the milk |
| `Milk_pH` | pH value of the milk |
| `Milk_Conductivity` | Electrical conductivity of the milk |
| `Somatic_Cell_Count` | Somatic cell count measurement |
| `Milk_Yield` | Milk production/yield |
| `Clotting` | Clotting indicator |
| `class1` | Binary classification attribute |

## Data Preprocessing

The preprocessing workflow consists of the following steps:

### 1. Dataset Loading

The dataset is loaded from a CSV file using the Pandas library.
```df = pd.read_csv(DATA_PATH)```
### 2. Dataset Exploration

The dataset is examined to understand its structure and characteristics. The following information is inspected:

Dataset dimensions
Data types
Summary statistics
Missing values
Duplicate records
Class distribution

These checks are performed before applying preprocessing operations to ensure that the dataset is properly understood.

### 3. Missing Value Handling

Missing values are checked for each column.

Numerical attributes are prepared to use their median values for missing-value replacement, while categorical attributes use their mode values when necessary.

If no missing values are present, no replacement is performed.

### 4. Duplicate Record Handling

Duplicate records are checked and removed when present.

This ensures that identical observations do not unnecessarily affect subsequent analysis or model training.

### 5. Incorrect Data Detection

Numerical attributes are checked for negative values that may represent invalid measurements.

When an invalid negative value is detected, it is replaced with the median value of the corresponding attribute.

If no invalid values are detected, the original values are retained.

### 6. Data Type Conversion

The Clotting and class1 attributes are converted to Boolean values.
```
df_cleaned["Clotting"] = df_cleaned["Clotting"].astype(bool)
df_cleaned["class1"] = df_cleaned["class1"].astype(bool)
```


This represents the binary nature of these attributes more appropriately for subsequent processing.

Feature Preparation

For machine learning, identifier and non-predictive columns are excluded from the feature set.

The following columns are excluded:

`Cow_ID
Day`

The remaining attributes can be used as input features for the machine learning models.

For regression, Milk_Yield is used as the target variable.

For classification, class1 is used as the target variable.

Data Visualization

Exploratory visualizations are used to better understand the dataset.

Milk Yield Distribution

A histogram is used to visualize the distribution of Milk_Yield.

Correlation Heatmap

A correlation heatmap is generated using the numerical attributes to examine relationships between variables.

Libraries Used
Pandas - Dataset loading, manipulation, and analysis
NumPy - Numerical operations
Matplotlib - Data visualization
Seaborn - Statistical data visualization
Scikit-learn - Machine learning preprocessing and feature engineering
Output

The preprocessing stage produces a cleaned and prepared dataset suitable for subsequent machine learning tasks.

The resulting data can be used for:

Regression analysis for Milk_Yield
Classification using class1
Feature scaling and dimensionality reduction
Machine learning model training and evaluation
Project Structure
cow_milk_mastitis
│
├── data_loader.py
├── preprocessing.py
├── regression.py
├── classification.py
├── cow_milk_mastitis_dataset.csv
│
└── outputs/

## Note on Preprocessing Operations
The preprocessing workflow includes checks and handling procedures for missing values, duplicate records, and potentially incorrect numerical values, as required by the specified preprocessing workflow.

However, the current dataset was already well-structured and contained no duplicate records, missing values, or identified invalid negative numerical values. As a result, several of the implemented cleaning procedures do not modify the dataset in practice.

These procedures have been retained to demonstrate the complete preprocessing workflow and to satisfy the requirements of the laboratory exercise. Consequently, portions of the preprocessing code may remain unused during execution because there are no corresponding data-quality issues to correct.

In a practical data-processing environment, unnecessary operations would normally be omitted or simplified when the dataset has already been verified to be clean. In this case, however, the additional procedures are intentionally preserved for completeness and documentation of the required methodology.
