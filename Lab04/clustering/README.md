# Cow Milk Mastitis Dataset - K-Means Clustering

## Overview

This project applies **K-Means Clustering** to the Cow Milk Mastitis Dataset.

The goal is to group cows based on similarities in their milk-related measurements without using the `class1` classification label during clustering.

The project also implements a custom **K-Means algorithm using TensorFlow** and a **KNN-based cluster assigner** for assigning new observations to existing clusters.

---

## Dataset

[DOWNLOAD DATASET HERE](https://www.kaggle.com/datasets/amithadityacp/cow-mastitisfrom-milk)

The dataset is stored in the project root:

```text
cow_milk_mastitis/
└── cow_milk_mastitis_dataset.csv
```

---

## Clustering Objective

Unlike classification, clustering does not use a known target label to create the groups.

The clustering process uses the following features:

| Feature | Description |
|---|---|
| `Milk_Temperature` | Temperature of the milk |
| `Milk_pH` | pH value of the milk |
| `Milk_Conductivity` | Electrical conductivity of the milk |
| `Somatic_Cell_Count` | Number of somatic cells in the milk |
| `Milk_Yield` | Milk production amount |
| `Clotting` | Clotting indicator |

The following columns are **not used** for clustering:

- `id` - record identifier
- `Cow_ID` - cow identifier
- `Day` - observation/day information
- `class1` - classification label

`class1` is intentionally excluded because K-Means is an **unsupervised learning method**. The clusters should be discovered from the feature data rather than created from the existing class label.

---

# Machine Learning Workflow

The project follows five main steps:

```text
Load Dataset
     ↓
Prepare and Scale Features
     ↓
Find Suitable Number of Clusters
     ↓
Run K-Means
     ↓
Analyze Clusters
     ↓
Assign New Observations Using KNN
```

---

## Step 1: Load Data

The `data_loader.py` file reads the CSV dataset and selects the features used for clustering.

Two versions of the feature data are retained:

### Raw Data

```python
X_raw
```

Contains the original measurements.

This is useful when interpreting the clusters because the values remain in their original units.

For example:

```text
Milk_Yield = 25.4
```

is easier to understand than a standardized value such as:

```text
Milk_Yield = 0.73
```

### Scaled Data

```python
X
```

Contains the standardized feature values.

Each feature is transformed to approximately:

```text
mean = 0
standard deviation = 1
```

This is important for K-Means because K-Means uses distance calculations.

Without scaling, a feature with large numerical values could dominate the distance calculation.

The project uses:

```python
StandardScaler()
```

---

# Step 2: Find the Number of Clusters

K-Means requires the number of clusters to be specified before training.

The project tests:

```python
k_values = [2, 3, 4, 5, 6, 7, 8]
```

For each value of `K`, the program calculates:

- Inertia
- Silhouette Score

---

## Elbow Method

**Inertia** measures how close the observations are to their assigned cluster centroids.

Lower inertia means the observations are closer to their cluster centers.

However, increasing the number of clusters will naturally decrease inertia.

Therefore, the goal is not simply to find the smallest inertia.

Instead, the **Elbow Method** looks for a point where increasing the number of clusters produces much smaller improvements.

The resulting graph is saved as:

```text
outputs/01_elbow.png
```

Example:

```text
Inertia
  │\
  │ \
  │  \
  │   \__
  │      \__
  │          \___
  └────────────────
     2  3  4  5  6  7  8
              K
```

The selected number of clusters is:

```python
N_CLUSTERS = 4
```

This value should be supported by inspection of the elbow graph and the clustering results.

---

## Silhouette Score

The silhouette score measures how well observations fit within their assigned clusters compared with other clusters.

The value ranges approximately from:

```text
-1 to 1
```

A general interpretation is:

| Silhouette Score | Interpretation |
|---|---|
| Close to 1 | Well-separated clusters |
| Around 0 | Overlapping clusters |
| Below 0 | Possible incorrect cluster assignments |

A low silhouette score does not mean that K-Means failed technically.

K-Means will always produce clusters when requested.

Instead, a low score suggests that the dataset may not contain strongly separated natural groups.

The program therefore checks the silhouette score after selecting `K`.

---

# Step 3: Run K-Means

After selecting the number of clusters, the program runs the custom TensorFlow K-Means implementation.

```python
km = TFKMeans(n_clusters=N_CLUSTERS)
labels = km.fit_predict(X)
```

Each observation receives a cluster label:

```text
0
1
2
3
```

for four clusters.

---

## How K-Means Works

The algorithm repeatedly performs two main operations.

### 1. Assign

Each observation is assigned to the nearest centroid.

Euclidean distance is used:

```text
distance =
sqrt(
    (x1 - c1)^2 +
    (x2 - c2)^2 +
    ...
)
```

The nearest centroid determines the observation's cluster.

### 2. Update

After assigning observations to clusters, the centroid of each cluster is recalculated using the mean of its members.

```python
tf.reduce_mean(members, axis=0)
```

The process continues until the centroids stop moving significantly or the maximum number of iterations is reached.

---

## Empty Cluster Handling

A cluster can potentially receive zero observations during an iteration.

The implementation handles this case by keeping the previous centroid:

```python
else:
    new_centroids.append(centroids[c])
```

This prevents an empty cluster from producing an invalid centroid.

---

## TensorFlow Implementation

The K-Means algorithm is implemented manually in:

```text
clustering/kmeans_tf.py
```

The implementation uses TensorFlow for:

- Distance calculations
- Tensor operations
- Cluster assignment
- Centroid calculations

NumPy is used for random centroid initialization.

The implementation does not simply call:

```python
sklearn.cluster.KMeans
```

The purpose is to demonstrate how K-Means works internally.

---

# Cluster Visualization

The clustering result is visualized using two selected features:

```text
Milk_pH
Milk_Conductivity
```

The output is saved as:

```text
outputs/02_clusters.png
```

The visualization only shows two dimensions.

The actual clustering process uses **all selected features**.

Therefore, the graph is a visual representation of the clusters rather than the complete six-dimensional clustering space.

---

# Step 4: Analyze Each Cluster

After clustering, the program combines the cluster labels with the original feature values.

```python
profile["cluster"] = labels
```

The mean value of each feature is then calculated for every cluster.

Example structure:

```text
         Milk_Temperature  Milk_pH  Milk_Conductivity  ...
cluster
0                  ...       ...          ...
1                  ...       ...          ...
2                  ...       ...          ...
3                  ...       ...          ...
```

This allows the clusters to be interpreted using the original units.

For example, a cluster may contain cows with relatively:

- Higher milk yield
- Lower milk pH
- Higher conductivity
- Higher somatic cell count

The actual characteristics depend on the dataset and the resulting K-Means solution.

The cluster summary is saved as:

```text
outputs/cluster_summary.csv
```

---

# Step 5: KNN Cluster Assignment

After K-Means creates the clusters, KNN is used to assign new observations to an existing cluster.

The custom implementation is located in:

```text
clustering/knn_tools.py
```

The process is:

```text
K-Means
   ↓
Existing observations receive cluster labels
   ↓
KNN learns those cluster labels
   ↓
New observation
   ↓
Find nearest observations
   ↓
Majority vote
   ↓
Predicted cluster
```

---

## KNN and Clustering

The KNN step is different from the original K-Means process.

### K-Means

Answers:

> "What groups can be formed from the existing observations?"

### KNN

Answers:

> "Given the groups already discovered, which group does this new observation most likely belong to?"

KNN therefore does not create new clusters.

It assigns observations to the existing clusters.

---

## KNN Parameter

The project uses:

```python
KNN_K = 5
```

This means that the five nearest known observations are used when assigning a new observation.

The cluster receiving the most votes becomes the predicted cluster.

---

## Simulating New Observations

The program treats:

```text
80% of observations
```

as known observations and:

```text
20% of observations
```

as new observations.

This is simulated using:

```python
n_known = int(len(X) * 0.8)
```

The K-Means cluster labels are already known for the entire dataset.

The labels of the final 20% are used only to compare the KNN assignment with the original K-Means assignments.

The comparison is reported as:

```text
KNN agreement with K-Means
```

This is **not classification accuracy against `class1`**.

It measures how often KNN reproduces the cluster assignments produced by K-Means.

---

# Output Files

The clustering program produces the following files:

```text
outputs/
├── 01_elbow.png
├── 02_clusters.png
├── cluster_summary.csv
└── clustered_cows.csv
```

### `01_elbow.png`

Elbow graph showing K versus inertia.

### `02_clusters.png`

Scatter plot showing the discovered clusters using:

- Milk pH
- Milk Conductivity

### `cluster_summary.csv`

Mean feature values for each cluster and the number of observations in each cluster.

### `clustered_cows.csv`

The original dataset with an additional:

```text
cluster
```

column.

---

# Project Structure

```text
cow_milk_mastitis/
│
├── clustering/
│   ├── data_loader.py
│   ├── kmeans_tf.py
│   ├── knn_tools.py
│   ├── visualize.py
│   ├── main.py
│   └── outputs/
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_cows.csv
│
├── regression/
│   └── ...
│
├── classification/
│   └── ...
│
├── preprocessing.py
├── cow_milk_mastitis_dataset.csv
└── README.md
```

---

# File Responsibilities

| File | Responsibility |
|---|---|
| `data_loader.py` | Load and scale the clustering data |
| `kmeans_tf.py` | Implement K-Means using TensorFlow |
| `knn_tools.py` | Implement KNN cluster assignment |
| `visualize.py` | Generate clustering visualizations |
| `main.py` | Run the complete clustering workflow |
| `outputs/` | Store graphs and CSV results |

---

# Libraries Used

The project uses:

```text
Python
NumPy
Pandas
TensorFlow
Scikit-learn
Matplotlib
```

### NumPy

Used for numerical operations and random centroid initialization.

### Pandas

Used for loading and analyzing the dataset.

### TensorFlow

Used to implement K-Means and KNN calculations.

### Scikit-learn

Used for:

- `StandardScaler`
- `silhouette_score`

### Matplotlib

Used to generate the clustering graphs.

---

# Design Decisions

## Why Standardize the Features?

K-Means relies on distance.

The features in the dataset have very different numerical scales.

For example:

```text
Milk pH
Milk Conductivity
Milk Yield
Somatic Cell Count
```

do not have comparable numerical ranges.

Standardization prevents a large-scale feature from dominating the distance calculation simply because its numbers are larger.

---

## Why Exclude `class1`?

`class1` is a known class label.

Using it as a clustering feature would allow the existing classification information to influence the clusters.

Since the purpose of clustering is to discover groups without known labels, `class1` is excluded.

It can still be useful later for comparing the discovered clusters with the known classes, but it should not be used to create the clusters.

---

## Why Keep Raw Data?

Clustering is performed using scaled data:

```python
X
```

but cluster interpretation uses:

```python
X_raw
```

This allows the mathematical algorithm to work correctly while keeping the results understandable in the original measurement units.

---

## Why Implement K-Means Manually?

The custom TensorFlow implementation demonstrates the actual operations behind K-Means:

1. Initialize centroids
2. Calculate distances
3. Assign observations
4. Calculate new centroids
5. Check convergence
6. Repeat

This makes the implementation useful for understanding the algorithm rather than treating K-Means as a black-box function.

---

# Important Interpretation Note

Clusters are **not automatically meaningful biological categories**.

K-Means will divide the observations according to the selected features and distance metric.

Therefore, a cluster should not automatically be interpreted as a medical or biological condition.

The cluster summary and silhouette score should be examined before making conclusions about what each group represents.

---

# Conclusion

This project demonstrates an unsupervised machine learning workflow using the Cow Milk Mastitis Dataset.

The workflow:

```text
Cow Milk Dataset
       ↓
Feature Selection
       ↓
Standardization
       ↓
Elbow Method
       ↓
K-Means Clustering
       ↓
Cluster Analysis
       ↓
KNN Cluster Assignment
```

The project demonstrates how K-Means can discover groups of cows based on milk-related measurements without using the existing `class1` label.

The custom TensorFlow implementation also demonstrates the internal mechanics of K-Means, while the KNN component provides a way to assign new observations to the clusters that have already been discovered.
