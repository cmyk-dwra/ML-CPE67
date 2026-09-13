# Cow Milk Mastitis Dataset - KNN Classification

## Overview

This project applies a machine learning classification workflow to the **Cow Milk Mastitis Dataset**.

The objective is to classify whether a cow is associated with mastitis using milk-related measurements.

The classification workflow consists of:

1. Loading the dataset
2. Preparing the features and target
3. Splitting the data into training, validation, and testing sets
4. Standardizing the feature values
5. Testing different K values
6. Selecting the best K based on validation accuracy
7. Training the final KNN classifier
8. Evaluating the model on unseen test data
9. Comparing the TensorFlow implementation with Scikit-learn
10. Comparing the model against a majority-class baseline
11. Visualizing and saving the results

---

## Dataset

[DOWNLOAD DATASET HERE](https://www.kaggle.com/datasets/amithadityacp/cow-mastitisfrom-milk)

### Relevant Features

| Feature | Description |
|---|---|
| `Milk_Temperature` | Temperature of the milk |
| `Milk_pH` | pH level of the milk |
| `Milk_Conductivity` | Electrical conductivity of the milk |
| `Somatic_Cell_Count` | Number of somatic cells in the milk |
| `Milk_Yield` | Milk production/yield |
| `Clotting` | Clotting measurement |
| `class1` | Classification target indicating the class associated with mastitis |

---

## Classification Objective

The objective of the classification model is to predict:

```
class1
```

using milk-related measurements as input features.

The feature matrix and target are prepared using:

```python
X = df[NUMERIC_FEATURES].copy()
y = df[TARGET]
```

The model learns patterns in the milk measurements and uses them to predict the class of previously unseen samples.

---

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded from the CSV file using the project's data loader.

```python
df = pd.read_csv(CSV_PATH)
```

The data loader is responsible for preparing the dataset for the KNN classification workflow.

---

### 2. Feature Preparation

The classification features are separated from the target variable.

The input features consist of:

```
Milk_Temperature
Milk_pH
Milk_Conductivity
Somatic_Cell_Count
Milk_Yield
Clotting
```

The target variable is:

```
class1
```

The features are converted into numerical arrays so that they can be processed by the machine learning model.

---

### 3. Train / Validation / Test Split

The dataset is divided into three subsets:

```
Training      60%
Validation    20%
Testing       20%
```

The training set is used to train the KNN model.

The validation set is used to compare different values of `K` and select the best one.

The test set is kept separate until the final evaluation.

This prevents the test data from influencing the selection of the model parameters.

---

### 4. Feature Scaling

KNN is a distance-based algorithm, so feature scaling is particularly important.

`StandardScaler` is used to standardize the features:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

The scaler is fitted using only the training data.

The same transformation is then applied to the validation and test data.

This prevents information from the validation or test sets from being used during preprocessing.

---

## K-Nearest Neighbors

K-Nearest Neighbors (KNN) classifies a new sample based on the classes of the nearest training samples.

The model uses a value called `K`, which determines the number of neighboring samples considered during prediction.

For example:

```
K = 3
```

means that the model examines the three closest training samples and uses their class labels to determine the prediction.

The classification process can be summarized as:

```
New Sample
     |
     v
Calculate distances
     |
     v
Find K nearest samples
     |
     v
Count class votes
     |
     v
Select majority class
     |
     v
Prediction
```

---

## TensorFlow KNN Implementation

This project includes a custom KNN implementation using TensorFlow.

The implementation is contained in:

```
classification/knn_tf.py
```

The main class is:

```python
TFKNNClassifier
```

### Model Training

The `fit()` method stores the training data as TensorFlow tensors.

```python
model = TFKNNClassifier(k=5)

model.fit(X_train, y_train)
```

Unlike many traditional machine learning models, KNN does not learn a set of coefficients during training.

Instead, it stores the training samples so that distances can be calculated when predictions are requested.

---

### Distance Calculation

The implementation calculates Euclidean distance between new samples and training samples.

The distance is based on:

```
distance = sqrt(
    (x1 - y1)^2 +
    (x2 - y2)^2 +
    ...
)
```

The TensorFlow implementation performs these calculations using tensor operations.

---

### Finding the Nearest Neighbors

The model identifies the `K` training samples with the smallest distances.

TensorFlow's `top_k()` function normally selects the largest values.

Therefore, the distances are negated before calling `top_k()`:

```python
_, indices = tf.math.top_k(
    -distances,
    k=self.k
)
```

Negating the distances converts the smallest original distances into the largest negative values.

For example:

```
Original distance:

1.0   3.0   5.0

Negative distance:

-1.0  -3.0  -5.0
```

The largest negative value corresponds to the smallest original distance.

---

### Majority Voting

After finding the nearest neighbors, their class labels are collected.

The labels are converted into one-hot representations and summed to count the votes for each class.

The class with the highest number of votes becomes the final prediction.

---

## Searching for the Best K

The project tests multiple K values:

```python
k_values = [1, 2, 3, 5, 10, 11, 15, 21, 30]
```

Each value is evaluated using the validation dataset.

```
Training Data
      |
      +---- K = 1  ----> Validation Accuracy
      |
      +---- K = 2  ----> Validation Accuracy
      |
      +---- K = 3  ----> Validation Accuracy
      |
      +---- ...
      |
      +---- K = 30 ----> Validation Accuracy
```

The K value with the highest validation accuracy is selected as the best K.

The test set is not used during this process.

This keeps the final test evaluation independent from model selection.

---

## K Value Visualization

The validation accuracy for each K value is saved as:

```
01_k_curve.png
```

The graph shows how the model's validation accuracy changes as the number of neighbors changes.

A very small K can make the model sensitive to individual training samples.

A larger K considers more neighboring samples and generally produces a smoother decision.

The graph helps identify a suitable K based on validation performance.

---

## Final Model

After selecting the best K, a new TensorFlow KNN model is trained using the training data.

```python
model = TFKNNClassifier(k=best_k)

model.fit(
    X_train,
    y_train
)
```

The final model then predicts the previously unseen test data:

```python
y_pred = model.predict(X_test)
```

---

## Model Evaluation

The final model is evaluated using the test dataset.

### Accuracy

Accuracy measures the proportion of correctly classified samples.

```
Accuracy =
Correct Predictions / Total Predictions
```

Higher accuracy indicates that a larger proportion of the test samples were classified correctly.

The accuracy is calculated using:

```python
accuracy = np.mean(y_pred == y_test)
```

---

### Classification Report

The classification report provides class-wise:

- Precision
- Recall
- F1-score
- Support

It is generated using:

```python
classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
)
```

### Precision

Precision measures how many samples predicted as a particular class were actually members of that class.

```
Precision =
True Positives / (True Positives + False Positives)
```

---

### Recall

Recall measures how many actual samples belonging to a class were correctly identified.

```
Recall =
True Positives / (True Positives + False Negatives)
```

---

### F1-Score

F1-score combines precision and recall into a single metric.

```
F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
```

---

## Confusion Matrix

A confusion matrix compares the actual classes against the predicted classes.

The rows represent the actual class.

The columns represent the predicted class.

The matrix is generated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

The resulting visualization is saved as:

```
02_confusion_matrix.png
```

The confusion matrix makes it possible to see which classes were correctly predicted and which classes were misclassified.

---

## Prediction Results

The program also displays individual prediction results.

Each test sample contains:

```
Actual Class
Predicted Class
Correct / Wrong
```

For example:

```
Actual: No Mastitis | Predicted: No Mastitis | CORRECT
Actual: Mastitis    | Predicted: Mastitis    | CORRECT
Actual: Mastitis    | Predicted: No Mastitis | WRONG
```

The prediction results are saved to:

```
predictions.csv
```

The CSV contains:

| Column | Description |
|---|---|
| `true_label` | Actual class |
| `predicted_label` | Class predicted by the KNN model |
| `correct` | Whether the prediction was correct |

---

## TensorFlow vs Scikit-learn

The project compares the custom TensorFlow implementation with the Scikit-learn KNN implementation.

Scikit-learn provides a standard KNN classifier:

```python
KNeighborsClassifier(
    n_neighbors=best_k
)
```

The same K value and data are used for both implementations.

The comparison checks:

```
TensorFlow KNN Accuracy
        vs
Scikit-learn KNN Accuracy
```

The project also compares the predictions produced by both implementations.

If the implementations are equivalent and deterministic under the same conditions, their predictions should generally match.

This comparison provides a practical check that the custom TensorFlow implementation is performing KNN correctly.

---

## Baseline Comparison

The KNN model is also compared against a simple majority-class baseline.

The baseline always predicts the most common class in the training data.

For example, if the training data contains:

```
No Mastitis : 700 samples
Mastitis    : 300 samples
```

the baseline always predicts:

```
No Mastitis
```

The baseline accuracy is then compared with the KNN accuracy.

```
Baseline Accuracy
        vs
KNN Accuracy
```

If KNN performs better than the baseline, this provides evidence that the model is extracting useful predictive information from the selected features rather than simply benefiting from class imbalance.

If KNN does not outperform the baseline, the selected features may provide limited predictive information for the classification task.

---

## Evaluation Module

The evaluation and visualization functions are separated into:

```
classification/evaluate.py
```

This module contains functions for:

### K Curve

```python
plot_k_curve(
    k_values,
    scores,
    out_path
)
```

Creates the validation accuracy graph for different K values.

### Confusion Matrix

```python
plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    out_path
)
```

Creates and saves the confusion matrix visualization.

### Classification Report

```python
print_report(
    y_true,
    y_pred,
    class_names
)
```

Displays precision, recall, F1-score, and support.

### Accuracy

```python
print_accuracy(
    y_true,
    y_pred
)
```

Calculates the classification accuracy.

### Save Predictions

```python
save_predictions(
    y_true,
    y_pred,
    class_names,
    out_path
)
```

Saves the actual and predicted class labels to a CSV file.

---

## Project Structure

```text
cow_milk_mastitis/
│
├── classification/
│   │
│   ├── data_loader.py
│   ├── knn_tf.py
│   ├── evaluate.py
│   ├── main.py
│   │
│   └── outputs/
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
│
├── regression/
│   ├── ...
│
├── preprocessing.py
├── cow_milk_mastitis_dataset.csv
└── README.md
```

---

## File Responsibilities

| File | Responsibility |
|---|---|
| `data_loader.py` | Loads, prepares, splits, and scales the dataset |
| `knn_tf.py` | Contains the custom TensorFlow KNN implementation |
| `evaluate.py` | Calculates evaluation metrics and generates result files/graphs |
| `main.py` | Runs the complete KNN classification workflow |
| `outputs/` | Stores generated graphs and prediction results |

---

## Libraries Used

The classification project uses the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow

### Main Components

```text
Pandas
    Dataset loading and data handling

NumPy
    Numerical operations and accuracy calculation

Scikit-learn
    Train/test splitting
    StandardScaler
    KNeighborsClassifier
    Classification metrics

TensorFlow
    Custom KNN implementation

Matplotlib
    Result visualization
```

---

## Output

Running:

```bash
python main.py
```

produces:

### Console Output

- Total number of dataset rows
- Feature names
- Classification classes
- Training/validation/testing sample counts
- Validation accuracy for each K value
- Best K value
- Final test accuracy
- Classification report
- Confusion matrix
- TensorFlow vs Scikit-learn comparison
- Baseline accuracy comparison

### Generated Files

```text
outputs/
├── 01_k_curve.png
├── 02_confusion_matrix.png
└── predictions.csv
```

---

## Important Design Decisions

### Separate Validation and Test Data

The validation set is used to choose the best K.

The test set is only used after the best K has been selected.

This prevents the test set from influencing model selection.

---

### Scaling Before KNN

KNN relies on distances between samples.

Therefore, features with very different numerical scales could otherwise dominate the distance calculation.

Standardization makes the feature scales more comparable before calculating distances.

---

### Custom TensorFlow Implementation

The TensorFlow KNN implementation demonstrates how KNN works internally instead of relying entirely on a pre-built classifier.

The implementation explicitly performs:

```text
Distance Calculation
        ↓
Nearest Neighbor Selection
        ↓
Majority Voting
        ↓
Class Prediction
```

Scikit-learn is then used as a reference implementation to verify the result.

---

## Conclusion

This project demonstrates a complete K-Nearest Neighbors classification workflow for the Cow Milk Mastitis Dataset.

The workflow combines:

- Feature preparation
- Train/validation/test splitting
- Feature scaling
- K-value selection
- Custom TensorFlow KNN
- Classification evaluation
- Confusion matrix analysis
- Scikit-learn verification
- Baseline comparison
- Result visualization
- Prediction export

The separation into `data_loader.py`, `knn_tf.py`, `evaluate.py`, and `main.py` keeps each part of the machine learning pipeline focused on a specific responsibility.

This structure also makes it easier to understand, test, and modify individual parts of the classification system without placing the entire workflow inside a single script.

---
