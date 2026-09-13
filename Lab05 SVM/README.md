# SVM Image Recognition - Cheetah vs Hyena

## Overview

This project uses a **Support Vector Machine (SVM)** to classify animal images into two classes:

- Cheetah
- Hyena

The project loads images from class-specific folders, preprocesses them into numerical feature vectors, splits the dataset into training and testing sets, trains an SVM classifier, evaluates its predictions, and saves the trained model and results.

The implementation is organized into separate Python modules so that data loading, preprocessing, dataset splitting, model training, prediction, and evaluation are handled independently.

---

## Dataset Structure

[DOWNLOAD DATASET HERE](https://www.kaggle.com/datasets/singhdatascientist/for-image-classification-of-cheetah-vs-hyena)

```text
train/
├── cheetah/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
└── hyena/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

The data loader detects the class directories automatically.

The classes are sorted alphabetically and assigned numerical labels using their position:

```text
cheetah → 0
hyena   → 1
```

Supported image formats are:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

---

## Project Structure

```text
Lab05/
├── __pycache__/
│
├── outputs/
│
├── train/
│   ├── cheetah/
│   └── hyena/
│
├── data_load.py
├── evaluate.py
├── main.py
├── preprocess.py
├── split_data.py
├── svm_model.py
└── test_svm.py
```

### File Responsibilities

| File | Purpose |
|---|---|
| `main.py` | Runs the complete SVM classification pipeline |
| `data_load.py` | Finds classes and loads images from the dataset |
| `preprocess.py` | Converts loaded images into the required feature representation |
| `split_data.py` | Splits the feature data into training and testing sets |
| `svm_model.py` | Trains the SVM and generates predictions |
| `evaluate.py` | Calculates accuracy, classification report, and confusion matrix |
| `test_svm.py` | Used for testing the SVM-related implementation |
| `train/` | Contains the image dataset organized by class |
| `outputs/` | Stores generated NumPy files, model files, class information, and evaluation results |

---

# Workflow

The program follows this pipeline:

```text
Image Dataset
     ↓
data_load.py
     ↓
Loaded Images + Labels
     ↓
preprocess.py
     ↓
Numerical Feature Vectors
     ↓
split_data.py
     ↓
Training Data + Testing Data
     ↓
svm_model.py
     ↓
Trained SVM
     ↓
Predictions
     ↓
evaluate.py
     ↓
Accuracy + Classification Report
     + Confusion Matrix
```

---

# Step 1: Load Dataset

The dataset is loaded using:

```python
images, labels, classes = load_data(
    DATA_PATH,
    IMG_SIZE,
    MAX_PER_CLASS
)
```

The current configuration in `main.py` is:

```python
DATA_PATH = "train"
IMG_SIZE = 100
MAX_PER_CLASS = 3000
```

`DATA_PATH` points to the directory containing the class folders.

The loader automatically detects the available classes instead of requiring the class names to be manually written into the program.

For every image, the loader:

1. Finds the image file.
2. Reads it using OpenCV.
3. Passes it to the preprocessing function.
4. Skips the image if it cannot be read or processed.
5. Stores the processed image and its numerical class label.

The loaded images and labels are also saved:

```text
outputs/
├── images.npy
├── labels.npy
└── classes.json
```

---

# Step 2: Preprocessing

The preprocessing stage converts the loaded images into numerical features suitable for an SVM.

The main program calls:

```python
X = to_features(images)
y = labels
```

The resulting feature matrix has the form:

```text
(number of images, number of features)
```

The feature shape is printed so the resulting representation can be inspected before training.

The exact image transformation is implemented in `preprocess.py`, keeping image processing separate from the model itself.

---

# Step 3: Split Dataset

The feature matrix and labels are passed to:

```python
X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
    TEST_SIZE
)
```

The current test size is:

```python
TEST_SIZE = 0.2
```

Therefore, approximately:

```text
80% → Training data
20% → Testing data
```

The resulting datasets are saved in the `outputs/` directory:

```text
X_train.npy
X_test.npy
y_train.npy
y_test.npy
```

Saving these intermediate datasets makes it possible to inspect or reuse the processed data without necessarily loading and preprocessing every image again.

---

# Step 4: Train SVM

The SVM is trained using:

```python
model, scaler = train_svm(
    X_train,
    y_train
)
```

The model and scaler are then saved using `joblib`:

```text
outputs/
├── svm_model.pkl
└── scaler.pkl
```

The scaler is saved alongside the model because preprocessing applied to the training features must also be applied consistently when making predictions.

---

# Step 5: Prediction

After training, the model predicts the labels of the test samples:

```python
predictions = predict_svm(
    model,
    scaler,
    X_test
)
```

The predictions are then compared against the true test labels stored in:

```python
y_test
```

---

# Step 6: Evaluation

Evaluation is handled by `evaluate.py`:

```python
evaluate_model(
    y_test,
    predictions,
    classes,
    save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
)
```

Three main evaluation results are produced.

## Accuracy

Accuracy measures the proportion of test images that were classified correctly.

```text
Accuracy = Correct Predictions / Total Predictions
```

The result is printed as a percentage.

---

## Classification Report

The classification report provides class-level metrics including:

- Precision
- Recall
- F1-score
- Support

This gives more information than accuracy alone, particularly when the number of samples belonging to different classes is not identical.

The report is generated using:

```python
classification_report(
    y_test,
    predictions,
    labels=labels,
    target_names=classes,
    zero_division=0
)
```

---

## Confusion Matrix

The confusion matrix shows how the predictions are distributed between the true and predicted classes.

For the two-class Cheetah vs Hyena problem, the matrix can be interpreted as:

```text
                 Predicted
              Cheetah  Hyena
Actual
Cheetah          ✓       ✗
Hyena            ✗       ✓
```

The matrix is printed to the terminal and also saved as:

```text
outputs/confusion_matrix.png
```

The rows represent the **true classes**, while the columns represent the **predicted classes**.

---

# Output Files

After running `main.py`, the `outputs/` directory contains the generated data and model results.

```text
outputs/
├── images.npy
├── labels.npy
├── classes.json
├── X_train.npy
├── X_test.npy
├── y_train.npy
├── y_test.npy
├── svm_model.pkl
├── scaler.pkl
└── confusion_matrix.png
```

### Description

| Output | Description |
|---|---|
| `images.npy` | Processed images loaded from the dataset |
| `labels.npy` | Numerical labels corresponding to the images |
| `classes.json` | Names of the detected classes |
| `X_train.npy` | Training feature vectors |
| `X_test.npy` | Testing feature vectors |
| `y_train.npy` | Training labels |
| `y_test.npy` | Testing labels |
| `svm_model.pkl` | Saved trained SVM model |
| `scaler.pkl` | Saved feature scaler |
| `confusion_matrix.png` | Visualization of classification results |

---

# Configuration

The main configuration values are defined in `main.py`:

```python
DATA_PATH = "train"
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 3000
```

### `DATA_PATH`

Specifies the directory containing the class folders.

```python
DATA_PATH = "train"
```

### `OUTPUT_DIR`

Specifies where generated files are stored.

```python
OUTPUT_DIR = "outputs"
```

### `IMG_SIZE`

Controls the image size used during preprocessing.

```python
IMG_SIZE = 100
```

### `TEST_SIZE`

Controls the proportion of the dataset reserved for testing.

```python
TEST_SIZE = 0.2
```

### `MAX_PER_CLASS`

Limits the number of images loaded from each class.

```python
MAX_PER_CLASS = 3000
```

Setting it to:

```python
MAX_PER_CLASS = None
```

allows the loader to use all available images, although this can increase processing time and memory usage.

---

# Libraries

The project uses the following Python libraries:

### NumPy

Used for numerical arrays and saving processed data in `.npy` format.

### OpenCV

Used to read image files.

### Matplotlib

Used to generate the confusion matrix visualization.

### Scikit-learn

Used for machine-learning utilities and evaluation metrics.

### Joblib

Used to save the trained SVM model and scaler.

---

# Design

The project separates the machine-learning pipeline into several modules.

This makes each stage easier to understand and modify.

```text
data_load.py
    ↓
preprocess.py
    ↓
split_data.py
    ↓
svm_model.py
    ↓
evaluate.py
```

`main.py` acts as the controller that connects these modules.

This separation also means that changing one stage does not require putting the entire machine-learning pipeline into a single large Python file.

---

# Why SVM?

A Support Vector Machine is a supervised classification algorithm that attempts to find a decision boundary separating classes.

For this project, the SVM receives numerical image features rather than the original image files directly.

Conceptually:

```text
Image
  ↓
Numerical Representation
  ↓
Feature Vector
  ↓
SVM
  ↓
Cheetah / Hyena
```

The SVM learns from the labeled training examples and uses the learned decision boundary to classify previously unseen test images.

---

# Running the Project

From the project directory, run:

```bash
python main.py
```

The program will:

1. Create the `outputs/` directory if necessary.
2. Load the image dataset.
3. Preprocess the images.
4. Split the dataset.
5. Train the SVM.
6. Predict the test labels.
7. Calculate evaluation metrics.
8. Save the trained model and generated results.

The terminal will display progress for each stage.

---

# Important Path Note

The dataset path is currently:

```python
DATA_PATH = "train"
```

This is a relative path.

The program therefore expects the `train` directory to be available relative to the directory from which the program is executed.

The intended project layout is:

```text
Lab05/
├── main.py
├── data_load.py
├── preprocess.py
├── split_data.py
├── svm_model.py
├── evaluate.py
│
├── train/
│   ├── cheetah/
│   └── hyena/
│
└── outputs/
```

If the program is executed from another working directory, the relative path may not resolve to the intended `train` folder.

---

# Conclusion

This project demonstrates a complete image-classification pipeline using an SVM.

The process begins with class-organized image data and ends with a trained classifier and quantitative evaluation:

```text
Dataset
  ↓
Image Loading
  ↓
Preprocessing
  ↓
Feature Extraction
  ↓
Train/Test Split
  ↓
SVM Training
  ↓
Prediction
  ↓
Evaluation
```

The modular structure allows the image-loading, preprocessing, splitting, SVM, and evaluation stages to remain independent while `main.py` coordinates the complete workflow.
