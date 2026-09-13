# ---------- Imports ----------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import data_loader
import evaluate
from knn_tf import TFKNNClassifier


# ---------- Output Directory ----------
OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------- Section Title ----------
def title(text):
    print("\n" + "-" * 60)
    print(text)
    print("-" * 60)


# ---------- Main Program ----------
def main():

    OUT_DIR.mkdir(exist_ok=True)

    # ================================================================
    # STEP 1 : Load and Prepare Data
    # ================================================================

    title("STEP 1 : LOAD DATA")

    data = data_loader.load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    class_names = data["class_names"]

    print(f"Total data      : {data['n_rows']} rows")
    print(f"Features        : {data['feature_names']}")
    print(f"Classes         : {class_names}")

    print(
        f"Data split      : "
        f"train {len(y_train)} / "
        f"validation {len(y_val)} / "
        f"test {len(y_test)}"
    )


    # ================================================================
    # STEP 2 : Find the Best K
    # ================================================================

    title("STEP 2 : SEARCH FOR THE BEST K")

    print("Testing different K values using validation data...\n")

    k_values = [1, 2, 3, 5, 10, 11, 15, 21, 30]

    validation_scores = []

    for k in k_values:

        model = TFKNNClassifier(k=k)

        model.fit(
            X_train,
            y_train
        )

        accuracy = model.score(
            X_val,
            y_val
        )

        validation_scores.append(accuracy)

        print(
            f"K = {k:>2} "
            f"| Validation Accuracy = "
            f"{accuracy * 100:.2f}%"
        )


    # ---------- Select Best K ----------
    best_index = int(
        np.argmax(validation_scores)
    )

    best_k = k_values[best_index]

    best_validation_accuracy = validation_scores[best_index]

    print(
        f"\nBest K: {best_k}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )


    # ---------- Save K Curve ----------
    evaluate.plot_k_curve(
        k_values,
        validation_scores,
        OUT_DIR / "01_k_curve.png"
    )


    # ================================================================
    # STEP 3 : Train Final Model and Evaluate Test Data
    # ================================================================

    title(
        f"STEP 3 : TRAIN FINAL KNN MODEL "
        f"(K = {best_k})"
    )

    # Train the final model using training data only
    model = TFKNNClassifier(
        k=best_k
    )

    model.fit(
        X_train,
        y_train
    )

    # Predict completely unseen test data
    y_pred = model.predict(
        X_test
    )

    # ---------- Test Accuracy ----------
    test_accuracy = float(
        np.mean(y_pred == y_test)
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


    # ---------- Classification Report ----------
    print(
        "\nClass-wise Classification Report:\n"
    )

    evaluate.print_report(
        y_test,
        y_pred,
        class_names
    )


    # ---------- Confusion Matrix ----------
    cm = evaluate.plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        OUT_DIR / "02_confusion_matrix.png"
    )

    print(
        "Confusion Matrix "
        "(rows = actual, columns = predicted):"
    )

    print(cm)


    # ================================================================
    # STEP 4 : Compare TensorFlow KNN with Scikit-learn KNN
    # ================================================================

    title(
        "STEP 4 : VERIFY KNN IMPLEMENTATION "
        "AGAINST SCIKIT-LEARN"
    )

    print(
        "Training Scikit-learn KNN with "
        f"K = {best_k}..."
    )

    sk_model = KNeighborsClassifier(
        n_neighbors=best_k
    )

    sk_model.fit(
        X_train,
        y_train
    )

    sk_pred = sk_model.predict(
        X_test
    )

    sklearn_accuracy = float(
        np.mean(sk_pred == y_test)
    )

    prediction_match = float(
        np.mean(sk_pred == y_pred)
    )

    print(
        f"\nTensorFlow KNN Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Scikit-learn KNN Accuracy: "
        f"{sklearn_accuracy:.4f}"
    )

    print(
        f"Matching Predictions    : "
        f"{prediction_match * 100:.2f}%"
    )


    # ================================================================
    # STEP 5 : Compare Against Majority-Class Baseline
    # ================================================================

    title(
        "STEP 5 : COMPARE AGAINST BASELINE"
    )

    # Convert y_train and y_test to numpy arrays safely to avoid scalar errors
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    # Predict the most common class for every test sample
    majority_class = np.bincount(
        y_train_arr
    ).argmax()

    baseline_accuracy = float(
        np.mean(
            y_test_arr == majority_class
        )
    )

    print(
        f"Baseline Accuracy "
        f"(always predict '{class_names[majority_class]}') "
        f": {baseline_accuracy * 100:.2f}%"
    )

    print(
        f"KNN Accuracy                              "
        f": {test_accuracy * 100:.2f}%"
    )


    # ================================================================
    # STEP 6 : Save Predictions
    # ================================================================

    title("STEP 6 : SAVE PREDICTIONS")

    evaluate.save_predictions(
        y_test,
        y_pred,
        class_names,
        OUT_DIR / "predictions.csv"
    )

    print(
        "Prediction results saved to:"
    )

    print(
        f"  {OUT_DIR / 'predictions.csv'}"
    )


    # ---------- Display Generated Outputs ----------
    print("\nGenerated outputs:")

    for file in sorted(OUT_DIR.iterdir()):

        print(
            f"  - outputs/{file.name}"
        )


# ---------- Program Entry Point ----------
if __name__ == "__main__":
    main()