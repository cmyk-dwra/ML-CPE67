# ---------- Imports ----------
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------- K Value Curve ----------
def plot_k_curve(k_values, scores, out_path):

    plt.figure(figsize=(7, 4.5))

    plt.plot(
        k_values,
        scores,
        "o-"
    )

    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.xticks(k_values)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=120
    )

    plt.close()


# ---------- Confusion Matrix ----------
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    out_path
):
    # --- FIX: Ensure class_names is a list of strings ---
    if isinstance(class_names, (str, int, np.integer)):
        class_names = [str(class_names)]
    else:
        class_names = [str(c) for c in class_names]
    # ----------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    for i in range(len(class_names)):

        for j in range(len(class_names)):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=120
    )

    plt.close()

    return cm


# ---------- Classification Report ----------
def print_report(
    y_true,
    y_pred,
    class_names
):
    # --- FIX: Ensure class_names is a list of strings ---
    if isinstance(class_names, (str, int, np.integer)):
        class_names = [str(class_names)]
    else:
        class_names = [str(c) for c in class_names]
    # ----------------------------------------------------

    print("\n" + "=" * 60)
    print("             KNN CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )


# ---------- Prediction Accuracy ----------
def print_accuracy(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    return accuracy


# ---------- Save Predictions ----------
def save_predictions(
    y_true,
    y_pred,
    class_names,
    out_path
):

    df = pd.DataFrame({
        "true_label": [
            class_names[i]
            for i in y_true
        ],

        "predicted_label": [
            class_names[i]
            for i in y_pred
        ],

        "correct": y_true == y_pred
    })

    df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    return df