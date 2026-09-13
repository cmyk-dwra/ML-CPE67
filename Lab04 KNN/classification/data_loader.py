from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "cow_milk_mastitis_dataset.csv"
)

TARGET = "class1"

NUMERIC_FEATURES = [
    "Milk_Temperature",
    "Milk_pH",
    "Milk_Conductivity",
    "Somatic_Cell_Count",
    "Milk_Yield",
    "Clotting",
]


def load_data(test_size=0.2, seed=42):

    # ---------- Step 1: Read CSV ----------
    df = pd.read_csv(CSV_PATH)

    # ---------- Step 2: Prepare Features ----------
    X = df[NUMERIC_FEATURES].copy()

    # ---------- Step 3: Prepare Target ----------
    class_names = sorted(df[TARGET].unique())

    y = df[TARGET].map(
        {name: i for i, name in enumerate(class_names)}
    )

    X = X.to_numpy(dtype="float32")
    y = y.to_numpy(dtype="int32")

    # ---------- Step 4: Train / Validation / Test Split ----------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=seed,
        stratify=y_temp
    )

    # ---------- Step 5: Feature Scaling ----------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,

        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES,
        "n_rows": len(df),
    }


if __name__ == "__main__":

    data = load_data()

    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("classes:", data["class_names"])