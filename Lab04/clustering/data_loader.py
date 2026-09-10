"""
Read Cow Milk Mastitis CSV
Prepare numerical features for clustering
Keep both raw and scaled data
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------- Dataset ----------
CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "cow_milk_mastitis_dataset.csv"
)


# ---------- Features Used for Clustering ----------
FEATURES = [
    "Milk_Temperature",
    "Milk_pH",
    "Milk_Conductivity",
    "Somatic_Cell_Count",
    "Milk_Yield",
    "Clotting",
]


def load_data():
    """
    Return:
        X       : scaled features used for clustering
        X_raw   : original feature values
        df      : complete dataset
        features: feature names
    """

    df = pd.read_csv(CSV_PATH)

    # ---------- Prepare Features ----------
    X_raw = df[FEATURES].to_numpy(dtype="float32")

    # ---------- Scale Features ----------
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype("float32")

    return {
        "X": X,
        "X_raw": X_raw,
        "df": df,
        "features": FEATURES
    }


if __name__ == "__main__":
    data = load_data()

    print("Data size:", data["X"].shape)
    print(
        "Mean after scaling:",
        data["X"].mean(axis=0).round(3)
    )