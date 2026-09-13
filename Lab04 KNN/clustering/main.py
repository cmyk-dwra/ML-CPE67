# ---------- Imports ----------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

import data_loader
import visualize
from kmeans_tf import TFKMeans
from knn_tools import KNNClusterAssigner


# ---------- Settings ----------
OUT_DIR = Path(__file__).resolve().parent / "outputs"

N_CLUSTERS = 4
KNN_K = 5


def title(text):
    print("\n" + "-" * 60)
    print(text)
    print("-" * 60)


def main():

    OUT_DIR.mkdir(exist_ok=True)

    # ---------- STEP 1 : Load Data ----------
    title("STEP 1 : LOAD DATA")

    data = data_loader.load_data()

    X = data["X"]
    X_raw = data["X_raw"]
    df = data["df"]
    features = data["features"]

    print(f"Data size: {X.shape[0]} rows x {X.shape[1]} features")
    print("Features used for clustering:")

    for feature in features:
        print(f"  - {feature}")

    # ---------- STEP 2 : Find Best K ----------
    title("STEP 2 : FIND THE BEST NUMBER OF CLUSTERS")

    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertias = []

    for k in k_values:
        km = TFKMeans(n_clusters=k)
        km.fit(X)

        silhouette = silhouette_score(X, km.labels_)
        inertias.append(km.inertia_)

        print(
            f"K = {k:>2} | "
            f"Inertia = {km.inertia_:8.1f} | "
            f"Silhouette = {silhouette:.3f}"
        )

    visualize.plot_elbow(
        k_values,
        inertias,
        OUT_DIR / "01_elbow.png"
    )

    print("\nElbow graph saved to outputs/01_elbow.png")
    print(f"Selected K = {N_CLUSTERS}")

    # ---------- STEP 3 : Run K-Means ----------
    title(f"STEP 3 : RUN K-MEANS (K = {N_CLUSTERS})")

    km = TFKMeans(n_clusters=N_CLUSTERS)
    labels = km.fit_predict(X)

    silhouette = silhouette_score(X, labels)

    print(f"Iterations       : {km.n_iter_}")
    print(f"Inertia          : {km.inertia_:.1f}")
    print(f"Silhouette score : {silhouette:.3f}")
    print(
        "Members per cluster:",
        np.bincount(labels).tolist()
    )

    if silhouette < 0.25:
        print("\n[Note] Low silhouette score.")
        print("       The clusters may not be clearly separated.")
        print("       K-Means always creates clusters.")

    # ---------- Cluster Visualization ----------
    visualize.plot_clusters(
        X_raw[:, [1, 2]],
        labels,
        OUT_DIR / "02_clusters.png"
    )

    # ---------- STEP 4 : Analyze Clusters ----------
    title("STEP 4 : ANALYZE EACH CLUSTER")

    profile = pd.DataFrame(
        X_raw.astype("float64"),
        columns=features
    )
    profile["cluster"] = labels

    summary = profile.groupby("cluster").mean().round(2)
    summary["member_count"] = np.bincount(labels)

    print(summary.to_string())

    summary.to_csv(
        OUT_DIR / "cluster_summary.csv",
        encoding="utf-8-sig"
    )

    # ---------- STEP 5 : KNN Cluster Assignment ----------
    title(f"STEP 5 : ASSIGN NEW DATA USING KNN (K = {KNN_K})")

    n_known = int(len(X) * 0.8)

    X_known = X[:n_known]
    labels_known = labels[:n_known]

    X_new = X[n_known:]
    labels_new = labels[n_known:]

    assigner = KNNClusterAssigner(k=KNN_K)
    assigner.fit(X_known, labels_known)

    knn_pred = assigner.predict(X_new)

    accuracy = float(
        np.mean(knn_pred == labels_new)
    )

    print(f"New observations: {len(X_new)}")
    print(
        f"KNN agreement with K-Means: "
        f"{accuracy * 100:.1f}%"
    )

    print("\nKNN can assign new observations")
    print("without running K-Means again.")

    # ---------- Save Results ----------
    title("SAVE RESULTS")

    result = df.copy()
    result["cluster"] = labels

    result.to_csv(
        OUT_DIR / "clustered_cows.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nGenerated outputs:")

    for file in sorted(OUT_DIR.iterdir()):
        print(f"  - outputs/{file.name}")


if __name__ == "__main__":
    main()