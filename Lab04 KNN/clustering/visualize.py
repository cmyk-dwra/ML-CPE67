# ---------- Imports ----------
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------- Elbow Plot ----------
def plot_elbow(k_values, inertias, out_path):

    plt.figure(figsize=(7, 4.5))

    plt.plot(k_values, inertias, "o-")

    plt.xlabel("K (Number of Clusters)")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------- Cluster Visualization ----------
def plot_clusters(X_2d, labels, out_path):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        s=50
    )

    plt.xlabel("Milk pH")
    plt.ylabel("Milk Conductivity")
    plt.title("Cow Milk Clustering")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()