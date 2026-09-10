# ---------- Imports ----------
import numpy as np
import tensorflow as tf


# ---------- TensorFlow K-Means ----------
class TFKMeans:

    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    # ---------- Distance Calculation ----------
    def _distance(self, X, centroids):
        diff = X[:, None, :] - centroids[None, :, :]
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

    # ---------- Model Training ----------
    def fit(self, X):
        X = tf.constant(X, dtype=tf.float32)
        n_samples = X.shape[0]

        # ---------- Initial Centroids ----------
        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(
            n_samples,
            size=self.n_clusters,
            replace=False
        )
        centroids = tf.gather(X, start_idx)

        # ---------- K-Means Iteration ----------
        for step in range(self.max_iter):

            # Assign each point to nearest centroid
            distances = self._distance(X, centroids)
            labels = tf.argmin(
                distances,
                axis=1,
                output_type=tf.int32
            )

            # Update centroids
            new_centroids = []

            for c in range(self.n_clusters):
                members = tf.boolean_mask(X, labels == c)

                if tf.shape(members)[0] > 0:
                    new_centroids.append(
                        tf.reduce_mean(members, axis=0)
                    )
                else:
                    new_centroids.append(centroids[c])

            new_centroids = tf.stack(new_centroids)

            # Check convergence
            moved = float(
                tf.reduce_max(
                    tf.abs(new_centroids - centroids)
                )
            )

            centroids = new_centroids

            if moved < 1e-4:
                break

        # ---------- Store Results ----------
        distances = self._distance(X, centroids)

        self.labels_ = tf.argmin(
            distances,
            axis=1,
            output_type=tf.int32
        ).numpy()

        self.centroids_ = centroids.numpy()
        self.n_iter_ = step + 1

        # Sum of squared distances to nearest centroid
        self.inertia_ = float(
            tf.reduce_sum(
                tf.square(
                    tf.reduce_min(distances, axis=1)
                )
            )
        )

        return self

    # ---------- Predict Clusters ----------
    def fit_predict(self, X):
        return self.fit(X).labels_