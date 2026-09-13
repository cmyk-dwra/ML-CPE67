# ---------- Imports ----------
import numpy as np
import tensorflow as tf


# ---------- TensorFlow KNN Classifier ----------
class TFKNNClassifier:
    def __init__(self, k=5):
        self.k = k
    # ---------- Model Training ----------
    def fit(self, X, y):
        """
        Store the training data as TensorFlow tensors.
        """
        self.X_train = tf.constant(
            X,
            dtype=tf.float32
        )

        self.y_train = tf.constant(
            y,
            dtype=tf.int32
        )

        self.n_classes = int(y.max()) + 1

        return self

    # ---------- Distance Calculation ----------
    def _distance(self, X_new):
        """
        Calculate Euclidean distance between
        each new sample and every training sample.

        distance = sqrt(
            (x1 - y1)^2 +
            (x2 - y2)^2 +
            ...
        )
        """

        diff = (
            X_new[:, None, :]
            - self.X_train[None, :, :]
        )

        return tf.sqrt(
            tf.reduce_sum(
                tf.square(diff),
                axis=2
            )
        )

    # ---------- Prediction ----------
    def predict(self, X):
        """
        Predict the class of each input sample
        using the K nearest training samples.
        """

        X = tf.constant(
            X,
            dtype=tf.float32
        )

        # Step 1: Calculate distances
        distances = self._distance(X)

        # Step 2: Select K nearest neighbors
        # top_k normally finds the largest values.
        # Negating the distances makes the smallest
        # distances become the largest values.
        _, indices = tf.math.top_k(
            -distances,
            k=self.k
        )

        # Get the class labels of the K neighbors
        neighbor_labels = tf.gather(
            self.y_train,
            indices
        )

        # Step 3: Count votes for each class
        one_hot = tf.one_hot(
            neighbor_labels,
            depth=self.n_classes
        )

        votes = tf.reduce_sum(
            one_hot,
            axis=1
        )

        # Select the class with the most votes
        predictions = tf.argmax(
            votes,
            axis=1
        )

        return predictions.numpy()

    # ---------- Model Evaluation ----------
    def score(self, X, y):
        """
        Calculate classification accuracy
        as the proportion of correct predictions.
        """

        predictions = self.predict(X)

        return float(
            np.mean(predictions == y)
        )