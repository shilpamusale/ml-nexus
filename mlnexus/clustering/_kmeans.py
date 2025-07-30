import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing  import List, Dict

class MyKMeans:
    """
    K-Means clustering algorithm implemented from scratch with type hints.
    """
    def __init__(self, k: int = 3, max_iterations: int = 100):
        """
        Initializes the KMeans instance.

        Args:
            k (int): The number of clusters.
            max_iterations (int): The maximum number of iterations to run.
        """
        self.k: int = k
        self.max_iterations: int = max_iterations
        # Dictionary to hold the data points for each cluster, keyed by cluster index
        self.clusters: Dict[int, List[npt.NDArray[np.float64]]] = {i: [] for i in range(self.k)}
        # The center points (centroids) of the clusters
        self.centroids: List[npt.NDArray[np.float64]] = []

    def fit(self, X: npt.NDArray[np.float64]) -> None:
        """
        Fits the K-Means model to the data.

        Args:
            X (npt.NDArray[np.float64]): The input data of shape (n_samples, n_features).
        """
        n_samples, _ = X.shape

        # 1. Initialize centroids by randomly selecting K points from the dataset
        random_sample_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = [X[idx] for idx in random_sample_idxs]

        # Optimization process: iterate until convergence or max_iterations
        for _ in range(self.max_iterations):
            # 2. Assign samples to the closest centroid (create clusters)
            self.clusters = self._create_clusters(X)

            # 3. Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids()

            # 4. Check for convergence (if centroids stop moving)
            if self._is_converged(centroids_old):
                break

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        """
        Predicts the cluster for each data point.

        Args:
            X (npt.NDArray[np.float64]): The input data.

        Returns:
            npt.NDArray[np.int_]: The cluster labels for each data point.
        """
        return self._get_cluster_labels(X)

    def _get_cluster_labels(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:
        """Assigns each data point to the closest centroid and returns the labels."""
        labels = np.empty(X.shape[0], dtype=int)
        for idx, sample in enumerate(X):
            distances = [self._euclidean_distance(sample, point) for point in self.centroids]
            cluster_idx = np.argmin(distances)
            labels[idx] = cluster_idx
        return labels

    def _create_clusters(self, X: npt.NDArray[np.float64]) -> Dict[int, List[npt.NDArray[np.float64]]]:
        """Assigns the data points to the closest centroids to create clusters."""
        clusters: Dict[int, List[npt.NDArray[np.float64]]] = {i: [] for i in range(self.k)}
        labels = self._get_cluster_labels(X)
        for idx, label in enumerate(labels):
            clusters[label].append(X[idx])
        return clusters

    def _get_centroids(self) -> List[npt.NDArray[np.float64]]:
        """Calculates the new centroids as the mean of the data points in each cluster."""
        n_features = self.centroids[0].shape[0] if self.centroids else 0
        new_centroids = np.zeros((self.k, n_features))
        for cluster_idx, cluster_points in self.clusters.items():
            if cluster_points:  # Avoid division by zero for empty clusters
                cluster_mean = np.mean(cluster_points, axis=0)
                new_centroids[cluster_idx] = cluster_mean
            else: # If a cluster is empty, re-use its old centroid
                new_centroids[cluster_idx] = self.centroids[cluster_idx]
        return list(new_centroids)

    def _is_converged(self, centroids_old: List[npt.NDArray[np.float64]]) -> bool:
        """Checks if the centroids have converged by comparing distances."""
        distances = [self._euclidean_distance(self.centroids[i], centroids_old[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _euclidean_distance(self, x1: npt.NDArray[np.float64], x2: npt.NDArray[np.float64]) -> float:
        """Calculates the Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2)**2))
