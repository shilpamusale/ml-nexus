import numpy as np
from collections import Counter
from typing import Optional, List, Tuple

class Node:
    """
    Represents a single node in the Decision Tree.
    
    An internal node has a feature and threshold for splitting the data.
    A leaf node has a value, which is the prediction.
    """
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None, left: Optional['Node'] = None, right: Optional['Node'] = None, *, value: Optional[int] = None):
        """
        Initializes a Node.
        
        Args:
            feature (Optional[int]): Index of the feature to split on.
            threshold (Optional[float]): Threshold value for the feature split.
            left (Optional['Node']): Left child node.
            right (Optional['Node']): Right child node.
            value (Optional[int]): The predicted class label (for a leaf node).
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """Checks if the node is a leaf node."""
        return self.value is not None


class MyDecisionTree:
    """
    A Decision Tree classifier implemented from scratch.
    """
    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_features: Optional[int] = None):
        """
        Initializes the Decision Tree.
        
        Args:
            min_samples_split (int): The minimum number of samples required to split an internal node.
            max_depth (int): The maximum depth of the tree.
            n_features (Optional[int]): The number of features to consider when looking for the best split.
                                         If None, all features will be considered.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Builds the decision tree from the training data.
        
        Args:
            X (np.ndarray): Training feature data of shape (n_samples, n_features).
            y (np.ndarray): Training target labels of shape (n_samples,).
        """
        # If n_features is not specified, use all features.
        # Otherwise, ensure it's not more than the available features.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grows the decision tree.
        
        Args:
            X (np.ndarray): Feature data for the current node.
            y (np.ndarray): Target labels for the current node.
            depth (int): The current depth of the node in the tree.
            
        Returns:
            Node: The root node of the constructed subtree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # 1. Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 2. Find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # 3. Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idxs: np.ndarray) -> Tuple[int, float]:
        """
        Finds the best feature and threshold to split the data on, maximizing information gain.
        
        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target labels.
            feat_idxs (np.ndarray): Indices of features to consider for splitting.
            
        Returns:
            Tuple[int, float]: A tuple containing the index of the best feature and the best threshold value.
        """
        best_gain = -1.0
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        
        return split_idx, split_thresh

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: float) -> float:
        """
        Calculates the information gain of a split using Gini Impurity.
        
        Args:
            y (np.ndarray): Parent target labels.
            X_column (np.ndarray): A single feature column.
            threshold (float): The threshold to split the feature on.
            
        Returns:
            float: The information gain.
        """
        # Parent Gini Impurity
        parent_gini = self._gini(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        # Calculate the weighted average Gini impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

        # Information gain is the reduction in impurity
        ig = parent_gini - child_gini
        return ig

    def _split(self, X_column: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits a feature column into two based on a threshold.
        
        Args:
            X_column (np.ndarray): The feature column to split.
            split_thresh (float): The threshold value.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of numpy arrays containing the indices for the left and right splits.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y: np.ndarray) -> float:
        """
        Calculates the Gini impurity for a set of labels.
        
        Args:
            y (np.ndarray): An array of labels.
            
        Returns:
            float: The Gini impurity.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1.0 - np.sum([p**2 for p in ps if p > 0])

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Finds the most common label in an array.
        
        Args:
            y (np.ndarray): An array of labels.
        
        Returns:
            int: The most frequent label.
        """
        counter = Counter(y)
        if not counter:
            # This case should ideally not be hit with proper checks, but as a fallback:
            return -1 
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions for a set of samples.
        
        Args:
            X (np.ndarray): The feature data to predict on.
            
        Returns:
            np.ndarray: An array of predicted labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Recursively traverses the tree to find the prediction for a single sample.
        
        Args:
            x (np.ndarray): A single sample's features.
            node (Node): The current node in the tree.
            
        Returns:
            int: The predicted label from a leaf node.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def print_tree(self, feature_names: Optional[List[str]] = None):
        """
        Prints a text representation of the decision tree.
        
        Args:
            feature_names (Optional[List[str]]): A list of names for the features.
        """
        self._print_tree(self.root, feature_names=feature_names)

    def _print_tree(self, node: Optional[Node], indent: str = " ", feature_names: Optional[List[str]] = None):
        """Helper function to recursively print the tree structure."""
        if not node:
            return
            
        if node.is_leaf_node():
            print(f"{indent}Predict: {node.value}")
            return
        
        feature_name = f"Feature {node.feature}"
        if feature_names and node.feature < len(feature_names):
            feature_name = feature_names[node.feature]

        print(f"{indent}If {feature_name} <= {node.threshold:.2f}:")
        self._print_tree(node.left, indent + "  ", feature_names)
        
        print(f"{indent}Else ({feature_name} > {node.threshold:.2f}):")
        self._print_tree(node.right, indent + "  ", feature_names)