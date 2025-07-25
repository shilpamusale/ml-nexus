import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.typing import NDArray


class MyLogisticRegression:
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 nums_iterations: int = 1000, 
                 random_state: Optional[int] = 42):
        
        self.learning_rate = learning_rate
        self.epochs = nums_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X:NDArray[np.float64], y:NDArray[np.int_]):
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 1

        for epoch in range(self.epochs):
            linear_function = np.dot(X, self.weights) + self.bias
            y_pred_proba = self._activation(linear_function)
            
            # Calculate the gradients
            dw = self.learning_rate * ((1 / n_samples) * np.dot(X.T, (y_pred_proba - y)))
            db = self.learning_rate * ((1 / n_samples) * np.sum(y_pred_proba - y))

            self.weights -= dw
            self.bias -= db
            if (epoch+1)>0 and (epoch+1) % 100 == 0:
                print(f"Epochs completed - {epoch+1}")

        print(f"~~~~~ Training Completed ~~~~~~")



    def predict(self, X:NDArray[np.float64]) -> NDArray[np.int_]:
        if self.weights is None or self.bias is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict function")
        
        linear_function = np.dot(X, self.weights) + self.bias
        y_pred_proba = self._activation(linear_function)
        y_pred_class = np.where(y_pred_proba>=0.5, 1, 0)
        return y_pred_class


    def _activation(self, z: float) -> float:
        # implement sigmoid function
        z_clipped = np.clip(z, -250, 250)
        return 1/ (1 + np.exp(-z_clipped))



