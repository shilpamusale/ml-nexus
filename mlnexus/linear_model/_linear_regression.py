import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score
from typing import Optional
from numpy.typing import NDArray

class MyLinearRegression:
    def __init__(self, 
                 learning_rate: float = 0.01,
                 nums_iterations: int = 1000,
                 random_state: Optional[int] = 42):
        self.learning_rate = learning_rate
        self.epochs = nums_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X: NDArray[np.float64], y:NDArray[np.float64]) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = self._activation(np.dot(X , self.weights) + self.bias)

            dw = ((1/n_samples) * np.dot(X.T, (y_pred - y)))
            db = ((1/n_samples) * np.sum(y_pred - y))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if (epoch+1)>0 and (epoch+1) % 100 == 0:
                print(f"Epochs completed - {epoch+1}")

        print(f"~~~~~ Training Completed ~~~~~~")



    def _activation(self, z: float):
        return z
    
    def predict(self, X:NDArray[np.float64]) -> NDArray[np.float64]:
        if self.weights is None or self.bias is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict function")
        return self._activation(np.dot(X, self.weights) + self.bias)



                 
        