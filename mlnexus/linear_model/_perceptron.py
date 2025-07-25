import numpy as np
import pandas as pd
from typing import Optional
from numpy.typing import NDArray 

class MyPerceptron:
    def __init__(self, learning_rate: float = 0.001, 
                random_state: Optional[int] = 42, 
                num_iterations: int=1000):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.epochs = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: NDArray[np.float64], y:NDArray[np.int_]):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(X.shape[0]):
                y_predict = self.activation(np.dot(self.weights, X[i]) + self.bias)
                if y_predict != y[i]:
                    # Adaline update rule
                    update =  self.learning_rate * (y[i] - y_predict)
                    self.weights = self.weights + update * X[i]
                    self.bias = self.bias + self.learning_rate * update

                    # # Perceptron rule
                    # update =  self.learning_rate * (y[i])
                    # self.weights = self.weights + update * X[i]
                    # self.bias = self.bias + self.learning_rate * update

                    
            
        print(f"Training complete!!!")

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        y_predictions = []

        for i in range(X.shape[0]):
            weighted_sum = np.dot(X[i], self.weights) + self.bias
            y_predictions.append(self._activation(weighted_sum))
        return y_predictions

    def _activation(self, weighted_sum: float):
        if weighted_sum >= 0:
            return 1
        else:
            return 0

