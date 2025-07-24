import numpy as np
import pandas as pd
from typing import Optional
from numpy.typing import NDArray 

class Perceptron:
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
            y_predictions.append(self.activation(weighted_sum))
        return y_predictions

    def activation(self, weighted_sum: float):
        if weighted_sum >= 0:
            return 1
        else:
            return 0

if __name__=="__main__":
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=0.5, random_state=2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
    percentron = Perceptron()
    percentron.fit(X_train,y_train)
    predictions = percentron.predict(X_test)

    print(f" Accuracy = {accuracy_score(y_test, predictions)}")

    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig("images/perceptron.jpg")