import numpy as np

from typing import Dict
from numpy.typing import NDArray


class MyKNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
    def fit(self, X_train: NDArray[np.float64],
                y_train: NDArray[np.float64]):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test: NDArray[np.float64]
                ) -> NDArray[np.int_]:
        predictions = [self._predict_point(test) for test in X_test]
        return np.array(predictions)
    
    def evaluate(self, X_test: NDArray[np.float64], 
                 y_test:NDArray[np.int_], 
                 predictions: NDArray[np.int_], 
                 positive_class_label: int, 
                 negative_class_label: int ) -> Dict:
        
        metrics = {}
        metrics["accuracy"] = np.sum(y_test == predictions) / len(y_test)
        true_positive = np.sum((y_test  == positive_class_label) & (predictions == positive_class_label))
        false_positive = np.sum((y_test == negative_class_label) & (predictions == positive_class_label))
        false_negative = np.sum((y_test == positive_class_label) & (predictions == negative_class_label))


        if (true_positive + false_positive) > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        
        if (true_positive + false_negative) > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0

        if (precision + recall) > 0:
            f1_score = 2 * ((precision * recall) / (precision + recall))
        else:
            f1_score

        
        metrics["precision"] = precision
        metrics["true_positive"] = true_positive
        metrics["false_positive"] = false_positive
        metrics["false_negative"] = false_negative
        metrics["recall"] = recall
        metrics["f1_score"] = f1_score

        return metrics
    
    def _predict_point(self, test_record:NDArray[np.float64]):

        # Step 1: Get the distances of the test_record with all records in the training dataset
        distances = np.sqrt(np.sum((self.X_train - test_record)**2, axis = 1))

        # Step 2: Get the indices of the closest k elements from the training dataset. 
        knn_indices = np.argsort(distances)[:self.k]

        # Step 3: Get the class label of the closet k elements. 
        knn_labels = self.y_train[knn_indices]

        # Step 4: Return the most common label 
        knn_preditcion = np.argmax(np.bincount(knn_labels))

        return knn_preditcion

class MyKNNRegression:
    def __init__(self, k: int = 5):
        self.k = k
    
    def fit(self, X_train: NDArray[np.float64], y_train: NDArray[np.float64]) -> None:
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        
        predictions = [self._predict_point(test_record) for test_record in X_test]

        return np.array(predictions)
    
    def _predict_point(self, test_record: NDArray[np.float64]) -> float:

        # step 1 : Calculate distances
        distances = np.sqrt(np.sum((self.X_train - test_record)**2, axis=1))

        # Step 2: Get the indices of the closest k elements from the training dataset. 
        knn_indices = np.argsort(distances)[:self.k]

        # Step 3: Get the average of the closest k elements target value
        knn_target = np.mean(self.y_train[knn_indices]) 

        return knn_target
    
    def evaluate(self, 
                 y_test: NDArray[np.float64],
                 predictions: NDArray[np.float64]
                 ) -> Dict:

        if len(y_test) == 0:
            raise ValueError("y_test cannot be empty")
        
        metrics = {}


        test_records = len(y_test)
        mae = 1/ test_records * (np.sum(np.absolute(y_test - predictions)))
        rsme = np.sqrt((1/test_records) * (np.sum((y_test - predictions)**2)))
        r2squared = 1 - (np.sum((y_test - predictions) ** 2)) / (np.sum((y_test - np.mean(y_test)) ** 2))
        metrics["Mean_Absolute_Error"] = mae
        metrics["Root_Mean_Square_Error"] = rsme
        metrics["r2_squared"] = r2squared

        return metrics
        

        
        






