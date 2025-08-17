import numpy as np
from sklearn.datasets import make_moons

# ~~~~~~~~~ Activation Functions ~~~~~~~~~~~~~~~~
def relu(Z):
    """ ReLU Activation function"""
    return np.maximum(0,Z)

def sigmod(Z):
    """ Sigmoid Activation Function"""
    return 1/ (1 + np.exp(-Z))


def forward_pass(X, parameters):
    """
    Implements full forward pass for MLP.
    
    """
    pass

