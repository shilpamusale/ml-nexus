# **ml-nexus: Machine Learning from Scratch**

An educational library of fundamental machine learning algorithms implemented from scratch using only NumPy.  
This project is designed to **deepen the understanding** of the math and logic behind each algorithm, serving as a key resource for **machine learning interview preparation**.

---

## 🚀 Why This Project?

The goal of `ml-nexus` is **not to replace** robust libraries like Scikit-learn, but to **deconstruct** them. By building these algorithms from the ground up, we gain critical insights into:

- **The Core Mechanics**  
  Understand the step-by-step logic, from weight initialization to update rules.

- **The Underlying Math**  
  Translate formulas for concepts like gradient descent and loss functions directly into working code.

- **API Design**  
  Learn how to structure code in a reusable, intuitive way, similar to professional libraries.

This repository is a **hands-on study guide** for anyone preparing for technical roles in machine learning.

---

## ✅ Implemented Algorithms

This library is a growing collection of classifiers, regressors, and tools.

### **Current Implementations**
- **Linear Models**
  - Perceptron

### **Coming Soon**
- Logistic Regression  
- Linear Regression  
- Support Vector Machines (SVM)  
- Decision Trees  

---

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ml-nexus-project.git
cd ml-nexus-project
```

### 2. Create a Virtual Environment

**For Unix/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows**:
```bash
python -m venv venv
.env\Scriptsctivate
```

### 3. Install the Package

Install the package and all required libraries using:

```bash
pip install .
```

This will install dependencies like `numpy`, `matplotlib`, and `scikit-learn`.

---

## ⚡ Quick Start: Using the Perceptron

Here’s how you can use the **Perceptron** classifier. For a more detailed walkthrough and visualization, check out the notebooks in the `/examples` directory.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from mlnexus.perceptron import Perceptron  # Assuming perceptron.py is in mlnexus/

# 1. Generate sample data
X, y = make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 2. Instantiate and train the model
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)

# 3. Make predictions
predictions = p.predict(X_test)

# 4. Evaluate the accuracy
accuracy = np.sum(y_test == predictions) / len(y_test)
print(f"Perceptron classification accuracy: {accuracy:.2f}")
```

---

## 🗂️ Project Structure

```
ml-nexus-project/
├── mlnexus/           # Core source code for the library
│   └── perceptron.py
├── examples/          # Jupyter notebooks with detailed visual walkthroughs
├── tests/             # Unit tests (work in progress)
├── pyproject.toml     # Project metadata and dependencies
└── README.md
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
