# Machine Learning Algorithms From Scratch 🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

> Understanding machine learning by building it from the ground up

## 🎯 About This Project

This repository contains **pure Python implementations** of popular machine learning algorithms built from scratch using only NumPy for numerical operations. Each algorithm is implemented with clear, readable code and detailed explanations to help you understand the mathematics and logic behind machine learning.

**Who is this for?**
- Students learning machine learning fundamentals
- Data scientists wanting to deepen their understanding
- Anyone preparing for technical interviews
- Developers curious about what happens "under the hood"

## 🌟 Why Learn Algorithms From Scratch?

- **Deep Understanding**: Go beyond black-box usage of libraries
- **Debug Better**: Know what's happening when things go wrong
- **Customize Smartly**: Modify algorithms for your specific needs
- **Interview Ready**: Confidently explain and implement core concepts
- **Appreciate Libraries**: Understand what scikit-learn does for you

## 📚 Algorithms Covered

### Supervised Learning

#### Regression
- Linear Regression (Ordinary Least Squares)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Polynomial Regression
- Logistic Regression (Binary Classification)

#### Classification
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian, Multinomial)
- Decision Trees (CART)
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting Machines

#### Neural Networks
- Perceptron
- Multi-Layer Perceptron (MLP)
- Backpropagation from Scratch

### Unsupervised Learning

#### Clustering
- K-Means Clustering
- DBSCAN
- Hierarchical Clustering
- Gaussian Mixture Models (GMM)

#### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Ensemble Methods
- Bagging
- Boosting (AdaBoost)
- Stacking

### Optimization & Loss Functions
- Gradient Descent (Batch, Stochastic, Mini-batch)
- Cost Functions (MSE, Cross-Entropy, Hinge Loss)

## 🗂️ Repository Structure

ml-algorithms-from-scratch/  
├── [algorithms/](algorithms/)  
│   ├── [supervised/](algorithms/supervised/)  
│   │   ├── [regression/](algorithms/supervised/regression/)  
│   │   │   ├── [linear_regression.py](algorithms/supervised/regression/linear_regression.py)  
│   │   │   ├── [ridge_regression.py](algorithms/supervised/regression/ridge_regression.py)  
│   │   │   ├── [lasso_regression.py](algorithms/supervised/regression/lasso_regression.py)  
│   │   │   ├── [polynomial_regression.py](algorithms/supervised/regression/polynomial_regression.py)  
│   │   │   └── [logistic_regression.py](algorithms/supervised/regression/logistic_regression.py)  
│   │   ├── [classification/](algorithms/supervised/classification/)  
│   │   │   ├── [knn.py](algorithms/supervised/classification/knn.py)  
│   │   │   ├── [naive_bayes.py](algorithms/supervised/classification/naive_bayes.py)  
│   │   │   ├── [decision_tree.py](algorithms/supervised/classification/decision_tree.py)  
│   │   │   ├── [svm.py](algorithms/supervised/classification/svm.py)  
│   │   │   ├── [random_forest.py](algorithms/supervised/classification/random_forest.py)  
│   │   │   └── [gradient_boosting.py](algorithms/supervised/classification/gradient_boosting.py)  
│   │   └── [neural_networks/](algorithms/supervised/neural_networks/)  
│   │       ├── [perceptron.py](algorithms/supervised/neural_networks/perceptron.py)  
│   │       ├── [mlp.py](algorithms/supervised/neural_networks/mlp.py)  
│   │       └── [backpropagation.py](algorithms/supervised/neural_networks/backpropagation.py)  
│   ├── [unsupervised/](algorithms/unsupervised/)  
│   │   ├── [clustering/](algorithms/unsupervised/clustering/)  
│   │   │   ├── [kmeans.py](algorithms/unsupervised/clustering/kmeans.py)  
│   │   │   ├── [dbscan.py](algorithms/unsupervised/clustering/dbscan.py)  
│   │   │   ├── [hierarchical.py](algorithms/unsupervised/clustering/hierarchical.py)  
│   │   │   └── [gmm.py](algorithms/unsupervised/clustering/gmm.py)  
│   │   └── [dimensionality_reduction/](algorithms/unsupervised/dimensionality_reduction/)  
│   │       ├── [pca.py](algorithms/unsupervised/dimensionality_reduction/pca.py)  
│   │       ├── [lda.py](algorithms/unsupervised/dimensionality_reduction/lda.py)  
│   │       └── [tsne.py](algorithms/unsupervised/dimensionality_reduction/tsne.py)  
│   ├── [ensemble/](algorithms/ensemble/)  
│   │   ├── [bagging.py](algorithms/ensemble/bagging.py)  
│   │   ├── [adaboost.py](algorithms/ensemble/adaboost.py)  
│   │   ├── [stacking.py](algorithms/ensemble/stacking.py)  
│   │   ├── [random_forest.py](algorithms/ensemble/random_forest.py)  
│   │   └── [gradient_boosting.py](algorithms/ensemble/gradient_boosting.py)  
│   └── [utils/](algorithms/utils/)  
│       ├── [metrics.py](algorithms/utils/metrics.py)  
│       ├── [preprocessing.py](algorithms/utils/preprocessing.py)  
│       ├── [visualization.py](algorithms/utils/visualization.py)  
│       └── [optimizers.py](algorithms/utils/optimizers.py)  
│ 
├── [tests/](tests/)  
│   ├── [test_linear_regression.py](tests/test_linear_regression.py)  
│   ├── [test_logistic_regression.py](tests/test_logistic_regression.py)  
│   ├── [test_ridge_regression.py](tests/test_ridge_regression.py)  
│   ├── [test_polynomial_regression.py](tests/test_polynomial_regression.py)  
│   ├── [test_knn.py](tests/test_knn.py)  
│   ├── [test_decision_tree.py](tests/test_decision_tree.py)  
│   ├── [test_kmeans.py](tests/test_kmeans.py)  
│   ├── [test_pca.py](tests/test_pca.py)  
│   └── ...  
├── [examples/](examples/)  
│   ├── [regression_example.py](examples/regression_example.py)  
│   ├── [classification_example.py](examples/classification_example.py)  
│   ├── [clustering_example.py](examples/clustering_example.py)  
│   └── [neural_network_example.py](examples/neural_network_example.py)  
├── [docs/](docs/)  
│   ├── [mathematical_foundations.md](docs/mathematical_foundations.md)  
│   ├── [algorithm_comparison.md](docs/algorithm_comparison.md)  
│   ├── [contributing.md](docs/contributing.md)  
│   └── [api_reference.md](docs/api_reference.md)  
├── [requirements.txt](requirements.txt)  
├── [LICENSE](LICENSE)  
└── [README.md](README.md)

## 🚀 Getting Started

### Prerequisites

Python 3.8 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rohanmistry231/Machine-Learning-Algorithms-From-Scratch.git
cd Machine-Learning-Algorithms-From-Scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Example

```python
from src.supervised.regression import LinearRegression
import numpy as np

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7]]))
print(predictions)  # [12, 14]
```

## 📖 Learning Path

**Recommended order for beginners:**

1. **Start with Linear Regression** - Simplest algorithm, introduces optimization
2. **Move to Logistic Regression** - Introduces classification and sigmoid function
3. **Try K-Nearest Neighbors** - Non-parametric, intuitive approach
4. **Explore Naive Bayes** - Introduces probability-based learning
5. **Tackle Decision Trees** - Foundation for ensemble methods
6. **Build up to Neural Networks** - Combine previous concepts

Each algorithm includes:
- ✅ Clean, commented implementation
- ✅ Mathematical explanation
- ✅ Step-by-step tutorial notebook
- ✅ Comparison with scikit-learn
- ✅ Complexity analysis
- ✅ Practical examples

## 🧮 Mathematical Foundations

Each algorithm implementation includes:
- **Theory**: Mathematical formulation and intuition
- **Derivations**: Step-by-step mathematical derivations
- **Pseudocode**: Algorithm in plain language
- **Implementation**: Python code with detailed comments
- **Visualization**: Plots showing how the algorithm works

## 🧪 Testing

All algorithms are tested against scikit-learn implementations to ensure correctness:

```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Adding new algorithms
- Improving documentation
- Fixing bugs
- Adding examples or tutorials

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 Code Style

- Clean, readable code following PEP 8
- Comprehensive docstrings (NumPy style)
- Type hints where applicable
- Meaningful variable names
- Comments explaining mathematical concepts

## 🎓 Educational Resources

- **Notebooks**: Interactive Jupyter notebooks for each algorithm
- **Documentation**: In-depth mathematical explanations
- **Visualizations**: Plots and animations showing algorithm behavior
- **Comparisons**: Performance comparisons with scikit-learn

## 📊 Performance Note

These implementations prioritize **clarity and education** over performance. For production use, always use optimized libraries like scikit-learn, TensorFlow, or PyTorch. However, understanding these fundamentals will make you a better user of those libraries.

## 🔗 References & Further Reading

- [Pattern Recognition and Machine Learning - Bishop](https://www.springer.com/gp/book/9780387310732)
- [The Elements of Statistical Learning - Hastie, Tibshirani, Friedman](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Machine Learning - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Show Your Support

If you find this repository helpful, please consider giving it a star! It helps others discover this resource.

## 📬 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Reach out via [your contact method]

---

**Happy Learning! 🚀** Remember: The best way to understand machine learning is to build it yourself.