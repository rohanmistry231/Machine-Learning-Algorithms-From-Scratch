# Contributing to ML Algorithms from Scratch

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Types of Contributions](#types-of-contributions)
- [Development Process](#development-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We expect all contributors to:

- Be respectful and inclusive
- Welcome newcomers and help them get oriented
- Focus on what is best for the community
- Show empathy towards other community members
- Be patient and understanding

### Unacceptable Behavior

The following behaviors are unacceptable:

- Harassment or discrimination of any kind
- Insulting, demeaning, or unwelcoming comments
- Personal attacks
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

---

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ml-algorithms-from-scratch.git
cd ml-algorithms-from-scratch

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/ml-algorithms-from-scratch.git
```

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black flake8 mypy
```

### Create a Branch

```bash
# Keep main updated
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/bug-description
```

---

## Types of Contributions

### 1. Adding New Algorithms

#### Algorithm Selection Criteria
- Algorithm should be well-established and documented
- Not already implemented in the library
- Should have clear educational value
- Clear use cases and practical applications

#### Implementation Checklist
- [ ] Implement the algorithm from scratch (no library calls except NumPy)
- [ ] Add comprehensive docstrings with mathematical formulations
- [ ] Include parameter descriptions
- [ ] Add working example in `if __name__ == "__main__"` block
- [ ] Test against sklearn equivalent (where available)
- [ ] Add to appropriate folder (supervised/unsupervised/ensemble)

### 2. Improving Existing Code

- Performance optimizations
- Bug fixes
- Better documentation
- Cleaner code refactoring
- Type hints improvements

### 3. Adding Tests

- Unit tests for algorithms
- Edge case testing
- Comparison with sklearn implementations
- Numerical stability tests

### 4. Documentation

- Detailed algorithm explanations
- Usage examples
- Mathematical formulations
- Comparison with other algorithms
- Tutorial notebooks

### 5. Bug Reports

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details

---

## Development Process

### Step 1: Understand the Algorithm

- Research the mathematical foundations
- Read relevant papers or textbooks
- Study existing implementations
- Understand use cases

### Step 2: Implement from Scratch

```python
# Template for new algorithm
"""
Algorithm Name Implementation from Scratch

Mathematical Foundation:
[Include mathematical formulations]

Parameters
----------
param1 : type
    Description
    
Attributes
----------
attribute1 : type
    Description
"""

import numpy as np
from typing import Optional

class AlgorithmName:
    """Class docstring with full description."""
    
    def __init__(self, param1: float = 0.01):
        """Initialize algorithm."""
        self.param1 = param1
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AlgorithmName':
        """Fit algorithm to data."""
        # Implementation
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Implementation
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate performance metric."""
        return score
    
    def get_params(self) -> dict:
        """Return algorithm parameters."""
        return {}

if __name__ == "__main__":
    # Working example
    pass
```

### Step 3: Test Thoroughly

```python
# Test against sklearn
from sklearn.algorithm import SklearnVersion

# Custom implementation
from algorithms.folder.algorithm import CustomImplementation

# Compare results
custom = CustomImplementation()
custom.fit(X_train, y_train)

sklearn_model = SklearnVersion()
sklearn_model.fit(X_train, y_train)

# Verify similar performance
assert np.allclose(custom.predict(X_test), sklearn_model.predict(X_test), atol=1e-5)
```

### Step 4: Document Thoroughly

- Add docstrings with mathematical notation
- Include references to papers/textbooks
- Explain parameters and return values
- Add usage examples

### Step 5: Update Project Documentation

- Add to relevant sections in README
- Update algorithm lists
- Add to algorithm_comparison.md if needed
- Update folder structure if needed

---

## Code Standards

### Style Guidelines

We follow PEP 8 with these additional standards:

#### Naming Conventions

```python
# Classes: PascalCase
class LinearRegression:
    pass

# Functions/Methods: snake_case
def fit_model(X, y):
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_LEARNING_RATE = 0.01

# Private/Internal: _leading_underscore
def _internal_helper(x):
    pass
```

#### Type Hints

```python
from typing import Optional, List, Tuple, Union
import numpy as np

def process_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process data with type hints."""
    return X, y
```

#### Docstrings (NumPy Style)

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassName':
    """
    Short description.
    
    Longer description explaining the method in detail.
    Can span multiple lines.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training features
    y : np.ndarray of shape (n_samples,)
        Target values
        
    Returns
    -------
    self : ClassName
        Fitted estimator
        
    Examples
    --------
    >>> model = ClassName()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    # Implementation
    return self
```

#### Comments

```python
# Use comments to explain WHY, not WHAT
# WHAT: Code should be self-explanatory
# WHY: Algorithm choice, edge cases, etc.

# Good: Explains algorithm insight
# Using L-BFGS for better convergence on non-convex problems
optimizer = LBFGS()

# Bad: Obvious from code
# Create an array
X = np.array([1, 2, 3])
```

### Code Organization

```python
# 1. Imports
import numpy as np
from typing import Optional

# 2. Constants
DEFAULT_LEARNING_RATE = 0.01

# 3. Class/Function definitions
class Algorithm:
    pass

# 4. Main block
if __name__ == "__main__":
    pass
```

### Formatting

Use `black` for automatic formatting:

```bash
black algorithms/
```

Check style with `flake8`:

```bash
flake8 algorithms/ --max-line-length=100
```

---

## Testing Guidelines

### Unit Test Template

```python
# tests/test_algorithm.py
import numpy as np
import pytest
from algorithms.supervised.classification import Algorithm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class TestAlgorithm:
    """Test suite for Algorithm."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_fit(self, sample_data):
        """Test that algorithm fits without error."""
        X_train, X_test, y_train, y_test = sample_data
        model = Algorithm()
        model.fit(X_train, y_train)
        assert model is not None
    
    def test_predict(self, sample_data):
        """Test predictions have correct shape."""
        X_train, X_test, y_train, y_test = sample_data
        model = Algorithm()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_score(self, sample_data):
        """Test score calculation."""
        X_train, X_test, y_train, y_test = sample_data
        model = Algorithm()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert 0 <= score <= 1
    
    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        model = Algorithm()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2, 3])  # Wrong length
        
        with pytest.raises(ValueError):
            model.fit(X, y)
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=algorithms tests/

# Run specific test
pytest tests/test_algorithm.py::TestAlgorithm::test_fit
```

---

## Documentation

### Adding Algorithm Documentation

1. **Docstrings:** In-code documentation with mathematical formulations
2. **README.md:** Add to algorithm list and mark as completed
3. **mathematical_foundations.md:** Add mathematical formulations
4. **algorithm_comparison.md:** Add comparison with similar algorithms
5. **Tutorials:** Create Jupyter notebook if complex algorithm

### Documentation Checklist

- [ ] Clear docstrings with mathematical notation
- [ ] Parameter descriptions
- [ ] Return value descriptions
- [ ] Examples in docstring
- [ ] References to papers/textbooks
- [ ] Working example in main block
- [ ] Updated README
- [ ] Added to algorithm comparison

---

## Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Verify code quality**
   ```bash
   black algorithms/
   flake8 algorithms/
   pytest tests/
   ```

3. **Update documentation**
   - README.md
   - Relevant .md files
   - Docstrings

### Creating Pull Request

1. **Push to your fork**
   ```bash
   git push origin your-feature-branch
   ```

2. **Create PR on GitHub**
   - Clear title: "Add XYZ Algorithm" or "Fix issue with ABC"
   - Description: What changes, why, related issues
   - Reference issues: "Closes #123"

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] New algorithm
   - [ ] Bug fix
   - [ ] Documentation
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Added tests
   - [ ] Tests pass locally
   - [ ] Tested against sklearn equivalent
   
   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated README
   - [ ] Added examples
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] No new warnings
   - [ ] Tests added/updated
   - [ ] Documentation updated
   ```

### PR Review

- Maintainers will review your code
- May request changes for:
  - Code quality
  - Performance
  - Documentation
  - Test coverage
- Respond to feedback promptly
- After approval, PR will be merged

---

## Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
Clear and concise description of the bug

## Steps to Reproduce
1. Load data
2. Run algorithm
3. Error occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Message/Traceback
[Paste full error]

## Environment
- Python version: 3.8+
- OS: Windows/Mac/Linux
- Library version: [version]

## Minimal Example
```python
# Code that reproduces the issue
```

## Additional Context
Any other relevant information
```

---

## Feature Requests

When requesting a feature:

- Clear description of the feature
- Use case and motivation
- Why it's valuable for the library
- References to papers/implementations

---

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Recognized in release notes
- Thanked in documentation

---

## Questions?

- Open an Issue for questions
- Check existing issues/discussions first
- Join community discussions

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to making ML education better! 🚀