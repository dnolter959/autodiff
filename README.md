![example workflow](https://code.harvard.edu/CS107/team14/actions/workflows/test.yml/badge.svg)
![second badge](https://code.harvard.edu/CS107/team14/actions/workflows/code_coverage.yml/badge.svg)

**Code Coverage Report:** https://code.harvard.edu/pages/CS107/team14/

# team14-autodiff

## Overiew

`team14-autodiff` is a package to perform automatic differentiation of scalar or vector functions with scalar or vector inputs. The package can perform "forward" or "reverse" mode automatic differentiation. Forward and Reverse mode will return the same results, but each mode has computational advantages depending on the nature of one's task. 

## Getting started

### Installation

We will provide separate (but similar) installation instructions for 1) typical users and 2) fellow developers.

If a user (typical or developer) wishes to install our package in a virtual environment, they may begin by running the following commands. Within a virtual environment, a user must install package dependencies (as specified below: numpy, pytest, toposort, pytest-cov); but this step is not necessary if these dependencies are already installed within the user's local environment. 

```sh
# Create and activate virtual environment
mkdir test_autodiff
cd test_autodiff
python3 -m venv .venv
source .venv/bin/activate
```

#### 1) Installation for typical package users
```sh
# Install package and necessary dependencies
python -m pip install -i https://test.pypi.org/simple/ team14-autodiff
python -m pip install numpy pytest pytest-cov toposort
```

#### 2) Installation for developers
```sh
# Clone repo
git clone git@code.harvard.edu:CS107/team14.git
cd team14

# Install necessary dependencies
python -m pip install numpy pytest pytest-cov toposort

# set PYTHONPATH
export PYTHONPATH="$(pwd -P)/src":${PYTHONPATH}

# Run tests
cd tests && ./run_tests.sh pytest -v && cd ..

# Run code coverage
./tests/check_coverage.sh pytest

# Look at Sphinx Docs
open docs/sphinx/build/html/index.html 

# Run test script
python3 driver_script.py
```

### How to Use

Below we show typical uses for our package. It can be used to calculate derivaties of scarlar function with scalar inputs, and also jacobians, which are useful in the case of multi-dimensional inputs and output functions.

**Notes and Reminders**

 - Derivatives and jacobians can be calculated using either forward or reverse mode; the default option is forward; if you would like to specify reverse mode include the option `mode="reverse"` in the `get_derivative` or  `get_jacobian` functions.
 - In the event that you would like to take a derivative of a function with multiple inputs, you must specify the variable with which you wish to take the derivative with respect to. In particular, you make this specification using a generalized `seed_vector`. See example in Case 2 below. 

More detailed documentation is available [here](https://code.harvard.edu/CS107/team14/blob/a317cdad86199a2ba208187f3bd9e13f92d4f555/docs/documentation.md). 

**Imports**
```python
import numpy as np
from autodiff.auto_diff import AutoDiff
from autodiff.utils.auto_diff_math import *
```

**Case 0: Evaluating a function at an input**
```python
f = lambda x: x**2 + 2*x
ad = AutoDiff(f)
x = 2

value = ad.get_value(x) # 8
```

**Case 1: $\mathbb{R} \rightarrow \mathbb{R}$**
```python
f = lambda x: x**2 + 2*x
ad = AutoDiff(f)
x = 2

derivative = ad.get_derivative(x) # 6 # (Defualt mode is forward)
derivative = ad.get_derivative(x, mode="reverse") # 6
```

**Case 2: $\mathbb{R}^n \rightarrow \mathbb{R}$ ($n \gt 1$)**
```python
f = lambda x: x[0]**2 + 2*x[1]
ad = AutoDiff(f)
x = [2, 3] # Order must match the indexing of x in f definition

jacobian = ad.get_jacobian(x) # [[4, 2]]

# Take derivative with respect to x[0]
seed_vector = np.array([1, 0])
derivative = ad.get_derivative(x, seed_vector) # 4

# Take derivative with respect to x[1]
seed_vector = np.array([0, 1])
derivative = ad.get_derivative(x, seed_vector) # 2
```

**Case 3: $\mathbb{R} \rightarrow \mathbb{R}^m$ ($m \gt 1$)**
```python
f1 = lambda x: x**2 + 2*x
f2 = lambda x: sin(x)
ad = AutoDiff([f1, f2])
x = 2

jacobian = ad.get_jacobian(x) # [[6], [cos(2)]]
derivative = ad.get_derivative(x) # [[6], [cos(2)]]
``` 

**Case 4: $\mathbb{R}^n \rightarrow \mathbb{R}^m$ ($n, m \gt 1$)**
```python
f1 = lambda x: x[0]**2 + 2*x[1]
f2 = lambda x: sin(x[0]) + 3*x[1]
ad = AutoDiff([f1, f2])
x = [2, 5] # Ordering specified by index of variables in f1, f2

jacobian = ad.get_jacobian(x) # [[4, 2], [cos(2), 3]]

seed_vector = np.array([1, 0])
derivative = ad.get_derivative(x, seed_vector) # [[4], [cos(2)]]

seed_vector = np.array([-2, 1])
derivative = ad.get_derivative(x, seed_vector) # [[-6], [-2cos(2) + 3]]
```

## Documentation
Complete, function-level documentation is available via Sphinx. To access it, follow these steps (also outlined above in developer-level installation.)

```sh
git clone git@code.harvard.edu:CS107/team14.git
open team14/docs/sphinx/build/html/index.html 
```

## Broader Impact

Automatic differentiation has wide usage in science and engineering due to its efficient calculation of gradient to machine precision. A notable example is its application in neural networks, serving as the foundation of back propagation. Such usage has been adopted by various fields. For example, automatic differentiation-powered deep convolutional neural network has been used in novel medical research to identify diseases such as Glaucoma and thyroid scintigram. These methods, along with other innovative machine learning techniques introduced to the medical and life sciences fields have profound impact on public health and well-being.

However, any algorithms that facilitate machine learning applications also augment the negative impact associated with these applications. To use the filed of medicine as an example again, while machine learning has been shown to be helpful, one should be extremely cautious about the role of algorithms in, say, medical diagnoses. Complete reliance on machine learning or artificial intelligence, regardless of how powerful they seem thanks to clever algorithms like automatic differentiation, can be extremely dangerous. 

## Software Inclusivity

The future development of this software package for any additional features, such as higher order derivatives, will aim to be open and inclusive. The current content of this software package is contributed equally by all members of our team with diverse backgrounds and experiences, and reflects equal appreciation and respect towards the diverse knowledge and expertise associated with these backgrounds. We hope that any future development continue to uphold these values while being open to the broader community. Specifically, we welcome everyone's contribution regardless of their background and the review of contributions, conducted by all members of the team, will be based on nothing else but the quality of the content. 
