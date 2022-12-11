## Introduction

In its simplest form, we define the derivative of a function as its rate of change. We should familiarize ourselves with an example of a function, the notation for a derivative, and the graphical representation of rate of change.

Say we are incredibly lucky in our investments and our return, $f(x)$, is modeled by the function $f(x) = x^2$. Here, $x$ can represent the dollar amount we invest. We would like to measure the rate at which our return, $f(x)$, changes with respect to a change in our investment dollar amount, $x$. The notation for such is represented mathematically as:

$$\frac{df}{dx} =lim_{\Delta x\to 0} \frac{\Delta f}{\Delta x}$$

For example, we may model the relationship between investment ($x$) and returns ($f(x)$) using a function $f:\mathbb{R}\to \mathbb{R}$, by $f(x) = x^2$. 

The derivative is modeled as $2\cdot x$ and we can interpret it as the instantaneous rate of change - slope - as seen in the illustration below. 

![](images/Figure1.png)

While the example above serves as a toy example to familiarize ourselves with the topic of differentiation and its graphical interpretation, the power of the derivative is not to be understated. Its origins date back to Isaac Newton and applications in physics and movement; however, it has since grown with applications in various different branches such as statistics, biology, finance, computer science, and many more fields.

There are three popular methods to calculate derivatives:

1) **Numerical**

2) **Symbolic**

3) **Automatic**

Numerical differentiation is the most basic and general introduction to calculating derivatives. In numerical differentiation we rely on the definition of the derivative, where we measure the amount of change in our function with a very small change in our input (x+h).

$$\frac{\delta{f(x)}}{\delta x} = lim_{h\to 0} \frac{f(x + h) - f(x)}{h}$$

However, numerical differentiation can have issues with round off errors that lead to not achieving machine precision and can struggle with computational time when many dependent variables exist. 

Symbolic differentiation attempts to manipulate formulas to create new formulas rather than performing numerical calculations. In doing so, we can in essence memorize derivatives of functions. However, symbolic interpretation is challenging to implement in computer programs and can be inefficient coding. 

Automatic differentiation focuses on certain core elements: the chain rule, elementary functions and, to a lesser extent, dual numbers. The benefits of automatic differentiation are that it does not suffer from the same round off errors that numerical differentiation is susceptible to and does not suffer from the overly expensive, inefficient methods of symbolic differentiation. For these reasons, automatic differentiation is ubiquitous in tasks requiring quick differentiation, such as optimization in machine learning.


## Background

Automatic differentiation builds off of two fundamental and relatively easy to understand concepts: elementary functions and the chain rule. 

### Elementary Functions

First, we can begin by providing an example of identifying elementary functions within a function. Consider the function: 

$f(x_{1}, x_{2}) = \exp{ (\sin{x_{1}^{2} + x_{2}^{2}}) + 2 * \cos{\frac{x_1}{x_2}}}$

In the function above we can identify several functions that would be considered elementary functions: multiplication, division, sin(), cos(), exponentiation, powers. Automatic differentiation breaks about functions such as $f(x)$ into the components of its elementary functions to act on intermediate steps in order to solve for its derivative. A more comprehensive list of elementary functions is included below: 

<br> 

| Category| Elementary Functions |
| --- | --- |
| Arithmetic | multiplication, addition, subtraction, division |
| Powers and Roots | $x^{2}$, $y^{1/2}$ |
| Trigonometric | sine, cosine, tangent, secant, cosecant, cotangent |
| Logarithmic | $\log(x)$ |
| Exponential | $\exp(x)$ |

### Chain Rule 

Utilizing the above elementary functions, automatic differentiation applies the ever important chain rule to the elementary functions in order to solve the derivative of more complex functions. As a quick recap of the chain rule, let us define the following function: 

$f(x) = \exp{4x}$

We can replace $4x$ with $u(x)$. This will allow us to do the following differentiation to get our desired derivative of $f(x)$ with respect to $x$. 

$\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx} = exp(u)\cdot 4 = 4\cdot exp(4x)$

### Computational Graph 

A computational graph allows us to see the ordered sequence of elementary functions, how we break down a more complex function from inside to outside (in forward mode), and how we can calculate intermediate steps to arrive at our final derivative result. 

In the computational graph below, we can see that we begin with the inputs to the function, independent variables denoted by subscripts -1 and 0 (these generally take values < 1). Additionally, we build on these with intermediate variables from $v_0, v_1, ...$. The intermediate variables parallel the elementary functions applied at each step until we arrive at the full complex model from the inside out (again in forward mode). We can follow the computational graph's arrows to see how the elementary functions are applied until we reach our desired result of differentiation.

Let us examine the utility of a computational graph with a complex function such as: 

$f(x_1, x_2) = [sin(\frac{x_1}{x_2}) + e^{x_2} ] \cdot [ \frac{x1}{x2} - e^{x_2}]$ 

We can see that elementary functions we will need are exp(), sin(), addition, subtraction, multiplication, and division. Additionally, we will need to create intermediate steps that build on the independent variables $x_1$ and $x_2$ in order to create all parts of the complex model. By following the arrows of the graph, we can see how we can begin at the independent variables and arrive back at the full complex function f(x). 

![](images/Figure2.png)

### Forward Mode and Evaluation Trace 

The evaluation trace allows us to utilize the components of our computational graph to aid us in solving the function value at a specific point and the partial derivatives with respect to each independent variable. The latter is done by utilizing seed vectors, p, which indicate the input variable to calculate the partial derivative of for the function. We will require one pass for each of the independent variables $(x_1, x_2)$. In the first pass, shown below, we set the seed vector $p = [1\;0]$.

| Forward Primal Trace           | Forward Tangent Trace ($p = [1 \; 0 ]$)                     |
| ---                            | ---                                                         |
| $v_{-1} = x_1 = 1.5$           | $D_pv_{-1} = 1$                                             |
| $v_0 = x_2 = 0.5$              | $D_pv_{0} = 0$                                              |
| $v_1 = \frac{v_{-1}}{v_0} = 3$ | $D_pv_1 = \frac{(v_0 D_pv_{-1} - v_{-1}D_pv_0)}{v_o^2} = 2$ |
| $v_2 = sin(v_1) = 0.141$       | $D_pv_2 = cos(v_1) \cdot D_pv_1 = -1.98$                    |
| $v_3 = exp(v_0) = 1.649$       | $D_pv_3 = v_3 \cdot D_pv_0 = 0$                             |
| $v_4 = v_1 - v_3 = 1.351$      | $D_pv_4 = D_pv_1 - D_pv_3 = 2$                              |
| $v_5 = v_2 + v_3 = 1.79$       | $D_pv_5 = D_pv_2 + D_pv_3 = 1.98$                           |
| $v_6 = v_5 \cdot v_4 = 2.418$  | $D_pv_5 \cdot v_4 - D_pv_4 \cdot v_5 = .905$               |


We point out that the left column gives us the result of our function $f(x_1 = 1.5, x_2 = 0.5)$. Meanwhile, the right column gives us the results of the partial derivative with respect to $x_1$. We would require another pass with $p =[0 \; 1]$ in order to solve for the partial derivative with respect to $x_2$.

### Reverse Mode

Another way to perform automatic differentiation is through the reverse mode. Unlike the forward mode, reverse mode does not need to perform one pass for each independent variables. Intead, to calculate a function's gradient, the reverse mode algorithm only requires two passes: forward and reverse pass. During the forward pass, reverse mode evaluates the primal trace, the values of each elementary operations and functions, and the sensitivity, the partial derivative between a parent and a child node. The algorithm also builds and stores the computational graph in the forward pass. During the reverse pass, reverse mode computes the adjoints, which are the partial derivatives with chain-rule incorporated. For node $v_i$, this is achieved by summing the products of each of its children node's adjoint $\bar{v}_j$ and the partial derivatives $\partial v_j/\partial v_i$, where $v_j$ is a child of $v_i$. 

For the example above, the reversre mode will work as follows

| Forward Pass (Top -> Bottom)                                                                                                 | Reverse Pass (Bottom -> Top)                                                                                                 |
| ---                                                                                                                          | ---                                                                                                                          |
| $v_{-1} = x_1 = 1.5$, $\frac{\partial v_{-1}}{\partial x_1} = 1$                                                             | $\bar{v}_{-1} = \bar{v}_1\cdot\frac{\partial v_{1}}{\partial v_{-1}} = .905$                                                 |
| $v_0 = x_2 = 0.5$, $\frac{\partial v_{0}}{\partial x_2} = 1$                                                                 | $\bar{v}_0 = \bar{v}_1\cdot\frac{\partial v_{1}}{\partial v_0}+\bar{v}_3\cdot\frac{\partial v_{3}}{\partial v_0} = -.925$    |
| $v_1 = \frac{v_{-1}}{v_0} = 3$, $\frac{\partial v_{1}}{\partial v_{-1}} = 2$, $\frac{\partial v_{1}}{\partial v_0} = -.444$  | $\bar{v}_1 = \bar{v}_2\cdot\frac{\partial v_{2}}{\partial v_1}+\bar{v}_4\cdot\frac{\partial v_{4}}{\partial v_1} = .453$     |
| $v_2 = sin(v_1) = 0.141$, $\frac{\partial v_{2}}{\partial v_{1}} = -.990$                                                    | $\bar{v}_2 = \bar{v}_5\cdot\frac{\partial v_{5}}{\partial v_2} = 1.351$                                                      |
| $v_3 = exp(v_0) = 1.649$, $\frac{\partial v_{3}}{\partial v_{0}} = 1.649$                                                    | $\bar{v}_3 = \bar{v}_4\cdot\frac{\partial v_{4}}{\partial v_3}+\bar{v}_5\cdot\frac{\partial v_{5}}{\partial v_3} = -.439$    |
| $v_4 = v_1 - v_3 = 1.351$, $\frac{\partial v_{4}}{\partial v_{1}} = 1$, $\frac{\partial v_{4}}{\partial v_3} = -1$           | $\bar{v}_4 = \bar{v}_6\cdot\frac{\partial v_{6}}{\partial v_4} = 1.79$                                                       |
| $v_5 = v_2 + v_3 = 1.79$, $\frac{\partial v_{5}}{\partial v_{2}} = 1$, $\frac{\partial v_{5}}{\partial v_3} = 1$             | $\bar{v}_5 = \bar{v}_6\cdot\frac{\partial v_{6}}{\partial v_5} = 1.351$                                                      |
| $v_6 = v_5 \cdot v_4 = 2.418$, $\frac{\partial v_{6}}{\partial v_{5}} = 1.351$, $\frac{\partial v_{6}}{\partial v_4} = 1.79$ | $\bar{v}_6 = 1$                                                                                                              |

## How to Use AutoDiff

The package will include a module for an `AutoDiff` class that utilizes the core data structure, the `DualNumber` objects. The user will interact with the `AutoDiff` module, without needing to interact with the `DualNumber` class. As such, user should import the `AutoDiff` module and the elementary functions for dual numbers. The user will initialize an `AutoDiff` object with a list of lambda functions representing a vector function $\mathbf{f}$. The user can then evaluate either a directional derivative, gradient, or Jacobian. and an associated `value` at which to evaluate. Example use cases are shown below.


### Installation

We will provide separate (but similar) installation instructions for 1) typical users and 2) fellow developers. In each case we will assume the user will install in a virtual environment, and will show correspond steps. 

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

### Imports
```python
import numpy as np
from autodiff.auto_diff import AutoDiff
from autodiff.utils.auto_diff_math import *
```

### Functions and Arguments

Below we discuss usage of the interface functions included in `AutoDiff`. For detailed documentation and a list of supported operations and mathematical functions, please see our sphinx documentation.

A function callable or list of callables is all that's needed to initiate an AutoDiff object. For example, here we initiate an AutoDiff object `ad` containing the vector function $\mathbf{f} = [f_1, f_2]^T$

```python
f1 = lambda(x): x[0]+x[1]
f2 = lambda(x): x[0]*x[1]
ad = AutoDiff([f1, f2])
```
Note that, as shown in the example above, if the function takes an $\mathbb{R}^m$ input, the argument for the function should be treated as an array with integer indices, and the index `k-1` should be used to indicate $x_k$ for $k\in \{1,...,m\}$, i.e. `x[0]` represents $x_1$.

To evaluate the function values at a given point $\mathbf{x}$ we use `get_value`, the function arguments are

```python
ad.get_value(point: Union[int, float, list, np.ndarray])
```

where `point` should be an `int` or `float` object if $x\in\mathbb{R}$ and a python list or numpy array if $\mathbf{x}\in\mathbb{R}^m$. An example would then be

```python
# evaluate function value at point
point = np.array([1, 1)]
ad.get_value(point)
```

Now entering the differentiation territory. To obtain the gradient of a scalar function or the Jacobian matrix of vector function, use `get_jacobian`

```python
get_jacobian(point: Union[int, float, list, np.ndarray], mode="forward"):
```

Note that the Jacobian matrix is the most generalized form of partial derivatives of $\mathbf{f}:\mathbb{R}^m\to\mathbb{R}^n$. As mentioned above, the `get_jacobian` function returns a **gradient** in case of scalar function $f:\mathbb{R}^m\to\mathbb{R}$. We will include a related demo later in this documentation.

The Jacobian can be computed through forward or reverse mode, specified via the `mode` argument, which takes one of the valid strings (case insensitive) `["f", "forward", "r", "reverse"`, with default set to forward mode.

We can also compute the **direciontal derivative** evaluated at point at direction (a seed vector). To do so we invoke the function `get_derivative` which takes these arguments

```python
get_derivative(point: Union[int, float, list, np.ndarray], seed_vector=None, mode="forward"):
```
 where the coordinates are passed to `point` and $k-1$ is passed to `var_index`. If the function has one single, scalar input, `var_index` is ignored and the partial derivative is the derivative evaluated at `point`. The `mode` argument works in the same way as in `get_jacobian`. 
 
 In the example below we have an input of $\mathbf{x}=[x_1, x_2]^T$ and we obtain the partial derivative with respect to $x_2$, evaluated at $[1, 1]^T$:
 
```python
point = np.array([1, 1)]
p = np.array([1,0])
ad.get_derivative(point, seed_vector = p)
```

The next function that may come in handy is `get_partial`. This function can be used to obtain only the partial derivative with respect to one specific independent variable when there are multiple:
```python
get_partial(point: Union[int, float, list, np.ndarray], var_index = None)
```
 where the coordinates are passed to `point` and $k-1$ is passed to `var_index`. If the function has one single input (in which case `point` must be a scalar), var_index is ignored and the partial derivative is the derivative evaluated at `point`. In the example below we have an input of $\mathbf{x}=[x_1, x_2]^T$ and we obtain the partial derivative with respect to $x_2$, evaluated at $[1, 1]^T$:

```python
point = np.array([1, 1)]
ad.get_partial(point, var_index = 1)
```
More sample usgaes of the aforementioned functions are inluded as Demos below. 

**Mathematical Functions Supported**
```python
# sine
sin(x)
# cosine
cos(x)
# tangent 
tan(x)
# exponential
exp(x)
# exponential with arbitrary base
exp_b(x, base)
# log with base e 
log(x)
# log with arbitrary base 
log_b(x, base)
# hyperbolic sine 
sinh(x)
# hyperbolic cosine 
cosh(x)
# hyperbolic tangent
tanh(x)
# square root
sqrt(x)
# arcsine
asin(x)
# arccosine
acos(x)
# arctan
atan(x)
# logistic (sigmoid)
logistic(x)
```

### Demos

**Case 1: $\mathbb{R} \rightarrow \mathbb{R}$**

In this case, users can use `get_partial`, `get_jacobian`, and `get_derivative` to achieve the same goal of computing derivatives.
```python
f = lambda x: x**2 + 2*x
ad = AutoDiff(f)
value = 2

# when x in R, the partial derivative w.r.t. x is the derivative
partial = ad.get_partial(value) # 6

# althugh users can use get_jacobian, note that by definition Jacobian is a matrix, 
# so this returns vector instead of a scalar
jacobian = ad.get_jacobian(value) # [6]

# not specifying the seed in the R -> R case results in seed = 1
# note that if seed is specified to a scalar s other than 1, the function
# returns the product of the derivative and s
derivative = ad.get_derivative(value) # 6
```

**Case 2: $\mathbb{R}^n \rightarrow \mathbb{R}$ ($n \gt 1$)**

This is when we have a scalar function with multiple independent variables. Note that in this case the Jacobian returns a row vector, which should be distinguished from a column vector returned in case of $\mathbb{R}\to\mathbb{R}^m$. This is consistent with the definition of a Jacobian matrix.

```python
f = lambda x: x[0]**2 + 2*x[1]
ad = AutoDiff(f)
value = [2, 3] # Order must match the indexing of x in f definition
# row vector
jacobian = ad.get_jacobian(value) # [[4, 2]]

# directional derivatives
seed_vector = np.array([1, 0])
derivative = ad.get_derivative(value, seed_vector) # 4

seed_vector = np.array([0, 1])
derivative = ad.get_derivative(value, seed_vector) # 2
```

**Case 3: $\mathbb{R} \rightarrow \mathbb{R}^m$ ($m \gt 1$)**

This is when we have a vector function with a single independent variable. Note that in this case the Jacobian returns a column vector, which should be distinguished from a row vector returned in case of $\mathbb{R}^n\to\mathbb{R}$. This is consistent with the definition of a Jacobian matrix.
```python
f1 = lambda x: x**2 + 2*x
f2 = lambda x: sin(x)

ad = AutoDiff([f1, f2])
value = 2
# column vector
jacobian = ad.get_jacobian(value) # [[6], [cos(2)]]

# directional derivative
seed_vector = np.array([1])
derivative = ad.get_derivative(value, seed_vector) # [[6], [cos(2)]]
``` 

**Case 4: $\mathbb{R}^n \rightarrow \mathbb{R}^m$ ($n, m \gt 1$)**

```python
f1 = lambda x: x[0]**2 + 2*x[1]
f2 = lambda x: sin(x[0]) + 3*x[1]
ad = AutoDiff([f1, f2])
value = [2, 5] # Ordering specified by index of variables in f1, f2

# m x n matrix
jacobian = ad.get_jacobian(value) # [[4, 2], [cos(2), 3]]

seed_vector = np.array([1, 0])
derivative = ad.get_derivative(value, seed_vector) # [[4], [cos(2)]]

seed_vector = np.array([-2, 1])
derivative = ad.get_derivative(value, seed_vector) # [[-6], [-2cos(2) + 3]]
```

## Software Organization

**Directory Structure**

Here is our package directory structure:
  
```
team14/
    |-- src/
    |   |-- autodiff/
    |       |-- __init__.py
    |       |-- auto_diff.py
    |       |-- utils/
    |       |   |-- __init__.py
    |       |   |-- dual_numbers.py
    |       |   |-- auto_diff_math.py
    |-- .github/
    |       |-- workflows/
    |       |   |-- code_coverage.yml
    |       |   |-- test.yml
    |-- tests/
    |   |-- __init__.py
    |   |-- check_coverage.sh
    |   |-- run_tests.sh
    |   |-- test_auto_diff.py
    |   |-- test_auto_diff_math.py
    |   |-- test_dual_numbers.py
    |-- docs/
        |-- milestone1.md
        |-- milestone1.pdf
        |-- milestone2.md
        |-- milestone2.pdf
    |-- driver_script.py
    |-- LICENSE
    |-- README.md
    |-- pyproject.toml

```

**Modules and Functiionality**
    
  - Modules for the AutoDiff package:
    - auto_diff.py: This module is the interface of the package. Users will initiate an AutoDiff object to carry out any necessary calculations. 
    - dual_numbers.py: The DualNumber class is defined in this module. Although users do not need to directly interact with the DualNumber objects, the AutoDiff objects carry out function calculations and differentiation using DualNumber objects.
    - comp_graph.py: The CompGrahNode class is defined in this module. Although users do not need to directly interact with the CompGraphNode objects, the AutoDiff objects carry out function calculations and differentiation using these objects.
    - auto_diff_math.py: The overload functions for auto_diff which carry out different operations based on whether inputs are real numbers, DualNumber objects, or CompGraphNode objects.
  - Third-party modules:
    - NumPy: used for mathematical operations in automatic differentiation.
    - Math: for mathematical constants like $\pi$ and $e$.
    - toposort: for sorting the computational graph which is a directed acyclic graph
- Test suite 
  - As indicated above, the test suite will be in the `tests/` directory, separated from the source files.
- Package distribution
  - PyPI with PEP517.
- Package installation is detailed above, separately for developers and typical users. Developers may wish to run our tests, look at our documentation, and run code coverage locally. To do so they must git clone our repository and run the commands specified above. In this situation they do not need to install the package from PyPI; they can simply set their PYTHONPATH as specified.

## Implementation Details

**Overview**

The package implements Automatic Differentiation through both the forward and reverse modes. The forward mode is implemented by appropriately translating variables into **dual numbers**, and then evaluating expressions containing dual numbers using the built-in order of operations defined within Python. Crucially, when we perform (binary or unary) operations in evaluating these expressions, we do so **using only** elemental operations which we explicitly define ourselves via "operation overloading" on `DualNumber`s (an object which we define), and which obey the characteristics of dual numbers. The resulting expression will itself be a dual number, the **real** part of which represents the evaluation of the function at the provided input, and the **dual** part of which represents the derivative of the functions evaluated at the provided inputs.   


Similarly, the reverse mode also utlizes operation overloading, such that the elementary operations and mathematical functions can act acccordingly when `CompGraphNode` objects, instances of the `CompGraphNode` class defined for purpose of reverse mode, are passed to these functions. Specifically, in addition to computing the values of the elementary operations, the overloaded functions also calculate the partial derivatives of each parent node with respect to its child nodes, and store the child-parent references. This allows construction of the computational graph which can then be used to calculate the adjoints and, ultimately, the partial derivatives $\frac{\partial f_i}{\partial x_k}$ during the reverse pass.

**Classes**

Core Class 1: `DualNumber`

- This class will be used inside the `AutoDiff` class; it is the foundation upon which our forward mode implementation is built.
- This class defines a `DualNumber` object which has two attributes `real` and `dual`
  - If not specified, the `dual` part of a `DualNumber` will default to 1
- We need to be able to perform elementary operations on `DualNumber`s in such a way that adheres to the behavior of dual numbers, as defined above.
- For example, for $z_1 = a_1 + b_1\epsilon$ and $z_2 = a_2  b_2 \epsilon$, we want that:
  - $z_1 + z_2 = (a_1 + a_2) + (b_1 + b_2)\epsilon$
  - $z_1z_2 = (a_1a_2) + (a_1b_2 + b_1a_2)\epsilon$
- In order to do this we will perform "operation overloading" on dunder methods, and define, for example:

```python
class DualNumber:
    def __add__(self, other):
        """Addition operator for dual numbers.
        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber or float or int
            The second dual number or a real number.
        Returns
        -------
        DualNumber
            The sum of the two dual numbers.
        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.
        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self), type(other)))
```

- These methods have been carefully constructed to handle cases of, say, adding a DualNumber to a scalar (no matter the order in which they are passed).
- We overload the following operators:
  - __add__, __radd__, __sub__, __rsub__, __mul__, __rmul__, __truediv__, __rtruediv__, __pow__, __rpow__, __neg__, __repr__, __eq__, __ne__, __lt__, __gt__, __le__, __ge__

Core Class 2: `CompGraphNode`

- This class will be used inside the `AutoDiff` class; it is the foundation upon which our reverse mode implementation is built.
- This class defines a `CompGraphNode` object which the following attributes
  - value: a real number representing the value of the node
  - parents: a list of reference to the node's parent nodes; default is None
  - partials: a list of partial derivatives $\frac{\partial v_i}{\partial v_j}$ where $v_i$ is a parent node and $v_j$ is the current node, in the same order as the list `parents`; default is None
  - _added_nodes: a dictionary storing the nodes that have already been added to the computational graph; default is None
  - adjoint: the value of adjoint used in reverse pass
  
- We need to be able to perform elementary operations on `CompGraphNode`s such that a computational graph is constructed during the forward pass and the intermediate partial derivatives are recorded, in addition to computing the result of the elementary operations.
- In order to do this we will perform "operation overloading" on dunder methods, and define, for example:

```python
class CompGraphNode:
    def __add__(self, other):
        """Addition operator for nodes.
        Parameters
        ----------
        self : CompGraphNode
            The first node.
        other : CompGraphNode or float or int
            The second node or a real number.
        Returns
        -------
        CompGraphNode
            The sum of the two nodes.
        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.
        """
        if ("add", self, other) in self._added_nodes:
            return self._added_nodes.get(("add", self, other))

        if isinstance(other, (CompGraphNode, int, float)):
            if isinstance(other, CompGraphNode):
                node = CompGraphNode(self.value + other.value,
                                     parents=[self, other],
                                     partials=[1, 1],
                                     added_nodes=self._added_nodes)

            else:
                node = CompGraphNode(self.value + other,
                                     parents=[self],
                                     partials=[1],
                                     added_nodes=self._added_nodes)

            # add to existing nodes
            self._added_nodes[("add", self, other)] = node
            return node

        raise TypeError(
            "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))
```

- We overload the following operators:
  - __add__, __radd__, __sub__, __rsub__, __mul__, __rmul__, __truediv__, __rtruediv__, __pow__, __rpow__, __neg__, __repr__
  
Module: `auto_diff_math.py`

- In addition to the operator overloading that we introduce in the `DualNumber` and `CompGraphNode` classes, we also specify our own definitions for other elementary mathematical operations which are needed for a complete AD implementation.
- We organize these additional overloading functions in a module which we import for use in the `AutoDiff` class defined below.
- These functions each follow the same structure: for a `DualNumber`, `a = DualNumber(real, dual)`, and a function `func`, if we pass `func(a)`, we will return another `DualNumber`, say `DualNumber(new_real, new_dual)` such that:
   - `new_real` is `func` applied to `real` 
   - `new_dual` is the derivative of `func` applied to `real` *times* `dual`
- These functions gracefully handle non-Dual numbers, by, for example, falling back to the standard implementation (e.g., `np.sin`) when passed a real number.
- By explicitly defining elemental operations in this way, we ensure that when evaluating expressions containing dual numbers, python will resolve to a final expression which is itself a dual number whose dual part represents the derivative of interest
- Here is an example of such a function.

```python
def sin(x):
    """Computes the sine of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the sine of.
        
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The sine of x.
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if isinstance(x, (int, float)):
        return math.sin(x)

    if isinstance(x, CompGraphNode):
        if ("sin", x, None) in x._added_nodes:
            return x._added_nodes.get(("sin", x, None))

        node = CompGraphNode(math.sin(x.value),
                             parents=[x],
                             partials=[math.cos(x.value)],
                             added_nodes=x._added_nodes)

        x._added_nodes[("sin", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.real), math.cos(x.real) * x.dual)

    raise TypeError(
        "sin() only accepts int, float, DualNumber, or CompGraphNode.")
```
- We implement the following:
  - sin, cos, tan, exp, log, sinh, cosh, tanh, arcsin, arccos, arctan, exponentials with any base, logistic, log with any base, sqrt

Interface Class: `AutoDiff`

- Users instantiate an `AutoDiff` object with one parameter `f`, which is assigned as an attribute of the object `self.f`.
  - `f` is either a *function* or *list of functions* ($f : \mathbb{R}^n \to \mathbb{R}^m$) over which to evaluate derivatives.
    - Example:
    - `f = lambda x : x**2 + 3*x` | `g = lambda x : 2x**2 - 14`
      - `ad1 = AutoDiff(f)` | `ad2 = AutoDiff([f, g])`
    - `h = lambda x : x[0]**2 + sin(x[1])`
      - `ad3 = AutoDiff(h)`
  - Note that in the event that a user wishes to pass a **multivariate function** (`h`, or `j` above), they must define a `lambda` function which takes a vector of input, and **index into x** appropriately within the functional expression.
  - It is important to note that `ad = AutoDiff([h, j])` will rely on the index values to assign variables and assume consistent indexing across multiple functions
  - Upon initialization the function also checks for valid input. For example, it will check the input represents valid mathematical functions.
- The AutoDiff implements four important instance methods which compute derivatives. 

- `get_jacobian`
  - Computes the Jacobian matrix for a given arbitrary function `f` mapping $\mathbb{R}^m\to\mathbb{R}^n$.  - 
  - args: 
    - `point`; the point at which to evaluate the Jacobian matrix.
  - The method performs forward mode AD by default. This implementation allows automatic differentiation of functions of $\mathbb{R}^m\to\mathbb{R}^n$.
  - The order of the columns correspond to the order arguments are passed to the functions
  - The order of the rows correspond to the index values of x (if multi-dimensional)

- `get_partial`
  - Computes the vector of partial derivatives of `f` evaluated at `point`
  - args: 
    - `point`; the point at which to evaluate the partial derivatives matrix
    - `var_index`; the variable index at which to evaluate partial derivatives
  - This function is called by get_jacobian; it calculates partial derivatives needed to construct the Jacobian matrix
  - For each partial derivative, $\frac{\partial f_1}{\partial x}$, $x_i$ will be converted into a DualNumber object `(x_i, 1)` while other variables $x_j, j\neq i$ will be converted into `DualNumber` objects `(x_j, 0)`, such that the differentiation will be done with respect to $x_i$.

- `get_derivative`
  - Computes the directional derivative evaluated at values in the direction and magnitude of seed_vector
  - args:
    - `point`; the point at which to evaluate the derivative
    - `seed_vector`; a scalar or array of number defining the seed of direction
  - This function uses the seed vector and Jacobian matrix to calculate and return a derivative for the specified `f` at `point`

- `get_value`
  - Calculates the value of a function at a given point
  - args:
    - `point`; the point at which to evaluate the function


# Extension

### Automatic Differentiation through Reverse Mode

We have implemented reverse mode to efficiently handle the calculation of derivatives in the case of $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ where $n \gg m$. The above session briefly discussed the inclusion of reverse mode while this session will go into more detail about the implementation.

- We retain the same user interface methods, such as `get_jacobian`, and `get_derivative`, which will include an additional parameter `mode`. The default of `mode` is `"forward"` but user can specify it to be `"reverse"`. For example:

    ```python
    f = lambda x: x[0]**2 + 3*x[1] + sin(x[2])
    ad = AutoDiff(f)
    der = f.get_derivative([1, 2, 3], [1, 0, 0], mode="reverse")
    ```

- When reverse mode is called, we carry out the implementation by 
  1) Performing the forward pass: compute and save partial derivatives and constructing the computational graph, and
  2) Performing the reverse pass: reconstruct the chain rule by exploiting parent-child relationships dictated by the graph structure.

- Specifically, the implementation is done as below
    - A `CompGraphNode` class (see Section Implementation for more details) is created of which the objects will be elements of a `CompGraph`. A `CompGraphNode` stores references to its parents and the corresponding partial derivatives $\frac{\partial v_j}{\partial v_i}$.
    - Inputs are initialized as 'CompGraphNode' objects of which the `value` attributes are the real number inputs and the `parents` and `partials` attributes are set to `None`, since inputs in the computational graph are nodes without parents.
    - The user-defined scalar or vector functions are called with `CompGraphNode`s passed as arguments. The functions are broken down to elementary operations, which are overloaded such that when each operation is carried out, a `CompGraphNode` is added to the graph through the child-parent link stored in each child node. This ensures the forward pass is completed as the function is evaluated.
        - Additionally, a hash table (dictionary) is used to keep track of nodes that have been added to the graph to avoid repeated nodes.
    - After forward pass is complete, the function output node serves as the root of the constructed computational graph. Note since our algorithms stores references to parent nodes in child nodes, instead of the other way around, the arrows in the constructed graph will have oppositive directions compared to the computational graph we drew in the Background session above, but the graph structure based on the parent-child relationships will be the same.
    - Evidently, the constructed graph is a directed acyclic graph (DAG). A topological sort is then imposed on the graph, which returns a sorted list of nodes such that all a child node comes after its parent nodes. The reverse pass to calculate the adjoints of each node is then started at the **end** of the sorted nodes, the output node. The reverse topological order ensures that a child node's adjoint will be computed before its parent nodes'.
    - After the reverse pass is complete, the partial derivatives $\frac{\partial f_i}{\partial x_k}$ are stored as the adjoint of the input nodes. The Jacobian matrix and directional derivatives are generated using the adjoints. 
    

# Broader Impact and Inclusivity Statement


## Broader Impact

Automatic differentiation has wide usage in science and engineering due to its efficient calculation of gradient to machine precision. A notable example is its application in neural networks, serving as the foundation of backpropagation. Such usage has been adopted by various fields. For example, automatic differentiation-powered deep convolutional neural network has been used in noval medical research to identify diseases such as Glaucoma and thyroid scintigram. These methods, along with other innovatie machine learning techniques introduced to the medical and life sciences fields have profound impact on public health and well-being.

However, any algorithms that facilitate machine learning applications also augment the negative impact associated with these applications. To use the filed of medicine as an example again, while machine learning has been shown to be helpful, one should be extremely cautious about the role of algorithims in, say, medical diagnoses. Complete reliance on machine learning or artificial intelligence, regardless of how powerful they seem thanks to clever algorithms like automatic differentiation, can be extremely dangeours. 


## Software Inclusivity
The future development of this software package for any additional features, such as higher order derivatives, will aim to be open and inclusive. The current content of this software package is contributed equally by all members of our team with diverse backgrounds and experiences, and reflects equal appreciation and respect towards the diverse knowledge and expertise associated with these backgrounds. We hope that any future development continue to uphold these values while being open to the broader community. Specifically, we welcome everyone's contribution regardless of their background and the review of contributions, conducted by all members of the team, will be based on nothing else but the quality of the content. 

# Future Enhancements

## Higher order derivatives
A natural next step for our project is a implement functionality for higher order derivatives. Calculating higher order derivatives of course has many applications in science and mathematics. For example, in physics, we use them to find the acceleration or jerk of an object moving through space. 

Second derivatives are particularly useful in identifying whether a critical point of a function is a minimum or maximum (or saddle point). This is useful in, for example, statistics when performing maximum likelihood estimation on a complex likelihood function of many parameters. 

Of course, our package can currently be used "as is" to compute higher order derivatives, in the sense that we can repeatedly call our differentiation functions on functional output. However, this naive approach is inefficient and, if we were to explore this option, we would need to research alternative, more efficient algorithms. 

## Host Documentation on a static webpage
Currently, if a user would like to view our Sphnix documentation, they need to clone our repo and view the appropriate index.html file locally. We would instead provide a better user experience by hosting our documentation on a static webpage (perhaps even one that is updated automatically on push to master), and linking to this webpage in our root-level README. 

## Default choice for forward/reverse mode
Some users may not be familiar with "forward" vs. "reverse" mode and simply want to calculate derivatives using our package, without knowing when it is wise to specify forward vs reverse mode. We could provide a default choice of forward/reverse mode based on the inputs that the user provides to the interface (for example, applying reverse mode with very high dimensional inputs).   

