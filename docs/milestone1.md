## Introduction

In its simplest form, we define the derivative of a function as its rate of change. We should familiarize ourselves with an example of a function, the notation for a derivative, and the graphical representation of rate of change.

Say we are incredibly lucky in our investments and our return, f(x), is modeled by the function f(x) = $x_2$. Here, x can represent the dollar amount we invest. We would like to measure the rate at which our return, f(x), changes with respect to a change in our investment dollar amount, x. The notation for such is represented mathematically as:

$\frac{df}{dx}$ = $lim_{\Delta x\to 0} \frac{\Delta f}{\Delta x}$

For example, we may model the relationship between investment ($x$) and returns ($f(x)$) using a function f : $R$ &rarr; $R$, by $f(x) = x^2$. 

The derivative is modeled as $2\cdot x$ and we can interpret it as the instantaneous rate of change - slope - as seen in the illustration below. 

![](files/Figure1.png)

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

$f(x_{1}, x_{2}) = exp( sin(x_{1}^{2} + x_{2}^{2}) + 2 * cos(\frac{x_1}{x_2}))$

In the function above we can identify several functions that would be considered elementary functions: multiplication, division, sin(), cos(), exponentiation, powers. Automatic differentiation breaks about functions such as f(x) into the components of its elementary functions to act on intermediate steps in order to solve for its derivative. A more comprehensive list of elementary functions is included below: 

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

$f(x) = exp(4x)$

We can replace $4x$ with u(x). This will allow us to do the following differentiation to get our desired derivative of f(x) with respect to x. 

$\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx} = exp(u)\cdot 4 = 4\cdot exp(4x)$

### Computational Graph (Forward Mode)

A computational graph allows us to see the ordered sequence of elementary functions, how we break down a more complex function from inside to outside (in forward mode), and how we can calculate intermediate steps to arrive at our final derivative result. 

In the computational graph below, we can see that we begin with the inputs to the function, independent variables denoted by subscripts -1 and 0 (these generally take values < 1). Additionally, we build on these with intermediate variables from $v_0, v_1, ...$. The intermediate variables parallel the elementary functions applied at each step until we arrive at the full complex model from the inside out (again in forward mode). We can follow the computational graph's arrows to see how the elementary functions are applied until we reach our desired result of differentiation.

Let us examine the utility of a computational graph with a complex function such as: 

$f(x_1, x_2) = [sin(\frac{x_1}{x_2} + \frac{x_1}{x_2} - e^{x_2} ] \cdot [ \frac{x1}{x2} - e^{x_2}]$ 

We can see that elementary functions we will need are exp(), sin(), addition, subtraction, multiplication, and division. Additionally, we will need to create intermediate steps that build on the independent variables $x_1$ and $x_2$ in order to create all parts of the complex model. By following the arrows of the graph, we can see how we can begin at the independent variables and arrive back at the full complex function f(x). 

![](files/Figure2.png)

### Evaluation Trace (Forward Mode)

The evaluation trace allows us to utilize the components of our computational graph to aid us in solving the function value at a specific point and the partial derivatives with respect to each independent variable. The latter is done by utilizing seed vectors, p, which indicate the input variable to calculate the partial derivative of for the function. We will require one pass for each of the independent variables (x1, x2). In the first pass, shown below, we set the seed vector p = [1,0].

| Forward Primal Trace| Forward Tangent Trace (p = [1 0])|
| --- | --- |
| $v_{-1} = x_1 = 1.5$| $D_pv_{-1} = 1$|
| $v_0 = x_2 = 0.5$ |  $D_pv_{0} = 0$|
| $v_1 = \frac{v_{-1}}{v_0} = 3$ | $D_pv_1 = \frac{(v_0 D_pv_{-1} - v_{-1}D_pv_0}{v_o^2} = 2$ |
| $v_2 = sin(v_1)$ = 0.141 | $D_pv_2 = cos(v_1) \cdot D_pv_1 = -1.98$|
| $v_3 = exp(v_0) = 1.649$ | $D_pv_3 = v_3 \cdot D_pv_0 = 0$ |
| $v_4 = v_1 - v_3 = 1.351$ | $D_pv_4 = D_pv_1 - D_pv_3 = 2$|
| $v_5 = v_2 + v_4 = 1.492$ | $D_pv_5 = D_pv_2 + D_pv_4 = 0.02$ |
| $v_6 = v_5 \cdot v_4 = 2.017$ | $D_pv_5 \cdot v_4 - D_pv_4 \cdot v_5 = 3.012$|


We point out that the left column gives us the result of our function f($x_1$ = 1.5, $x_2$ = 0.5). Meanwhile, the right column gives us the results of the partial derivative with respect to $x_1$. We would require another pass with p =[0 1] in order to solve for the partial derivative with respect to $x_2$


## How to Use AutoDiff

The package will include a module for an AutoDiff class that utilizes the core data structure, the DualNumber objects. The user will interact with the AutoDiff module, without needing to interact with the DualNumber class. As such, user should import the AutoDiff module. The user will initialize an AutoDiff object with a list of strings representing a vector function $\mathbf{f}$, and an associated `value` at which to evalauate. The user can then evaluate either a derivative, gradient, or Jacobian. For example:

```python
value = {"x": 2, "y" : 5}
f = lambda x, y : x**2 + 2y
ad = AutoDiff(f, value)
derivative = ad.get_derivative("x") # 4
derivative = ad.get_derivative("y") # 2
gradient  = ad.get_gradient() # [4, 2]

value = {"x": 2, "y" : 5}
f1 = lambda x, y : x**2 + 2y
f2 = lambda x, y : sin(x) + 3y
f = [f1, f2]
ad = AutoDiff(f, value)
jacobian = ad.get_jacobian() # [[4, 2], [cos(2), 3]]
```

## Software Organization

```
team14/
    ├── src/
    │   ├── __init__.py
    │   ├── adiff.py
    │   ├── utils/
    |   |   ├── __init__.py     
    │   │   ├── dual_numbers.py
    │   │   └── helpers.py
    │   └── examples/
    |       ├── __init__.py
    │       ├── example_1.py
    |       └── ...
    ├── tests/
    ├── docs/
    │   └── milestone1
    ├── LICENSE
    └── README.md
```

- What modules do you plan on including? What is their basic functionality?
  - NumPy: used for mathematical operations in automatic differentiation.
  - Math: for mathematical constants like $\pi$ and $e$

- Where will your test suite live?
  - As indicated above, the test suite will be in the `tests/` directory, separated from the source files.

- How will you distribute your package (e.g. PyPI with PEP517/518 or simply setuptools)?
  - PyPI with PEP517.

- Other considerations?
  - If the operations included in the `dual_numbers` module prove to be too extensive for a single file we will consider changing it into a directory and separating the dual number related operations in different modules

## Implementation

**Overview**

The package will implement Automatic Differentiation by appropriately translating variables into **dual numbers**, and then simply evaluating expressions containing dual numbers using the built-in order of operations defined within Python. Crucially, when we perform (binary or unary) operations in evaluating these expressions, we will do so **using only** elemental operations which we explicitly define ourselves via "operation overloading" on `DualNumber`s (an object which we define), and which obey the characteristics of dual numbers. The resulting expression will itself be a dual number, the **real** part of which represents the evaluation of the function at the provided input, and the **dual** part of which represents the derivative of the functions evaluated at the provided inputs.  

**Classes**

Class 1: `AutoDiff`

- This is the only class that users will directly interact with.
- Users instantiate an `AutoDiff` object with two parameters, `f` and `value`
  - `f` is either a *function* or *list of functions* ($f : R^n \rightarrow R^m$) over which to evaluate derivatives. 
  - `value` is a *dictionary* (`str` : `float`), representing the value(s) at which the user seeks to evaluate the derivative
  - Some examples:
    - `ad = AutoDiff(lambda x: sin(x) + 14, {"x" : 0})`
    - `f1 = lambda x : x**2; f2 = lambda x, y : x + y` 
    - `ad = AutoDiff([f1, f2], {"x" : 0, "y" : 1})`
- Upon initialization the function will also check for valid input. For example, it will check for valid correspondence in variable names between functions and value names
 - The variable strings supplied in `value` must match the variable names supplied in `f`
 - Variables will ultimately be passed to f as kwargs, so the following two cases will return the same derivative
    - `AutoDiff(lambda x, y: x - y, {"x" : 0, "y" : 1})`
    - `AutoDiff(lambda y, x: x - y, {"x" : 0, "y" : 1})`
- It will have a class method called `get_derivative` which performs forward mode AD 
  - `get_derivative` will operate on `self` and return the specified derivative
- It will also have class methods called `get_gradient` and `get_jacobian` which similarly operate on self and return either a gradient or jacobian, provided these these methods are compatible with the AutoDiff instance attributes provided
- Each of these functions will compute partial derivatives by evaluating expression involving dual number using elementary operations which we explicitly define on the `DualNumbers` class (discussed below) 
  
Class 2: `DualNumber`

- This class will be used inside the `AutoDiff` class; it is the foundation upon which our implementation is built
- This class defines a `DualNumber` object which has two attributes `real` and `dual`
  - If not specified, the `dual` part of a `DualNumber` will default to 1
- We need to be able to perform elementary operations on `DualNumber`s in such a way that adheres to the behavior of dual numbers, as defined above.
- For example, for $z_1 = a_1 + b_1\epsilon$ and $z_2 = a_2  b_2 \epsilon$, we want that:
  - $z_1 + z_2 = (a_1 + a_2) + (b_1 + b_2)\epsilon$
  - $z_1z_2 = (a_1a_2) + (a_1b_2 + b_1a_2)\epsilon$
- In order to do this we will perform "operation overloading" on dunder methods, and define, for example:

```python
class DualNumber:
    def __init__(self, real, dual=1):
        self.real=real
        self.dual=1
    
    def __add__(self, other):
        pass #TODO
        
    def __mul__(self, other):
        pass #TODO

    def __radd__(self, other):
        pass #TODO
    
    def __rmul__(self, other):
        pass #TODO

    def __pow__(self, other):
        pass #TODO

    # and more
```
- These methods will be carefully constructed to handle cases of, say, adding a DualNumber to a scalar (no matter the order in which they are passed)

Class 3: `AutoDiffMath`

- For those operations for which dunder methods are not defined, we will define a separate set of functions which perform these operations on `DualNumber`s. 
- We include these functions as static methods in a separate class which we import for use in the `AutoDiff` class defined above
- We need to import `numpy` and `math` for use in these functions
- These functions will be defined in a module and imported so that a user may call, e.g., `sin(DualNumber(3))` directly
- These functions each follow the same structure: for a `DualNumber`, `a = DualNumber(real, dual)`, and a function `func`, if we pass `func(a)`, we will return another `DualNumber`, say `DualNumber(new_real, new_dual)` such that:
   - `new_real` is `func` applied to `real` 
   - `new_dual` is the derivative of `func` applied to `real` *times* `dual`
- These functions will ultimately need to gracefully handle non-Dual numbers, by, for example, falling back to the standard implementation (e.g., `np.sin`) when passed a real number.
- By explicitly defining elemental operations in this way, we ensure that when evaluating expressions containing dual numbers, python will resolve to a final expression which is itself a dual number whose dual part represents the derivative of interest
- Here are some such functions.

```python
import numpy as np
import math

class AutoDiffMath:
    def __init__(self):
        pass
        
    @staticmethod
    def sin(DualNumber: x):  
        return DualNumber(np.sin(x.real), np.cos(x.real)*x.dual)
        # TODO: Add handling of non dual numbers
        
    @staticmethod
    def cos(DualNumber: x):  
        return DualNumber(np.cos(x.real), -np.sin(x.real)*x.dual)
        # TODO: Add handling of non dual numbers
        
    # ... and more
```

- Below is skeleton code for the `AutoDiff` class which relies upon the DualNumber objects discussed above.

```python
from utils import *

class AutoDiff:
    def __init__(self, f, value):
        self.format_check(f)
        self.f = f
        self.value = value
        
    def format_check(self):
        '''check that vector function passed as strings are correctly formated'''
        pass #TODO
        
    def get_derivative(self, df_wrt):
      dual_var = { df_wrt : DualNumber(self.value[df_wrt]) }
      other_vars = {(k, self.value[k]) for k != dual_var)}
      kwargs = {**dual_var, **other_vars}
      derivative = self.f(**kwargs).dual
      return derivative
      
    def get_gradient(self):
      pass #TODO
      
    def get_jacobian(self):
      pass #TODO
```

**Other Comments**

- We will not need a graph class to resemble the computational graph in forward mode since, as discussed above, we can avoid storing this information by simply casting certain variables to dual numbers and using python's built in "order of operations" to evaluate these expressions in the ways we define. However, we may need to implement a graph class if we proceed to implement reverse mode at a later stage in the project.
- ADD COMMENTS ABOUT REVERSE MODE/SAVING NODES/CLASS STRUCTURE 

## Licensing

We will use the *MIT License*, since we want our library to be open for anyone to modify it. This library will not provide all uses or versions of automatic differentiation available or the ones being still developed. Due to the limited timeframe of the project we only aim to provide the backbones for automatic differentiation and some of the basic algorithms in automatic differentiation. Thus we want anyone that wants to make changes to the library to fill their specific needs to be able to do so, either in a commercial setting or not.
