## Introduction

In its simplest form, we define the derivative of a function as its rate of change. We should familiarize ourselves with an example of a function, the notation for a derivative, and the graphical representation of rate of change.

Say we are incredibly lucky in our investments and our return, f(x), is modeled by the function f(x) = x2. Here, x can represent the dollar amount we invest. We would like to measure the rate at which our return, f(x), changes with respect to a change in our investment dollar amount, x. The notation for such is represented mathematically as:

$\frac{df}{dx}$ = $lim_{x\to 0} \frac{\Delta f}{\Delta x}$

For example, we may model the relationship between investment ($x$) and returns ($f(x)$) using a function f : $R$ &rarr; $R$, by $f(x) = x^2$. 

![](files/Figure1.png)

While the example above serves as a toy example to familiarize ourselves with the topic of differentiation and its graphical interpretation, the power of the derivative is not to be understated. Its origins data back to Isaac Newton and an application in physics and movement; however, it has since grown with applications in various different branches such as statistics, biology, finance, computer science, and many more fields.

There are three popular methods to calculate derivatives:

1) **Numerical**

2) **Symbolic**

3) **Automatic**

Numerical differentiation is the most basic and general introduction to calculating derivatives. In numerical differentiation we rely on the definition of the derivative, where we measure the amount of change in our function with a very small change in our input (x+h).

$$\frac{\delta{f(x)}}{\delta x} = lim_{h\to 0} \frac{f(x + h) - f(x)}{h}$$

However, numerical differentiation can have issues with round off errors that lead to not achieving machine precision and can struggle with computational time when many dependent variables exist. 

Symbolic differentiation attempts to manipulate formulas to create new formulas rather than performing numerical calculations. In doing so, we can in essence memorize derivatices of functiosn. However, symbolic interpretation is challenging to implement in computer programs and can be inefficient coding. 

Automatic differentiation focuses on certain core elements: the chain rule, elementary functions and, to a lesser extent, dual numbers. The benefits of automatic differentiation are that it does not suffer form the same round off errors that numerical differentiation is susceptible to and does not suffer from the overly expensive, inefficient methods of symbolic differentiation. For these reasons, automatic differentiation is ubiquitous in tasks requiring quick differentiation, such as optimization in machine learning.


## Background

Automatic differentiation builds off of two fundamental and relatively easy to understand concepts: elementary functions and the chain rule. 

### Elementary Functions

First, we can begin by providing an example of identifying elementary functions within a function. Consider the function: 

$f(x_{1}, x_{2}) = exp( sin(x_{1}^{2} + x_{2}^{2}) + 2 * cos(\frac{x_1}{x_2}))$

In the function above we can identify several functions that would be considered elementary functions: multiplication, division, sin(), cos(), exponentiation, powers. Automatic differentiation breaks about functions such as f(x) into the components of its elementary functions to act on intermediate steps in order to solve for its derivative. A more comprehensive list of variables is included below: 

<br> 

| Category| Elementary Functions |
| --- | --- |
| Arithmetic | multiplication, addition, subtraction, division |
| Powers and Roots | $x^{2}$, $y^{1/2}$ |
| Trigonometric | sine, cosine, tangent, secant, cosecant, cotangent |
| Logorithmic | $\log(x)$ |
| Exponential | $\exp(x)$ |



### Chain Rule 

Utilizing the above elementary functions, automatic differentiation applies the ever important chain rule to the elementary functions in order to solve the derivative of more complex functions. As  aquick recap of the chain rule, let us define the following function: 

$f(x) = log(7x^{2})$

We can replace $7x^{2}$ with u(x). This will allow us to do the following differentiation to get our desired derivative of f(x) with respect to x. 

**potentially include more steps below**

$\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx} =  \frac{2}{x}$ 



### Computational Graph

A computational graph allows us to see the ordered sequence of elementary functions, how we break down a more complex function from inisde to outside (in forward mode), and how we can calculate intermediate steps to arrive at our final derivative result. 

In the computational graph below, we can see that we begin with the inputs to the function, independent variables denoted by subscripts -1 and 0 (these generally take values <1). Additionally, we build on these with intermediate variables from $v_0, v_1, ...$. The intermediate variables parallel the elementary functions applied at each step until we arrive at the full complex model from the inside out (again in forward mode). We can follow the computational graph's arrows to see how the elementary functions are applied until we reach our desired result of differentiation.

Let us examine the utility of a computational graph with a complex function such as: 

$f(x_1, x_2) = [sin(\frac{x_1}{x_2} + \frac{x_1}{x_2} - e^{x_2} ] \cdot [ \frac{x1}{x2} - e^{x_2}]$ 

We can see that elementary functions we will need are exp(), sin(), addition, subtraction, multiplication, and division. Additionally, we will need to create intermediate steps that build on the independent variables $x_1$ and $x_2$ in order to create all parts of the complex model. By following the arrows of the graph, we can see how we can begin at the independent variables and arrive back at the full complex function f(x). 

![](files/Figure2.png)

provide explanation of evaluation trace and example 

explain forward mode and example 

explain reverse mode and example 

## How to Use PackageName

The package will include a module for an AutoDiff class that utilizes the core data structure, the DualNumber objects. User will interact with the AutoDiff module, without needing to interact with the DualNumber class. As such, user should import the AutoDiff module. User will initialize an AutomaticDifferentiator object with a list of strings representing a vector function $\mathbf{f}$. User can then use either forward or backward mode to evaluate the vector function at a point $\mathbf{x}$, represented by a dictionary with keys corresponding to the variable names in the user defined function. For example:

```python
function_output = ["sin(x1)+x2", "exp(x2)*ln(x1)"]
ad = AutoDiff(function_output)
inputs = {"x1": 2, "x2":1}
df1 = ad.forward(inputs)
df2 = ad.backward(inputs)
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

The package will implement Automatic Differentiation by appropriately translating variables into **dual numbers**, and then simply evaluating expressions containing dual numbers using the built-in order of operations defined within Python. Crucially, when we perform (binary or unary) operations in evaluating these expressions, we will do so **using only** elemental operations which we explicitly define ourselves on `DualNumber`s (an object which we define), and which obey the characteristics of dual numbers. The resulting expression will itself be a dual number, the **real** part of which represents the evaluation of the function at the provided input, and the **dual** part of which represents the derivative of the functions evaluated the provided inputs.  

**Classes**

Class 1: `AutoDiff`

- This is the only class that users will directly interact with.
- Users instantiate an `AutoDiff` object with two parameters, `f` and `value`
  - `f` is either a *string* or *list of strings* representing the functions ($f : R^n \rightarrow R^m$) over which to evaluate the derivative. 
  - `value` is a *dictionary* (`str` : `float`), representing the value(s) at which the user seeks to evaluate the derivative
  - Some examples:
    - `ad = AutoDiff("sin(x) + 14", {"x" : 0})`
    - `ad = AutoDiff("sin(xy)", {"x" : 0, "y" : 1})`
    - `ad = AutoDiff("sin(xz)", {"x" : 0, "z" : 1})`
    - `ad = AutoDiff(["sin(x)", "x+y"], {"x" : 0, "y" : 1})`
    - `ad = AutoDiff(["sin(xyz)", "x+y", "y+z^2"], {"x" : 0, "y" : 1, "z" : 0})`
- Upon initialization the function will also check for valid input. For example, it will check:
  - Parenthesis correctly applied
  - Valid function names
  - Valid correspondence in variable names between functions and value names
- It will have a class method called `forward` which perform forward mode AD 
  - `forward` will operate on `self` and return:
    - A **scalar** of the specified derivative **if** provided a 1D input and 1D function
    - A **gradient** **if** provided a vector input and 1D output
    - A **Jacobian** **if** provided a vector input and vector output
  - Forward will calculate partial derivatives by simply converting functional string expressions into python functions which operate on DualNumbers, and evaluating these expressions using elementary operations which we explicitly define on DualNumbers (discussed below) 
  
Class 2: `DualNumber`

- This class will be used inside the `AutoDiff` class; it is the foundation upon which our implementation is build
- This class defines a DualNumber object which has two attributes `real` and `dual`
  - If not specified, the `dual` part of a DualNumber will default to 1
- We need to be able to perform elementary operations on DualNumbers in such a way that adheres to the behavior of dual numbers, as defined above.
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
    
    def __eq__(self, other):
        pass #TODO
    
    def __lt__(self, other):
        pass #TODO
    
    def __gt__(self, other):
        pass #TODO
    
    def __str__(self):
        pass #TODO

    # and more
```
- These methods will be carefully constructed to handle cases of, say, adding a DualNumber to a scalar (no matter the order in which they are passed)

Class 3: `AutoDiffMath`

- For those operations for which dunder methods are not defined, we will define a separate set of functions which perform these operations on DualNumbers. 
- We include these functions as static methods in a separate class which we import for use in the `AutoDiff` class defined above
- We need to import `numpy` and `math` for use in these functions
- These functions each follow the same structure: for a DualNumber, `a = DualNumber(real, dual)`, and a function `func`, if we pass `func(a)`, we will return another DualNumber, say `DualNumber(new_real, new_dual)` such that:
   - `new_real` is `func` applied to `real` 
   - `new_dual` is the derivative of `func` applied to `real` *times* `dual` (by the chain rule)
- By explicitly defining elemental operations in this way, we ensure that when evaluating expressions containing dual numbers, python will resolve to a final expression which is itself a dual number whose dual part represents the derivative of interest
- Here are some such functions:

```python
import numpy as np
import math

class AutoDiffMath:
    def __init__(self):
        pass
        
    @staticmethod
    def sin(DualNumber: x):  
        return DualNumber(np.sin(x.real), np.cos(x.real)*x.dual)
        
    @staticmethod
    def cos(DualNumber: x):  
        return DualNumber(np.cos(x.real), -np.sin(x.real)*x.dual)
        
    @staticmethod
    def log(DualNumber: x, float: base=np.e):  
        return DualNumber(np.log(x.real)/np.log(base), 1/(x.real*np.log(base))*x.dual)
               
    @staticmethod
    def exp(DualNumber: x):  
        return DualNumber(np.exp(x.real), np.exp(x.real)*x.dual)           
    
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
        
    def _format_check(self):
        '''check that vector function passed as strings are correctly formated'''
        pass
        
    def _parse(self, f, xi):
        '''parse f expressed in string to the overloading functions 
           defined for DualNumber
           f: The function to parse
           xi: Differentiate with respect to xi (xj treated as constant for j!=i)
           return: A function involving (potentially) a combination of DualNumbers and scalars which
           takes as input the entire dictionary self.value, passed via **kwargs
        '''
        return func
    
    def forward(self):
        jacobian = np.empty((len(self.value), (len(self.f)))
        for j, func in enumerate(self.f):
          for i, val in enumerate(self.value)
            parsed_func = parse(func, val)
            derivative  = parsed_func(self.value)
            jacobian[i, j] = derivative
            
        return jacobian
```

- Note that, as written,  `forward` always returns a Jacobian
  - In the event that a user passes in a scalar function and input, we can appropriately index into the Jacobian to return a scalar
  - The indexing of the Jacobian will correspond to the order in which functions and values were passed to `AutoDiff`. We will explore ways to potentially disambiguate this output. 
  - Further, the format_check class method will handle the case of converting a single string function into a single-element list for compatibility with `forward`.

**Other Comments**

- We will not need a graph class to resemble the computational graph in forward mode since, as discussed above, we can avoid storing this information by simply casting certain variables to dual numbers and using python's built in "order of operations" to evaluate these expressions in the ways we define. However, we may need to implement a graph class if we proceed to implement reverse mode at a later stage in the project.  

**Example**

Say a user runs:

```python
f = "sin(x)+3y"
value = {"x" : 0, "y" : 4}
ad = AutoDiff(f, value)
derivative = ad.forward()
print(derivative)
```

What is happening "behind the scenes"?

- Step 1: `AutoDiff(f, value)` will first check for valid input, and then create and instance of `AutoDiff` with `self.f = f` and `self.value = value`
- Step 2: `ad.forward()` will initialize an empty `jacobian` matrix of dimension 2x1
- Step 3: Fill in jacobian[0, 0] via `forward`, which will first calculate the derivative of `f` with respect to `x`
  - `parse(func, val)` will `parse` `f` with respect to `x` and return a function, `DualNumber.sin(DualNumber(x)) + 3*y` which takes two inputs, `x` and `y` 
    - Note that it only converted `x` to a DualNumber because we are differentiating with respect to `x` in this pass
  - `parsed_func(self.value)` will evaluate the parsed expression at the value `{"x" : 0, "y" : 4}`
  - The dual part of this expression will be the derivative of `f` with respect to `x`
  - We add the `derivative` to the appropriate index in the jacobian
- Step 4: Fill in jacobian[0, 1] via `forward`, which will calculate the derivative of `f` with respect to `y`
  - `parse(func, val)` will `parse` `f` with respect to `y` and return a function, `np.sin(x) + 3*DualNumber(y)` which takes two inputs, `x` and `y` 
    - Note that it only converted `y` to a DualNumber because we are differentiating with respect to `y` in this pass
  - `parsed_func(self.value)` will evaluate the parsed expression at the value `{"x" : 0, "y" : 4}`
  - The dual part of this expression will be the derivative of `f` with respect to `y`
  - We add the `derivative` to the appropriate index in the jacobian
- Step 5: Return the Jacobian

## Licensing

We will use the *MIT License*, since we want our library to be open for anyone to modify it. This library will not provide all uses or versions of automatic differentiation available or the ones being still developed. Due to the limited timeframe of the project we only aim to provide the backbones for automatic differentiation and some of the basic algorithms in automatic differentiation. Thus we want anyone that wants to make changes to the library to fill their specific needs to be able to do so, either in a commercial setting or not.
