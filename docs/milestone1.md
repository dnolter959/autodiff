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

- What classes do you need and what will you implement first?

  - We will implement three classes: DualNumber, AutoDiffMath, and AutoDiff, in that order.

- What are the core data structures? How will you incorporate dual numbers?

  - The core data data structures include the inputs:

    - `function_output`:
      - Type: `List[str]`
      - Description: list of strings where each value represents the output of the function $f: R^n \rightarrow R^m$ in each dimension. Thus `function_output[i]` represents the return of function $f$ at dimension $i+1$.

    - `inputs`:
      - Type: `Dict[str, float]`
      - Description: dictionary of input values for vector $x$ to function $f: R^n \rightarrow R^m$. Key represents the name of the input variable that is used in `function_output` and value is the number we calculate the derivative at.
    - `ad`:
      - Type: `AutoDiff`
      - Description: class where the forward or backward automatic differention is performed.
  
  - Dual numbers will be used inside the `AutoDiff` class to perform forward automatic differentiation using the `DualNumber` class.

- What method and name attributes will your classes have?

  - The `DualNumber` class will have two instance variables representing the real and dual part of the dual number. It will overwrite most of the dunder methods that are used in mathmatical expressions, such as binary operations, comparison, equality, etc.

```python
class DualNumber:
    def __init__(self, real, dual=1):
        self.real=real
        self.dual=1
    
    def __add__(self, other):
        pass
        
    def __mul__(self, other):
        pass

    def __pow__(self, other):
        pass
    
    def __radd__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
    
    def __eq__(self, other):
        pass
    
    def __lt__(self, other):
        pass
    
    def __gt__(self, other):
        pass
    
    def __str__(self):
        pass

    # and more
```

The `AutoDiffMath` class is used to carry out elementary math functions for the forward mode in automatic differentiation.

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
    def log(DualNumber: x, float: base=math.e):  
        return DualNumber(np.log(x.real)/np.log(base), 1/(x.real*np.log(base))*x.dual)
               
    @staticmethod
    def exp(DualNumber: x):  
        return DualNumber(np.exp(x.real), np.exp(x.real)*x.dual)           
    
    # ... and more
```

```
f = [
    "x1+sin(x2)"
    ,"ln(x1*(x2**2))"
]
x = {'x1': 1, 'x2':pi}

f_parsed = 
    # resolve variables not being diff with respect to 
    [
     [DN(x1,1) + sin(x2),    x1+sin(DN(x2, 1)) = 1+sin(DN(x2,1))]
    ,[ln(DN(x1, 1)*(x2**2)), ln(x1*(DN(x2, 1)**2))]
] = [
     [DN(x1, 1) + 0,         1+sin(DN(x2, 1))]
    ,[ln(DN(x1, 1)*pi_sq),   ln(DN(x2, 1)**2)]
] = [
     [DN(1, 1) + 0,          1+sin(DN(pi, 1))]
    ,[ln(DN(1, 1)*pi_sq),    ln(DN(pi, 1)**2)]
] = [
     [DN(1, 1),              1+DN(cos(pi), -sin(pi)*1)]
    ,[ln(DN(pi_sq, pi_sq)),  ln(DN(pi_sq, 2*pi)]
] = [
     [DN(1, 1),              1+DN(1, 0)]
    ,[DN(pi_sq, 1),          DN(ln(pi_sq), 1/pi_sq*(2*pi))]
] = [
     [DN(1, 1),              DN(2, 0)]
    ,[DN(pi_sq, 1),          DN(ln(pi_sq), 2/pi)]
]
```

The `AudoDiff` class is the interface of the package. Users will initiate an `AutoDiff` object. The instance variables include the vector function passed as a list of strings and a seed vector. The class will implement two methods, one for forward mode and one for reverse, each taking a point $\mathbf{x}\in\mathbb{R}^m$ for evaluation. The class will also implement helper functions for forward and reverse mode, such as those to check the validity of the functions passed as strings, the correspondence between variables in $\mathbf{f}$ and $\mathbf{x}$, parsing the vector function passed as strings into an evaluable function using methods from AutoDiffMath, etc.

```python
from utils import *

class AutoDiff:
    def __init__(self, f, seed):
        self.format_check(f)
        self.f = f
        self.seed = seed
        
    @classmethod    
    def from_func(self, f):
        '''constructor for initializing with a list of functions'''
        pass
    
    def _format_check(self):
        '''check that vector function passed as strings are correctly formated'''
        # check parathesis
        # check function names 
        pass
        
    # This would be the key function for forward mode.
    # For example, parse f = "x1 - exp(-2(sin(4x1))^2)" to 
    # DN(x1) - DN.exp(-2*DN.pow(DN.sin(4 * DN(x1)), 2)).
    # This way the DualNumber object would handle all the calculation.
    def _parse(self, x, xi):
        '''parse functions expressed in string to the overloading functions 
           defined for DualNumber
           xi: differentiate with respect to xi (xj treated as constant for j!=i)
        '''
        return func
    
    def forward(self, x):
        eval(x)
        jacobian = np.array()
        # very rough psudocode - haven't thought thru things like dimensions & efficiency
        for i in len(x):
            for j in len(f):
                f_parsed = _parse(self.f[i])
                x_dual = DualNumber(x[i])
                jacobian[i,j] = f_parsed(x_dual)
        
        return jacobian
        
    def reverse(self, x):
        pass
```

- Will you need some graph class to resemble the computational graph in forward mode or maybe later for reverse mode? Note that in milestone 2 you propose an extension for your project, an example could be reverse mode.

  - We will need a graph class for the computational graph for reverse mode. We will not need a graph class for forward mode, as python implicitly carries out the calculation based on the computational graph, once the string function is correctly parsed and evaluated as a function that calls the overwritten methods from AutoDiffMath class.

Think about how your basic operator overloading template should look like. How will you deal with elementary functions like sin, sqrt, log, and exp (and many others)?

## Licensing
Licensing is an essential consideration when you create new software. You should choose a suitable license for your project. A comprehensive list of licenses can be found here. The license you choose depends on factors such as what other software or libraries you use in your code (copyleft, copyright). will you have to deal with patents? How can others advertise software that makes use of your code (or parts thereof)? You may consult the following reading to aid you in choosing a license:

