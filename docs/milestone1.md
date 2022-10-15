## Introduction
Describe the problem the software solves and why it's important to solve that problem.

## Background
Describe (briefly) the mathematical background and concepts as you see fit. You do not need to give a treatise on automatic differentiation or dual numbers. Just give the essential ideas (e.g. the chain rule, the graph structure of calculations, elementary functions, etc). Do not copy and paste any of the lecture notes. We will easily be able to tell if you did this as it does not show that you truly understand the problem at hand.

## How to Use PackageName
How do you envision that a user will interact with your package? What should they import? How can they instantiate AD objects?

Note: This section should be a mix of pseudo code and text. It should not include any actual operations yet. Remember, you have not yet written any code at this point.

f = a vector function with an input
x0 = input
ad = ADForward(f, seed_vec)
ad.eval(x0)

## Software Organization
Discuss how you plan on organizing your software package.

- What will the directory structure look like?
- What modules do you plan on including? What is their basic functionality?
- Where will your test suite live?
- How will you distribute your package (e.g. PyPI with PEP517/518 or simply setuptools)?
- Other considerations?


## Implementation
Discuss how you plan on implementing the forward mode of automatic differentiation.

- What are the core data structures?
- What classes will you implement?
    * AD 
    * ADForward(AD) 
    * DualNumber 
    * 
- What method and name attributes will your classes have?
    * DualNumber 
       - attributes: self.real, self.dual
       - methods for binary operations, __add__, __radd__, __sub__
       - methods for elementary functions
       
- What external dependencies will you rely on?
- How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?
- Be sure to consider a variety of use cases. For example, don't limit your design to scalar functions of scalar values. Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors. Don't forget that people will want to use your library in algorithms like Newton's method (among others).

## Licensing
Licensing is an essential consideration when you create new software. You should choose a suitable license for your project. A comprehensive list of licenses can be found here. The license you choose depends on factors such as what other software or libraries you use in your code (copyleft, copyright). will you have to deal with patents? How can others advertise software that makes use of your code (or parts thereof)? You may consult the following reading to aid you in choosing a license:
