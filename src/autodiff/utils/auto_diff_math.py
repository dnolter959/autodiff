"""Module containing overloaded functions to handle dual numbers and computational graph nodes."""

import math
import numpy as np
from autodiff.utils.dual_numbers import DualNumber
from autodiff.utils.comp_graph import CompGraphNode 

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

        node = CompGraphNode(math.sin(x.value), parents = [x], 
                             partials = [math.cos(x.value)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("sin", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.real), math.cos(x.real) * x.dual)

    raise TypeError("sin() only accepts int, float, DualNumber, or CompGraphNode.")


def cos(x):
    """Computes the cosine of a real number, a DualNumber object, or a CompGraphNode object.    

    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the cosine of.
        
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The cosine of x.
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if isinstance(x, (int, float)):
        return math.cos(x)

    if isinstance(x, CompGraphNode):
        if ("cos", x, None) in x._added_nodes:
            return x._added_nodes.get(("cos", x, None))

        node = CompGraphNode(math.cos(x.value), parents = [x], 
                             partials = [-math.sin(x.value)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("cos", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.cos(x.real), -math.sin(x.real) * x.dual)

    raise TypeError("cos() only accepts int, float, DualNumber, or CompGraphNode.")


def tan(x):
    """Computes the tangent of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the tangent of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The tangent of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
        
    """
    if isinstance(x, (int, float)):
        return math.tan(x)

    if isinstance(x, CompGraphNode):
        if ("tan", x, None) in x._added_nodes:
            return x._added_nodes.get(("tan", x, None))

        node = CompGraphNode(math.tan(x.value), parents = [x], 
                             partials = [1/ (math.cos(x.value)**2)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("tan", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.tan(x.real),
                          (1 / (math.cos(x.real)**2)) * x.dual)
    raise TypeError("tan() only accepts int, float, DualNumber, or CompGraphNode.")


def exp(x):
    """Computes the exponential of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the exponential of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The exponential of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
        
    """
    if isinstance(x, (int, float)):
        return math.exp(x)

    if isinstance(x, CompGraphNode):
        if ("exp", x, None) in x._added_nodes:
            return x._added_nodes.get(("exp", x, None))

        node = CompGraphNode(math.exp(x.value), parents = [x], 
                             partials=[math.exp(x.value)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("exp", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.exp(x.real), math.exp(x.real) * x.dual)

    raise TypeError("exp() only accepts int, float, DualNumber, or CompGraphNode.")

def exp_b(x, base):
    """Computes the exponential (any base) of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the exp of of.
    base : int or float
        The base to use.
  
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The exponential of x with base defined
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if isinstance(x, (int, float)):
        return base**x.real

    if isinstance(x, CompGraphNode):
        if ("exp_b", x, base) in x._added_nodes:
            return x._added_nodes.get(("exp_b", x, base))

        node = CompGraphNode(base**x.value, parents = [x], 
                             partials=[math.log(base) * base**x.value], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("exp_b", x, base)] = node
        return node        

    if isinstance(x, DualNumber):
        return DualNumber(base**x.real, math.log(base) * base**x.real * x.dual)

    raise TypeError("log() only accepts int, float, DualNumber, or CompGraphNode.")

def log(x):
    """Computes the natural logarithm of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the natural logarithm of.
        
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The natural logarithm of x.
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if isinstance(x, (int, float)):
        return math.log(x)

    if isinstance(x, CompGraphNode):
        if ("log", x, None) in x._added_nodes:
            return x._added_nodes.get(("log", x, None))

        node = CompGraphNode(math.log(x.value), parents = [x], 
                             partials=[1/x.value], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("log", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.log(x.real), x.dual / x.real)

    raise TypeError("log() only accepts int, float, DualNumber, or CompGraphNode.")

def log_b(x, base):
    """Computes the logarithm (any base) of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the logarithm of.
    base : int or float
        The base to use.
  
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The logarithm of x with base defined
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if not isinstance(base, (int, float)):
        raise TypeError("log_b() only accepts int or float as base.")

    if isinstance(x, (int, float)):
        return math.log(x.real) / math.log(base)

    if isinstance(x, CompGraphNode):
        if ("log_b", x, base) in x._added_nodes:
            return x._added_nodes.get(("log_b", x, base))

        node = CompGraphNode(math.log(x.value) / math.log(base), 
                             parents = [x], 
                             partials=[(1 / x.value) * (1 / math.log(base))], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("log_b", x, base)] = node
        return node    

    if isinstance(x, DualNumber):
        return DualNumber(
            math.log(x.real) / math.log(base),
            (1 / x.real) * (1 / math.log(base)) * x.dual)

    raise TypeError("log() only accepts int, float, DualNumber, or CompGraphNode.")

def sinh(x):
    """Computes the hyperbolic sine of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the hyperbolic sine of.
        
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The hyperbolic sine (sinh) of x.
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if isinstance(x, (int, float)):
        return math.sinh(x)

    if isinstance(x, CompGraphNode):
        if ("sinh", x, None) in x._added_nodes:
            return x._added_nodes.get(("sinh", x, None))

        node = CompGraphNode(math.sinh(x.value), parents = [x], 
                             partials=[math.cosh(x.value)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("sinh", x, None)] = node
        return node    
    
    if isinstance(x, DualNumber):
        return DualNumber(math.sinh(x.real), math.cosh(x.real) * x.dual)

    raise TypeError("sinh() only accepts int, float, DualNumber, or CompGraphNode.")


def cosh(x):
    """Computes the hyperbolic cosine of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the hyperbolic cosine of.
        
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The hyperbolic cosine (cosh) of x.
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """

    if isinstance(x, (int, float)):
        return math.cosh(x)

    if isinstance(x, CompGraphNode):
        if ("cosh", x, None) in x._added_nodes:
            return x._added_nodes.get(("cosh", x, None))

        node = CompGraphNode(math.cosh(x.value), parents = [x], 
                             partials=[math.sinh(x.value)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("cosh", x, None)] = node
        return node    
    
    if isinstance(x, DualNumber):
        return DualNumber(math.cosh(x.real), math.sinh(x.real) * x.dual)

    raise TypeError("cosh() only accepts int, float, DualNumber, or CompGraphNode.")


def tanh(x):
    """Computes the hyperbolic tangent of a real number, a DualNumber object, or a CompGraphNode object.
    
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the hyperbolic tangent of.
        
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The hyperbolic tangent (cosh) of x.
        
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
    
    """
    if isinstance(x, (int, float)):
        return math.tanh(x)

    if isinstance(x, CompGraphNode):
        if ("tanh", x, None) in x._added_nodes:
            return x._added_nodes.get(("tanh", x, None))

        node = CompGraphNode(math.tanh(x.value), parents = [x], 
                             partials=[1/(math.cosh(x.value)**2)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("tanh", x, None)] = node
        return node    
    
    if isinstance(x, DualNumber):
        return DualNumber(math.tanh(x.real), 
                          (1/(math.cosh(x.real)**2) * x.dual))

    raise TypeError("tanh() only accepts int, float, DualNumber, or CompGraphNode.")

def sqrt(x):
    """Computes the square root of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the square root of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The sqrt of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
        
    """
    if isinstance(x, (int, float)):
        return math.sqrt(x)
    
    if isinstance(x, CompGraphNode):
        if ("sqrt", x, None) in x._added_nodes:
            return x._added_nodes.get(("sqrt", x, None))

        node = CompGraphNode(math.sqrt(x.value), parents = [x], 
                             partials=[0.5/math.sqrt(x.value)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("sqrt", x, None)] = node
        return node    
    
    if isinstance(x, DualNumber):
        return DualNumber(math.sqrt(x.real),
                          (0.5 / math.sqrt(x.real)) * x.dual)

    raise TypeError("sqrt() only accepts int, float, DualNumber, or CompGraphNode.")


def asin(x):
    """Computes the arcsine of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the arcsine of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The arcsine of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
        
    """

    if isinstance(x, (int, float)):
        if x > 1 or x < -1:
            raise ValueError("Range of values must be -1 < x < 1")
        return math.asin(x)

    if isinstance(x, CompGraphNode):
        if x.value > 1 or x.value < -1:
            raise ValueError("Range of values must be -1 < x < 1")
        if ("asin", x, None) in x._added_nodes:
            return x._added_nodes.get(("asin", x, None))

        node = CompGraphNode(math.asin(x.value), parents = [x], 
                             partials=[(1 / math.sqrt(1 - (x.value**2))) ], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("asin", x, None)] = node
        return node        

    if isinstance(x, DualNumber):
        if x.real > 1 or x.real < -1:
            raise ValueError("Range of values must be -1 < x < 1")
        return DualNumber(np.arcsin(x.real),
                          (1 / np.sqrt(1 - (x.real**2))) * x.dual)

    raise TypeError("asin() only accepts int, float, DualNumber, or CompGraphNode.")


def acos(x):
    """Computes the arccosine of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the arccosine of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The arccosine of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
        
    """
    if isinstance(x, (int, float)):
        if x > 1 or x < -1:
            raise ValueError("Range of values must be -1 < x < 1")
        return math.acos(x)

    if isinstance(x, CompGraphNode):
        if x.value > 1 or x.value < -1:
            raise ValueError("Range of values must be -1 < x < 1")
        if ("acos", x, None) in x._added_nodes:
            return x._added_nodes.get(("acos", x, None))

        node = CompGraphNode(math.acos(x.value), parents = [x], 
                             partials=[-1 * (1 / math.sqrt(1 - x.value**2))], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("acos", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        if x.real > 1 or x.real < -1:
            raise ValueError("Range of values must be -1 < x < 1")
        return DualNumber(math.acos(x.real),
                          -1 * (1 / math.sqrt(1 - x.real**2)) * x.dual)

    raise TypeError("acos() only accepts int, float, DualNumber, or CompGraphNode.")


def atan(x):
    """Computes the arctan of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the arctangent of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The arctangent of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode
        
    """
    if isinstance(x, (int, float)):
        return math.atan(x)

    if isinstance(x, CompGraphNode):
        if ("atan", x, None) in x._added_nodes:
            return x._added_nodes.get(("atan", x, None))

        node = CompGraphNode(math.atan(x.value), parents = [x], 
                             partials=[1 / (1 + x.value**2)], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("atan", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(math.atan(x.real), 1 / (1 + x.real**2) * x.dual)
    raise TypeError("atan() only accepts int, float, DualNumber, or CompGraphNode.")


def logistic(x):
    """Computes the logistic (sigmoid) of a real number, a DualNumber object, or a CompGraphNode object.
        
    Parameters
    ----------
    x : int, float, DualNumber, or CompGraphNode
        The value to compute the sigmoid of.
            
    Returns
    -------
    int, float, DualNumber, or CompGraphNode
        The sigmoid of x.
            
    Raises
    ------
    TypeError
        If x is not a int, float, DualNumber, or CompGraphNode

    """
    if isinstance(x, (int, float)):
        return 1 / (1 + math.exp(-x.real))

    if isinstance(x, CompGraphNode):
        if ("logistic", x, None) in x._added_nodes:
            return x._added_nodes.get(("logistic", x, None))

        node = CompGraphNode(1 / (1 + math.exp(-x.value)), parents = [x], 
                             partials=[(math.exp(x.value) /
                           ((math.exp(x.value) + 1) *
                            (math.exp(x.value) + 1)))], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("logistic", x, None)] = node
        return node

    if isinstance(x, DualNumber):
        return DualNumber(1 / (1 + math.exp(-x.real)),
                          (math.exp(x.real) /
                           ((math.exp(x.real) + 1) *
                            (math.exp(x.real) + 1))) * x.dual)

    raise TypeError(
            "logistic() only accepts int, float, DualNumber, or CompGraphNode.")
