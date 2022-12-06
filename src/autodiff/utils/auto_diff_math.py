"""Module containing overloaded functions to handle dual numbers."""

import math
import numpy as np
from autodiff.utils.dual_numbers import DualNumber
from autodiff.utils.comp_graph import CompGraphNode 

# def _add_to_existing_nodes(child, parent, child_func, child_val_1, child_val_2=None):

#     existing_nodes = parent._exsiting_nodes

#     if (function, val_1, val_2) in existing_nodes:
#         return existing_nodes.get(child_func, child_val_1, child_val_2)
#         node._parents += [parent]

#     else:
#         existing_nodes[(child_func, child_val_1, child_val_2)] = node

#     return

def sin(x):
    """Computes the sine of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the sine of.
        
    Returns
    -------
    DualNumber or int or float
        The sine of x.
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
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

    raise TypeError("sin() only accepts DualNumbers, ints, or floats.")

    # if isinstance(x, DualNumber):
    #     return DualNumber(math.sin(x.real), math.cos(x.real) * x.dual)
    # elif isinstance(x, (int, float)):
    #     return DualNumber(math.sin(x), 0)
    # else:
    #     raise TypeError("sin() only accepts DualNumbers, ints, or floats.")

def cos(x):
    """Computes the cosine of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the cosine of.
        
    Returns
    -------
    DualNumber or int or float
        The cosine of x.
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
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

    raise TypeError("cos() only accepts DualNumbers, ints, or floats.")


    # if isinstance(x, DualNumber):
    #     return DualNumber(math.cos(x.real), -math.sin(x.real) * x.dual)
    # elif isinstance(x, (int, float)):
    #     return DualNumber(math.cos(x), 0)
    # else:
    #     raise TypeError("cos() only accepts DualNumbers, ints, or floats.")


def tan(x):
    """Computes the tangent of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the tangent of.
            
    Returns
    -------
    DualNumber or int or float
        The tangent of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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
    raise TypeError("tan() only accepts DualNumbers, ints, or floats.")


def exp(x):
    """Computes the exponential of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the exponential of.
            
    Returns
    -------
    DualNumber or int or float
        The exponential of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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

    raise TypeError("exp() only accepts DualNumbers, ints, or floats.")

def exp_b(x, base):
    """Computes the exponential (any base) of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the exp of of.
    base : int or float
        The base to use.
  
    Returns
    -------
    DualNumber or int or float
        The exponential of x with base defined
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
    """
    if isinstance(x, (int, float)):
        return base**x.real

    if isinstance(x, CompGraphNode):
        if ("exp_b", x, None) in x._added_nodes:
            return x._added_nodes.get(("exp_b", x, None))

        node = CompGraphNode(base**x.value, parents = [x], 
                             partials=[math.log(base) * base**x.value], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("exp_b", x, None)] = node
        return node        

    if isinstance(x, DualNumber):
        return DualNumber(base**x.real, math.log(base) * base**x.real * x.dual)

    raise TypeError("log() only accepts DualNumbers, ints, or floats.")

def log(x):
    """Computes the natural logarithm of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the natural logarithm of.
        
    Returns
    -------
    DualNumber or int or float
        The natural logarithm of x.
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
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

    raise TypeError("log() only accepts DualNumbers, ints, or floats.")

def log_b(x, base):
    """Computes the logarithm (any base) of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the logarithm of.
    base : int or float
        The base to use.
  
    Returns
    -------
    DualNumber or int or float
        The logarithm of x with base defined
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
    """
    if isinstance(x, (int, float)):
        return math.log(x.real) / math.log(base)

    if isinstance(x, CompGraphNode):
        if ("log_b", x, None) in x._added_nodes:
            return x._added_nodes.get(("log_b", x, None))

        node = CompGraphNode(math.log(x.value) / math.log(base), 
                             parents = [x], 
                             partials=[(1 / x.value) * (1 / math.log(base))], 
                             added_nodes = x._added_nodes)

        x._added_nodes[("log_b", x, None)] = node
        return node    

    if isinstance(x, DualNumber):
        return DualNumber(
            math.log(x.real) / math.log(base),
            (1 / x.real) * (1 / math.log(base)) * x.dual)

    raise TypeError("log() only accepts DualNumbers, ints, or floats.")

def sinh(x):
    """Computes the hyperbolic sine of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the hyperbolic sine of.
        
    Returns
    -------
    DualNumber or int or float
        The hyperbolic sine (sinh) of x.
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
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

    raise TypeError("sinh() only accepts DualNumbers, ints, or floats.")


def cosh(x):
    """Computes the hyperbolic cosine of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the hyperbolic cosine of.
        
    Returns
    -------
    DualNumber or int or float
        The hyperbolic cosine (cosh) of x.
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
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

    raise TypeError("cosh() only accepts DualNumbers, ints, or floats.")


def tanh(x):
    """Computes the hyperbolic tangent of a DualNumber or a numpy array of DualNumbers.
    
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the hyperbolic tangent of.
        
    Returns
    -------
    DualNumber or int or float
        The hyperbolic tangent (cosh) of x.
        
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
    
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

    raise TypeError("tanh() only accepts DualNumbers, ints, or floats.")

def sqrt(x):
    """Computes the square root of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the square root of.
            
    Returns
    -------
    DualNumber or int or float
        The sqrt of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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

    raise TypeError("sqrt() only accepts DualNumbers, ints, or floats.")


def asin(x):
    """Computes the arcsine of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the arcsine of.
            
    Returns
    -------
    DualNumber or int or float
        The arcsine of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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

    raise TypeError("asin() only accepts DualNumbers, ints, or floats.")


def acos(x):
    """Computes the arccosine of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the arccosine of.
            
    Returns
    -------
    DualNumber or int or float
        The arccosine of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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

    raise TypeError("acos() only accepts DualNumbers, ints, or floats.")


def atan(x):
    """Computes the arctan of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the arctangent of.
            
    Returns
    -------
    DualNumber or int or float
        The arctangent of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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
    raise TypeError("atan() only accepts DualNumbers, ints, or floats.")


def logistic(x):
    """Computes the logistic (sigmoid) of a DualNumber or a numpy array of DualNumbers.
        
    Parameters
    ----------
    x : DualNumber or int or float
        The value to compute the sigmoid of.
            
    Returns
    -------
    DualNumber or int or float
        The sigmoid of x.
            
    Raises
    ------
    TypeError
        If x is not a DualNumber or int or float.
        
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
            "logistic() only accepts DualNumbers, ints, or floats.")
