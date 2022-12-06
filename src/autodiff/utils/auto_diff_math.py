"""Module containing overloaded functions to handle dual numbers."""

import math
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

    # if isinstance(x, DualNumber):
    #     return DualNumber(math.exp(x.real), math.exp(x.real) * x.dual)
    # elif isinstance(x, (int, float)):
    #     return DualNumber(math.exp(x), 0)
    # else:
    #     raise TypeError("exp() only accepts DualNumbers, ints, or floats.")


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
