"""Module containing overloaded functions to handle dual numbers."""

import math

from .dual_numbers import DualNumber


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
    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.real), math.cos(x.real) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.sin(x), 0)
    else:
        raise TypeError("sin() only accepts DualNumbers, ints, or floats.")


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
    if isinstance(x, DualNumber):
        return DualNumber(math.cos(x.real), -math.sin(x.real) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.cos(x), 0)
    else:
        raise TypeError("cos() only accepts DualNumbers, ints, or floats.")


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
    if isinstance(x, DualNumber):
        return DualNumber(math.tan(x.real),
                          (1 / (math.cos(x.real)**2)) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.tan(x), 0)
    else:
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
    if isinstance(x, DualNumber):
        return DualNumber(math.exp(x.real), math.exp(x.real) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.exp(x), 0)
    else:
        raise TypeError("exp() only accepts DualNumbers, ints, or floats.")


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
    if isinstance(x, DualNumber):
        return DualNumber(math.log(x.real), x.dual / x.real)
    elif isinstance(x, (int, float)):
        return DualNumber(math.log(x), 0)
    else:
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
    if isinstance(x, DualNumber):
        return DualNumber(math.sinh(x.real), math.cosh(x.real) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.sinh(x), 0)
    else:
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
    if isinstance(x, DualNumber):
        return DualNumber(math.cosh(x.real), math.sinh(x.real) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.cosh(x), 0)
    else:
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
    if isinstance(x, DualNumber):
        return DualNumber(math.tanh(x.real),
                          (1 / math.cosh(x.real)**2) * x.dual)
    elif isinstance(x, (int, float)):
        return DualNumber(math.tanh(x), 0)
    else:
        raise TypeError("tanh() only accepts DualNumbers, ints, or floats.")
