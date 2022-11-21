import ast
import inspect
import numpy as np
from typing import Callable, Union

from utils.dual_numbers import DualNumber
from utils.auto_diff_math import *

class AutoDiff:

    """ A class to perform automatic differentiation on scalar and vector functions 

    ...

    Attributes
    ----------
    f: list of functions 
    curr_values: stores the most recent set of values evaluated at 
    curr_seed: stores the most recent seed vector
    curr_derivative: stores the most recent directional derivative computed
    curr_jacobian: stores the most recent Jacobian matrix computed

    Methods
    -------
    get_partial(self, values, var_index = None):
        Computes the partial derivative evaluated at values 

    get_jacobian(self, values):
        Computes the Jacobian matrix evaluated at values

    get_derivative(self, values, seed_vector):
        Computes the directional derivative evaluated at values in the direction and 
        magnitude of seed_vector
    """

    def __init__(self, f: Union[list, Callable, int, float]):
        """
        Constructs attributes for an AutoDiff object.

        Positional arguments:
        f: single or a sequence of mathematical functions
        """

        # convert to list for ease of processing
        if not isinstance(f, list):
            f = [f]
        
        # check type of each function 
        if sum([not isinstance(func, (Callable, int, float)) for func in f])>0:
            raise TypeError("Invalid input type")

        self.f = f
        
        # store the last computed values
        self.curr_values = None 
        self.curr_seed = None
        self.curr_derivative = None
        self.curr_jacobian = None
   
    def __str__(self):
        """ return a description of the functions"""

        if len(self.f) < 1:
            return None
        
        if len(self.f) == 1:
            res = "AutoDiff object of a scalar function:\n"
        else:
            res = "AutoDiff object of a vector function:\n"

        for func in self.f:
            res = res + inspect.getsource(func).split(':')[1].strip()+"\n"

        return res

    def _check_values(self, values):
        """ confirm that the passed list of variables match those in the vector function
            and their values are not None
        
        Positional arguments: 
        values: scalar or sequence of numbers
        """
        assert isinstance(values, (int, float, list, np.ndarray)), "Invalid input value type"

        if isinstance(values, list):
            if sum([not isinstance(v, (int,float)) for v in values])>0:
                raise TypeError("Invalid input value type")

        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise TypeError("Invalid input array dimension")

        return


    def get_partial(self, values: Union[int,float,list,np.ndarray], var_index: int = None):
        """ obtain partial derivative with respect to a variable, specified by the index in 
            the sequence if there is a sequence of variables

        Positional arguments: 
        values: scalar or sequence of numeric values for the functions to evaluate at

        Keyword arguments:
        var_index: index of variable in the sequence to partially differentiate with respect to

        Return: the partial derivative evaluated at the point of values
        """

        assert var_index is None or isinstance(var_index, int)

        self._check_values(values)

        if isinstance(values, (int, float)): 
            values_dual = DualNumber(values, 1)
        else:
            # the variable to differentiate w.r.t. has a dual part of 1
            values_dual = np.array(
                    [DualNumber(v, 1) if ind == var_index else DualNumber(v,0) for 
                     ind, v in enumerate(values)])
        
        ret = np.array([])
        for func in self.f:
            if isinstance(func, (int, float)):
                ret = np.append(ret, 0)
            elif isinstance(func(values_dual), (int, float)):
                ret = np.append(ret, 0)
            else:
                ret = np.append(ret, func(values_dual).dual)
        if len(ret) == 1:
            return ret[0]
        
        return ret

    def get_jacobian(self, values: Union[int,float,list,np.ndarray]):
        """ the passed list of variables match those in the vector function
            and their values are not None
        
        Positional arguments: 
        values: a scalar or array of numbers defining the point for evaluation

        Return: the Jacobian matrix as an array
        """
        self._check_values(values)

        compare = values==self.curr_values
        if ((isinstance(compare, bool) and compare or isinstance(compare, np.ndarray) 
             and compare.all()) and self.jacobian is not None):
            return self.jacobian

        if isinstance(values, (int,float)):
            ret = self.get_partial(values)
        else:
            if isinstance(values, list):
                values = np.array(values)

            ret = []
            for i in range(len(values)):
                ret += [self.get_partial(values, var_index=i)]

        self.curr_derivative = None
        self.curr_seed = None
        self.curr_values = values
        self.curr_jacobian = np.transpose(np.array(ret))

        return self.curr_jacobian

    def get_derivative(self, values: Union[int, float, list, np.ndarray], 
                       seed_vector: Union[int, float, list, np.ndarray]):
        """ calculate the directional derivative given values and the seed vector

        Positional arguments: 
        values: a scalar or array of numbers defining the point for evaluation
        seed_vector: a scalar or array of number defining the seed of direction

        Return: the directional derivative based on the seed
        """
        self._check_values(values)
        
        if not isinstance(seed_vector, (int, float, np.ndarray)):
            raise TypeError("Invalid input type of seed")
        if isinstance(seed_vector, np.ndarray) and seed_vector.ndim != 1:
            raise ValueError("Invalid dimension of seed")

        # convert single value of seed to array
        if isinstance(seed_vector, (int, float)):
            seed_vector = np.array([seed_vector])
        
        # check dimension match of n and seed
        if isinstance(values, (int, float)):
            values_arr = np.array([values])
        else:
            values_arr = values
        if len(values_arr) != len(seed_vector):
            raise ValueError("Dimension mismatch")
            
        # check if the values are the same as last set of computed values
        compare = self.curr_values == values
        if (isinstance(compare, bool) and compare or isinstance(compare, np.ndarray) 
             and compare.all()):

            compare = self.curr_seed == seed_vector
            if ((isinstance(compare, bool) and compare or isinstance(compare, np.ndarray) 
                 and compare.all()) and self.curr_derivative is not None): 
                return self.curr_derivative

            elif self.curr_jacobian is not None:
                self.curr_seed = seed_vector 
                self.curr_derivative = np.dot(self.curr_jacobian, seed_vector)
                return self.curr_derivative

        self.curr_jacobian = self.get_jacobian(values)
        self.curr_seed = seed_vector
        self.curr_derivative = np.dot(self.curr_jacobian, seed_vector)
        
        return self.curr_derivative
