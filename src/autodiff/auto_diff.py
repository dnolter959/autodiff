import inspect
import numpy as np
import re
from typing import Callable, Union

from autodiff.utils.dual_numbers import DualNumber
from autodiff.utils.auto_diff_math import *

class AutoDiff:

    """ A class to perform automatic differentiation on scalar and vector functions 

    ...

    Attributes
    ----------
    f: list of functions 
    curr_point: stores the most recent point evaluated at 
    curr_seed: stores the most recent seed vector
    curr_derivative: stores the most recent directional derivative computed
    curr_jacobian: stores the most recent Jacobian matrix computed

    Methods
    -------
    get_value(self, point):
        Evaluates f at the given point

    get_partial(self, point, var_index = None):
        Computes the partial derivatives evaluated at the given point 

    get_jacobian(self, point):
        Computes the Jacobian matrix evaluated at the given point

    get_derivative(self, point, seed_vector):
        Computes the directional derivative evaluated at the point in the direction and 
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
        
        # store the last computed point
        self.curr_point = None 
        self.curr_seed = None
        self.curr_derivative = None
        self.curr_jacobian = None
   
    def __str__(self):
        """ returns a description of the functions"""

        if len(self.f) < 1:
            return None
        
        if len(self.f) == 1:
            res = "AutoDiff object of a scalar function:\n"
        else:
            res = "AutoDiff object of a vector function:\n"

        for func in self.f:
            res = res + inspect.getsource(func).split(':')[1].strip()+"\n"

        return res

    def __repr__(self):
        """ returns a representation of the object""" 
        f = [inspect.getsource(func).split('=')[1].strip() for func in self.f]
        ret = re.sub("'", "", f"AutoDiff({f})")
        return f"'{ret}'"

    def __eq__(self, other):
        """ compare equality of two AutoDiff objects 
            Two Autodiff objects are equal when they have the same vector function 

        """
        if not isinstance(other, AutoDiff):
            raise TypeError("Invalid comparison")

        if len(self.f) != len(other.f):
            return False
        
        # check if each function is the same 
        eq = [a.__code__.co_code==b.__code__.co_code for a in self.f for b in other.f]
        return sum(eq) == len(self.f)


    def _check_vector(self, vector):
        """ confirm that the passed vector is numeric and is either 1 or 2-D 
        
        Positional arguments: 
        vector: scalar or sequence of numbers
        """
        assert isinstance(vector, (int, float, list, np.ndarray)), "Invalid input value type"

        if isinstance(vector, list):
            if sum([not isinstance(v, (int,float)) for v in vector])>0:
                raise TypeError("Invalid input type")

        if isinstance(vector, np.ndarray):
            if vector.ndim != 1:
                raise TypeError("Invalid input array dimension")

        return

    def get_value(self, point: Union[int,float,list,np.ndarray]):
        """ evaluate f at point

        Positional arguments: 
        point: scalar or sequence of numeric values for the functions to evaluate at

        Return: the function value at point 
        """

        self._check_vector(point)
        
        if len(self.f) == 1:
            return self.f[0](point).real

        return [func(point).real for func in self.f]

    def get_partial(self, point: Union[int,float,list,np.ndarray], var_index: int = None):
        """ obtain partial derivative with respect to a variable, specified by the index in 
            the sequence if there is a sequence of variables

        Positional arguments: 
        point: scalar or sequence of numeric values for the functions to evaluate at

        Keyword arguments:
        var_index: index of variable in the sequence to partially differentiate with respect to

        Return: the partial derivative evaluated at the point of values
        """

        assert var_index is None or isinstance(var_index, int)

        self._check_vector(point)

        if isinstance(point, (int, float)): 
            point_dual = DualNumber(point, 1)
        else:
            # the variable to differentiate w.r.t. has a dual part of 1
            point_dual = np.array(
                    [DualNumber(v, 1) if ind == var_index else DualNumber(v,0) for 
                     ind, v in enumerate(point)])
        
        ret = np.array([])
        for func in self.f:
            if isinstance(func, (int, float)):
                ret = np.append(ret, 0)
            elif isinstance(func(point_dual), (int, float)):
                ret = np.append(ret, 0)
            else:
                ret = np.append(ret, func(point_dual).dual)

        # return as a scalar if there is only 1 input
        if len(ret) == 1:
            return ret[0]
        # return as an array 
        return ret

    def get_jacobian(self, point: Union[int,float,list,np.ndarray]):
        """ the passed list of variables match those in the vector function
            and their point are not None
        
        Positional arguments: 
        point: a scalar or array of numbers defining the point for evaluation

        Return: the Jacobian matrix as an array
        """
        self._check_vector(point)

        compare = point==self.curr_point
        if ((isinstance(compare, bool) and compare or isinstance(compare, np.ndarray) 
             and compare.all()) and self.jacobian is not None):
            return self.jacobian

        if isinstance(point, (int,float)):
            ret = self.get_partial(point)
        else:
            if isinstance(point, list):
                point = np.array(point)

            ret = []
            for i in range(len(point)):
                ret += [self.get_partial(point, var_index=i)]
        
        # Jacobian should always be returned as matrices
        if isinstance(ret, (int,float)):
            ret = [ret]

        self.curr_derivative = None
        self.curr_seed = None
        self.curr_point = point
        self.curr_jacobian = np.transpose(np.array(ret))

        return self.curr_jacobian

    def get_derivative(self, point: Union[int, float, list, np.ndarray], 
                       seed_vector: Union[int, float, list, np.ndarray]):
        """ calculate the directional derivative given point and the seed vector

        Positional arguments: 
        point: a scalar or array of numbers defining the point for evaluation
        seed_vector: a scalar or array of number defining the seed of direction

        Return: the directional derivative based on the seed
        """
        self._check_vector(point)
        self._check_vector(seed_vector)
        
        # check dimension match of n and seed by converting both to arrays
        if isinstance(seed_vector, (int, float)):
            seed_vector_arr = np.array([seed_vector])
        elif isinstance(seed_vector, list):
            seed_vector_arr = np.array(seed_vector)
        else:
            seed_vector_arr = seed_vector

        if isinstance(point, (int, float)):
            point_arr = np.array([point])
        elif isinstance(point, list):
            point_arr = np.array(point)
        else:
            point_arr = point

        if len(point_arr) != len(seed_vector_arr):
            raise ValueError("Dimension mismatch")
           
        # check if the point are the same as last set of computed point
        compare = self.curr_point == point
        if (isinstance(compare, bool) and compare or isinstance(compare, np.ndarray) 
             and compare.all()):

            # check if the seed is the same as last set of computed point
            compare = self.curr_seed == seed_vector
            if ((isinstance(compare, bool) and compare or isinstance(compare, np.ndarray) 
                 and compare.all()) and self.curr_derivative is not None): 
                return self.curr_derivative

            elif self.curr_jacobian is not None:
                self.curr_seed = seed_vector 
                self.curr_derivative = np.dot(self.curr_jacobian, seed_vector)
                if self.curr_derivative.ndim==1 and len(self.curr_derivative)==1:
                    self.curr_derivative = self.curr_derivative[0]
                return self.curr_derivative

        self.curr_jacobian = self.get_jacobian(point)
        self.curr_seed = seed_vector
        self.curr_derivative = np.dot(self.curr_jacobian, seed_vector)
        if self.curr_derivative.ndim==1 and len(self.curr_derivative)==1:
            self.curr_derivative = self.curr_derivative[0]
        
        return self.curr_derivative

