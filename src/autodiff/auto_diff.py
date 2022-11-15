##
import ast
import numpy as np
from typing import Callable, Union

from utils.dual_numbers import DualNumber
from utils.auto_diff_math import *

##
class AutoDiff:

    def __init__(self, f: Union[list, Callable]):
        if not isinstance(f, list):
            f = [f]

        if sum([not isinstance(func, Callable) for func in f])>0:
            raise ValueError("Invalid function input")

        self.f = f
        #self.variables = set(f.__code__.co_varnames)
        
        # store the last computed values
        self.curr_values = None 
        self.curr_seed = None
        self.curr_derivative = None
        self.curr_jacobian = None
        self.curr_gradient = None
    
   
    def _check_values(self, values):
        """ confirm that the passed list of variables match those in the vector function
            and their values are not None
        
        Positional arguments: 
        values: scalar or sequence of numbers
        """
        assert isinstance(values, (int, float, list, np.ndarray)), "Invalid input value type"

        if isinstance(values, list):
            if sum([not isinstance(v, (int,float)) for v in values])>0:
                raise ValueError("Invalid input value type")

        return


    def get_partial(self, values: Union[int,float,list,np.ndarray], var_index: int = None):
        """ obtain partial derivative with respect to a variable, specified by the index in 
            the sequence if there is a sequence of variables

        Positional arguments: 
        values: scalar or sequence of numbers

        Keyword arguments:
        var_index: index of variable in the sequence
        """

        self._check_values(values)
        assert isinstance(var_index, int)

        # the variable to differentiate w.r.t. has a dual part of 1
        values_dual = np.array(
                [DualNumber(v, 1) if ind == var_index else DualNumber(v,0) for 
                 ind, v in enumerate(values)])
        
        ret = np.array([])
        for func in self.f:
            ret = np.append(ret, func(values_dual).dual)

        return ret

    def get_jacobian(self, values: Union[int,float,list,np.ndarray]):
        """ the passed list of variables match those in the vector function
            and their values are not None
        
        Positional arguments: 
        values: a scalar or array of m 

        Return: the Jacobian matrix
        """
        self._check_values(values)

        if isinstance(values, (int,float)):
            values = np.array([values])
        elif isinstance(values, list):
            values = np.array(values)

        compare = values==self.curr_values
        if compare.all() and self.jacobian is not None:
            return self.jacobian
        
        
        if isinstance(values, (int, float)):
            values = np.array([values])

        ret = []
        for i in range(len(values)):
            ret += [self.get_partial(values, var_index=i)]

        self.curr_derivative = None
        self.curr_seed = None
        self.curr_values = values
        self.curr_jacobian = np.transpose(np.array(ret))

        return self.curr_jacobian

    def get_derivative(self, values: Union[int,float,list,np.ndarray], seed_vector: np.ndarray):
        """ calculate the directional derivative given values and the seed vector

        Positional arguments: 
        values: scalar or sequence of numbers
        seed_vector: array

        Return: the directional derivative based on the seed
        """


        self._check_values(values)

        compare = self.curr_values == values
        if compare.all():
            compare = self.curr_seed == seed_vector
            if compare.all() and self.curr_derivative is not None:
                return self.curr_derivative
            elif self.curr_jacobian is not None:
                self.curr_seed = seed_vector 
                self.curr_derivative = np.dot(self.curr_jacobian, seed_vector)
                return self.curr_derivative

        self.curr_jacobian = self.get_jacobian(values)
        self.curr_seed = seed_vector
        self.curr_derivative = np.dot(self.curr_jacobian, seed_vector)
        
        return self.curr_derivative

##


g = lambda x: x[0]+x[1]+x[0]*x[1]
f = lambda x: x[0]*sin(x[1])
h = lambda x: 5*x[0]**2

ad = AutoDiff([g,f, h])
#print(ad.get_partial(np.array([4,np.pi]), 1))
j = ad.get_jacobian(np.array([4,np.pi]))
#print(j)
print(np.dot(j, np.array([1,1])))
print(ad.get_derivative(np.array([4,np.pi]), np.array([1,1])))


