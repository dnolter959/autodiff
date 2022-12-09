import inspect
import numpy as np
import re
from typing import Callable, Union
from toposort import toposort_flatten

from autodiff.utils.dual_numbers import DualNumber
from autodiff.utils.comp_graph import CompGraphNode
from autodiff.utils.auto_diff_math import *


class AutoDiff:
    """ A class to perform automatic differentiation on scalar and vector functions 

    ...

    Attributes
    ----------
    f: list of functions 
    point: stores the most recent point evaluated at 
    seed: stores the most recent seed vector
    jacobian: stores the most recent Jacobian matrix computed
    derivative: stores the most recent directional derivative computed

    Methods
    -------
    get_value(self, point):
        Evaluates f at the given point

    get_partial(self, point, var_index = None):
        Computes the partial derivatives evaluated at the given point 

    get_jacobian(self, point, mode="forward"):
        Computes the Jacobian matrix evaluated at the given point

    get_derivative(self, point, seed_vector, mode="forward"):
        Computes the directional derivative evaluated at the point in the direction and 
        magnitude of seed_vector
    """
    def __init__(self, f: Union[list, Callable, int, float]):
        """
        Constructs an AutoDiff object.

        Parameters
        ----------
        f: a single or a list of mathematical functions

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If f is not a Callable, a list of Callables, or a real number
        """

        # convert to list for ease of processing
        if not isinstance(f, list):
            f = [f]

        # check type of each function
        if sum([not isinstance(func, (Callable, int, float))
                for func in f]) > 0:
            raise TypeError("Invalid input type")

        self.f = f

        # store the computed output for the last input
        self.point = None
        self.seed = None
        self.derivative = None
        self.jacobian = None

        self.computational_graph = None

    def __str__(self):
        """ returns a description of the functions contained in the AutoDiff object """

        if len(self.f) < 1:
            return None

        if len(self.f) == 1:
            res = "AutoDiff object of a scalar function:\n"
        else:
            res = "AutoDiff object of a vector function:\n"

        for func in self.f:
            res = res + inspect.getsource(func).split(':')[1].strip() + "\n"

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
        eq = [
            a.__code__.co_code == b.__code__.co_code for a in self.f
            for b in other.f
        ]
        return sum(eq) == len(self.f)

    def _check_vector(self, vector):
        """ confirm that the passed vector is numeric and is either 1 or 2-D 
        
        Parameters
        ----------
        vector: a single or sequence of numbers
        """
        if not isinstance(vector, (int, float, list, np.ndarray)):
            raise TypeError("Invalid input type")

        if isinstance(vector, list):
            if sum([not isinstance(v, (int, float)) for v in vector]) > 0:
                raise TypeError("Invalid input type")

        if isinstance(vector, np.ndarray):
            if sum([not isinstance(v.item(), (int, float))
                    for v in vector]) > 0:
                raise TypeError("Invalid input type")
            if vector.ndim != 1:
                raise TypeError("Invalid input array dimension")

        return

    def get_value(self, point: Union[int, float, list, np.ndarray]):
        """ evaluate f at point

        Parameters
        ----------
        point: int, float, list, or numpy ndarray 
            a single or a sequence of numbers defining the point for the functions to evaluate at

        Returns
        -------
        int, float, numpy ndarray 
            the function evaluated at point 

        Raises
        ------
        TypeError
            If point is not an int, float, list, or numpy ndarray or has incorrect dimension
        """

        self._check_vector(point)

        if len(self.f) == 1:
            if isinstance(self.f[0], (int, float)):
                return self.f[0](point).real

        values = []
        for func in self.f:
            val = func(point)
            assert isinstance(val, (int, float, DualNumber,
                                    CompGraphNode)), "invalid function value"

            if isinstance(val, (int, float)):
                values += [val]
            elif isinstance(val, DualNumber):
                values += [val.real]
            else:
                values += [val.value]

        if len(values) == 1:
            return values[0]
        return values

    def get_partial(self,
                    point: Union[int, float, list, np.ndarray],
                    var_index: int = None):
        """ obtain partial derivative with respect to a coordinate, specified by the index corresponding to 
            that coordinate in the point sequence; forward mode is employed for this method

        Parameters
        ----------
        point: int, float, list, or numpy ndarray
            a single or a sequence of numbers defining the point for the functions to evaluate at
        var_index: int
            index of variable in the sequence to partially differentiate with respect to, default is None; 
            if the point is a scalar, var_index is ignored

        Returns
        -------
            int, float, numpy ndarray
            the partial derivative evaluated at the point of values

        Raises
        ------
        TypeError
            If point is not an int, float, list, or numpy ndarray or has incorrect dimension
        """

        assert var_index is None or isinstance(var_index, int)

        self._check_vector(point)

        if isinstance(point, (int, float)):
            point_dual = DualNumber(point, 1)
        elif isinstance(point, np.ndarray):
            point_dual = np.array([
                DualNumber(v.item(), 1) if ind == var_index else DualNumber(
                    v.item(), 0) for ind, v in enumerate(point)
            ])
        else:
            # the variable to differentiate w.r.t. has a dual part of 1
            point_dual = np.array([
                DualNumber(v, 1) if ind == var_index else DualNumber(v, 0)
                for ind, v in enumerate(point)
            ])

        ret = np.array([])
        for func in self.f:
            if isinstance(func, (int, float)):
                ret = np.append(ret, 0)
            elif isinstance(func(point_dual), (int, float)):
                ret = np.append(ret, 0)
            elif isinstance(func(point_dual), DualNumber):
                ret = np.append(ret, func(point_dual).dual)

        # return as a scalar if there is only 1 input
        if len(ret) == 1:
            return ret[0]
        # return as an array
        return ret

    def get_jacobian(self,
                     point: Union[int, float, list, np.ndarray],
                     mode="forward"):
        """ compute the Jacobian matrix evaluated at point using the specified mode
        
        Parameters
        ----------
        point: int, float, list, or numpy ndarray
            a single or a sequence of numbers defining the point for the functions to evaluate at
        mode: {"forward", "f", "reverse", "r"}
            option to perform automatic differentiation using forward or reverse mode; default is "forward"

        Returns
        -------
        numpy ndarray 
            the Jacobian matrix as an array

        Raises
        ------
        TypeError
            If point is not an int, float, list, or numpy ndarray or has incorrect dimension
        """

        self._check_vector(point)

        compare = point == self.point
        if ((isinstance(compare, bool) and compare
             or isinstance(compare, np.ndarray) and compare.all())
                and self.jacobian is not None):
            return self.jacobian

        self.derivative = None
        self.seed = None
        self.point = point

        if mode in ["forward", "f"]:
            jacobian = self._get_jacobian_forward(point)
        elif mode in ["reverse", "r"]:
            jacobian = self._get_jacobian_reverse(point)
        else:
            raise ValueError("Invalid mode")

        # reshape Jacobian matrix
        # Jacobian matrix should be a row vector for scalar function with multiple input
        if len(self.f) == 1 and isinstance(
                point, (list, np.ndarray)) and len(point) > 1:
            jacobian = jacobian.reshape(1, -1)
        # Jacobian matrix should be a column vector for vector function with single input
        elif (len(self.f) > 1
              and (isinstance(point, (int, float))
                   or isinstance(point,
                                 (list, np.ndarray)) and len(point) == 1)):
            jacobian = jacobian.reshape(-1, 1)

        self.jacobian = jacobian
        return self.jacobian

    def _get_jacobian_forward(self, point: Union[int, float, list,
                                                 np.ndarray]):
        """ computes the Jacobian matrix using forward mode """

        assert isinstance(point, (int, float, list, np.ndarray))

        if isinstance(point, (int, float)):
            ret = self.get_partial(point)
        else:
            if isinstance(point, list):
                point = np.array(point)

            ret = []
            for i in range(len(point)):
                # append the partial derivatives computed for each coordinate
                ret += [self.get_partial(point, var_index=i)]

        # Jacobian should always be returned as matrices
        if isinstance(ret, (int, float)):
            ret = [ret]

        jacobian = np.transpose(np.array(ret))
        return jacobian

    def _get_jacobian_reverse(self, point: Union[int, float, list,
                                                 np.ndarray]):
        """ computes the Jacobian matrix using reverse mode """

        assert isinstance(point, (int, float, list, np.ndarray))

        jacobian = []
        self.computational_graph = []
        
        # for creating an array of 0s in case of constant function or function returning constants
        if isinstance(point, (int, float)):
            shape = (1, )
        else:
            shape = (len(point), )

        for func in self.f:
            if isinstance(func, (int, float)):
                jacobian += [np.zeros(shape)]

            else:
                # forward pass
                added_nodes = {}

                # convert input to CompGraphNodes
                if isinstance(point, (int, float)):
                    input_nodes = CompGraphNode(point, added_nodes=added_nodes)
                elif isinstance(point, np.ndarray):
                    input_nodes = [
                        CompGraphNode(p.item(), added_nodes=added_nodes)
                        for p in point
                    ]
                else:
                    input_nodes = [
                        CompGraphNode(p, added_nodes=added_nodes) for p in point
                    ]

                output_node = func(input_nodes)

                if isinstance(output_node, (int, float)):
                    jacobian += [np.zeros(shape)]
                    self.computational_graph += [None]

                else:
                    # reverse pass
                    if isinstance(input_nodes, CompGraphNode):
                        input_nodes = [input_nodes]

                    # build adjacency list (a dictionary) for toposort
                    # the inputs do not have parents
                    adj_list = {n: set() for n in input_nodes}

                    # add intermediate nodes to the adjacency list
                    for node in added_nodes.values():
                        adj_list[node] = set(node.parents)

                    # toposort the computational graph
                    sorted_list = toposort_flatten(adj_list, sort=False)

                        

                    # set last node's adjoint
                    output_node.adjoint = 1

                    # compute adjoint for each node
                    for node in reversed(sorted_list):
                        if node.parents is not None and node.partials is not None:
                            for parent, partial in zip(node.parents, node.partials):
                                parent.adjoint += node.adjoint * partial

                    # the chain-rule incorporated partial is stored as the input nodes adjoint
                    jacobian += [np.array([node.adjoint for node in input_nodes])]
                    # store computational_graph
                    self.computational_graph += [adj_list]

        jacobian = np.array(jacobian)
        if jacobian.size == 1:
            jacobian = jacobian.flatten()

        return jacobian

    def get_derivative(self,
                       point: Union[int, float, list, np.ndarray],
                       seed_vector: Union[int, float, list, np.ndarray],
                       mode="forward"):
        """ calculate the directional derivative given point and the seed vector

        Parameters
        ----------
        point: int, float, list, or numpy ndarray
            a single or a sequence of numbers defining the point for the functions to evaluate at
        seed_vector: int, float, list, or numpy ndarray
            a single or a sequence of numbers defining the seed of direction
        mode: {"forward", "f", "reverse", "r"}
            option to perform automatic differentiation using forward or reverse mode; default is "forward"

        Returns
        -------
        int, float, or numpy ndarray
            the directional derivative based on the seed

        Raises 
        ------
        TypeError
            If point or seed_vector is not an int, float, list, or numpy ndarray or has incorrect dimension
        
        ValueError 
            If mode is not one of the accepted strings
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
        compare = self.point == point
        if (isinstance(compare, bool) and compare
                or isinstance(compare, np.ndarray) and compare.all()):

            # check if the seed is the same as last set of computed point
            compare = self.seed == seed_vector_arr
            if ((isinstance(compare, bool) and compare
                 or isinstance(compare, np.ndarray) and compare.all())
                    and self.derivative is not None):
                return self.derivative

        self.jacobian = self.get_jacobian(point, mode)
        self.seed = seed_vector_arr

        derivative = np.dot(self.jacobian, seed_vector_arr.reshape(-1, 1))

        if derivative.ndim == 1 and len(derivative) == 1:
            derivative = derivative[0]

        if derivative.size == 1:
            derivative = derivative.flatten()[0]

        self.derivative = derivative
        return self.derivative
