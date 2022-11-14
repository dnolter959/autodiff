"""Module contains the dual number class for automatic differentiation."""

import math
import numpy as np

class DualNumber:
    
    def __init__(self, real, dual=1):
        """class DualNumber

        A class for representing dual numbers, which are used for automatic
        differentiation.

        Parameters
        ----------
        real : float
            The real part of the dual number.
        dual : float, optional
            The dual part of the dual number. Defaults to 1.

        """
        self.real = real
        self.dual = dual

    def __add__(self, other):
        """Addition operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber or float or int
            The second dual number or a real number.

        Returns
        -------
        DualNumber
            The sum of the two dual numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))

    def __radd__(self, other):
        """Addition operator for dual numbers.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : DualNumber
            The second dual number.

        Returns
        -------
        DualNumber
            The sum of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        return self + other
    
    def __sub__(self, other):
        """Subtraction operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber or float or int
            The second dual number or a real number.

        Returns
        -------
        DualNumber
            The difference of the two dual numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real - other, self.dual)
        else:
            raise TypeError("unsupported operand type(s) for -: '{}' and '{}'".format(type(self), type(other)))
    
    def __rsub__(self, other):
        """Subtraction operator for dual numbers.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : DualNumber
            The second dual number.

        Returns
        -------
        DualNumber
            The difference of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        return other + (-self)

    def __mul__(self, other):
        """Multiplication operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber or float or int
            The second dual number or a real number.

        Returns
        -------
        DualNumber
            The product of the two dual numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real * other, self.dual * other)
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(type(self), type(other)))

    def __rmul__(self, other):
        """Multiplication operator for dual numbers.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : DualNumber 
            The second dual number.

        Returns
        -------
        DualNumber
            The product of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        return self * other

    def __truediv__(self, other):
        """Division operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber or float or int
            The second dual number or a real number.

        Returns
        -------
        DualNumber
            The quotient of the two dual numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
        elif isinstance(other, (int, float)):
            return DualNumber(self.real / other, self.dual / other)
        else:
            raise TypeError("unsupported operand type(s) for /: '{}' and '{}'".format(type(self), type(other)))

    def __rtruediv__(self, other):
        """Division operator for dual numbers.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : DualNumber
            The second dual number.

        Returns
        -------
        DualNumber
            The quotient of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a dual number or a real number.

        """
        return other * (self ** -1)

    def __pow__(self, other):
        """Power operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber or float or int
            The second dual number or a real number.

        Returns
        -------
        DualNumber
            The power of the two dual numbers.

        Raises
        ------
        TypeError
            If the other operand is a dual number with a non-zero dual part.
        TypeError
            If the other operand is not a dual number with a non-zero dual part or a real number.

        """
        if isinstance(other, DualNumber):
            if self.dual != 0:
                return DualNumber(
                    self.real**other.real,
                    self.real**(-1 + other.real) *
                    (self.dual * other.real +
                     self.real * other.dual * np.log(self.real)))
            return DualNumber(
                self.real**other.real,
                other.real * self.real**(other.real - 1) * self.dual)
        elif isinstance(other, (int, float)):
            return DualNumber(self.real ** other, other * self.real ** (other - 1) * self.dual)
        else:
            raise TypeError("unsupported operand type(s) for **: '{}' and '{}'".format(type(self), type(other)))

    def __rpow__(self, other):
        """Power operator for dual numbers.

        Parameters
        ----------
        self : int or float
            The first real number.
        other : DualNumber
            The second dual number.

        Returns
        -------
        DualNumber
            The power of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is a dual number with a non-zero dual part.
        TypeError
            If the other operand is not a dual number with a non-zero dual part or a real number.

        """
        return DualNumber(other ** self.real, other ** self.real * self.dual * math.log(other))

    def __neg__(self):
        """Negation operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The dual number.

        Returns
        -------
        DualNumber
            The negated dual number.

        """
        return DualNumber(-self.real, -self.dual)

    def __repr__(self):
        """Representation of a dual number.

        Parameters
        ----------
        self : DualNumber
            The dual number.

        Returns
        -------
        str
            The representation of the dual number.

        """
        return "DualNumber({}, {})".format(self.real, self.dual)

    def __eq__(self, other):
        """Equality operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber
            The second dual number.

        Returns
        -------
        bool
            True if the two dual numbers are equal, False otherwise.

        Raises
        ------
        TypeError
            If the other operand is not a dual number.

        """
        if not isinstance(other, DualNumber):
            raise TypeError("unsupported operand type(s) for ==: '{}' and '{}'".format(type(self), type(other)))
        return self.real == other.real and self.dual == other.dual

    def __ne__(self, other):
        """Unequality operator for dual numbers.

        Parameters
        ----------
        self : DualNumber
            The first dual number.
        other : DualNumber
            The second dual number.

        Returns
        -------
        bool
            True if the two dual numbers are not equal, False otherwise.

        Raises
        ------
        TypeError
            If the other operand is not a dual number.

        """
        return not self.__eq__(other)
