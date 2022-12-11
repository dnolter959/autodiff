"""Module contains the node class for automatic differentiation."""

import numpy as np

class CompGraphNode:
    """class CompGraphNode

    A class for representing nodes, which are used for automatic
    differentiation.
    """
    def __init__(self,
                 value,
                 parents=None,
                 partials=None,
                 adjoint=0,
                 added_nodes=None):
        """Constructs a CompGraphNode object.

        Parameters
        ----------
        value : float
            A a real number representing the value of the node.
        parents : float, optional
            A list of reference to the node's parent nodes; default is None.
        partials : float, optional
            a list of partial derivatives in the same order as the list parents; default is None.
        adjoint : float, optional
            The value of adjoint used in reverse pass.
        added_nodes : float, optional
            A dictionary storing the nodes that have already been added to the computational graph; default is None.

        """
        assert isinstance(value, (int, float))
        self.value = value

        # each element of the lists (if not None) must be real number
        assert (parents is None or isinstance(parents, list)
                and sum([not isinstance(x, CompGraphNode)
                         for x in parents]) == 0)

        assert (partials is None or isinstance(partials, list)
                and sum([not isinstance(x, (int, float))
                         for x in partials]) == 0)

        # number of partial derivatives must match number of parents
        assert (parents is None and partials is None
                or isinstance(parents, list) and isinstance(partials, list)
                and len(parents) == len(partials))

        self.partials = partials
        self.parents = parents

        # adjoint for reverse pass
        assert isinstance(adjoint, (int, float))
        self.adjoint = adjoint

        # dict of existing nodes identified by function (as str) and
        # the passed values
        if added_nodes is None:
            added_nodes = {}
        self._added_nodes = added_nodes

    def __add__(self, other):
        """Addition operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first node.
        other : CompGraphNode or float or int
            The second node or a real number.

        Returns
        -------
        CompGraphNode
            The sum of the two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.

        """
        if ("add", self, other) in self._added_nodes:
            return self._added_nodes.get(("add", self, other))

        if isinstance(other, (CompGraphNode, int, float)):
            if isinstance(other, CompGraphNode):
                node = CompGraphNode(self.value + other.value,
                                     parents=[self, other],
                                     partials=[1, 1],
                                     added_nodes=self._added_nodes)

            else:
                node = CompGraphNode(self.value + other,
                                     parents=[self],
                                     partials=[1],
                                     added_nodes=self._added_nodes)

            # add to existing nodes
            self._added_nodes[("add", self, other)] = node
            return node

        raise TypeError(
            "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))

    def __radd__(self, other):
        """Reflexive addition operator for nodes.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : CompGraphNode
            The second node.

        Returns
        -------
        CompGraphNode
            The sum of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.

        """
        return self + other

    def __sub__(self, other):
        """Subtraction operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first node.
        other : CompGraphNode or float or int
            The second node or a real number.

        Returns
        -------
        CompGraphNode
            The difference of the two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.

        """
        if ("sub", self, other) in self._added_nodes:
            return self._added_nodes.get(("sub", self, other))

        if isinstance(other, (CompGraphNode, int, float)):
            if isinstance(other, CompGraphNode):
                node = CompGraphNode(self.value - other.value,
                                     parents=[self, other],
                                     partials=[1, -1],
                                     added_nodes=self._added_nodes)

            else:
                node = CompGraphNode(self.value - other,
                                     parents=[self],
                                     partials=[1],
                                     added_nodes=self._added_nodes)

            # add to existing nodes
            self._added_nodes[("sub", self, other)] = node
            return node

        raise TypeError(
            "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))

    def __rsub__(self, other):
        """Reflexive subtraction operator for nodes.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : CompGraphNode
            The second node.

        Returns
        -------
        CompGraphNode
            The difference of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.

        """
        return other + (-self)

    def __mul__(self, other):
        """Multiplication operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first node.
        other : CompGraphNode or float or int
            The second node or a real number.

        Returns
        -------
        CompGraphNode
            The product of the two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.

        """

        if ("mul", self, other) in self._added_nodes:
            return self._added_nodes.get(("mul", self, other))

        if isinstance(other, (CompGraphNode, int, float)):
            if isinstance(other, CompGraphNode):
                node = CompGraphNode(self.value * other.value,
                                     parents=[self, other],
                                     partials=[other.value, self.value],
                                     added_nodes=self._added_nodes)

            else:
                node = CompGraphNode(self.value * other,
                                     parents=[self],
                                     partials=[other],
                                     added_nodes=self._added_nodes)

            # add to existing nodes
            self._added_nodes[("mul", self, other)] = node
            return node

        raise TypeError(
            "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))

    def __rmul__(self, other):
        """Reflexive multiplication operator for nodes.

        Parameters
        ----------
        self : float or int
            The first real number.
        other : CompGraphNode
            The second node.

        Returns
        -------
        CompGraphNode
            The product of the two numbers.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.

        """
        return self * other


    def __truediv__(self, other):
        """Division operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first node.
        other : CompGraphNode or float or int
            The second node or a real number.

        Returns
        -------
        CompGraphNode
            The quotient of the two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.
        """

        if ("div", self, other) in self._added_nodes:
            return self._added_nodes.get(("div", self, other))

        if isinstance(other, (CompGraphNode, int, float)):
            if isinstance(other, CompGraphNode):
                node = CompGraphNode(self.value / other.value,
                                     parents=[self, other],
                                     partials=[1/other.value, -self.value/other.value**2],
                                     added_nodes=self._added_nodes)

            else:
                node = CompGraphNode(self.value / other,
                                     parents=[self],
                                     partials=[1/other],
                                     added_nodes=self._added_nodes)

            # add to existing nodes
            self._added_nodes[("div", self, other)] = node
            return node

        raise TypeError(
            "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))


    def __rtruediv__(self, other):
        """Reflexive division operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first real number.
        other : CompGraphNode or float or int
            The second node.

        Returns
        -------
        CompGraphNode
            The quotient of the two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.
        """

        return other * (self**-1)

    def __pow__(self, other):
        """Power operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first node.
        other : CompGraphNode or float or int
            The second node or real number.

        Returns
        -------
        CompGraphNode
            The power of two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.
        """

        if ("pow", self, other) in self._added_nodes:
            return self._added_nodes.get(("pow", self, other))

        if isinstance(other, (CompGraphNode, int, float)):
            if isinstance(other, CompGraphNode):
                node = CompGraphNode(self.value**other.value,
                                     parents=[self, other],
                                     partials=[
                                       other.value * self.value**(other.value - 1),
                                       self.value**other.value * np.log(self.value)
                                     ],
                                     added_nodes=self._added_nodes)

            else:
                node = CompGraphNode(self.value**other,
                                     parents=[self],
                                     partials=[other * self.value**(other - 1)],
                                     added_nodes=self._added_nodes)

            # add to existing nodes
            self._added_nodes[("pow", self, other)] = node
            return node

        raise TypeError(
            "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))


    def __rpow__(self, other):
        """Reflexive power operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The first real number.
        other : CompGraphNode or float or int
            The second node.

        Returns
        -------
        CompGraphNode
            The power of two nodes.

        Raises
        ------
        TypeError
            If the other operand is not a node or a real number.
        """
        node = CompGraphNode(other**self.value,
                                     parents=[self],
                                     partials=[other**self.value*np.log(other)],
                                     added_nodes=self._added_nodes)

        self._added_nodes[("rpow", self, other)] = node
        return node

    def __neg__(self):
        """Negation operator for nodes.

        Parameters
        ----------
        self : CompGraphNode
            The node.

        Returns
        -------
        CompGraphNode
            The negated node.

        """
        if ("neg", self, None) in self._added_nodes:
            return self._added_nodes.get(("neg", self, None))

        node = CompGraphNode(-self.value,
                             parents=[self],
                             partials=[-1],
                             added_nodes=self._added_nodes)

        self._added_nodes[("neg", self, None)] = node

        return node

    def __repr__(self):
        """Representation of a node.

        Parameters
        ----------
        self : DualNumber
            The dual number.
        Returns
        -------
        str
            The representation of the dual number.
        """
        return "CompGraphNode({})".format(self.value)
