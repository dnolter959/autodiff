class CompGraphNode:
    def __init__(self, value, parents=None, partials=None, adjoint=0, added_nodes=None):

        assert isinstance(value, (int, float))
        self.value = value
        
        # each element of the lists (if not None) must be real number
        assert (parents is None or isinstance(parents, list) and 
                sum([not isinstance(x, CompGraphNode) for x in parents])==0)

        assert (partials is None or isinstance(partials, list) and 
                sum([not isinstance(x, (int, float)) for x in partials])==0)
        
        # number of partial derivatives must match number of parents
        assert (parents is None and partials is None or 
                isinstance(parents, list) and isinstance(partials, list) and 
                len(parents)==len(partials))

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
                node = CompGraphNode(self.value + other.value, parents = [self, other], 
                                     partials = [1, 1], added_nodes = self._added_nodes)

            else:
                node = CompGraphNode(self.value + other, parents = [self], 
                                 partials = [1], added_nodes = self._added_nodes)

            # add to existing nodes
            self._added_nodes[("add", self, other)] = node
            return node
        
        raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))


    def __radd__(self, other):
        """Addition operator for nodes.

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

        if isinstance(other, CompGraphNode):
            node = CompGraphNode(self.value - other.value, parents = [self, other], 
                                 partials = [1, -1], added_nodes = self._added_nodes)

            self._added_nodes[("sub", self, other)] = node
            return node

        elif isinstance(other, (int, float)):
            self._added_nodes[("sub", self, other)] = node
            return CompGraphNode(self.value - other, parents = [self], 
                                 partials = [1], added_nodes = self._added_nodes)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(
                    type(self), type(other)))


    def __rsub__(self, other):
        """Subtraction operator for nodes.

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

        
    def __neg__(self):
        """Negation operator for dual numbers.

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

        node = CompGraphNode(-self.value, parents = [self], 
                             partials = [-1], added_nodes = self._added_nodes)

        self._added_nodes[("neg", self, None)] = node

        return node 

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
                node = CompGraphNode(self.value * other.value, parents = [self, other], 
                                     partials = [other.value, self.value], added_nodes = self._added_nodes)

            else:
                node = CompGraphNode(self.value * other, parents = [self], 
                                 partials = [other], added_nodes = self._added_nodes)

            # add to existing nodes
            self._added_nodes[("mul", self, other)] = node
            return node
        
        raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(
                type(self), type(other)))


    def __rmul__(self, other):
        """Multiplication operator for nodes.

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

    # for toposort comparison
    def __lt__(self, other):
        """

        Parameters
        ----------
        self : CompGraphNode
            The node.

        Returns
        -------

        """
        return other.parents is None or self in other.parents

    def __gt__(self, other):
        """

        Parameters
        ----------
        self : CompGraphNode
            The node.

        Returns
        -------

        """
        return other < self
