"""
This test suite (a module) runs tests for CompGraphNodes
of the autodiff package.
"""

import numpy as np
import pytest

from autodiff.utils.comp_graph import CompGraphNode


class TestCompGraphNode:
    """Test class for CompGraphNode"""

    def test_init(self):
        node = CompGraphNode(2)

        assert node.value == 2
        assert node.adjoint == 0
        assert node.parents is None
        assert node.partials is None
        assert len(node._added_nodes) == 0

        node2 = CompGraphNode(5, parents=[node], partials=[2], adjoint=1)

        assert node2.value == 5
        assert node2.adjoint == 1
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1

        with pytest.raises(AssertionError):
            CompGraphNode("string")

        with pytest.raises(AssertionError):
            CompGraphNode(2, parents=[node, "string"])

        with pytest.raises(AssertionError):
            CompGraphNode(2, partials=[node])

        with pytest.raises(AssertionError):
            CompGraphNode(2, partials=[5], parents=[node, node2])

    def test_addition(self):
        node = CompGraphNode(2)
        node2 = CompGraphNode(5)

        # CompGraphNode + CompGraphNode
        node3 = node + node2
        assert node3.value == 7
        assert node3.adjoint == 0
        assert len(node3.parents) == 2
        assert len(node3.partials) == 2
        assert node3.partials[0] == 1 and node3.partials[1] == 1
        assert len(node3._added_nodes) == 1

        # CompGraphNode + int
        node4 = node + 3
        assert node4.value == 5
        assert node4.adjoint == 0
        assert len(node4.parents) == 1
        assert len(node4.partials) == 1
        assert len(node4._added_nodes.keys()) == 2

        # CompGraphNode + float
        node5 = node + 3.0
        assert node5.value == 5.0
        assert node5.adjoint == 0
        assert len(node5.parents) == 1
        assert len(node5.partials) == 1
        assert len(node5._added_nodes.keys()) == 2

        assert node5._added_nodes[("add", node, 3)] == node4
        assert id(node5) == id(node4)

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            node + "string"
            "string" + node

    def test_reflective_addition(self):
        node = CompGraphNode(2)

        # int + CompGraphNode
        node2 = 3 + node
        assert node2.value == 5
        assert node2.adjoint == 0
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1
        assert len(node2._added_nodes) == 1

        # float + CompGraphNode
        node3 = 3.0 + node
        assert node3.value == 5.0
        assert node3.adjoint == 0
        assert len(node3.parents) == 1
        assert len(node3.partials) == 1
        assert node3.partials[0] == 1
        assert len(node3._added_nodes.keys()) == 1

        assert node3._added_nodes[("add", node, 3)] == node2
        assert id(node2) == id(node3)

    def test_subtraction(self):
        node = CompGraphNode(2)
        node2 = CompGraphNode(5)

        # CompGraphNode - CompGraphNode
        node3 = node - node2
        assert node3.value == -3
        assert node3.adjoint == 0
        assert len(node3.parents) == 2
        assert len(node3.partials) == 2
        assert node3.partials[0] == 1 and node3.partials[1] == -1
        assert len(node3._added_nodes) == 1

        # CompGraphNode - int
        node4 = node - 3
        assert node4.value == -1
        assert node4.adjoint == 0
        assert len(node4.parents) == 1
        assert len(node4.partials) == 1
        assert len(node4._added_nodes.keys()) == 2

        # CompGraphNode - float
        node5 = node - 3.0
        assert node5.value == -1.0
        assert node5.adjoint == 0
        assert len(node5.parents) == 1
        assert len(node5.partials) == 1
        print(node5._added_nodes)
        assert len(node5._added_nodes.keys()) == 2

        assert node5._added_nodes[("sub", node, 3)] == node4
        assert id(node5) == id(node4)

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            node - "string"
            "string" - node

    def test_reflective_subtraction(self):
        node = CompGraphNode(2)

        # int - CompGraphNode
        node2 = 3 - node
        assert node2.value == 1
        assert node2.adjoint == 0
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1
        print(node2._added_nodes)
        assert len(node2._added_nodes) == 2

        # float - CompGraphNode
        node3 = 3.0 - node
        assert node3.value == 1.0
        assert node3.adjoint == 0
        assert len(node3.parents) == 1
        assert len(node3.partials) == 1
        assert node3.partials[0] == 1
        assert len(node3._added_nodes.keys()) == 2

        print(node3._added_nodes)
        #assert node3._added_nodes[("add", 3, node)] == node2
        assert id(node2) == id(node3)

    def test_mul(self):
        node = CompGraphNode(2)
        node2 = CompGraphNode(5)

        # CompGraphNode * CompGraphNode
        node3 = node * node2
        assert node3.value == 10
        assert node3.adjoint == 0
        assert len(node3.parents) == 2
        assert len(node3.partials) == 2
        assert node3.partials[0] == 5 and node3.partials[1] == 2
        assert len(node3._added_nodes) == 1

        # CompGraphNode * int
        node4 = node * 3
        assert node4.value == 6
        assert node4.adjoint == 0
        assert len(node4.parents) == 1
        assert len(node4.partials) == 1
        assert node4.partials[0] == 3
        assert len(node4._added_nodes.keys()) == 2

        # CompGraphNode * float
        node5 = node * 3.0
        assert node5.value == 6.0
        assert node5.adjoint == 0
        assert len(node5.parents) == 1
        assert len(node5.partials) == 1
        assert len(node5._added_nodes.keys()) == 2

        assert node5._added_nodes[("mul", node, 3)] == node4
        assert id(node5) == id(node4)

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            node * "string"
            "string" * node

    def test_reflective_mul(self):
        node = CompGraphNode(2)

        # int * CompGraphNode
        node2 = 3 * node
        assert node2.value == 6
        assert node2.adjoint == 0
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1
        assert len(node2._added_nodes) == 1

        # float * CompGraphNode
        node3 = 3.0 * node
        assert node3.value == 6.0
        assert node3.adjoint == 0
        assert len(node3.parents) == 1
        assert len(node3.partials) == 1
        assert node3.partials[0] == 3
        assert len(node3._added_nodes.keys()) == 1

        assert node3._added_nodes[("mul", node, 3)] == node2
        assert id(node2) == id(node3)

    def test_div(self):
        node = CompGraphNode(2)
        node2 = CompGraphNode(5)

        # CompGraphNode / CompGraphNode
        node3 = node / node2
        assert node3.value == 0.4
        assert node3.adjoint == 0
        assert len(node3.parents) == 2
        assert len(node3.partials) == 2
        assert node3.partials[0] == 0.2 and node3.partials[1] == -0.08
        assert len(node3._added_nodes) == 0

        # CompGraphNode / int
        node4 = node / 3
        assert node4.value == 2 / 3
        assert node4.adjoint == 0
        assert len(node4.parents) == 1
        assert len(node4.partials) == 1
        assert node4.partials[0] == 1 / 3
        assert len(node4._added_nodes.keys()) == 1

        # CompGraphNode / float
        node5 = node / 3.0
        assert node5.value == 2 / 3
        assert node5.adjoint == 0
        assert len(node5.parents) == 1
        assert len(node5.partials) == 1
        assert len(node5._added_nodes.keys()) == 2

        assert node5._added_nodes[("div", node, 3)] == node4
        assert id(node5) == id(node4)

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            node / "string"
            "string" / node

    def test_reflective_div(self):
        node = CompGraphNode(2)

        # int / CompGraphNode
        node2 = 3 / node
        assert node2.value == 3 / 2
        assert node2.adjoint == 0
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1
        assert node2.partials[0] == -0.75
        assert len(node2._added_nodes) == 0

        # float / CompGraphNode
        node3 = 3.0 / node
        assert node3.value == 3 / 2
        assert node3.adjoint == 0
        assert len(node3.parents) == 1
        assert len(node3.partials) == 1
        assert node3.partials[0] == -0.75
        assert len(node3._added_nodes.keys()) == 1

        assert node3._added_nodes[("div", 3, node)] == node2
        assert id(node2) == id(node3)

    def test_pow(self):
        node = CompGraphNode(2)
        node2 = CompGraphNode(5)

        # CompGraphNode ** CompGraphNode
        node3 = node**node2
        assert node3.value == 32
        assert node3.adjoint == 0
        assert len(node3.parents) == 2
        assert len(node3.partials) == 2
        assert node3.partials[0] == 5 * 16 and node3.partials[
            1] == 32 * np.log(2)
        assert len(node3._added_nodes) == 0

        # CompGraphNode ** int
        node4 = node**3
        assert node4.value == 8
        assert node4.adjoint == 0
        assert len(node4.parents) == 1
        assert len(node4.partials) == 1
        assert node4.partials[0] == 3 * 8 / 2
        assert len(node4._added_nodes.keys()) == 1

        # CompGraphNode ** float
        node5 = node**3.0
        assert node5.value == 8
        assert node5.adjoint == 0
        assert len(node5.parents) == 1
        assert len(node5.partials) == 1
        assert len(node5._added_nodes.keys()) == 2

        assert node5._added_nodes[("pow", node, 3)] == node4
        assert id(node5) == id(node4)

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            node**"string"
            "string"**node

    def test_reflective_pow(self):
        node = CompGraphNode(2)

        # int ** CompGraphNode
        node2 = 3**node
        assert node2.value == 9
        assert node2.adjoint == 0
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1
        assert len(node2._added_nodes) == 0

        # float ** CompGraphNode
        node3 = 3.0**node
        assert node3.value == 9
        assert node3.adjoint == 0
        assert len(node3.parents) == 1
        assert len(node3.partials) == 1
        assert node3.partials[0] == 9 * np.log(3)
        assert len(node3._added_nodes.keys()) == 1

        assert node3._added_nodes[("pow", 3, node)] == node2
        assert id(node2) == id(node3)

    def test_neg(self):
        node = CompGraphNode(2)

        node2 = -node
        assert node2.value == -2
        assert node2.adjoint == 0
        assert len(node2.parents) == 1
        assert len(node2.partials) == 1
        assert node2.partials[0] == -1
        assert len(node2._added_nodes) == 0

    def test_repr(self):
        node = CompGraphNode(2)

        assert repr(node) == "CompGraphNode(2)"

    def test_less_than(self):
        node = CompGraphNode(2)
        node2 = node + 3
        node3 = 2 * node2

        assert node2 < node
        assert node3 < node2

        with pytest.raises(TypeError):
            node < 2

    def test_greater_than(self):
        node = CompGraphNode(2)
        node2 = node - 1
        node3 = 2**node2

        assert node > node2
        assert node2 > node3

        with pytest.raises(TypeError):
            node > 2
