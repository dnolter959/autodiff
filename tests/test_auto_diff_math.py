"""
This test suite (a module) runs tests for auto_diff_math of the
autodiff package.
"""

import pytest
import math
import numpy as np

# import names to test
from autodiff.utils.auto_diff_math import *


class TestAutoDiffMath:
    """Test class for dual number types"""

    def test_sin(self):
        z1 = DualNumber(np.pi / 4)
        z2 = np.pi / 4
        z3 = CompGraphNode(np.pi / 4)

        assert sin(z1).real == math.sin(np.pi / 4)
        assert sin(z1).dual == math.cos(np.pi / 4)

        assert sin(z2) == math.sin(np.pi / 4)

        z4 = sin(z3)
        assert z4.value == math.sin(np.pi / 4)
        assert z4.parents[0] == z3
        assert z4.partials[0] == math.cos(np.pi / 4)

        assert z4._added_nodes[("sin", z3, None)] == z4
        assert id(z4) == id(sin(z3))

        with pytest.raises(TypeError):
            sin("string")

    def test_cos(self):
        z1 = DualNumber(np.pi)
        z2 = np.pi
        z3 = CompGraphNode(np.pi)

        assert cos(z1).real == math.cos(np.pi)
        assert cos(z1).dual == -math.sin(np.pi)

        assert cos(z2) == math.cos(np.pi)

        z4 = cos(z3)
        assert z4.value == math.cos(np.pi)
        assert z4.parents[0] == z3
        assert z4.partials[0] == -math.sin(np.pi)

        assert z4._added_nodes[("cos", z3, None)] == z4
        assert id(z4) == id(cos(z3))

        with pytest.raises(TypeError):
            cos("string")

    def test_tan(self):
        z1 = DualNumber(np.pi)
        z2 = np.pi
        z3 = CompGraphNode(np.pi)

        assert tan(z1).real == math.tan(np.pi)
        assert tan(z1).dual == 1 / math.cos(np.pi)**2

        assert tan(z2) == math.tan(np.pi)

        z4 = tan(z3)
        assert z4.value == math.tan(np.pi)
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 / math.cos(np.pi)**2

        assert z4._added_nodes[("tan", z3, None)] == z4
        assert id(z4) == id(tan(z3))

        with pytest.raises(TypeError):
            tan("string")

    def test_exp(self):
        z1 = DualNumber(1)
        z2 = 1
        z3 = CompGraphNode(1)

        assert exp(z1).real == np.e
        assert exp(z1).dual == np.e

        assert exp(z2) == np.e

        z4 = exp(z3)
        assert z4.value == np.e
        assert z4.parents[0] == z3
        assert z4.partials[0] == np.e

        assert z4._added_nodes[("exp", z3, None)] == z4
        assert id(z4) == id(exp(z3))

        with pytest.raises(TypeError):
            exp("string")

    def test_log(self):
        z1 = DualNumber(5)
        z2 = 5
        z3 = CompGraphNode(5)

        assert log(z1).real == np.log(5)
        assert log(z1).dual == 1 / 5

        assert log(z2) == np.log(5)

        z4 = log(z3)
        assert z4.value == np.log(5)
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 / 5

        assert z4._added_nodes[("log", z3, None)] == z4
        assert id(z4) == id(log(z3))

        with pytest.raises(TypeError):
            log("string")

    def test_sinh(self):
        z1 = DualNumber(5)
        z2 = 5
        z3 = CompGraphNode(5)

        assert sinh(z1).real == pytest.approx(np.sinh(5))
        assert sinh(z1).dual == pytest.approx(np.cosh(5))

        assert sinh(z2) == pytest.approx(np.sinh(5))

        z4 = sinh(z3)
        assert z4.value == pytest.approx(np.sinh(5))
        assert z4.parents[0] == z3
        assert z4.partials[0] == pytest.approx(np.cosh(5))

        assert z4._added_nodes[("sinh", z3, None)] == z4
        assert id(z4) == id(sinh(z3))

        with pytest.raises(TypeError):
            sinh("string")

    def test_cosh(self):
        z1 = DualNumber(5)
        z2 = 5
        z3 = CompGraphNode(5)

        assert cosh(z1).real == pytest.approx(np.cosh(5))
        assert cosh(z1).dual == pytest.approx(np.sinh(5))

        assert cosh(z2) == pytest.approx(np.cosh(5))

        z4 = cosh(z3)
        assert z4.value == pytest.approx(np.cosh(5))
        assert z4.parents[0] == z3
        assert z4.partials[0] == pytest.approx(np.sinh(5))

        assert z4._added_nodes[("cosh", z3, None)] == z4
        assert id(z4) == id(cosh(z3))

        with pytest.raises(TypeError):
            cosh("string")

    def test_tanh(self):
        z1 = DualNumber(0)
        z2 = 0
        z3 = CompGraphNode(0)

        assert tanh(z1).real == np.tanh(0)
        assert tanh(z1).dual == 1 - np.tanh(0)**2

        assert tanh(z2) == np.tanh(0)

        z4 = tanh(z3)
        assert z4.value == np.tanh(0)
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 - np.tanh(0)**2

        assert z4._added_nodes[("tanh", z3, None)] == z4
        assert id(z4) == id(tanh(z3))

        with pytest.raises(TypeError):
            tanh("string")

    def test_asin(self):
        z1 = DualNumber(0.5)
        z2 = 0.5
        z3 = CompGraphNode(0.5)

        assert asin(z1).real == math.asin(0.5)
        assert asin(z1).dual == 1 / np.sqrt(1 - 0.5**2)

        assert asin(z2) == math.asin(0.5)

        z4 = asin(z3)
        assert z4.value == math.asin(0.5)
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 / np.sqrt(1 - 0.5**2)

        assert z4._added_nodes[("asin", z3, None)] == z4
        assert id(z4) == id(asin(z3))

        with pytest.raises(TypeError):
            asin("string")

    def test_acos(self):
        z1 = DualNumber(0.5)
        z2 = 0.5
        z3 = CompGraphNode(0.5)

        assert acos(z1).real == math.acos(0.5)
        assert acos(z1).dual == -1 / np.sqrt(1 - 0.5**2)

        assert acos(z2) == math.acos(0.5)

        z4 = acos(z3)
        assert z4.value == math.acos(0.5)
        assert z4.parents[0] == z3
        assert z4.partials[0] == -1 / np.sqrt(1 - 0.5**2)

        assert z4._added_nodes[("acos", z3, None)] == z4
        assert id(z4) == id(acos(z3))

        with pytest.raises(TypeError):
            acos("string")

    def test_atan(self):
        z1 = DualNumber(0.5)
        z2 = 0.5
        z3 = CompGraphNode(0.5)

        assert atan(z1).real == math.atan(0.5)
        assert atan(z1).dual == 1 / (1 + 0.5**2)

        assert atan(z2) == math.atan(0.5)

        z4 = atan(z3)
        assert z4.value == math.atan(0.5)
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 / (1 + 0.5**2)

        assert z4._added_nodes[("atan", z3, None)] == z4
        assert id(z4) == id(atan(z3))

        with pytest.raises(TypeError):
            atan("string")

    def test_log_b(self):
        z1 = DualNumber(4)
        z2 = 4
        z3 = CompGraphNode(4)

        assert log_b(z1, 2).real == 2
        assert log_b(z1, 2).dual == 1 / (4 * np.log(2))

        assert log_b(z2, 2) == 2

        z4 = log_b(z3, 2)
        assert z4.value == 2
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 / (4 * np.log(2))

        assert z4._added_nodes[("log_b", z3, 2)] == z4
        assert id(z4) == id(log_b(z3, 2))

        with pytest.raises(TypeError):
            log_b("string", 2)

    def test_exp_b(self):
        z1 = DualNumber(2)
        z2 = 2
        z3 = CompGraphNode(2)

        assert exp_b(z1, 2).real == 4
        assert exp_b(z1, 2).dual == 4 * np.log(2)

        assert exp_b(z2, 2) == 4

        z4 = exp_b(z3, 2)
        assert z4.value == 4
        assert z4.parents[0] == z3
        assert z4.partials[0] == 4 * np.log(2)

        assert z4._added_nodes[("exp_b", z3, 2)] == z4
        assert id(z4) == id(exp_b(z3, 2))

        with pytest.raises(TypeError):
            exp_b("string", 2)

    def test_sqrt(self):
        z1 = DualNumber(4)
        z2 = 4
        z3 = CompGraphNode(4)

        assert sqrt(z1).real == 2
        assert sqrt(z1).dual == 1 / (2 * np.sqrt(4))

        assert sqrt(z2) == 2

        z4 = sqrt(z3)
        assert z4.value == 2
        assert z4.parents[0] == z3
        assert z4.partials[0] == 1 / (2 * np.sqrt(4))

        assert z4._added_nodes[("sqrt", z3, None)] == z4
        assert id(z4) == id(sqrt(z3))

        with pytest.raises(TypeError):
            sqrt("string")

    def test_logistic(self):
        z1 = DualNumber(0)
        z2 = 0
        z3 = CompGraphNode(0)

        assert logistic(z1).real == 0.5
        assert logistic(z1).dual == 0.25

        assert logistic(z2) == 0.5

        z4 = logistic(z3)
        assert z4.value == 0.5
        assert z4.parents[0] == z3
        assert z4.partials[0] == 0.25

        assert z4._added_nodes[("logistic", z3, None)] == z4
        assert id(z4) == id(logistic(z3))

        with pytest.raises(TypeError):
            logistic("string")
