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

        assert sin(z1).real == math.sin(np.pi / 4)
        assert sin(z1).dual == math.cos(np.pi / 4)

        assert sin(z2).real == math.sin(np.pi / 4)
        assert sin(z2).dual == 0

        with pytest.raises(TypeError):
            sin("string")

    def test_cos(self):
        z1 = DualNumber(np.pi)
        z2 = np.pi

        assert cos(z1).real == math.cos(np.pi)
        assert cos(z1).dual == -math.sin(np.pi)

        assert cos(z2).real == math.cos(np.pi)
        assert cos(z2).dual == 0

        with pytest.raises(TypeError):
            cos("string")

    def test_tan(self):
        z1 = DualNumber(np.pi)
        z2 = np.pi

        assert tan(z1).real == math.tan(np.pi)
        assert tan(z1).dual == 1 / math.cos(np.pi)**2

        assert tan(z2).real == math.tan(np.pi)
        assert tan(z2).dual == 0

        with pytest.raises(TypeError):
            tan("string")

    def test_exp(self):
        z1 = DualNumber(1)
        z2 = 1

        assert exp(z1).real == np.e
        assert exp(z1).dual == np.e

        assert exp(z2).real == np.e
        assert exp(z2).dual == 0

        with pytest.raises(TypeError):
            exp("string")

    def test_log(self):
        z1 = DualNumber(5)
        z2 = 5

        assert log(z1).real == np.log(5)
        assert log(z1).dual == 1 / 5

        assert log(z2).real == np.log(5)
        assert log(z2).dual == 0

        with pytest.raises(TypeError):
            log("string")

    def test_sinh(self):
        z1 = DualNumber(5)
        z2 = 5

        assert sinh(z1).real == np.sinh(5)
        assert sinh(z1).dual == np.cosh(5)

        assert sinh(z2).real == np.sinh(5)
        assert sinh(z2).dual == 0

        with pytest.raises(TypeError):
            sinh("string")

    def test_cosh(self):
        z1 = DualNumber(5)
        z2 = 5

        assert cosh(z1).real == np.cosh(5)
        assert cosh(z1).dual == np.sinh(5)

        assert cosh(z2).real == np.cosh(5)
        assert cosh(z2).dual == 0

        with pytest.raises(TypeError):
            cosh("string")

    def test_tanh(self):
        z1 = DualNumber(0)
        z2 = 0

        assert tanh(z1).real == np.tanh(0)
        assert tanh(z1).dual == 1 - np.tanh(0)**2

        assert tanh(z2).real == np.tanh(0)
        assert tanh(z2).dual == 0

        with pytest.raises(TypeError):
            tanh("string")
