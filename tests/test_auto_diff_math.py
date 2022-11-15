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
        z1 = DualNumber(np.pi/4)
        z2 = np.pi/4

        assert sin(z1).real == math.sin(np.pi/4)
        assert sin(z1).dual == math.cos(np.pi/4)
        
        assert sin(z2).real == math.sin(np.pi/4)
        assert sin(z2).dual == 0

    def test_cos(self):
        z3 = DualNumber(np.pi)
        z4 = np.pi

        assert cos(z3).real == math.cos(np.pi)
        assert cos(z3).dual == -math.sin(np.pi)

        assert cos(z4).real == math.cos(np.pi)
        assert cos(z4).dual == 0
        

    def test_tan(self):
        z5 = DualNumber(np.pi)
        z6 = np.pi

        assert tan(z5).real == math.tan(np.pi)
        assert tan(z5).dual == 1/math.cos(np.pi) ** 2

        assert tan(z6).real == math.tan(np.pi)
        assert tan(z6).dual == 0

    def test_exp(self):
        z7 = DualNumber(1)
        z8 = 1

        assert exp(z7).real == np.e
        assert exp(z7).dual == np.e

        assert exp(z8).real == np.e
        assert exp(z8).dual == 0

    def test_log(self):
        z9 = DualNumber(5)
        z10 = 5

        assert log(z9).real == np.log(5)
        assert log(z9).dual == 1/5

        assert log(z10).real == np.log(5)
        assert log(z10).dual == 0

    def test_sinh(self):
        pass

    def test_cosh(self):
        pass

    def test_tanh(self):
        pass
