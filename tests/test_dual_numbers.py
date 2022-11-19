"""
This test suite (a module) runs tests for dual_numbers of the
autodiff package.
"""

import pytest

import numpy as np

# import names to test
from autodiff.utils.dual_numbers import DualNumber

class TestDualNumber:
    """Test class for dual number types"""
    def test_init(self):
        z1 = DualNumber(2, 1)
        assert z1.real == 2
        assert z1.dual == 1

    def test_addition(self):
        z1 = DualNumber(1, 2)
        z2 = DualNumber(5, 6)

        # DualNumber + DualNumber
        assert (z1 + z2).real == 6
        assert (z1 + z2).dual == 8
        assert (z2 + z1).real == 6
        assert (z2 + z1).dual == 8

        # DualNumber + int
        assert (z1 + 3).real == 4
        assert (z1 + 3).dual == 2

        # DualNumber + float
        assert (z1 + 3.0).real == 4.0
        assert (z1 + 3.0).dual == 2.0

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            z1 + "string"
            "string" + z1

    def test_reflective_addition(self):
        z1 = DualNumber(1, 2)

        # int + DualNumber
        assert (z1 + 3).real == (3 + z1).real == 4
        assert (z1 + 3).dual == (3 + z1).dual == 2

        # float + DualNumber
        assert (z1 + 3.0).real == (3.0 + z1).real == 4.0
        assert (z1 + 3.0).dual == (3.0 + z1).dual == 2.0

    def test_subtraction(self):
        z1 = DualNumber(1, 2)
        z2 = DualNumber(5, 6)

        # DualNumber + DualNumber
        assert (z1 - z2).real == -4
        assert (z1 - z2).dual == -4
        assert (z2 - z1).real == 4
        assert (z2 - z1).dual == 4

        # DualNumber + int
        assert (z1 - 3).real == -2
        assert (z1 - 3).dual == 2

        # DualNumber + float
        assert (z1 - 3.0).real == -2.0
        assert (z1 - 3.0).dual == 2

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            z1 - "string"
            "string" - z1

    def test_reflective_subtraction(self):
        z1 = DualNumber(5, 5)

        # int - DualNumber 
        assert (1 - z1).real == -4
        assert (1 - z1).dual == -5

        # float - DualNumber
        assert (1.0 - z1).real == -4.0
        assert (1.0 - z1).dual == -5.0

    def test_multiplication(self):
        z1 = DualNumber(1, 2)
        z2 = DualNumber(5, 6)

        # DualNumber * DualNumber
        assert (z1 * z2).real == 5
        assert (z1 * z2).dual == 16
        assert (z2 * z1).real == 5
        assert (z2 * z1).dual == 16

        # DualNumber * int
        assert (z1 * 3).real == 3
        assert (z1 * 3).dual == 6

        # DualNumber * float
        assert (z1 * 3.0).real == 3.0
        assert (z1 * 3.0).dual == 6.0

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            z1 * "string"
            "string" * z1

    def test_reflective_multiplication(self):
        z1 = DualNumber(1, 2)

        # int * DualNumber
        assert (z1 * 3).real == (3 * z1).real == 3
        assert (z1 * 3).dual == (3 * z1).dual == 6

        # float * DualNumber
        assert (z1 * 3.0).real == (3.0 * z1).real == 3.0
        assert (z1 * 3.0).dual == (3.0 * z1).dual == 6.0
    
    def test_true_division(self):
        z1 = DualNumber(5, 5)
        z2 = DualNumber(1, 1)

        # DualNumber / DualNumber
        assert (z1 / z2).real == 5
        assert (z1 / z2).dual == 0

        # DualNumber / int
        assert (z1 / 5).real == 1
        assert (z1 / 5).dual == 1

        # DualNumber / float
        assert (z1 / 5.0).real == 1.0
        assert (z1 / 5.0).dual == 1.0

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            z1 / "string"
            "string" / z1

    def test_reflective_true_division(self):
        z1 = DualNumber(5, 5)

        # int / DualNumber 
        assert (5 / z1).real == 1
        assert (5 / z1).dual == -1

        # float / DualNumber
        assert (5.0 / z1).real == 1.0
        assert (5.0 / z1).dual == -1.0

    def test_power(self):
        z1 = DualNumber(5, 5)

        # DualNumber ** int
        assert (z1 ** 3).real == 125
        assert (z1 ** 3).dual == 375

        # DualNumber ** float
        assert (z1 ** 3.0).real == 125.0
        assert (z1 ** 3.0).dual == 375.0

        # Handle Non-Supported Types (String)
        with pytest.raises(TypeError):
            z1 ** "string"
            "string" ** z1

        # # Dual Number ** Dual Number
        # z2 = DualNumber(1, 1)
        # assert (z1 ** z2).real == 5
        # assert (z1 ** z2).dual == (5 ** 0) * (1 * 5 + np.log(5) * 5 * 1)

    def test_reflective_power(self):
        pass

    def test_negation(self):
        z1 = DualNumber(5, 5)
         # -DualNumber
        assert (-z1).real == -5
        assert (-z1).dual == -5

    def test_repr(self):
        z1 = DualNumber(5, 5)
        # repr(DualNumber)
        assert repr(z1) == 'DualNumber(5, 5)'

    def test_equal(self):
        z1 = DualNumber(2, 2)
        z2 = DualNumber(2, 1)
        z3 = DualNumber(2, 2)

        # DualNumber == DualNumber
        assert z1 == z3
        assert not (z1 == z2)

        # Handle Int/Float Comparisons
        with pytest.raises(TypeError):
            z1 == 2
            z1 == 2.0

    def test_not_equal(self):
        z1 = DualNumber(2, 2)
        z2 = DualNumber(2, 1)
        z3 = DualNumber(2, 2)

        # DualNumber != DualNumber
        assert z1 != z2
        assert not (z1 != z3)

        # Handle Int/Float Comparisons
        with pytest.raises(TypeError):
            z1 != 2
            z1 != 2.0
