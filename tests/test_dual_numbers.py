"""
This test suite (a module) runs tests for dual_numbers of the
autodiff package.
"""

import pytest

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

        # DualNumber + int
        assert (z1 + 3).real == (3 + z1).real == 4
        assert (z1 + 3).dual == (3 + z1).dual == 2

        # DualNumber + float
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
        pass

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

    def test_reflexive_multiplication(self):
        z1 = DualNumber(1, 2)

        # DualNumber * int
        assert (z1 * 3).real == (3 * z1).real == 3
        assert (z1 * 3).dual == (3 * z1).dual == 6

        # DualNumber * float
        assert (z1 * 3.0).real == (3.0 * z1).real == 3.0
        assert (z1 * 3.0).dual == (3.0 * z1).dual == 6.0

    def test_reflexive_true_division(self):
        pass

    def test_power(self):
        pass

    def test_reflexive_power(self):
        pass

    def test_negation(self):
        pass

    def test_repr(self):
        pass

    def test_equal(self):
        pass

    def test_reflective_multiplication(self):
        pass

    def test_not_equal(self):
        pass
