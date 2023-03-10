import pytest
import math
import numpy as np
from pytest import approx

# import names to test
from autodiff.auto_diff import AutoDiff
from autodiff.utils.dual_numbers import DualNumber
from autodiff.utils.auto_diff_math import *


class TestAutoDiff:
    def test_construction(self):
        # initialize with None
        with pytest.raises(TypeError):
            ad = AutoDiff(None)

        # initialize with non function or not list of functions
        with pytest.raises(TypeError):
            ad = AutoDiff("f(x)=x+1")

        # initialize with scalar function
        g = lambda x: x[0] + x[1] + x[0] * x[1]
        ad = AutoDiff(g)
        assert ad.f == [g]

        # initialize with scalar function which is a constant
        g = 5
        ad = AutoDiff(g)
        assert ad.f == [g]

        # initialize with list of functions
        f = lambda x: x[0] * sin(x[1])
        h = lambda x: 5 * x[0]**2
        ad = AutoDiff([f, h])
        assert ad.f == [f, h]

        # initialize with list of function and constants
        ad = AutoDiff([f, h, 5])
        assert ad.f == [f, h, 5]

        # initialize with list of function and string
        with pytest.raises(TypeError):
            ad = AutoDiff([f, h, "a+1"])

    def test_eq(self):
        f = lambda x: x + 1
        g = lambda y: y + 1

        ad1 = AutoDiff(f)
        ad2 = AutoDiff(g)
        assert ad1 == ad2

        with pytest.raises(TypeError):
            ad1 == "AutoDiff Obj"

        h = lambda x: x + 1 + 1
        ad3 = AutoDiff(h)
        assert ad1 != ad3

        ad4 = AutoDiff([f, h])
        assert ad1 != ad4

    def test_str(self):
        # change f attribute of AD object directly
        f = lambda x: x + 1
        ad = AutoDiff([f])
        ad.f = None
        assert str(ad) == ""
        ad.f = f
        assert str(ad) == ""
        ad.f = []
        assert str(ad) == ""

        # initialize with scalar function
        g = lambda x: x[0] + x[1] + x[0] * x[1]
        ad = AutoDiff(g)
        assert str(ad).strip(
        ) == "AutoDiff object of a scalar function:\nx[0] + x[1] + x[0] * x[1]"

        # initialize with vector function
        f = lambda x: x[0] * sin(x[1])
        g = lambda x: x[0] + x[1] + x[0] * x[1]
        h = lambda x: 5 * x[0]**2
        ad = AutoDiff([f, g, h])
        assert str(ad).strip() == (
            "AutoDiff object of a vector function:\n" +
            "x[0] * sin(x[1])\nx[0] + x[1] + x[0] * x[1]\n5 * x[0]**2")

    def test_get_value(self):
        # scalar constant function with m=1
        f = 3
        x = 1.5
        assert AutoDiff(f).get_value(x) == 3

        # scalar constant function specified as lambda function with m=1
        f = lambda x: 3
        x = 1.5
        assert AutoDiff(f).get_value(x) == 3

        # vector constant function with m=1
        f = 3
        g = 2
        x = 1.5
        compare = AutoDiff([f, g]).get_value(x) == np.array([3, 2])
        assert compare.all()

        # scalar function with m=1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        assert AutoDiff(f).get_value(
            x) == -x + np.cos(x) * np.sin(x) + 5 * x**4

        # scalar function with m=2
        f = lambda x: x[0] * sin(x[1])
        x = np.array([1, math.pi])
        assert AutoDiff(f).get_value(x) == x[0] * np.sin(x[1])

        # scalar function with a nested list (invalid input)
        f = lambda x: x[0] * sin(x[1])
        with pytest.raises(TypeError):
            val = AutoDiff(f).get_value([1, [math.pi]])

        # scalar function with 2-d array (invalid input)
        with pytest.raises(TypeError):
            val = AutoDiff(f).get_value(np.array([[1], [math.pi]]))

        # vector function with m=1
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        h = lambda x: 5
        x = 10
        f_v = np.exp(x) * (-x**(-1 / 2))
        g_v = np.cos(x) + np.log(x)
        h_v = 5
        assert np.allclose(AutoDiff([f, g, h]).get_value(x),
                           np.array([f_v, g_v, h_v]),
                           rtol=1e-15,
                           atol=1e-15)

        # vector function with m=3
        x = np.array([-1, 10, 105.5])
        f = lambda x: exp(x[1]) * (-x[2]**(-1 / 2))
        g = lambda x: cos(x[0]) + log(x[1]) * x[2]
        h = lambda x: 5 + x[0]
        f_v = np.exp(x[1]) * (-x[2]**(-1 / 2))
        g_v = np.cos(x[0]) + np.log(x[1]) * x[2]
        h_v = 5 + x[0]
        assert np.allclose(AutoDiff([f, g, h]).get_value(x),
                           np.array([f_v, g_v, h_v]),
                           rtol=1e-15,
                           atol=1e-15)

    def test_get_partial(self):
        # scalar function with m=1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        assert AutoDiff(f).get_partial(
            x) == -1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2

        # scalar constant function with m=1
        f = 2
        x = 1.5
        assert AutoDiff(f).get_partial(x) == 0

        # scalar constant function with m=1
        f = lambda x: 2
        x = 1.5
        assert AutoDiff(f).get_partial(x) == 0

        # scalar function with constant return with m=1
        f = lambda x: sin(10)
        x = 1.5
        assert AutoDiff(f).get_partial(x) == 0

        # scalar function with m=2 passed as list
        f = lambda x: x[0] * sin(x[1])
        assert (AutoDiff(f).get_partial([1, math.pi], 0) == approx(0.0)
                and AutoDiff(f).get_partial(np.array([1, math.pi]),
                                            1) == approx(-1))

        # scalar function with a nested list (invalid input)
        f = lambda x: x[0] * sin(x[1])
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_partial([1, [math.pi]], 0)

        # scalar function with 2-d array (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_partial(np.array([[1], [math.pi]]), 0)

        # scalar function with non-integer index
        with pytest.raises(AssertionError):
            ad = AutoDiff(f)
            ad.get_partial(np.array([1, math.pi]), 0.5)

        # scalar function with out-of-bound index
        with pytest.raises(IndexError):
            ad = AutoDiff(f)
            ad.get_partial(np.array([1, math.pi]), 2)

        # vector function with m=1
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        h = lambda x: 5
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        h_p = 0
        assert AutoDiff([f, g,
                         h]).get_partial(x) == approx(np.array([f_p, g_p,
                                                                h_p]))

        # vector function with m=3
        f = lambda x: exp(x[1]) * (-x[2]**(-1 / 2))
        g = lambda x: cos(x[0]) + log(x[1]) * x[2]
        h = lambda x: 5 + x[0]
        x = np.array([-1, 10, 105.5])
        f_p_0 = 0
        g_p_0 = -np.sin(x[0])
        h_p_0 = 1
        f_p_1 = -np.exp(x[1]) / x[2]**(1 / 2)
        g_p_1 = x[2] / x[1]
        h_p_1 = 0
        f_p_2 = np.exp(x[1]) / (2 * x[2]**(3 / 2))
        g_p_2 = np.log(x[1])
        h_p_2 = 0
        assert (AutoDiff([f, g, h]).get_partial(x, 0) == approx(
            np.array([f_p_0, g_p_0, h_p_0]))
                and AutoDiff([f, g, h]).get_partial(x, 1) == approx(
                    np.array([f_p_1, g_p_1, h_p_1]))
                and AutoDiff([f, g, h]).get_partial(x, 2) == approx(
                    np.array([f_p_2, g_p_2, h_p_2])))

        # vector function with m=3 passed as list
        f = lambda x: exp(x[1]) * (-x[2]**(-1 / 2))
        g = lambda x: cos(x[0]) + log(x[1]) * x[2]
        h = lambda x: 5 + x[0]
        x = [-1, 10, 105.5]
        f_p_0 = 0
        g_p_0 = -np.sin(x[0])
        h_p_0 = 1
        f_p_1 = -np.exp(x[1]) / x[2]**(1 / 2)
        g_p_1 = x[2] / x[1]
        h_p_1 = 0
        f_p_2 = np.exp(x[1]) / (2 * x[2]**(3 / 2))
        g_p_2 = np.log(x[1])
        h_p_2 = 0
        assert (AutoDiff([f, g, h]).get_partial(x, 0) == approx(
            np.array([f_p_0, g_p_0, h_p_0]))
                and AutoDiff([f, g, h]).get_partial(x, 1) == approx(
                    np.array([f_p_1, g_p_1, h_p_1]))
                and AutoDiff([f, g, h]).get_partial(x, 2) == approx(
                    np.array([f_p_2, g_p_2, h_p_2])))

    def test_get_jacobian(self):
        # scalar function with m=1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        assert AutoDiff(f).get_jacobian(x) == np.array(
            [-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2])

        # scalar function with m=2
        f = lambda x: x[0] * sin(x[1])
        assert AutoDiff(f).get_jacobian(np.array([1, math.pi])) == approx(
            np.array([[0.0, -1]]))

        # scalar function with m=2 passed as list
        f = lambda x: x[0] * sin(x[1])
        assert AutoDiff(f).get_jacobian([1, math.pi
                                         ]) == approx(np.array([[0.0, -1]]))

        # scalar function with a nested list (invalid input)
        f = lambda x: x[0] * sin(x[1])
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_jacobian([1, [math.pi]])

        # scalar function with 2-d array (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_jacobian(np.array([[1], [math.pi]]))

        # vector function with m=1
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        h = lambda x: 5
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        h_p = 0
        assert AutoDiff([f, g, h]).get_jacobian(x) == approx(
            np.array([[f_p], [g_p], [h_p]]))

        # vector function with m=3
        f = lambda x: exp(x[1]) * (-x[2]**(-1 / 2))
        g = lambda x: cos(x[0]) + log(x[1]) * x[2]
        h = lambda x: 5 + x[0]
        x = np.array([-1, 10, 105.5])
        f_p_0 = 0
        g_p_0 = -np.sin(x[0])
        h_p_0 = 1
        f_p_1 = -np.exp(x[1]) / x[2]**(1 / 2)
        g_p_1 = x[2] / x[1]
        h_p_1 = 0
        f_p_2 = np.exp(x[1]) / (2 * x[2]**(3 / 2))
        g_p_2 = np.log(x[1])
        h_p_2 = 0

        res = np.array([[f_p_0, f_p_1, f_p_2], [g_p_0, g_p_1, g_p_2],
                        [h_p_0, h_p_1, h_p_2]])
        assert AutoDiff([f, g, h]).get_jacobian(x) == approx(res)

        # vector function with m=3 passed as list
        f = lambda x: exp(x[1]) * (-x[2]**(-1 / 2))
        g = lambda x: cos(x[0]) + log(x[1]) * x[2]
        h = lambda x: 5 + x[0]
        x = [-1, 10, 105.5]
        f_p_0 = 0
        g_p_0 = -np.sin(x[0])
        h_p_0 = 1
        f_p_1 = -np.exp(x[1]) / x[2]**(1 / 2)
        g_p_1 = x[2] / x[1]
        h_p_1 = 0
        f_p_2 = np.exp(x[1]) / (2 * x[2]**(3 / 2))
        g_p_2 = np.log(x[1])
        h_p_2 = 0

        res = np.array([[f_p_0, f_p_1, f_p_2], [g_p_0, g_p_1, g_p_2],
                        [h_p_0, h_p_1, h_p_2]])
        assert AutoDiff([f, g, h]).get_jacobian(x) == approx(res)

    def test_get_derivative(self):
        # scalar function with m=1 and default_seed
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        assert AutoDiff(f).get_derivative(x) == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]),
            np.array([1]))

        # scalar function with m=1 and scalar seed
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = 5
        assert AutoDiff(f).get_derivative(x, p) == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]),
            np.array([p]))

        # scalar function with m=1 and vector seed with length 1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = np.array([5])
        assert AutoDiff(f).get_derivative(x, p) == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]), p)

        # scalar function with m=1 passed as vector and seed with lengths 1
        f = lambda x: -x[0] + cos(x[0]) * sin(x[0]) + 5 * x[0]**4
        x = np.array([1.5])
        p = np.array([5])
        assert AutoDiff(f).get_derivative(x, p) == np.array([
            np.dot(
                np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]),
                p)
        ])

        # scalar function with m=1 passed as vector and default seed
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = np.array([1])
        assert AutoDiff(f).get_derivative(x) == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]), p)

        # scalar function with m=1 and seed with length>1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = np.array([1, 5])
        with pytest.raises(ValueError):
            ad = AutoDiff(f).get_derivative(x, p)

        # scalar function with m=2 and seed with length=2
        f = lambda x: x[0] * sin(x[1])
        x = np.array([1, math.pi])
        p = np.array([1, 5])
        assert AutoDiff(f).get_derivative(x, p) == approx(
            np.dot(np.array([0.0, -1]), p))

        # scalar function with m=2 and default seed
        f = lambda x: x[0] * sin(x[1])
        x = np.array([1, math.pi])
        with pytest.raises(ValueError):
            AutoDiff(f).get_derivative(x)

        # scalar function with m=2 and seed with length=2, passed as lists
        f = lambda x: x[0] * sin(x[1])
        x = [1, math.pi]
        p = [1, 5]
        assert AutoDiff(f).get_derivative(x, p) == approx(
            np.dot(np.array([0.0, -1]), p))

        # scalar function with a nested list (invalid input)
        f = lambda x: x[0] * sin(x[1])
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_derivative([1, [math.pi]], np.array([0, 1]))

        # scalar function with 2-d array values (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            x = np.array([[1], [math.pi]])
            p = np.array([1, 5])
            ad.get_derivative(x, p)

        # scalar function with m=1 and 2-d array seed (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            x = np.array([1, math.pi])
            p = np.array([[1], [5]])
            ad.get_derivative(x, p)

        # n=2 vector function with m=1 and len(seed)>1 (invalid input)
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        p = np.array([1, 2.0, -1.5])

        with pytest.raises(ValueError):
            d = AutoDiff([f, g]).get_derivative(x, p)

        # n=2 vector function with m=1 and scalar seed
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        p = -200.5
        assert AutoDiff([f, g]).get_derivative(x, p) == approx(
            np.dot(np.array([[f_p], [g_p]]), p))

        # n=2 vector function with m=1 and a seed array with length 1
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        p = np.transpose(np.array([-200.5]))
        assert AutoDiff([f, g]).get_derivative(x, p) == approx(
            np.dot(np.array([[f_p], [g_p]]), p.reshape(-1, 1)))

        # n=3 vector function with m=3
        f = lambda x: exp(x[1]) * (-x[2]**(-1 / 2))
        g = lambda x: cos(x[0]) + log(x[1]) * x[2]
        h = lambda x: 5 + x[0]
        x = np.array([-1, 10, 105.5])
        p = np.array([1, 1, 1])
        f_p_0 = 0
        g_p_0 = -np.sin(x[0])
        h_p_0 = 1
        f_p_1 = -np.exp(x[1]) / x[2]**(1 / 2)
        g_p_1 = x[2] / x[1]
        h_p_1 = 0
        f_p_2 = np.exp(x[1]) / (2 * x[2]**(3 / 2))
        g_p_2 = np.log(x[1])
        h_p_2 = 0

        res = np.array([[f_p_0, f_p_1, f_p_2], [g_p_0, g_p_1, g_p_2],
                        [h_p_0, h_p_1, h_p_2]])
        ad = AutoDiff([f, g, h])
        assert ad.get_derivative(x, p) == approx(np.dot(res, p.reshape(-1, 1)))

        assert (np.array_equal(ad.point, x) and np.array_equal(ad.seed, p)
                and np.allclose(ad.jacobian, res, rtol=1e-15, atol=1e-15)
                and np.array_equal(ad.derivative, np.dot(
                    res, p.reshape(-1, 1))))

        # same function and seed as above with new point
        x = np.array([10.2, 31, 0.055])
        f_p_0 = 0
        g_p_0 = -np.sin(x[0])
        h_p_0 = 1
        f_p_1 = -np.exp(x[1]) / x[2]**(1 / 2)
        g_p_1 = x[2] / x[1]
        h_p_1 = 0
        f_p_2 = np.exp(x[1]) / (2 * x[2]**(3 / 2))
        g_p_2 = np.log(x[1])
        h_p_2 = 0
        res = np.array([[f_p_0, f_p_1, f_p_2], [g_p_0, g_p_1, g_p_2],
                        [h_p_0, h_p_1, h_p_2]])
        ad = AutoDiff([f, g, h])
        assert ad.get_derivative(x, p) == approx(np.dot(res, p.reshape(-1, 1)))
        assert (np.array_equal(ad.point, x) and np.array_equal(ad.seed, p)
                and np.allclose(ad.jacobian, res, rtol=1e-15, atol=1e-15)
                and np.array_equal(ad.derivative, np.dot(
                    res, p.reshape(-1, 1))))

        # same function and value as above with new seed
        p = np.array([1, 1, 0])
        assert (AutoDiff([f, g, h]).get_derivative(x, p) == approx(
            np.dot(res, p.reshape(-1, 1))))

    # test correct storage and calls of computed values (attributes of AutoDiff objects)
    def test_cache(self):

        f = lambda x: 1 / 2 * x**2
        ad = AutoDiff(f)
        # 1)
        assert ad.get_jacobian(1) == 1.0
        # 2) call get_jacobian again with the same point as 1)
        assert ad.get_jacobian(1) == 1.0
        # 3) call get_derivative with the same point as 2)
        assert ad.get_derivative(1, 1) == 1.0
        # 4) call get_derivative with the same point and seed as 3)
        assert ad.get_derivative(1, 1) == 1.0
        # 5) call get_derivative with the same point but different seed
        assert ad.get_derivative(1, 2) == 2.0
        # 6) call get_jacobian again with the same point
        assert ad.get_jacobian(1) == 1.0
        # 7) call get_derivative with same point as 4) but
        assert ad.get_derivative(1, 1) == 1.0
        # 8) call get_derivative with different point
        assert ad.get_derivative(2, 1) == 2.0
        # 9) call get_jacobian with different point
        assert ad.get_jacobian(3) == 3.0
        # 10) call get_derivative with different point
        assert ad.get_derivative(4, [1]) == 4.0
