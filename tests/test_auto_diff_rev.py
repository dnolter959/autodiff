import pytest
import math
import numpy as np
from pytest import approx

# import names to test
from autodiff.auto_diff import AutoDiff
from autodiff.utils.dual_numbers import DualNumber
from autodiff.utils.auto_diff_math import *


class TestAutoDiffReverse:
    def test_init(self):
        # test initialization
        x = AutoDiff(lambda x: x**2 - 2 * x)
        assert x.point is None
        assert x.seed is None
        assert x.derivative is None
        assert x.jacobian is None
        assert x.computational_graph is None

    def test_check_vector(self):
        x = AutoDiff(lambda x: x**2 - 2 * x)

        with pytest.raises(TypeError):
            x._check_vector("1")

        with pytest.raises(TypeError):
            x._check_vector([1, 2, "3"])

        with pytest.raises(TypeError):
            x._check_vector(np.array([1, 2, "3"]))

    def test_get_value(self):
        x = AutoDiff(lambda x: x**2 - 2 * x)
        assert x.get_value(1) == -1

        x2 = AutoDiff([lambda x: x**2 - 2 * x, lambda y: y + 3])
        assert x2.get_value(1)[0] == -1 and x2.get_value(1)[1] == 4

    def test_get_jacobian(self):
        # invalid mode
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        with pytest.raises(ValueError):
            AutoDiff(f).get_jacobian(x, mode="a")

        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        with pytest.raises(ValueError):
            AutoDiff(f).get_jacobian(x, mode=1)

        # scalar constant function with m=1
        f = 2
        x = 1.5
        assert AutoDiff(f).get_jacobian(x, mode='r') == 0

        # scalar constant function with m=1
        f = lambda x: 2
        x = 1.5
        assert AutoDiff(f).get_jacobian(x, mode='r') == 0

        # scalar function with m=1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        assert AutoDiff(f).get_jacobian(x, mode="r") == np.array(
            [-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2])

        # scalar function with m=2
        f = lambda x: x[0] * sin(x[1])
        assert AutoDiff(f).get_jacobian(np.array([1, math.pi]),
                                        mode="reverse") == approx(
                                            np.array([[0.0, -1]]))

        # scalar function with m=2 passed as list
        f = lambda x: x[0] * sin(x[1])
        assert AutoDiff(f).get_jacobian([1, math.pi], mode="r") == approx(
            np.array([[0.0, -1]]))

        # scalar function with a nested list (invalid input)
        f = lambda x: x[0] * sin(x[1])
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_jacobian([1, [math.pi]], mode="r")

        # scalar function with 2-d array (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_jacobian(np.array([[1], [math.pi]]), mode="r")

        # vector function with m=1
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        h = lambda x: 5
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        h_p = 0
        assert AutoDiff([f, g, h]).get_jacobian(x, mode="R") == approx(
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
        assert AutoDiff([f, g, h]).get_jacobian(x,
                                                mode="REVERSE") == approx(res)

    def test_get_derivative(self):
        # scalar function with m=1 and default_seed
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        assert AutoDiff(f).get_derivative(x, mode="r") == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]),
            np.array([1]))

        # scalar function with m=1 and scalar seed
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = 5
        assert AutoDiff(f).get_derivative(x, p, mode="r") == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]),
            np.array([p]))

        # scalar function with m=1 and vector seed with length 1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = np.array([5])
        assert AutoDiff(f).get_derivative(x, p, mode="r") == np.dot(
            np.array([-1 + 20 * x**3 + (np.cos(x))**2 - (np.sin(x))**2]), p)

        # scalar function with m=1 and seed with length>1
        f = lambda x: -x + cos(x) * sin(x) + 5 * x**4
        x = 1.5
        p = np.array([1, 5])
        with pytest.raises(ValueError):
            ad = AutoDiff(f).get_derivative(x, p, mode="r")

        # scalar function with m=2 and seed with length=2
        f = lambda x: x[0] * sin(x[1])
        x = np.array([1, math.pi])
        p = np.array([1, 5])
        assert AutoDiff(f).get_derivative(x, p, mode="r") == approx(
            np.dot(np.array([0.0, -1]), p))

        # scalar function with a nested list (invalid input)
        f = lambda x: x[0] * sin(x[1])
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.get_derivative([1, [math.pi]], np.array([0, 1]), mode="r")

        # scalar function with 2-d array values (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            x = np.array([[1], [math.pi]])
            p = np.array([1, 5])
            ad.get_derivative(x, p, mode="r")

        # scalar function with m=1 and 2-d array seed (invalid input)
        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            x = np.array([1, math.pi])
            p = np.array([[1], [5]])
            ad.get_derivative(x, p, mode="r")

        # n=2 vector function with m=1 and len(seed)>1 (invalid input)
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        p = np.array([1, 2.0, -1.5])

        with pytest.raises(ValueError):
            d = AutoDiff([f, g]).get_derivative(x, p, mode="r")

        # n=2 vector function with m=1 and scalar seed
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        p = -200.5
        assert AutoDiff([f, g]).get_derivative(x, p, mode="r") == approx(
            np.dot(np.array([[f_p], [g_p]]), p))

        # n=2 vector function with m=1 and a seed array with length 1
        f = lambda x: exp(x) * (-x**(-1 / 2))
        g = lambda x: cos(x) + log(x)
        x = 10
        f_p = (np.exp(x) * (1 - 2 * x)) / (2 * x**(3 / 2))
        g_p = -np.sin(x) + 1 / x
        p = np.transpose(np.array([-200.5]))
        assert AutoDiff([f, g]).get_derivative(x, p, mode="r") == approx(
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
        assert ad.get_derivative(x, p, mode="r") == approx(
            np.dot(res, p.reshape(-1, 1)))

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
        assert (AutoDiff([f, g, h]).get_derivative(x, p, mode="r") == approx(
            np.dot(res, p.reshape(-1, 1))))

    def test_forward_reverse_match(self):
        # complicated scalar function with 1d input
        f = lambda x: 1 / x + x * x**2 - cos(1 / x) + sin(cos(1 / x)) - log_b(
            x, 5) / sinh(x**2)
        x = 0.5
        adf = AutoDiff(f)
        adr = AutoDiff(f)

        assert adf.get_derivative(x, mode="f") == approx(
            adr.get_derivative(x, mode="r"))

        # complicated vector function with 1d input
        f = lambda x: 1 / x + x * x**2 - cos(1 / x) + sin(cos(1 / x)) - log_b(
            x, 5) / sin(x**2)
        g = lambda x: x * (1 / x + x * x**2 - cos(1 / x) + sin(cos(1 / x)) -
                           log_b(x, 5) / sin(x**2)) - exp_b(x / 20, 6) - cos(x)
        h = lambda x: x**(-1 / 2) - tan(x**(-1 / 2)) * (x + x**(
            -0.5)) - 1000 * x * 20 * exp(x - 6) + (x - 6)**15
        x = 100
        adf = AutoDiff([f, g, h])
        adr = AutoDiff([f, g, h])
        assert np.allclose(adf.get_derivative(x, mode="f"),
                           adr.get_derivative(x, mode="r"),
                           rtol=1e-15,
                           atol=1e-15)

        # complicated vector function with 2d input
        f = lambda x: sin(x[0] * (x[1] + 1) - cos(x[1] + 1) / x[0]**2) - x[
            0]**2 + x[0] / (x[0] + 1) * exp(x[1]**2)
        g = lambda x: x[1] * (1 / x[0] + x[1] * x[0]**2 - cos(1 / x[0]) + sin(
            cos(1 / x[0])) - log_b(x[0], 5) / sin(x[0]**2)) - exp_b(
                x[0] / 20, 6) - cos(x[1])
        h = lambda x: x[0]**(-1 / 2) - tan(x[0]**(-1 / 2)) * (x[0] + x[0]**(
            -0.5)) - 1000 * x[1] * 20 * exp(x[0] - 6) + (x[0] - 6)**15
        x = np.array([0.5, -11])
        adf = AutoDiff([f, g, h])
        adr = AutoDiff([f, g, h])
        p = np.array([2, 3])
        assert np.allclose(adf.get_derivative(x, mode="f", seed_vector=p),
                           adr.get_derivative(x, mode="r", seed_vector=p),
                           rtol=1e-15,
                           atol=1e-15)
