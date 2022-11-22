#!/usr/bin/env python
"""In this script we implement the newton method using automatic differentiation"""
import numpy as np
from autodiff.auto_diff import AutoDiff
from autodiff.utils.auto_diff_math import *

f = lambda x: x**2 - 5 * x + 2 * exp(x) - sin(x) - 4
x_k = 0.5
tol = 1e-4
max_it = 1000
for k in range(max_it):
    der = AutoDiff(f).get_derivative(x_k, 1)
    val = AutoDiff(f).get_value(x_k)
    dx_k = -val / der
    if abs(dx_k) < tol:
        root = x_k + dx_k
        break
    x_k += dx_k

# f has 2 roots: -0.42 and 1.662
assert round(root, 3) in [-0.42, 1.662]

print("Root: ", root)
print("Number of steps: ", k)
