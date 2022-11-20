"""In this script we implement the newton method using automatic differentiation"""

from autodiff.auto_diff import AutoDiff
from autodiff.utils.auto_diff_math import *

f = lambda x: x**2 - 5 * x + 2 * exp(x) - sin(x) - 4
# f has 2 roots: -0.42 and 1.662

ad_class = AutoDiff(f)

# Newton's method
eps = 1e-4  # tolerance
x = 0.5  # initial guess
steps = 0  # number of steps
while abs(f(x)) > eps:
    steps += 1
    x = x - f(x) / ad_class.derivative(x)
    if steps > 100:
        print("No convergence")
        break

assert round(x, 3) in [-0.42, 1.662]

print("Root: ", x)
print("Number of steps: ", steps)
