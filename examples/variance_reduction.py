"""
Example showcasing the variance reduction techiques emploeyed in the estimation
of the non-myopic acquisition function.
"""


import os

os.environ["NUMBA_DISABLE_JIT"] = "1"


import matplotlib.pyplot as plt
import numpy as np

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Kernel, Rbf, fit
from globopt.nonmyopic.acquisition import acquisition, deterministic_acquisition

plt.style.use("bmh")

# define the function and its domain
lb, ub = np.array([-3.0]), np.array([+3.0])
f = Simple1dProblem.f

# create data points - X has shape (batch, n_samples, dim), with batch=1
X = np.array([-2.62, -1.99, 0.14, 1.01, 2.62]).reshape(1, -1, 1)
y = f(X)
mdl = fit(Rbf(Kernel.ThinPlateSpline, 0.01), X, y)

# define the target point for which we want to compute the non-myopic acqusition and
# other parameters
x_target = np.reshape(-1.0, (1, 1, 1))
c1 = 1.0
c2 = 0.5
horizon = 3
discount = 0.9

# compute the non-myopic acquisition for `x_target` with deterministic dynamics - no
# MC integration is required
a_deterministic = deterministic_acquisition(
    x_target, mdl, horizon, discount, c1, c2, "rollout", lb, ub
).item()
# x_ = np.random.rand(1, horizon, 1)
# a__ = deterministic_acquisition(x_, mdl, horizon, discount, c1, c2, "mpc").item()

# compute the non-myopic acquisition function for x with no variance reduction but high
# MC iterations to get a baseline
a_mc = acquisition(
    x_target,
    mdl,
    horizon,
    discount,
    c1,
    c2,
    "rollout",
    lb,
    ub,
    mc_iters=4,
    quasi_mc=False,
    common_random_numbers=False,
    seed=69,
    return_as_list=True,
).item()

quit()
