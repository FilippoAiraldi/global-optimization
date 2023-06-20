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
args = (x_target, mdl, horizon, discount, c1, c2, "rollout", lb, ub)
a_deterministic = deterministic_acquisition(*args).item()

# compute reference MC estimate with a large number of MC iterations
a_target = acquisition(
    *args,
    mc_iters=2**64,
    quasi_mc=False,
    common_random_numbers=False,
    antithetic_variates=False,
    seed=69,
).item()

# compute the non-myopic acquisition function for x with different variance reductions
mc_iters = 2**6
a_no_vr = acquisition(
    *args,
    mc_iters=mc_iters,
    quasi_mc=False,
    common_random_numbers=False,
    antithetic_variates=False,
    seed=69,
    return_as_list=True,
)
a_qmc = acquisition(
    *args,
    mc_iters=mc_iters,
    quasi_mc=True,
    common_random_numbers=False,
    antithetic_variates=False,
    seed=69,
    return_as_list=True,
)


# plotting
iters_ = np.arange(1, mc_iters + 1)
_, ax = plt.subplots(1, 1, constrained_layout=True)
ax.semilogy(iters_, np.abs(np.cumsum(a_no_vr) / iters_ - a_target), label="No VR")
ax.semilogy(iters_, np.abs(np.cumsum(a_qmc) / iters_ - a_target), label="Quasi-MC")
ax.set_xlabel("MC iterations")
ax.set_ylabel("Estimation error")
plt.show()
