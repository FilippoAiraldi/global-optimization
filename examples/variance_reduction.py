"""
Example showcasing the variance reduction techiques emploeyed in the estimation
of the non-myopic acquisition function.
"""


from datetime import datetime
from time import perf_counter

import ray

ray.init()

from joblib import Parallel, delayed

print(f"{datetime.now()} | Importing... ")
t0 = perf_counter()
import matplotlib.pyplot as plt
import numpy as np

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Kernel, Rbf, fit
from globopt.nonmyopic.acquisition import (
    acquisition,
    acquisition_joblib,
    deterministic_acquisition,
)
from globopt.util.ray import wait_tasks

print(f"{datetime.now()} | time = {perf_counter() - t0:.3f}s")

plt.style.use("bmh")


# define the function and its domain
print(f"{datetime.now()} | Setup... ")
t0 = perf_counter()
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
print(f"{datetime.now()} | time = {perf_counter() - t0:.3f}s")

# compute the non-myopic acquisition for `x_target` with deterministic dynamics - no
# MC integration is required
print(f"{datetime.now()} | Deterministic... ")
t0 = perf_counter()
kwargs = {
    "x": x_target,
    "mdl": mdl,
    "horizon": horizon,
    "discount": discount,
    "c1": c1,
    "c2": c2,
    "rollout": True,
    "lb": lb,
    "ub": ub,
}
a_deterministic = deterministic_acquisition(**kwargs, seed=420).item()
print(f"{datetime.now()} | time = {perf_counter() - t0:.3f}s")


# warm-up
acquisition_joblib(
    **kwargs,
    mc_iters=2**2,
    quasi_mc=False,
    common_random_numbers=True,
    antithetic_variates=False,
    return_as_list=True,
    n_jobs=1,
)

p = 13
N = 10

print(f"{datetime.now()} | MC (n_jobs=1)... ")
t0 = perf_counter()
with Parallel(n_jobs=-1) as parallel:
    a_target1 = np.squeeze(
        parallel(
            delayed(acquisition_joblib)(
                **kwargs,
                mc_iters=2**p,
                quasi_mc=False,
                common_random_numbers=True,
                antithetic_variates=False,
                return_as_list=True,
                n_jobs=1,
            )
            for _ in range(N)
        )
    )
print(f"{datetime.now()} | time = {perf_counter() - t0:.3f}s")

print(f"{datetime.now()} | MC (n_jobs=-1)... ")
t0 = perf_counter()
with Parallel(n_jobs=-1) as parallel:
    a_target2 = np.squeeze(
        parallel(
            delayed(acquisition_joblib)(
                **kwargs,
                mc_iters=2**p,
                quasi_mc=False,
                common_random_numbers=True,
                antithetic_variates=False,
                return_as_list=True,
                n_jobs=-1,
            )
            for _ in range(N)
        )
    )
print(f"{datetime.now()} | time = {perf_counter() - t0:.3f}s")

print(f"{datetime.now()} | MC (ray)... ")
t0 = perf_counter()
a_target3 = np.squeeze(
    wait_tasks(
        [
            acquisition.remote(
                **kwargs,
                mc_iters=2**p,
                quasi_mc=False,
                common_random_numbers=True,
                antithetic_variates=False,
                return_as_list=True,
            )
            for _ in range(N)
        ]
    )
)
print(f"{datetime.now()} | time = {perf_counter() - t0:.3f}s")

print(a_target1.mean(-1))
print(a_target2.mean(-1))
print(a_target3.mean(-1))
np.savez("a_targets.npz", a_target1=a_target1, a_target2=a_target2, a_target3=a_target3)


# # compute the non-myopic acquisition function for x with different variance reductions
# kwargs["mc_iters"] = 2**7
# a_no_vr = acquisition(
#     **kwargs, quasi_mc=False, antithetic_variates=False, return_as_list=True
# )
# a_qmc = acquisition(
#     **kwargs, quasi_mc=True, antithetic_variates=False, return_as_list=True
# )
# a_qmc_atv = acquisition(
#     **kwargs, quasi_mc=True, antithetic_variates=True, return_as_list=True
# )


# # plotting
# iters_ = np.arange(1, kwargs["mc_iters"] + 1)
# _, ax = plt.subplots(1, 1, constrained_layout=True)
# for a, lbl in zip(
#     [a_no_vr, a_qmc, a_qmc_atv],
#     ["Standard MC", "Quasi-MC", "Quasi-MC + Antithetic Variates"],
# ):
#     mean_estimate = np.cumsum(a) / iters_
#     ax.semilogy(iters_, np.abs(mean_estimate - a_target), label=lbl)
# ax.set_xlabel("MC iterations")
# ax.set_ylabel("Estimation error")
# plt.show()
