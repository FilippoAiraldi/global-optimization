"""
Example showcasing the variance reduction techiques emploeyed in the estimation
of the non-myopic acquisition function.
"""


import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

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

# compute the non-myopic acquisition for `x_target` with deterministic dynamics (without
# Monte Carlo integration) and with stochastic dynamics (via MC)
a_deterministic = deterministic_acquisition(**kwargs, seed=420).item()
a_target = np.squeeze(
    Parallel(n_jobs=-1, verbose=1)(
        delayed(acquisition)(
            **kwargs,
            mc_iters=2**13,
            quasi_mc=False,
            common_random_numbers=False,
            antithetic_variates=False,
            parallel={"n_jobs": -1, "verbose": 1, "backend": "loky"},
            return_as_list=True,
            seed=420 + i,
        )
        for i in range(3)
    )
)


# TODO: ablation studies repeated for different seeds (target can be run only once)


print(a_target.mean(-1))
np.savez("examples/variance_reduction.npz", a_target=a_target)


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
