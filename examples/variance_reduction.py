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
# Monte Carlo integration)
seed = 420
a_deterministic = deterministic_acquisition(**kwargs, seed=seed).item()

# compute the acquisition with stochastic dynamics (via MC and no variance reduction).
# Use a huge number of MC iterations to get a good estimate of the target.
try:
    # try to load the data from disk since these computations are quite long
    data = np.load("examples/variance_reduction.npz")
    a_target = data["a_target"]
    a_qmc = data["a_qmc"]
    print("Loaded data from disk.")

except FileNotFoundError:
    print("Computing data from scratch...")
    N = 3

    # run the computations
    with Parallel(n_jobs=-1, backend="loky", verbose=1) as parallel:
        a_target = np.squeeze(
            parallel(
                delayed(acquisition)(
                    **kwargs,
                    mc_iters=2**14,
                    quasi_mc=False,
                    common_random_numbers=False,  # CRN have no influece here
                    antithetic_variates=False,
                    parallel={"n_jobs": -1, "verbose": 1, "backend": "loky"},
                    return_as_list=True,
                    seed=seed + i,
                )
                for i in range(N)
            )
        )

        # use Quasi-MC pseudo-random numbers
        a_qmc = np.squeeze(
            parallel(
                delayed(acquisition)(
                    **kwargs,
                    mc_iters=2**11,
                    quasi_mc=True,
                    common_random_numbers=False,
                    antithetic_variates=False,
                    parallel={"n_jobs": -1, "verbose": 1, "backend": "loky"},
                    return_as_list=True,
                    seed=seed + i,
                )
                for i in range(N)
            )
        )

    np.savez("examples/variance_reduction.npz", a_target=a_target, a_qmc=a_qmc)


# plotting
a_target_avg = a_target.mean(-1).mean(-1)
_, ax = plt.subplots(1, 1, constrained_layout=True)
for a, lbl in zip([a_target, a_qmc], ["Standard MC", "Quasi-MC"]):
    iters = np.arange(1, a.shape[-1] + 1)
    avg = np.cumsum(a, axis=-1) / iters.reshape(1, -1)
    ax.semilogy(iters, np.abs(avg - a_target_avg), label=lbl)
ax.set_xlabel("MC iterations")
ax.set_ylabel("Estimation error")
ax.set_xlim(2**7, 2**11)
plt.show()
