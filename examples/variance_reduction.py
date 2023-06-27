"""
Example showcasing the variance reduction techiques emploeyed in the estimation
of the non-myopic acquisition function.
"""


import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from globopt.core.problems import Simple1dProblem
from globopt.core.regression import Kernel, Rbf, fit
from globopt.nonmyopic.acquisition import acquisition

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
a_deterministic = acquisition(**kwargs, mc_iters=0, seed=seed).item()

# compute the acquisition with stochastic dynamics (via MC and no variance reduction).
# Use a huge number of MC iterations to get a good estimate of the target.
try:
    # try to load the data from disk since these computations are quite long
    data = np.load("examples/variance_reduction.npz")
    a_target = data["a_target"]
    a_qmc = data["a_qmc"]
    a_qmc_av = data["a_qmc_av"]
    print("Loaded from disk.")

except FileNotFoundError:
    print("Computing from scratch...")
    N = 10  # number of independent runs

    # run the computations
    with Parallel(n_jobs=-1, backend="loky", verbose=1) as parallel:
        a_target = np.squeeze(
            parallel(
                delayed(acquisition)(
                    **kwargs,
                    mc_iters=2**14,
                    parallel={"n_jobs": -1, "backend": "loky", "verbose": 1},
                    return_iters=True,
                    seed=seed + i,
                    common_random_numbers=False,  # CRN have no influece here
                    quasi_mc=False,
                    antithetic_variates=False,
                )
                for i in range(N)
            )
        )

        # use Quasi-MC pseudo-random numbers
        a_qmc = np.squeeze(
            parallel(
                delayed(acquisition)(
                    **kwargs,
                    mc_iters=2**12,
                    parallel={"n_jobs": -1, "backend": "loky", "verbose": 1},
                    return_iters=True,
                    seed=seed + i,
                    common_random_numbers=False,
                    quasi_mc=True,
                    antithetic_variates=False,
                )
                for i in range(N)
            )
        )

        a_qmc_av = np.squeeze(
            parallel(
                delayed(acquisition)(
                    **kwargs,
                    mc_iters=2**12,
                    parallel={"n_jobs": -1, "backend": "loky", "verbose": 1},
                    return_iters=True,
                    seed=seed + i,
                    common_random_numbers=False,
                    quasi_mc=True,
                    antithetic_variates=True,
                )
                for i in range(N)
            )
        )

    np.savez(
        "examples/variance_reduction.npz",
        a_target=a_target,
        a_qmc=a_qmc,
        a_qmc_av=a_qmc_av,
    )


# plotting
a_target_avg = a_target.mean(-1, keepdims=True)
fig, ax = plt.subplots(constrained_layout=True)
for a, lbl in zip(
    [a_target, a_qmc, a_qmc_av], ["Standard MC", "Quasi-MC", "Quasi-MC + AV"]
):
    iters = np.arange(1, a.shape[-1] + 1)
    delta = np.abs(np.cumsum(a, -1) / iters.reshape(1, -1) - a_target_avg)
    delta_avg = delta.mean(0)
    delta_std = delta.std(0)
    ax.semilogy(iters, delta_avg, label=lbl)
    ax.fill_between(iters, delta_avg - delta_std, delta_avg + delta_std, alpha=0.2)
ax.set_xlabel("Iterations")
ax.set_ylabel("Estimation Error")
ax.set_xlim(2**7, 2**12)
ax.set_ylim(1e-4, 1e-0)
ax.legend()
plt.show()
