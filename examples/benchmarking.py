import os
from itertools import zip_longest

os.environ["NUMBA_DISABLE_JIT"] = "1"  # no need for jit in this example

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from scipy.io import savemat

from globopt.myopic.algorithm import GO, RBFRegression
from globopt.util.benchmark import get_available_benchmark_tests, get_benchmark_test
from globopt.util.callback import BestSoFarCallback

plt.style.use("bmh")


def get_now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fnv1a(s: str) -> int:
    FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
    return sum((FNV_OFFSET ^ b) * FNV_PRIME**i for i, b in enumerate(s.encode()))


def solve_problem(name: str, times: int) -> tuple[Problem, npt.NDArray[np.floating]]:
    # create the problem
    problem, max_n_iter = get_benchmark_test(name)

    # create the algorithm
    n_var = problem.n_var
    algorithm = GO(
        regression=RBFRegression(eps=1.0775 / n_var),
        init_points=2 * n_var,
        acquisition_min_algorithm=PSO(pop_size=20),
        acquisition_min_kwargs={
            "termination": DefaultSingleObjectiveTermination(
                ftol=1e-4, n_max_gen=2000, period=10
            )
        },
        acquisition_fun_kwargs={"c1": 1.5078 / n_var, "c2": 1.4246 / n_var},
    )

    # solve the problem multiple times with different seeds
    seed = fnv1a(name) % 2**32
    out = np.empty((times, max_n_iter))
    for i in range(times):
        res = minimize(
            problem,
            algorithm,
            termination=("n_iter", max_n_iter),
            callback=BestSoFarCallback(),
            verbose=False,
            seed=seed + i * 2**7,
        )
        out[i] = res.algorithm.callback.data["best"]
    return problem, out


# run the benchmarking
testnames = get_available_benchmark_tests()
testnames = ["ackley"]
f = lambda name: solve_problem(name, 3)
results = Parallel(n_jobs=1, verbose=10)(delayed(f)(name) for name in testnames)

# save the results to disk
nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
savemat(f"results_{nowstr}.mat", {n: out for n, (_, out) in zip(testnames, results)})

# plot results
n_rows = 3
n_cols = np.ceil(len(results) / n_rows).astype(int)
_, axs = plt.subplots(n_cols, n_rows, constrained_layout=False)
axs = np.atleast_2d(axs)
for ax, problem, out in zip_longest(axs.flat, *zip(*results)):
    if problem is None or out is None:
        # if there are more axes than results, set the axis off
        ax.set_axis_off()
    else:
        # compute the average, worst and best solution, and the optimal point
        evals = np.arange(1, out.shape[1] + 1)
        fmin = np.full(out.shape[1], problem.pareto_front())
        avg = out.mean(axis=0)
        worst = out.max(axis=0)
        best = out.min(axis=0)

        # plot the results
        h = ax.plot(evals, avg)[0]
        ax.plot(evals, worst, color=h.get_color(), lw=0.5)
        ax.plot(evals, best, color=h.get_color(), lw=0.5)
        ax.fill_between(evals, worst, best, alpha=0.2, color=h.get_color(), lw=2)
        ax.plot(evals, fmin, "--", color="grey", zorder=-1000)
        ax.set_title(problem.__class__.__name__.lower())
        ax.set_xlim(1, evals[-1])
for ax in axs[-1]:
    ax.set_xlabel("number of function evaluations")
plt.show()
