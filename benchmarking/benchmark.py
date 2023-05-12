import argparse
from datetime import datetime
from itertools import zip_longest
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.ticker import MaxNLocator
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from scipy.io import savemat

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Array, Idw, Rbf
from globopt.myopic.algorithm import GO
from globopt.nonmyopic.algorithm import NonMyopicGO
from globopt.util.callback import BestSoFarCallback

plt.style.use("bmh")


def fnv1a(s: str) -> int:
    """Hashes a string using the FNV-1a algorithm."""
    FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
    return sum((FNV_OFFSET ^ b) * FNV_PRIME**i for i, b in enumerate(s.encode()))


def solve_problem(
    algorithms: Literal["myopic", "nonmyopic", "all"],
    problem_name: str,
    n_trials: int,
    seed: int,
) -> tuple[Problem, Array, Array]:
    """Solve the problem with the given algorithms."""
    problem, max_n_iter, regression = get_benchmark_problem(problem_name)
    n_var = problem.n_var
    # TODO: these values are to be fine-tuned
    eps, c1, c2 = 1.0775 / n_var, 1.5078 / n_var, 1.4246 / n_var
    kwargs = {
        "regressor": Rbf(eps=eps) if regression == "rbf" else Idw(),
        "init_points": 2 * n_var,
        "acquisition_min_algorithm": PSO(pop_size=10),  # size will be scaled with n_var
        "acquisition_min_kwargs": {
            "termination": DefaultSingleObjectiveTermination(
                ftol=1e-4, n_max_gen=300, period=10
            )
        },
        "acquisition_fun_kwargs": {"c1": c1, "c2": c2},
    }
    ma = nma = None
    if algorithms in ("all", "myopic"):
        ma = GO(**kwargs)
    if algorithms in ("all", "nonmyopic"):
        horizon, discount = 2, 1.0  # TODO: these values are to be fine-tuned
        nma = NonMyopicGO(**kwargs, horizon=horizon, discount=discount)

    seed += fnv1a(problem_name)
    mout = np.empty((n_trials, max_n_iter)) if ma is not None else np.nan
    nmout = np.empty((n_trials, max_n_iter)) if nma is not None else np.nan
    for i in range(n_trials):
        print(f"Solving {problem_name.upper()}, iteration {i + 1}")
        seed_ = (seed + i * 2**7) % 2**32
        for algo, out in zip((ma, nma), (mout, nmout)):
            if algo is not None:
                res = minimize(
                    problem,
                    algo,
                    termination=("n_iter", max_n_iter),
                    callback=BestSoFarCallback(),
                    verbose=False,
                    seed=seed_,
                )
                out[i] = res.algorithm.callback.data["best"]  # type: ignore[index]
    return problem, mout, nmout  # type: ignore[return-value]


def run_benchmarks(
    algorithms: Literal["myopic", "nonmyopic", "all"],
    testnames: list[str],
    n_trials: int,
    seed: int,
) -> list[tuple[Problem, Array, Array]]:
    """Run the benchmarks for the given algorithsm and problems."""
    if testnames == ["all"]:
        testnames = get_available_benchmark_problems()

    results: list[tuple[Problem, Array, Array]] = Parallel(n_jobs=-1, verbose=10)(
        delayed(solve_problem)(algorithms, name, n_trials, seed) for name in testnames
    )

    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    data: dict[str, Array] = {}
    for name, (_, mout, nmout) in zip(testnames, results):
        data[f"myopic_{name}"] = mout
        data[f"nonmyopic_{name}"] = nmout
    savemat(f"bm_{nowstr}.mat", data)

    return results


def plot_results(res: list[tuple[Problem, Array, Array]]) -> None:
    n_rows = 2
    n_cols = np.ceil(len(res) / n_rows).astype(int)
    _, axs = plt.subplots(n_cols, n_rows, constrained_layout=True)
    axs = np.atleast_2d(axs)
    lbls = ("Myopic", "Non-myopic")

    for i, (ax, problem, mout, nmout) in enumerate(zip_longest(axs.flat, *zip(*res))):
        if problem is None:
            # if there are more axes than results, set the axis off
            ax.set_axis_off()
        else:
            for out, lbl in zip((mout, nmout), lbls):
                if np.isscalar(out):
                    continue

                # compute the average, worst and best solution, and the optimal point
                evals = np.arange(1, out.shape[1] + 1)
                fmin = np.full(out.shape[1], problem.pareto_front())
                avg = out.mean(axis=0)
                worst = out.max(axis=0)
                best = out.min(axis=0)

                # plot the results
                c = ax.plot(evals, avg, label=lbl)[0].get_color()
                ax.plot(evals, worst, color=c, lw=0.25)
                ax.plot(evals, best, color=c, lw=0.25)
                ax.fill_between(evals, worst, best, alpha=0.2, color=c)

            ax.plot(evals, fmin, "--", color="grey", zorder=-10000)
            ax.set_title(problem.__class__.__name__.lower(), fontsize=11)
            ax.set_xlim(1, evals[-1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if i == 0:
            ax.legend()
    for ax in axs[-1]:
        ax.set_xlabel("number of function evaluations")
    plt.show()


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Fine-tuning of GO strategies via Bayesian Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithms",
        choices=("myopic", "nonmyopic", "all"),
        default="all",
        help="Algorithms to test in the benchmarks.",
    )
    parser.add_argument(
        "--problems",
        choices=["all"]
        + get_available_benchmark_problems()
        + get_available_simple_problems(),
        nargs="+",
        default=["all"],
        help="Problems to include in the benchmarking.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of trials to run per problem."
    )
    parser.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plots the results at the end of the run.",
    )
    args = parser.parse_args()

    # run the benchmarks
    results = run_benchmarks(args.algorithms, args.problems, args.n_trials, args.seed)

    # plot the results, if requested
    if args.plot:
        plot_results(results)
