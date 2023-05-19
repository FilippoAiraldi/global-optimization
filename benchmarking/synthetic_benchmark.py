import argparse
from datetime import datetime
from itertools import groupby, product, zip_longest
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.ticker import MaxNLocator
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from scipy.io import savemat

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Array, Idw, Rbf
from globopt.myopic.algorithm import GO, Algorithm
from globopt.nonmyopic.algorithm import NonMyopicGO
from globopt.util.callback import BestSoFarCallback

plt.style.use("bmh")

MAX_SEED = 2**32
BENCHMARK_PROBLEMS = get_available_benchmark_problems()
SIMPLE_PROBLEMS = get_available_simple_problems()
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619


def fnv1a(s: str) -> int:
    """Hashes a string using the FNV-1a algorithm."""
    return sum((FNV_OFFSET ^ b) * FNV_PRIME**i for i, b in enumerate(s.encode()))


def get_algorithm(h: int, n_var: int, regression: Literal["rbf", "idw"]) -> Algorithm:
    """Returns the algorithm to be used for the given horizon and regression."""
    if h == 1:
        cls = GO
        c1, c2, eps = 1.5078, 1.4246, 1.0775
        kwargs = {}
    else:
        cls = NonMyopicGO  # type: ignore[assignment]
        c1, c2, eps = 1.0887, 2.7034, 1.8473
        kwargs = {"horizon": h, "discount": 0.8277}
        # c1, c2, eps = 1.2389, 2.5306, 0.8609
        # kwargs = {"horizon": h, "discount": 0.8151}
    termination = DefaultSingleObjectiveTermination(ftol=1e-4, n_max_gen=300, period=10)
    return cls(
        regressor=Rbf(eps=eps / n_var) if regression == "rbf" else Idw(),
        init_points=2 * n_var,
        acquisition_min_algorithm=PSO(pop_size=10),  # size will be scaled with n_var
        acquisition_min_kwargs={"termination": termination},
        acquisition_fun_kwargs={"c1": c1 / n_var, "c2": c2 / n_var},
        **kwargs,  # type: ignore[arg-type]
    )


def run_benchmark(problem_name: str, h: int, seed: int) -> list[float]:
    """Solves the problem with the given horizon and seed."""
    seed += (h ^ fnv1a(problem_name)) % MAX_SEED
    problem, max_n_iter, regression = get_benchmark_problem(problem_name)
    algorithm = get_algorithm(h, problem.n_var, regression)
    callback = BestSoFarCallback()
    minimize(
        problem,
        algorithm,
        termination=("n_iter", max_n_iter),
        copy_algorithm=False,  # no need to copy the algorithm, it is freshly created
        callback=callback,
        verbose=False,
        seed=seed,
    )
    return callback.data["best"]


def run_benchmarks(
    problem_names: list[str], horizons: list[int], n_trials: int, seed: int
) -> dict[str, Array]:
    """Run the benchmarks for the given problems and horizons, repeated per the number
    of trials."""
    if problem_names == ["all"]:
        problem_names = BENCHMARK_PROBLEMS

    def _run(name: str, h: int, n_trial: int) -> tuple[str, list[float]]:
        print(f"Solving {name.upper()}, iteration {n_trial + 1}")
        return f"{name}_h{h}", run_benchmark(name, h, seed + n_trial)

    results: list[tuple[str, list[float]]] = Parallel(n_jobs=-1, verbose=10)(
        delayed(_run)(name, h, trial)
        for name, h, trial in product(problem_names, horizons, range(n_trials))
    )
    data: dict[str, Array] = {
        k: np.array([e[1] for e in g]) for k, g in groupby(results, key=lambda o: o[0])
    }

    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    savemat(f"sbm_{nowstr}.mat", data)
    return data


def plot_results(data_: dict[str, Array]) -> None:
    data: dict[str, dict[int, Array]] = {}
    for k, v in data_.items():
        name, h_name = k.split("_")
        h = int(h_name[1:])
        if name not in data:
            data[name] = {}
        data[name][h] = v

    n_rows = 2
    n_cols = np.ceil(len(data) / n_rows).astype(int)
    fig, axs = plt.subplots(n_cols, n_rows, constrained_layout=True)
    axs = np.atleast_2d(axs)
    for i, (ax, problem_name) in enumerate(zip_longest(axs.flat, data.keys())):
        if problem_name is None:
            # if there are more axes than results, set the axis off
            ax.set_axis_off()
            continue

        for horizon, results in data[problem_name].items():
            # compute the average, worst and best solution
            evals = np.arange(1, results.shape[1] + 1)
            avg = results.mean(axis=0)
            worst = results.max(axis=0)
            best = results.min(axis=0)

            # plot the results
            lbl = f"h={horizon}" if i == 0 else None
            c = ax.plot(evals, avg, label=lbl)[0].get_color()
            ax.plot(evals, worst, color=c, lw=0.25)
            ax.plot(evals, best, color=c, lw=0.25)
            ax.fill_between(evals, worst, best, alpha=0.2, color=c)

        # plot the optimal point
        problem = get_benchmark_problem(problem_name)[0]
        fmin = np.full(results.shape[1], problem.pareto_front())
        ax.plot(evals, fmin, "--", color="grey", zorder=-10000)
        ax.set_title(problem.__class__.__name__.lower(), fontsize=11)
        ax.set_xlim(1, evals[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    for ax in axs[-1]:
        ax.set_xlabel("number of function evaluations")
    fig.legend(loc="outside lower center", ncol=len(next(iter(data.values()))))
    plt.show()


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Fine-tuning of GO strategies via Bayesian Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        choices=["all"] + BENCHMARK_PROBLEMS + SIMPLE_PROBLEMS,
        nargs="+",
        default=["all"],
        help="Problems to include in the benchmarking.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Horizons for non-myopic strategies to benchmark.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Number of trials to run per problem."
    )
    parser.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plots the results at the end of the run.",
    )
    args = parser.parse_args()

    # run the benchmarks
    results = run_benchmarks(args.problems, args.horizons, args.n_trials, args.seed)

    # plot the results, if requested
    if args.plot:
        plot_results(results)
