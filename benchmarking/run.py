"""
Benchmarking of myopic and non-myopic Global Optimization strategies on synthetic
problems.
"""


import argparse
from datetime import datetime
from itertools import groupby, product
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
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

MAX_SEED = 2**32
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
BENCHMARK_PROBLEMS = get_available_benchmark_problems()
SIMPLE_PROBLEMS = get_available_simple_problems()


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
        kwargs = {"horizon": h, "discount": 0.8277, "shrink_horizon": True}
        # c1, c2, eps = 1.2389, 2.5306, 0.8609
        # kwargs = {"horizon": h, "discount": 0.8151}
    termination = DefaultSingleObjectiveTermination(ftol=1e-4, n_max_gen=300, period=10)
    return cls(
        regression=Rbf(eps=eps / n_var) if regression == "rbf" else Idw(),
        init_points=2 * n_var,
        acquisition_min_algorithm=PSO(pop_size=10),  # size will be scaled with n_var
        acquisition_min_kwargs={"termination": termination},
        c1=c1 / n_var,
        c2=c2 / n_var,
        **kwargs,  # type: ignore[arg-type]
    )


def run_benchmark(problem_name: str, h: int, seed: int) -> list[float]:
    """Solves the problem with the given horizon and seed."""
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
        seed=(seed ^ fnv1a(problem_name)) % MAX_SEED,
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
        print(f"Solving {name.upper()}, h={h}, iter={n_trial + 1}")
        return f"{name}_h{h}", run_benchmark(name, h, seed + n_trial)

    results: list[tuple[str, list[float]]] = Parallel(n_jobs=-1, verbose=100)(
        delayed(_run)(name, h, trial)
        for name, h, trial in product(problem_names, horizons, range(n_trials))
    )
    data: dict[str, Array] = {
        k: np.array([e[1] for e in g]) for k, g in groupby(results, key=lambda o: o[0])
    }

    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    savemat(f"sbm_{nowstr}.mat", data)
    return data


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Benchmarking of GO strategies on synthetic problems.",
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
    args = parser.parse_args()

    # run the benchmarks
    results = run_benchmarks(args.problems, args.horizons, args.n_trials, args.seed)
