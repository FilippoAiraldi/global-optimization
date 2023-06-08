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
from pymoo.core.callback import CallbackCollection
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
from globopt.util.callback import BestSoFarCallback, DPStageCostCallback
from globopt.util.random import make_seeds

BENCHMARK_PROBLEMS = get_available_benchmark_problems()
SIMPLE_PROBLEMS = get_available_simple_problems()


def get_algorithm(h: int, n_var: int, regression: Literal["rbf", "idw"]) -> Algorithm:
    """Returns the algorithm to be used for the given horizon and regression."""
    c1, c2, eps = 1.5078, 1.4246, 1.0775
    termination = DefaultSingleObjectiveTermination(ftol=1e-4, n_max_gen=300, period=5)
    kwargs = {
        "regression": Rbf(eps=eps / n_var) if regression == "rbf" else Idw(),
        "init_points": 2 * n_var,
        "acquisition_min_algorithm": PSO(20),
        "acquisition_min_kwargs": {"termination": termination},
        "c1": c1 / n_var,
        "c2": c2 / n_var,
    }
    if h == 1:
        cls = GO
    else:
        cls = NonMyopicGO  # type: ignore[assignment]
        kwargs.update({"horizon": h, "discount": 0.9, "rollout_algorithm": PSO(20)})
    return cls(**kwargs)


def run_benchmark(problem_name: str, h: int, seed: int) -> tuple[list[float], float]:
    """Solves the problem with the given horizon and seed, and returns as result the
    performance of the algorithm in terms of best-so-far and total cost."""
    problem, max_n_iter, regression = get_benchmark_problem(problem_name)
    algorithm = get_algorithm(h, problem.n_var, regression)
    bsf_callback = BestSoFarCallback()
    dp_callback = DPStageCostCallback()
    minimize(
        problem,
        algorithm,
        termination=("n_iter", max_n_iter),
        callback=CallbackCollection(bsf_callback, dp_callback),
        verbose=False,
        copy_algorithm=False,  # no need to copy the algorithm, it is freshly created
        seed=seed,
    )
    return bsf_callback.data["best"], sum(dp_callback.data["cost"])


def run_benchmarks(
    problem_names: list[str], horizons: list[int], n_trials: int, seed: int
) -> dict[str, Array]:
    """Run the benchmarks for the given problems and horizons, repeated per the number
    of trials."""
    if problem_names == ["all"]:
        problem_names = BENCHMARK_PROBLEMS

    def _run(name: str, h: int, n_trial: int, seed: int) -> tuple[str, list[float]]:
        print(f"{name.upper()}: h={h}, iter={n_trial + 1} | seed={seed}")
        bsf, J = run_benchmark(name, h, seed)
        bsf.append(J)
        return f"{name}_h{h}", bsf

    seeds = make_seeds(str(seed) + "".join(problem_names))
    results: list[tuple[str, list[float]]] = Parallel(n_jobs=-1, verbose=100)(
        delayed(_run)(name, h, trial, next(seeds))
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
