"""
Benchmarking of myopic and non-myopic Global Optimization strategies on synthetic
problems.
"""


import argparse
from datetime import datetime
from itertools import chain, groupby, product

import numpy as np
from joblib import Parallel, delayed
from scipy.io import savemat
from vpso.typing import Array2d

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Idw, Rbf
from globopt.myopic.algorithm import go
from globopt.nonmyopic.algorithm import nmgo
from globopt.util.callback import (
    BestSoFarCallback,
    CallbackCollection,
    DpStageCostCallback,
)

BENCHMARK_PROBLEMS = get_available_benchmark_problems()
SIMPLE_PROBLEMS = get_available_simple_problems()
BATCHES = 2


def run_benchmark(problem_name: str, h: int, seed: int) -> tuple[list[float], float]:
    """Solves the problem with the given horizon and seed, and returns as result the
    performance of the algorithm in terms of best-so-far and total cost."""
    problem, maxiter, regression_type = get_benchmark_problem(problem_name)
    bsf_callback = BestSoFarCallback()
    dp_callback = DpStageCostCallback()
    callbacks = CallbackCollection(bsf_callback, dp_callback)
    c1, c2, eps = 1.5078, 1.4246, 1.0775
    rollout = True
    kwargs = {
        "func": problem.f,
        "lb": problem.lb,
        "ub": problem.ub,
        "mdl": Rbf(eps=eps / problem.dim) if regression_type == "rbf" else Idw(),
        "init_points": problem.dim,
        "c1": c1 / problem.dim,
        "c2": c2 / problem.dim,
        "maxiter": maxiter,
        "seed": seed,
        "callback": callbacks,
        "pso_kwargs": {
            "swarmsize": 5 * problem.dim * (rollout or h),
            "xtol": 1e-9,
            "ftol": 1e-9,
            "maxiter": 300,
            "patience": 10,
        },
    }
    if h == 1:
        _ = go(**kwargs)
    else:
        _ = nmgo(
            horizon=h,
            discount=0.9,
            rollout=rollout,
            mc_iters=0,
            parallel=Parallel(n_jobs=1, verbose=0, backend="loky"),
            **kwargs,
        )
    return bsf_callback, sum(dp_callback)


def run_benchmarks(
    problems: list[str], horizons: list[int], trials: int, seed: int
) -> dict[str, Array2d]:
    """Run the benchmarks for the given problems and horizons, repeated per the number
    of trials."""
    if problems == ["all"]:
        problems = BENCHMARK_PROBLEMS

    # split the main loop in batches (seems to speed up computations)
    assert trials % BATCHES == 0, f"number of trials not divisible by {BATCHES} batches"
    trials_per_batch = trials // BATCHES

    # create seeds (are independent of the horizons)
    seeds = dict(
        zip(
            problems,
            np.random.SeedSequence(seed)
            .generate_state(len(problems) * trials)
            .reshape(-1, BATCHES, trials_per_batch),
        )
    )

    def _run(name: str, h: int, batch: int, trial: int) -> tuple[str, list[float]]:
        bsf, J = run_benchmark(name, h, seeds[name][batch, trial])
        bsf.append(J)
        return f"{name}_h{h}", bsf

    data = chain.from_iterable(
        Parallel(n_jobs=1, verbose=100, backend="loky")(
            delayed(_run)(n, h, b, t)
            for n, h, t in product(problems, horizons, range(trials_per_batch))
        )
        for b in range(BATCHES)
    )
    results: dict[str, Array2d] = {
        k: np.asarray([e[1] for e in g]) for k, g in groupby(data, key=lambda o: o[0])
    }

    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    savemat(f"sbm_{nowstr}.mat", results)
    return results


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
