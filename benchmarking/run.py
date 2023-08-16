"""
Benchmarking of myopic and non-myopic Global Optimization strategies on synthetic
problems.
"""


import os

os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
from contextlib import contextmanager
from datetime import datetime
from itertools import product
from multiprocessing import shared_memory
from time import sleep

import numpy as np
from joblib import Parallel, delayed

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


@contextmanager
def shm_lock(
    shm_name: str, sleep_interval: float = 1.0, max_wait_intervals: int = 100
) -> None:
    """Emulates a lock using a shared memory object."""
    shm = shared_memory.SharedMemory(name=shm_name)
    for _ in range(max_wait_intervals):
        if shm.buf[0] == 0:
            break
        sleep(sleep_interval)
    else:
        raise RuntimeError("Timeout waiting for shared memory lock.")
    shm.buf[0] = 1
    try:
        yield
    finally:
        shm.buf[0] = 0
        shm.close()


def run_problem(
    problem_name: str, horizon: int, seed: int, output_csv: str, shm_name: str
) -> None:
    """Solves the given problem with the given algorithm (based on the specified
    horizon), and saves as result the performance of the run in terms of best-so-far
    and total cost."""
    rollout = True
    c1, c2, eps = 1.5078, 1.4246, 1.0775
    bsf_callback = BestSoFarCallback()
    dp_callback = DpStageCostCallback()
    callbacks = CallbackCollection(bsf_callback, dp_callback)
    problem, maxiter, regression_type = get_benchmark_problem(problem_name)
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
            "swarmsize": 5 * problem.dim * (rollout or horizon),
            "xtol": 1e-9,
            "ftol": 1e-9,
            "maxiter": 300,
            "patience": 10,
        },
    }
    if horizon == 1:
        go(**kwargs)
    else:
        nmgo(
            horizon=horizon,
            discount=0.9,
            rollout=rollout,
            mc_iters=0,
            parallel=None,
            **kwargs,
        )
    cost = sum(dp_callback)
    bests = ",".join(map(str, bsf_callback))
    with shm_lock(shm_name), open(output_csv, "a") as f:
        f.write(f"{problem_name},{horizon},{cost},{bests}\n")


def run_benchmarks(
    problems: list[str], horizons: list[int], trials: int, seed: int, n_jobs: int
) -> None:
    """Run the benchmarks for the given problems and horizons, repeated per the number
    of trials."""
    if problems == ["all"]:
        problems = BENCHMARK_PROBLEMS

    # seeds are independent of the horizons
    N = len(problems)
    seedseq = np.random.SeedSequence(seed)
    seeds = dict(zip(problems, np.split(seedseq.generate_state(N * trials), N)))

    # create name of csv that will be filled with the results of each iteration
    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = f"results_{nowstr}.csv"

    # create shared mem lock and launch each benchmarking iteration in parallel
    shm = shared_memory.SharedMemory(create=True, size=1)
    shm.buf[0] = 0  # set to unlocked
    try:
        Parallel(n_jobs=n_jobs, verbose=100, backend="loky")(
            delayed(run_problem)(p, h, seeds[p][t], csv, shm.name)
            for t, p, h in product(range(trials), problems, horizons)
        )
    finally:
        shm.close()
        shm.unlink()


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
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel processes."
    )
    parser.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    args = parser.parse_args()

    # run the benchmarks
    run_benchmarks(args.problems, args.horizons, args.n_trials, args.seed, args.n_jobs)
