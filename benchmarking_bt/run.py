"""
Benchmarking of myopic and non-myopic Global Optimization strategies on synthetic
problems.
"""

import argparse
import fcntl
from datetime import datetime
from itertools import product
from math import ceil, log2

import numpy as np
import torch
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from joblib import Parallel, delayed
from scipy.stats.qmc import LatinHypercube
from torch import Tensor

from globopt.myopic_acquisitions import IdwAcquisitionFunction
from globopt.problems import get_available_benchmark_problems, get_benchmark_problem
from globopt.regression import Idw, Rbf

BENCHMARK_PROBLEMS = get_available_benchmark_problems()
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
NUM_GPUS = torch.cuda.device_count()


def fnv1a_64(s: str, base_seed: int = 0) -> int:
    """Creates a 64-bit hash of the given string using the FNV-1a algorithm."""
    hash64 = FNV_OFFSET + base_seed
    for char in s:
        hash64 ^= ord(char)
        hash64 *= FNV_PRIME
        hash64 &= 0xFFFFFFFFFFFFFFFF  # Ensure hash is 64-bit
    return hash64


def lock_write(filename: str, data: str) -> None:
    """Appends data to file, locking it to prevent concurrent writes."""
    with open(filename, "a", encoding="utf-8") as f:
        try:
            fcntl.lockf(f, fcntl.LOCK_EX)
            f.write(data + "\n")
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def run_problem(
    task_id: int, problem_name: str, horizon: int, seed: int, output_csv: str
) -> None:
    torch.set_default_device(f"cuda:{task_id % NUM_GPUS}")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    myopic = horizon == 1
    problem, maxiter, regression_type = get_benchmark_problem(problem_name)
    use_rbf = regression_type == "rbf"

    # set some hyperparameters
    ndim = problem.dim
    n_init = ndim * 2
    c1 = 1.0 / ndim
    c2 = 0.5 / ndim
    if use_rbf:
        eps = 1.0 / ndim
        Minv_and_coeffs = None
    num_restarts = 16 * ndim
    raw_samples = 32 * ndim

    # set various RNG
    np_random = np.random.default_rng(seed)
    mk_seed = lambda: int(np_random.integers(0, 2**32 - 1))
    lhs = LatinHypercube(ndim, seed=mk_seed())
    if not myopic:
        mc_samples = 2 ** ceil(log2(256 * horizon))
        SobolQMCNormalSampler(mc_samples, seed=mk_seed())

    # draw random initial points
    bounds: Tensor = problem.bounds
    X = torch.as_tensor(lhs.random(n_init)) * (bounds[1] - bounds[0]) + bounds[0]
    Y = problem(X)

    # run optimization loop
    best_so_far: list[float] = [Y.amin().item()]
    stage_rewards: list[float] = []
    for _ in range(maxiter):
        # fit model
        mdl = Rbf(X, Y, eps, Minv_and_coeffs=Minv_and_coeffs) if use_rbf else Idw(X, Y)

        # minimize acquisition function
        if myopic:
            acqfun = IdwAcquisitionFunction(mdl, c1, c2)
        else:
            raise NotImplementedError
        X_opt, acq_opt = optimize_acqf(
            acqfun, bounds, 1, num_restarts, raw_samples, {"seed": mk_seed()}
        )

        # evaluate objective function at the new point, and append it to training data
        X = torch.cat((X, X_opt))
        Y = torch.cat((Y, problem(X_opt)))
        if use_rbf:
            Minv_and_coeffs = mdl.Minv_and_coeffs

        # compute and save best-so-far and incurred cost
        best_so_far.append(Y.amin().item())
        if myopic:
            stage_rewards.append(acq_opt.item())
        else:
            raise NotImplementedError  # TODO: compute myopic acquisition manually

    # save results, delete references and free memory (at least, try to)
    rewards = ",".join(map(str, stage_rewards))
    bests = ",".join(map(str, best_so_far))
    lock_write(output_csv, f"{problem_name};{horizon};{rewards};{bests}")
    del problem, mdl, acqfun, X, Y, X_opt, best_so_far, stage_rewards
    torch.cuda.empty_cache()


def run_benchmarks(
    problems: list[str],
    horizons: list[int],
    trials: int,
    seed: int,
    n_jobs: int,
    csv: str,
) -> None:
    """Run the benchmarks for the given problems and horizons, repeated per the number
    of trials."""
    if problems == ["all"]:
        problems = BENCHMARK_PROBLEMS
    # create one seed per each trial. These are independent of the horizons, so that
    # myopic and non-myopic methods start with the same initial conditions; moreover,
    # they are crafted out of the name of the problem, so that new benchmarks can be
    # added without having to simulate everything again. Lastly, they are also
    # independent of the number of trials, so that new trials can be appended freely.
    seeds = {
        p: np.random.SeedSequence(fnv1a_64(p, seed)).generate_state(trials)
        for p in problems
    }
    tasks = product(range(trials), problems, horizons)  # TODO: rework task filtering
    n_jobs = NUM_GPUS * min(1, n_jobs)
    Parallel(n_jobs=n_jobs, verbose=100, backend="loky")(
        delayed(run_problem)(i, p, h, seeds[p][t], csv)
        for i, (t, p, h) in enumerate(tasks)
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available!"

    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Benchmarking of GO strategies on synthetic problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        choices=["all"] + BENCHMARK_PROBLEMS,
        nargs="+",
        default=["all"],
        help="Problems to include in the benchmarking.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6],
        help="Horizons for non-myopic strategies to benchmark.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Number of trials to run per problem."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=2, help="Number (positive) of parallel processes."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--csv", type=str, default="", help="Output csv filename.")
    args = parser.parse_args()

    # if the output csv is not specified, create it
    if args.csv is None or args.csv == "":
        nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv = f"results_{nowstr}.csv"
    elif not args.csv.endswith(".csv"):
        args.csv += ".csv"

    # run the benchmarks
    run_benchmarks(
        args.problems, args.horizons, args.n_trials, args.seed, args.n_jobs, args.csv
    )


#  python benchmarking/run.py --problems=all --horizons 1 2 4 6 --n-trials=30 --n-jobs=6 --seed=0
