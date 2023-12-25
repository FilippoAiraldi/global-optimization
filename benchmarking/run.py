"""
Benchmarking of myopic and non-myopic Global Optimization strategies on synthetic
problems.
"""

import argparse
import fcntl
from collections.abc import Iterable
from datetime import datetime
from functools import partial
from itertools import chain, cycle, product
from pathlib import Path
from time import perf_counter
from typing import Union

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from joblib import Parallel, delayed
from scipy.stats.qmc import LatinHypercube
from status import filter_tasks_by_status
from torch import Tensor

from globopt.myopic_acquisitions import (
    GaussHermiteSampler,
    IdwAcquisitionFunction,
    qIdwAcquisitionFunction,
)
from globopt.problems import get_available_benchmark_problems, get_benchmark_problem
from globopt.regression import Idw, Rbf

BENCHMARK_PROBLEMS = get_available_benchmark_problems()
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def convert_methods_arg(method: str) -> Iterable[str]:
    """Given a `method` argument, decodes the horizon values from it, and returns an
    iterable of (method, horizon)-strings (if the method is myopic, the horizon is not
    included in the string)."""
    if method in ["ei", "myopic", "s-myopic"]:
        yield method
    elif method.startswith("rollout") or method.startswith("multi-tree"):
        method, *horizons = method.split(".")
        for horizon in horizons:
            if int(horizon) < 2:
                raise argparse.ArgumentTypeError(
                    "Horizons for non-myopic methods must be greater than 1; got "
                    f"{method} with horizon {horizon} instead."
                )
            yield f"{method}.{horizon}"
    else:
        raise argparse.ArgumentTypeError(f"Unrecognized method {method}.")


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
    problem_name: str, method: str, seed: int, csv: str, device: str
) -> None:
    """Runs the given problem, with the given horizon, and writes the results to csv."""
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    problem, maxiter, regression_type = get_benchmark_problem(problem_name)

    # set hyperparameters
    ndim = problem.dim
    n_init = ndim * 2
    c1 = torch.scalar_tensor(1.0 / ndim)
    c2 = torch.scalar_tensor(0.5 / ndim)
    eps = torch.scalar_tensor(1.0 / ndim)
    num_restarts = 16 * ndim
    raw_samples = 16 * 8 * ndim
    gh_sampler = GaussHermiteSampler(sample_shape=torch.Size([16]))
    # mc_samples = 2 ** ceil(log2(256 * horizon))
    # sampler = SobolQMCNormalSampler(mc_samples, seed=mk_seed())

    # draw random initial points via LHS
    np_random = np.random.default_rng(seed)
    mk_seed = lambda: int(np_random.integers(0, 2**32 - 1))
    lhs = LatinHypercube(ndim, seed=mk_seed())
    bounds: Tensor = problem.bounds
    X = torch.as_tensor(lhs.random(n_init)) * (bounds[1] - bounds[0]) + bounds[0]
    Y = problem(X)

    # define mdoel and acquisition function getters
    if method == "ei":

        def get_mdl(X: Tensor, Y: Tensor, _) -> SingleTaskGP:
            Y = Y.unsqueeze(-1)
            mdl = SingleTaskGP(
                X, standardize(Y), input_transform=Normalize(ndim, bounds=bounds)
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(mdl.likelihood, mdl))
            mdl.train_Y = Y
            return mdl

        def get_next_point_and_reward(model: SingleTaskGP) -> tuple[Tensor, float]:
            acqfun = ExpectedImprovement(model, model.train_Y.amin(), maximize=False)
            X_opt, _ = optimize_acqf(
                acqfun, bounds, 1, num_restarts, raw_samples, {"seed": mk_seed()}
            )
            return X_opt, torch.nan

    elif method == "myopic" or method == "s-myopic":
        if regression_type == "rbf":

            def get_mdl(X: Tensor, Y: Tensor, prev_mdl: Rbf) -> Rbf:
                Mc = None if prev_mdl is None else prev_mdl.Minv_and_coeffs
                return Rbf(X, Y, eps, Minv_and_coeffs=Mc)

        elif regression_type == "idw":

            def get_mdl(X: Tensor, Y: Tensor, _) -> Idw:
                return Idw(X, Y)

        else:
            raise RuntimeError(f"Unrecognized regression type {regression_type}.")

        acqf_cls = (
            IdwAcquisitionFunction
            if method == "myopic"
            else partial(qIdwAcquisitionFunction, sampler=gh_sampler)
        )

        def get_next_point_and_reward(model: Union[Rbf, Idw]) -> tuple[Tensor, float]:
            acqfun = acqf_cls(model, c1, c2)
            X_opt, acq_opt = optimize_acqf(
                acqfun, bounds, 1, num_restarts, raw_samples, {"seed": mk_seed()}
            )
            return X_opt, acq_opt.item()

    else:
        raise NotImplementedError

    # run optimization loop
    prev_mdl = None
    best_so_far: list[float] = [Y.amin().item()]
    stage_reward: list[float] = []
    iteration_times: list[float] = []
    for _ in range(maxiter):
        start_time = perf_counter()

        # fit model and optimize acquisition function to get the next point to sample
        mdl = get_mdl(X, Y, prev_mdl)
        X_opt, acq_opt = get_next_point_and_reward(mdl)

        # evaluate objective function at the new point, and append it to training data
        X = torch.cat((X, X_opt))
        Y = torch.cat((Y, problem(X_opt)))

        # compute and save best-so-far and incurred cost
        best_so_far.append(Y.amin().item())
        stage_reward.append(acq_opt)
        prev_mdl = mdl
        iteration_times.append(perf_counter() - start_time)

    # save results, delete references and free memory (at least, try to)
    rewards = ",".join(map(str, stage_reward))
    bests = ",".join(map(str, best_so_far))
    times = ",".join(map(str, iteration_times))
    lock_write(csv, f"{problem_name};{method};{rewards};{bests};{times}")
    del problem, mdl, X, Y, X_opt, best_so_far, stage_reward
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_benchmarks(
    methods_and_horizons: Iterable[str],
    problems: list[str],
    n_trials: int,
    seed: int,
    n_jobs: int,
    csv: str,
    devices: list[torch.device],
) -> None:
    """Runs the benchmarks for the given problems, methods and horizons, repeated per
    the number of trials, distributively across the given devices."""
    # for each problem, create one seed per trial. These are independent of the method
    # and horizon, so that myopic and non-myopic algorithms start with the same initial
    # conditions; moreover, they are crafted out of the name of the problem, so that new
    # benchmarks (with different, new names) can be added without having to simulate all
    # again. Lastly, they are also subsquent, so that new trials can be appended freely.
    if problems == ["all"]:
        problems = BENCHMARK_PROBLEMS
    seeds = {
        p: np.random.SeedSequence(fnv1a_64(p, seed)).generate_state(n_trials)
        for p in problems
    }
    tasks = filter_tasks_by_status(
        product(range(n_trials), problems, methods_and_horizons), csv
    )
    Parallel(n_jobs=n_jobs, verbose=100, backend="loky")(
        delayed(run_problem)(prob, methodhor, seeds[prob][trial], csv, device)
        for (trial, prob, methodhor), device in zip(tasks, cycle(devices))
    )


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Benchmarking of Global Optimization strategies on synthetic "
        "benchmark problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Benchmarking options")
    group.add_argument(
        "--methods",
        type=convert_methods_arg,
        nargs="+",
        help="Methods to run the benchmarking on. Greedy algorithms include `ei` and "
        " `myopic`. Non-myopic algorithms are `rollout` and `multi-tree`, where the "
        "horizons to simulate can be specified with a dot folloewd by (one or more) "
        "horizons, e.g., `rollout.2.3`. These horizons should be larger than 1. "
        "Methods are: `ei`, `myopic`, `s-myopic`.",
        required=True,
    )
    group.add_argument(
        "--problems",
        choices=["all"] + BENCHMARK_PROBLEMS,
        nargs="+",
        default=["all"],
        help="Problems to include in the benchmarking.",
    )
    group.add_argument(
        "--n-trials", type=int, default=30, help="Number  of trials to run per problem."
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument(
        "--n-jobs", type=int, default=2, help="Number (positive) of parallel processes."
    )
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument("--csv", type=str, default="", help="Output csv filename.")
    group.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu"],
        help="List of torch devices to use, e.g., `cpu`, `cuda:0`, etc..",
    )
    args = parser.parse_args()

    # if the output csv is not specified, create it, and write header if anew
    if args.csv is None or args.csv == "":
        args.csv = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    elif not args.csv.endswith(".csv"):
        args.csv += ".csv"
    if not Path(args.csv).is_file():
        lock_write(args.csv, "problem;method;stage-reward;best-so-far;time")

    # run the benchmarks
    run_benchmarks(
        chain.from_iterable(args.methods),
        args.problems,
        args.n_trials,
        args.seed,
        args.n_jobs,
        args.csv,
        args.devices,
    )
