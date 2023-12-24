"""
Benchmarking of myopic and non-myopic Global Optimization strategies on synthetic
problems.
"""

import argparse
import fcntl
from collections.abc import Iterable
from datetime import datetime
from itertools import chain, cycle, product
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from joblib import Parallel, delayed
from scipy.stats.qmc import LatinHypercube
from torch import Tensor

from globopt.myopic_acquisitions import IdwAcquisitionFunction
from globopt.problems import get_available_benchmark_problems, get_benchmark_problem
from globopt.regression import Idw, Rbf

BENCHMARK_PROBLEMS = get_available_benchmark_problems()
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def convert_methods_arg(method: str) -> Iterable[str]:
    """Given a `method` argument, decodes the horizon values from it, and returns an
    iterable of (method, horizon)-strings (if the method is myopic, the horizon is not
    included in the string)."""
    if method.startswith("ei") or method.startswith("myopic"):
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
    c1 = 1.0 / ndim
    c2 = 0.5 / ndim
    eps = 1.0 / ndim
    num_restarts = 16 * ndim
    raw_samples = 32 * ndim

    # draw random initial points via LHS
    np_random = np.random.default_rng(seed)
    mk_seed = lambda: int(np_random.integers(0, 2**32 - 1))
    lhs = LatinHypercube(ndim, seed=mk_seed())
    bounds: Tensor = problem.bounds
    X = torch.as_tensor(lhs.random(n_init)) * (bounds[1] - bounds[0]) + bounds[0]
    Y = problem(X)
    # mc_samples = 2 ** ceil(log2(256 * horizon))
    # sampler = SobolQMCNormalSampler(mc_samples, seed=mk_seed())

    # define mdoel and acquisition function getters
    # TODO: different getters for each method
    if method == "ei":

        def get_mdl(X: Tensor, Y: Tensor, prev_mdl: SingleTaskGP) -> SingleTaskGP:
            # TODO: standardize Y, create model, fit it, load dict state
            pass

        get_acqfun = lambda mdl, Y: ExpectedImprovement(mdl, Y.amin(), maximize=False)

    elif method == "myopic":
        if regression_type == "rbf":

            def get_mdl(X: Tensor, Y: Tensor, prev_mdl: Rbf) -> Rbf:
                Mc = None if prev_mdl is None else prev_mdl.Minv_and_coeffs
                return Rbf(X, Y, eps, Minv_and_coeffs=Mc)

        elif regression_type == "idw":
            get_mdl = lambda X, Y, _: Idw(X, Y)
        else:
            raise RuntimeError(f"Unrecognized regression type {regression_type}.")

        get_acqfun = lambda mdl, *_: IdwAcquisitionFunction(mdl, c1, c2)
    else:
        raise NotImplementedError

    # run optimization loop
    prev_mdl = None
    best_so_far: list[float] = [Y.amin().item()]
    stage_reward: list[float] = []
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(maxiter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(maxiter)]
    for i in range(maxiter):
        start_events[i].record()

        # fit model and optimize acquisition function
        mdl = get_mdl(X, Y, prev_mdl)
        acqfun = get_acqfun(mdl, Y)
        X_opt, acq_opt = optimize_acqf(
            acqfun, bounds, 1, num_restarts, raw_samples, {"seed": mk_seed()}
        )

        # evaluate objective function at the new point, and append it to training data
        X = torch.cat((X, X_opt))
        Y = torch.cat((Y, problem(X_opt)))

        # compute and save best-so-far and incurred cost
        best_so_far.append(Y.amin().item())
        stage_reward.append(
            acq_opt.item()
            if isinstance(acqfun, IdwAcquisitionFunction)
            else IdwAcquisitionFunction(mdl, c1, c2)(X_opt).item()
        )
        prev_mdl = mdl
        end_events[i].record()

    # save results, delete references and free memory (at least, try to)
    torch.cuda.synchronize()
    rewards = ",".join(map(str, stage_reward))
    bests = ",".join(map(str, best_so_far))
    times = ",".join(str(s.elapsed_time(e)) for s, e in zip(start_events, end_events))
    lock_write(csv, f"{problem_name};{method};{rewards};{bests};{times}")
    del problem, mdl, acqfun, X, Y, X_opt, best_so_far, stage_reward
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
    # TODO: rework task filtering
    tasks = product(range(n_trials), problems, methods_and_horizons)
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
        "horizons, e.g., `rollout.2.3`. These horizons should be larger than 1.",
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
