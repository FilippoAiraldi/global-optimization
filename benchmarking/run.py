"""
Benchmarking myopic and non-myopic Global Optimization strategies on benchmark problems.
"""

import argparse
import fcntl
import gc
from collections.abc import Iterable
from datetime import datetime
from itertools import cycle, product
from pathlib import Path
from time import perf_counter
from traceback import format_exc
from typing import Optional, Union
from warnings import filterwarnings, warn

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_step_lookahead import warmstart_multistep
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from joblib import Parallel, delayed
from status import filter_tasks_by_status
from torch import Tensor

from globopt import (
    GaussHermiteSampler,
    IdwAcquisitionFunction,
    Ms,
    make_idw_acq_factory,
    qIdwAcquisitionFunction,
)
from globopt.problems import get_available_benchmark_problems, get_benchmark_problem
from globopt.regression import Idw, Rbf

BENCHMARK_PROBLEMS = get_available_benchmark_problems()
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def check_methods_arg(method: str) -> str:
    """Given a `method` argument, check its value."""
    if method in ["random", "ei", "myopic", "myopic-s"]:
        return method
    elif method.startswith("ms"):
        sampler_type, *fantasies = method[3:].split(".")
        if sampler_type != "gh" and sampler_type != "mc":
            raise argparse.ArgumentTypeError(
                f"Sampler type must be either `gh` or `mc`; got {sampler_type} instead."
            )
        if len(fantasies) == 0:
            raise argparse.ArgumentTypeError(
                "Multi-step methods must have at least one fantasy."
            )
        if not all(f.isdigit() and int(f) > 0 for f in fantasies):
            raise argparse.ArgumentTypeError(
                "Fantasies must be integers (greater than or equal to 0)."
            )
        return method
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
    filterwarnings("ignore", "Optimization failed", RuntimeWarning, "botorch")
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
    n_restarts = 16 * ndim
    raw_samples = 16 * 8 * ndim

    # draw random initial points
    np_random = np.random.default_rng(seed)
    mk_seed = lambda: int(np_random.integers(0, 2**32 - 1))
    bounds: Tensor = problem.bounds
    X = (
        torch.as_tensor(np_random.random((n_init, ndim))) * (bounds[1] - bounds[0])
        + bounds[0]
    )
    Y = problem(X)

    # define mdoel and acquisition function getters
    if method == "random":

        def next_obs(*_, **__) -> tuple[Tensor, Tensor, None]:
            X_opt = torch.rand(1, ndim) * (bounds[1] - bounds[0]) + bounds[0]
            return X_opt, torch.nan, None

    elif method == "ei":

        def next_obs(
            X: Tensor, Y: Tensor, *_, **__
        ) -> tuple[Tensor, Tensor, SingleTaskGP]:
            Y_ = Y.unsqueeze(-1)
            mdl = SingleTaskGP(
                X, standardize(Y_), input_transform=Normalize(ndim, bounds=bounds)
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(mdl.likelihood, mdl))
            acqfun = ExpectedImprovement(mdl, Y.amin(), maximize=False)
            X_opt, _ = optimize_acqf(
                acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed()}
            )
            return X_opt, torch.nan, mdl

    else:
        if regression_type == "rbf":

            def get_mdl(X: Tensor, Y: Tensor, prev_mdl: Optional[Rbf]) -> Rbf:
                state = None if prev_mdl is None else prev_mdl.state
                return Rbf(X, Y, eps, init_state=state)

        else:  # regression_type == "idw":

            def get_mdl(X: Tensor, Y: Tensor, _) -> Idw:
                return Idw(X, Y)

        if method == "myopic":

            def next_obs(
                X: Tensor, Y: Tensor, prev_mdl: Union[None, Idw, Rbf], *_, **__
            ) -> tuple[Tensor, Tensor, Union[Idw, Rbf]]:
                mdl = get_mdl(X, Y, prev_mdl)
                acqfun = IdwAcquisitionFunction(mdl, c1, c2)
                X_opt, _ = optimize_acqf(
                    acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed()}
                )
                return X_opt, torch.nan, mdl

        elif method == "myopic-s":
            # NOTE: the stage reward is slightly different between myopic and myopic-s
            gh_sampler = GaussHermiteSampler(sample_shape=torch.Size([16]))

            def next_obs(
                X: Tensor, Y: Tensor, prev_mdl: Union[None, Idw, Rbf], *_, **__
            ) -> tuple[Tensor, Tensor, Union[Idw, Rbf]]:
                mdl = get_mdl(X, Y, prev_mdl)
                acqfun = qIdwAcquisitionFunction(mdl, c1, c2, sampler=gh_sampler)
                X_opt, _ = optimize_acqf(
                    acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed()}
                )
                return X_opt, torch.nan, mdl

        elif method.startswith("ms"):
            sampler_type, *fantasies_str = method[3:].split(".")
            fantasies = list(map(int, fantasies_str))

            horizon = len(fantasies) + 1
            maxfun = 15_000
            valfunc_sampler = GaussHermiteSampler(torch.Size([16]))
            kwargs_factory = make_idw_acq_factory(c1, c2)

            if sampler_type == "gh":
                fantasies_samplers = [
                    GaussHermiteSampler(torch.Size([f])) for f in fantasies
                ]
            else:
                fantasies_samplers = [
                    SobolQMCNormalSampler(torch.Size([f]), seed=mk_seed())
                    for f in fantasies
                ]

            def next_obs(
                X: Tensor,
                Y: Tensor,
                prev_mdl: Union[None, Idw, Rbf],
                prev_full_opt: Tensor,
                budget: int,
            ) -> tuple[Tensor, Tensor, Union[Idw, Rbf]]:
                mdl = get_mdl(X, Y, prev_mdl)
                remaining_horizon = min(horizon, budget)
                if remaining_horizon == 1:
                    acqfun = qIdwAcquisitionFunction(mdl, c1, c2, valfunc_sampler)
                    X_opt, _ = optimize_acqf(
                        acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed()}
                    )
                    return X_opt, torch.nan, mdl

                n_restarts_ = n_restarts * remaining_horizon
                raw_samples_ = raw_samples * remaining_horizon
                acqfun = Ms(
                    mdl,
                    fantasies_samplers,
                    qIdwAcquisitionFunction,
                    kwargs_factory,
                    valfunc_sampler=valfunc_sampler,
                )
                if prev_full_opt is torch.nan:
                    prev_full_opt = None
                else:
                    prev_full_opt = warmstart_multistep(
                        acqfun, bounds, n_restarts_, raw_samples_, prev_full_opt
                    )
                full_opt, tree_vals = optimize_acqf(
                    acqfun,
                    bounds,
                    acqfun.get_augmented_q_batch_size(1),
                    n_restarts_,
                    raw_samples_,
                    batch_initial_conditions=prev_full_opt,
                    return_best_only=False,
                    return_full_tree=True,
                    options={"seed": mk_seed(), "maxfun": maxfun},
                )
                best_tree_idx = tree_vals.argmax()
                X_opt = acqfun.extract_candidates(full_opt[best_tree_idx])
                return X_opt, full_opt, mdl

        else:
            raise NotImplementedError(f"Method {method} not implemented.")

    # run optimization loop
    mdl: Optional[Model] = None
    obs_opt: Tensor = torch.nan
    acq_opt: float = float("nan")
    full_opt: Tensor = torch.nan
    bests: list[float] = [Y.amin().item()]
    rewards: list[float] = []
    timings: list[float] = []
    try:
        for iteration in range(maxiter):
            # fit model and optimize acquisition to get the next point to sample
            start_time = perf_counter()
            obs_opt, full_opt, mdl = next_obs(X, Y, mdl, full_opt, maxiter - iteration)
            timings.append(perf_counter() - start_time)

            # compute stage reward (if applicable)
            if isinstance(mdl, (Idw, Rbf)):
                stage_cost_acqfun = IdwAcquisitionFunction(mdl, c1, c2)
                acq_opt = stage_cost_acqfun(obs_opt).item()
            else:
                acq_opt = float("nan")
            rewards.append(acq_opt)

            # evaluate objective function at new point, and append it to training data
            X = torch.cat((X, obs_opt))
            Y = torch.cat((Y, problem(obs_opt)))
            bests.append(Y.amin().item())

        # save results, delete references and free memory (at least, try to)
        rewards = ",".join(map(str, rewards))
        bests = ",".join(map(str, bests))
        timings = ",".join(map(str, timings))
        lock_write(csv, f"{problem_name};{method};{rewards};{bests};{timings}")
    except Exception:
        warn(
            f"Exception raised in `{problem_name}`, `{method}`:\n{format_exc()}",
            RuntimeWarning,
        )
    finally:
        del problem, X, Y, mdl, obs_opt, acq_opt, full_opt, bests, rewards, timings
        gc.collect()
        if device.startswith("cuda"):
            with torch.no_grad():
                torch.cuda.empty_cache()


def run_benchmarks(
    methods: Iterable[str],
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
    tasks = filter_tasks_by_status(product(range(n_trials), problems, methods), csv)
    Parallel(n_jobs=n_jobs, verbose=100, backend="loky")(
        delayed(run_problem)(prob, method, seeds[prob][trial], csv, device)
        for (trial, prob, method), device in zip(tasks, cycle(devices))
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
        type=check_methods_arg,
        nargs="+",
        help="Methods to run the benchmarking on. Greedy algorithms include `ei` and "
        " `myopic`. Non-myopic multi-step algorithms have the following semantic: "
        "`ms-sampler.m1.m2. ...`, where `ms` stands for multi-step, `sampler` is either"
        "`gh` or `mc` (for Gauss Hermite and Monte Carlo, respectively), while `m1`, "
        "`m2` and so on are the number of fantasies at each stage. The overall horizon "
        "of an `ms` method is the number of fantasies plus one.",
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
        args.methods,
        args.problems,
        args.n_trials,
        args.seed,
        args.n_jobs,
        args.csv,
        args.devices,
    )
