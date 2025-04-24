"""
Benchmarking myopic and non-myopic Global Optimization strategies on benchmark problems.
"""

import argparse
import gc
import os
import sys
from collections.abc import Iterable
from itertools import cycle, product
from pathlib import Path
from time import perf_counter
from traceback import format_exc
from typing import Callable, Literal, Optional, Union
from warnings import filterwarnings, warn

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.multi_step_lookahead import warmstart_multistep
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.synthetic import SyntheticTestFunction
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

sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarking.utils import (
    check_methods_arg,
    create_csv_if_needed,
    fnv1a_64,
    lock_write,
    mk_seed,
    torch_seed,
)

BENCHMARK_PROBLEMS = get_available_benchmark_problems()


def run_problem(
    problem_name: str,
    problem: SyntheticTestFunction,
    regression_type: Literal["rbf", "idw"],
    method: str,
    maxiter: int,
    rng: np.random.Generator,
    csv: str,
    device: torch.device,
    n_init: Optional[int] = None,
    callback: Optional[Callable[[SyntheticTestFunction], str]] = None,
) -> None:
    """Solves the given problem with the given method, and writes the results to csv."""
    # set hyperparameters
    ndim = problem.dim
    if n_init is None:
        n_init = ndim * 2
    c1 = torch.scalar_tensor(1.0 / ndim)
    c2 = torch.scalar_tensor(0.5 / ndim)
    eps = torch.scalar_tensor(1.0 / ndim)
    n_restarts = 10 * ndim
    raw_samples = max(n_restarts, 512)

    # draw random initial points
    bounds: Tensor = problem.bounds
    lb, ub = bounds
    span = ub - lb
    X = torch.as_tensor(rng.random((n_init, ndim))) * span + lb
    Y = problem(X)

    # create seed functions - one for the optimizer, the other for other uses. In this
    # way, all methods' optimizer runs are seeded equally
    other_rng = rng.spawn(1)[0]

    # define mdoel and acquisition function getters
    if method == "random":

        def next_obs(*_, **__) -> tuple[Tensor, Tensor, None]:
            X_opt = torch.rand(1, ndim) * span + lb
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
            acqfun = LogExpectedImprovement(mdl, Y.amin(), maximize=False)
            X_opt, _ = optimize_acqf(
                acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed(rng)}
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
                    acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed(rng)}
                )
                return X_opt, torch.nan, mdl

        elif method == "myopic-s":
            gh_sampler = GaussHermiteSampler(sample_shape=torch.Size([16]))

            def next_obs(
                X: Tensor, Y: Tensor, prev_mdl: Union[None, Idw, Rbf], *_, **__
            ) -> tuple[Tensor, Tensor, Union[Idw, Rbf]]:
                mdl = get_mdl(X, Y, prev_mdl)
                acqfun = qIdwAcquisitionFunction(mdl, c1, c2, sampler=gh_sampler)
                X_opt, _ = optimize_acqf(
                    acqfun, bounds, 1, n_restarts, raw_samples, {"seed": mk_seed(rng)}
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
                    SobolQMCNormalSampler(torch.Size([f]), seed=mk_seed(other_rng))
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
                h = min(horizon, budget)
                if h == 1:
                    acqfun = qIdwAcquisitionFunction(mdl, c1, c2, valfunc_sampler)
                    X_opt, _ = optimize_acqf(
                        acqfun,
                        bounds,
                        1,
                        n_restarts,
                        raw_samples,
                        {"seed": mk_seed(rng)},
                    )
                    return X_opt, torch.nan, mdl

                n_restarts_ = n_restarts * h * 2 // 3
                raw_samples_ = max(n_restarts_, 512)
                acqfun = Ms(
                    mdl,
                    fantasies_samplers[: h - 1],
                    qIdwAcquisitionFunction,
                    kwargs_factory,
                    valfunc_sampler=valfunc_sampler,
                )
                q = acqfun.get_augmented_q_batch_size(1)
                if prev_full_opt is torch.nan:
                    prev_full_opt = None
                else:
                    prev_full_opt = warmstart_multistep(
                        acqfun,
                        bounds,
                        n_restarts_,
                        raw_samples_,
                        prev_full_opt[:n_restarts_, :q],
                    )
                full_opt, tree_vals = optimize_acqf(
                    acqfun,
                    bounds,
                    q,
                    n_restarts_,
                    raw_samples_,
                    batch_initial_conditions=prev_full_opt,
                    return_best_only=False,
                    return_full_tree=True,
                    options={"seed": mk_seed(rng), "maxfun": maxfun},
                )
                best_tree_idx = tree_vals.argmax()
                X_opt = acqfun.extract_candidates(full_opt[best_tree_idx])
                return X_opt, full_opt, mdl

        else:
            raise NotImplementedError(f"Method {method} not implemented.")

    # run optimization loop
    mdl: Optional[Model] = None
    obs_opt: Tensor = torch.nan
    full_opt: Tensor = torch.nan
    bests: list[float] = [Y.amin().item()]
    timings: list[float] = []
    try:
        for iteration in range(maxiter):
            # fit model and optimize acquisition to get the next point to sample
            start_time = perf_counter()
            obs_opt, full_opt, mdl = next_obs(X, Y, mdl, full_opt, maxiter - iteration)
            timings.append(perf_counter() - start_time)

            # evaluate objective function at new point, and append it to training data
            X = torch.cat((X, obs_opt))
            Y = torch.cat((Y, problem(obs_opt)))
            bests.append(Y.amin().item())

        # save results, delete references and free memory (at least, try to) - call also
        # the callback for (optional) additional data to be saved
        bests = ",".join(map(str, bests))
        timings = ",".join(map(str, timings))
        data = f"{problem_name};{method};{bests};{timings}"
        if callback is not None:
            data += f";{callback(problem)}"
        lock_write(csv, data)
    except Exception:
        warn(
            f"Exception raised in `{problem_name}`, `{method}`:\n{format_exc()}",
            RuntimeWarning,
        )
    finally:
        del problem, X, Y, mdl, obs_opt, full_opt, bests, timings
        gc.collect()
        if device.type == "cuda":
            with torch.no_grad():
                torch.cuda.empty_cache()


def run_benchmark(
    problem_name: str,
    method: str,
    seed: np.random.SeedSequence,
    csv: str,
    device: torch.device,
    n_init: Optional[int] = None,
    setup_callback: Optional[Callable[[], None]] = None,
    save_callback: Optional[Callable[[SyntheticTestFunction], str]] = None,
) -> None:
    """Sets default values and then runs the given benchmarks"""
    filterwarnings("ignore", "Optimization failed", module="botorch")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    np_random = np.random.default_rng(seed)
    torch_seed(mk_seed(np_random))
    if setup_callback is not None:
        setup_callback()
    problem, maxiter, regression_type = get_benchmark_problem(problem_name)
    run_problem(
        problem_name,
        problem,
        regression_type,
        method,
        maxiter,
        np_random,
        csv,
        device,
        n_init,
        save_callback,
    )


def run_benchmarks(
    methods: Iterable[str],
    problems: list[str],
    n_trials: int,
    seed: int,
    n_jobs: int,
    csv: str,
    devices: list[torch.device],
    n_init: Optional[int] = None,
    setup_callback: Optional[Callable[[], None]] = None,
    save_callback: Optional[Callable[[SyntheticTestFunction], str]] = None,
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
        p: np.random.SeedSequence(fnv1a_64(p, seed)).spawn(n_trials) for p in problems
    }
    tasks = filter_tasks_by_status(product(range(n_trials), problems, methods), csv)
    list(
        Parallel(n_jobs=n_jobs, verbose=100, return_as="generator_unordered")(
            delayed(run_benchmark)(
                prob,
                method,
                seeds[prob][trial],
                csv,
                device,
                n_init,
                setup_callback,
                save_callback,
            )
            for (trial, prob, method), device in zip(tasks, cycle(devices))
        )
    )


def parse_args(name: str, multiproblem: bool = True) -> argparse.Namespace:
    """Parses the command-line arguments for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description=f"Benchmarking of Global Optimization strategies on {name}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Benchmarking options")
    group.add_argument(
        "--methods",
        type=check_methods_arg,
        nargs="+",
        help="Methods to run. Greedy algorithms include `ei` and `myopic`. Non-myopic "
        "multi-step algorithms have the following semantic: `ms-sampler.m1.m2. ...`, "
        "where `ms` stands for multi-step, `sampler` is either `gh` or `mc` (for "
        "Gauss-Hermite or Monte Carlo, respectively), while `m1`, `m2`, and so on, are "
        "the number of fantasies at each stage. The overall horizon of a multi-step "
        "method is the number of fantasies plus one.",
        required=True,
    )
    if multiproblem:
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
        "--n-jobs", type=int, default=1, help="Number (positive) of parallel processes."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args("synthetic/real benchmark problems")
    csv = create_csv_if_needed(args.csv, "problem;method;best-so-far;time")

    # ensure each job runs only on one CPU
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ[
        "OMP_NUM_THREADS"
    ] = "1"
    # torch.set_num_threads(1)  # this must be done inside each job

    run_benchmarks(
        args.methods,
        args.problems,
        args.n_trials,
        args.seed,
        args.n_jobs,
        csv,
        list(map(torch.device, args.devices)),
    )
