"""A script to tighten the bounds of constrained benchmark problems based on their
feasible region."""

import argparse
from typing import Callable

import numpy as np
import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution

from globopt.problems import (
    get_available_constrained_benchmark_problems,
    get_benchmark_problem,
)

BENCHMARK_CONSTRAINED_PROBLEMS = get_available_constrained_benchmark_problems()


def tighten_bounds(
    problem: ConstrainedBaseTestProblem, rng: np.random.Generator
) -> np.ndarray:
    nc = problem.num_constraints
    bounds = Bounds(*problem.bounds.numpy())
    constraints = NonlinearConstraint(
        lambda x: problem.evaluate_slack_true(torch.from_numpy(x)).numpy().reshape(nc),
        0.0,
        np.inf,
    )

    def differential_evolution_wrapper(objective: Callable) -> float:
        res = differential_evolution(
            objective,
            bounds,
            strategy="currenttobest1bin",
            maxiter=100_000,
            popsize=100,
            tol=5e-3,
            mutation=(0.8, 1.0),
            constraints=constraints,
            rng=rng,
        )
        return res.fun

    dim = problem.dim
    new_bounds = np.empty((2, dim), dtype=np.float64)
    for i in range(dim):
        new_bounds[0, i] = differential_evolution_wrapper(lambda x: x[i])
        new_bounds[1, i] = -differential_evolution_wrapper(lambda x: -x[i])
    return new_bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tightening of bounds for constrained problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problems",
        choices=["all"] + BENCHMARK_CONSTRAINED_PROBLEMS,
        nargs="+",
        default=["all"],
        help="Problems to include in the benchmarking.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()
    if args.problems == ["all"]:
        args.problems = BENCHMARK_CONSTRAINED_PROBLEMS
    rng = np.random.default_rng(args.seed)

    torch.set_default_dtype(torch.float64)

    for problem_name in args.problems:
        problem = get_benchmark_problem(problem_name)[0]
        tight_bounds = tighten_bounds(problem, rng)
        print(f"{problem_name.upper()}:\nlb={tight_bounds[0]}; ub={tight_bounds[1]}")
