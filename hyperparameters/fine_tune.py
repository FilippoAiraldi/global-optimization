"""
Fine-tuning, i.e., hyperparameter optimization, of non-myopic Global Optimization
strategies via Optuna's Bayesian Optimization and other algorithms.
"""


import argparse
from functools import partial
from typing import Literal

import optuna
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Idw, Rbf
from globopt.nonmyopic.algorithm import NonMyopicGO
from globopt.util.callback import BestSoFarCallback

MAX_SEED = 2**32
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
BENCHMARK_PROBLEMS = get_available_benchmark_problems()
SIMPLE_PROBLEMS = get_available_simple_problems()


def fnv1a(s: str) -> int:
    """Hashes a string using the FNV-1a algorithm."""
    return sum((FNV_OFFSET ^ b) * FNV_PRIME**i for i, b in enumerate(s.encode()))


class TrackSecondHalfObjectiveCallback(Callback):
    """
    Callback for computing the current objective of an optuna trial according to the
    performance of the optimization strategy in the second half of the run, as per [1].

    References
    ----------
    [1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
        functions. Computational Optimization and Applications, 77(2):571–595, 2020
    """

    def __init__(self) -> None:
        super().__init__()
        self.total = 0.0

    def notify(self, algorithm: Algorithm) -> None:
        iter = algorithm.n_iter - 1
        if iter >= algorithm.termination.n_max_gen // 2:
            self.total += (iter + 1) * algorithm.opt[0].F.item()


def optimize(
    problem: Problem,
    objective: Literal["gap", "second-half"],
    regression: Literal["rbf", "idw"],
    max_iter: int,
    N: int,
    seed: int,
    trial: optuna.trial.Trial,
) -> float:
    # suggest algorithm's parameters
    n_var = problem.n_var
    c1 = trial.suggest_float("c1", 0.0, 3.0) / n_var
    c2 = trial.suggest_float("c2", 0.0, 3.0) / n_var
    horizon = trial.suggest_int("horizon", 2, 5)
    discount = trial.suggest_float("discount", 0.6, 1.0)
    trial.suggest_discrete_uniform
    if regression == "rbf":
        eps = trial.suggest_float("eps", 0.1, 3.0) / n_var
        regressor = Rbf(eps=eps)
    else:
        regressor = Idw()  # type: ignore

    # instantiate algorithm and problem
    algorithm = NonMyopicGO(
        regression=regressor,
        init_points=2 * n_var,
        acquisition_min_algorithm=PSO(pop_size=10),  # size will be scaled with n_var
        acquisition_min_kwargs={
            "termination": DefaultSingleObjectiveTermination(
                ftol=1e-4, n_max_gen=300, period=10
            )
        },
        acquisition_fun_kwargs={"c1": c1, "c2": c2},
        horizon=horizon,
        discount=discount,
    )

    if objective == "gap":
        f_opt = problem.pareto_front().item()

        def run_once(seed: int) -> tuple[float, float]:
            callback = BestSoFarCallback()
            minimize(
                problem, algorithm, ("n_iter", max_iter), callback=callback, seed=seed
            )
            f_init = callback.data["best"][0]
            f_final = callback.data["best"][-1]
            return (f_init - f_final) / (f_init - f_opt), f_final

    else:

        def run_once(seed: int) -> tuple[float, float]:
            callback = TrackSecondHalfObjectiveCallback()
            res = minimize(
                problem, algorithm, ("n_iter", max_iter), callback=callback, seed=seed
            )
            return callback.total, res.opt[0].F.item()

    # run the minimization N times
    seed += trial.number ^ fnv1a(problem.__class__.__name__)
    total = 0.0
    final_minimum = float("inf")
    for n in range(N):
        contribution, this_minimum = run_once((seed * (n + 1)) % MAX_SEED)
        final_minimum = min(final_minimum, this_minimum)
        total += contribution / N
        trial.report(total, n)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("final-minimum", final_minimum)
    return total


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tuning of GO strategies via Bayesian Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        choices=["all"] + BENCHMARK_PROBLEMS + SIMPLE_PROBLEMS,
        help="Problem to fine tune the algorithm to.",
        required=True,
    )
    parser.add_argument(
        "--objective",
        choices=["gap", "second-half"],
        help="Objective to optimize.",
        required=True,
    )
    parser.add_argument(
        "--sampler",
        choices=["random", "tpe", "cmaes"],
        help="Optuna sampler to use.",
        required=True,
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of trials to run."
    )
    parser.add_argument(
        "--n-avg",
        type=int,
        default=20,
        help="Number of runs per trial for averaging.",
    )
    parser.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    args = parser.parse_args()
    problem = args.problem
    objective = args.objective
    n_trials = args.n_trials
    n_avg = args.n_avg
    seed = args.seed

    # instantiate the problem to fine tune the algorithm to
    problem_instance, iters, regression = get_benchmark_problem(problem)

    # create the study
    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)  # type: ignore[assignment]
    else:
        sampler = optuna.samplers.CmaEsSampler(seed=seed)  # type: ignore[assignment]
    pruner = optuna.pruners.NopPruner()
    study_name = (
        f"{problem}-objective-{objective}-trials-{n_trials}-avg-{n_avg}-seed-{seed}"
        + f"-sampler-{sampler.__class__.__name__[:-7].lower()}"
        + f"-pruner-{pruner.__class__.__name__[:-6].lower()}"
    )
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///hyperparameters/hpo.db",
        direction="maximize" if objective == "gap" else "minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # run the optimization
    obj = partial(optimize, problem_instance, objective, regression, iters, n_avg, seed)
    study.optimize(obj, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # print the results - saving is done automatically in the db
    print("BEST VALUE:", study.best_value, "\nBEST PARAMS:", study.best_params)
    # optuna-dashboard sqlite:///hyperparameters/hpo.db
