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


MAX_SEED = 2**32
FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
BENCHMARK_PROBLEMS = get_available_benchmark_problems()
SIMPLE_PROBLEMS = get_available_simple_problems()


def fnv1a(s: str) -> int:
    """Hashes a string using the FNV-1a algorithm."""
    return sum((FNV_OFFSET ^ b) * FNV_PRIME**i for i, b in enumerate(s.encode()))


class TrackOptunaObjectiveCallback(Callback):
    """Callback for computing the current performance of an optuna trial."""

    def __init__(self) -> None:
        super().__init__()
        self.total = 0.0

    def notify(self, algorithm: Algorithm) -> None:
        iter = algorithm.n_iter - 1
        if iter >= algorithm.termination.n_max_gen // 2:
            self.total += (iter + 1) * algorithm.opt[0].F.item()


def objective(
    problem: Problem,
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

    # run the minimization N times
    total = 0.0
    final_minimum = float("inf")
    seed += trial.number ^ fnv1a(problem.__class__.__name__)
    for n in range(N):
        callback = TrackOptunaObjectiveCallback()
        res = minimize(
            problem,
            algorithm,
            ("n_iter", max_iter),
            callback=callback,
            seed=(seed * (n + 1)) % MAX_SEED,
        )
        final_minimum = min(final_minimum, res.opt[0].F.item())
        total += callback.total
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
    n_trials = args.n_trials
    n_avg = args.n_avg
    seed = args.seed

    # instantiate the problem to fine tune the algorithm to
    problem_instance, iters, regression = get_benchmark_problem(problem)

    # create the study
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.NopPruner()
    storage = "sqlite:///benchmarking/results/fine-tunings.db"
    study_name = (
        f"{problem}-trials-{n_trials}-avg-{n_avg}-seed-{seed}"
        + f"-sampler-{sampler.__class__.__name__[:-7].lower()}"
        + f"-pruner-{pruner.__class__.__name__[:-6].lower()}"
    )
    study = optuna.create_study(
        storage=storage, sampler=sampler, pruner=pruner, study_name=study_name
    )

    # run the optimization
    obj = partial(objective, problem_instance, regression, iters, n_avg, seed)
    study.optimize(obj, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # print the results - saving is done automatically in the db
    print("BEST VALUE:", study.best_value, "\nBEST PARAMS:", study.best_params)
    # optuna-dashboard sqlite:///benchmarking/results/fine-tunings.db
