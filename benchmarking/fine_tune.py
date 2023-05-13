import argparse
from functools import partial

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
from globopt.core.regression import Rbf
from globopt.nonmyopic.algorithm import NonMyopicGO


class ReportBestSoFarToOptunaCallback(Callback):
    """Callback for reporting the current best solution found so far to optuna."""

    def __init__(self, trial: optuna.trial.Trial) -> None:
        super().__init__()
        self.trial = trial

    def notify(self, algorithm: Algorithm) -> None:
        self.trial.report(algorithm.opt[0].F.item(), algorithm.n_iter - 1)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def fnv1a(s: str) -> int:
    """Hashes a string using the FNV-1a algorithm."""
    FNV_OFFSET, FNV_PRIME = 2166136261, 16777619
    return sum((FNV_OFFSET ^ b) * FNV_PRIME**i for i, b in enumerate(s.encode()))


def objective(
    problem: Problem, max_iter: int, seed: int, trial: optuna.trial.Trial
) -> float:
    # suggest algorithm's parameters
    n_var = problem.n_var
    c1 = trial.suggest_float("c1", 0.0, 3.0) / n_var
    c2 = trial.suggest_float("c2", 0.0, 3.0) / n_var
    eps = trial.suggest_float("eps", 0.1, 3.0) / n_var
    horizon = trial.suggest_int("horizon", 2, 5)
    discount = trial.suggest_float("discount", 0.6, 1.0)

    # instantiate algorithm and problem
    algorithm = NonMyopicGO(
        regressor=Rbf(eps=eps),
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

    # run the minimization
    res = minimize(
        problem,
        algorithm,
        termination=("n_iter", max_iter),
        callback=ReportBestSoFarToOptunaCallback(trial),
        verbose=False,
        seed=(seed + trial.number ^ fnv1a(problem.__class__.__name__)) % 2**32,
    )
    return res.opt[0].F.item()


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tuning of GO strategies via Bayesian Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        choices=["all"]
        + get_available_benchmark_problems()
        + get_available_simple_problems(),
        help="Problem to fine tune the algorithm to.",
        required=True,
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials to run."
    )
    parser.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    args = parser.parse_args()

    # instantiate the problem to fine tune the algorithm to
    problem, iters, _ = get_benchmark_problem(args.problem)

    # create the study
    study_name = f"{args.problem}_iters_{iters}_trials_{args.n_trials}_seed_{args.seed}"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///benchmarking/fine-tunings.db",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.NopPruner(),
    )

    # run the optimization
    obj = partial(objective, problem, iters, args.seed)
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=True, n_jobs=1)

    # print the results - saving is done automatically in the db
    print(study.best_params)
    # optuna-dashboard sqlite:///benchmarking/fine-tunings.db
