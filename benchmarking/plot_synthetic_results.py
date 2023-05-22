import argparse
from itertools import zip_longest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from prettytable import PrettyTable
from scipy.io import loadmat

from globopt.core.problems import (
    get_available_benchmark_problems,
    get_available_simple_problems,
    get_benchmark_problem,
)
from globopt.core.regression import Array

plt.style.use("bmh")

PROBLEMS = get_available_benchmark_problems() + get_available_simple_problems()


def load_data(filename: str) -> dict[str, dict[int, Array]]:
    """Loads the data from the given file."""
    data = loadmat(filename)
    for k in ("__header__", "__version__", "__globals__"):
        data.pop(k, None)
    data_: dict[str, dict[int, Array]] = {}
    for k, v in data.items():
        name, h_name = k.split("_")
        h = int(h_name[1:])
        if name not in data_:
            data_[name] = {}
        data_[name][h] = v
    return data_


def plot_results(data: dict[str, dict[int, Array]], figtitle: Optional[str]) -> None:
    """Plots the results in the given dictionary."""
    n_cols = 2
    n_rows = np.ceil(len(data) / n_cols).astype(int)
    fig, axs = plt.subplots(
        n_rows, n_cols, constrained_layout=True, figsize=(n_cols * 3.5, n_rows * 1.75)
    )
    axs = np.atleast_2d(axs)
    for i, (ax, problem_name) in enumerate(zip_longest(axs.flat, sorted(data.keys()))):
        if problem_name is None:
            # if there are more axes than results, set the axis off
            ax.set_axis_off()
            continue

        for horizon, results in data[problem_name].items():
            # compute the average, worst and best solution
            evals = np.arange(1, results.shape[1] + 1)
            avg = results.mean(axis=0)
            worst = results.max(axis=0)
            best = results.min(axis=0)

            # plot the results
            lbl = f"h={horizon}" if i == 0 else None
            c = ax.plot(evals, avg, label=lbl)[0].get_color()
            ax.plot(evals, worst, color=c, lw=0.25)
            ax.plot(evals, best, color=c, lw=0.25)
            ax.fill_between(evals, worst, best, alpha=0.2, color=c)

        # plot the optimal point
        problem = get_benchmark_problem(problem_name)[0]
        fmin = np.full(results.shape[1], problem.pareto_front())
        ax.plot(evals, fmin, "--", color="grey", zorder=-10000)
        ax.set_title(problem.__class__.__name__.lower(), fontsize=11)
        ax.set_xlim(1, evals[-1])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    for ax in axs[-1]:
        ax.set_xlabel("number of function evaluations")
    fig.legend(loc="outside lower center", ncol=len(next(iter(data.values()))))
    fig.suptitle(figtitle, fontsize=12)


def print_summary(data: dict[str, dict[int, Array]], tabletitle: Optional[str]) -> None:
    """Prints the summary of the results in the given dictionary."""
    horizons = [f"h={h}" for h in next(iter(data.values())).keys()]
    problem_names = sorted(data.keys())
    table = PrettyTable()
    table.field_names = ["Function name", ""] + horizons
    table.title = tabletitle
    table.float_format = ".3"

    for problem_name in problem_names:
        f_opt = get_benchmark_problem(problem_name)[0].pareto_front().item()
        means = []
        medians = []
        for _, results in data[problem_name].items():
            gaps = (results[:, 0] - results[:, -1]) / (results[:, 0] - f_opt)
            means.append(gaps.mean())
            medians.append(np.median(gaps))
        best_mean = np.argmax(means)
        best_median = np.argmax(medians)

        means_str = [f"{m:.3f}" for m in means]
        medians_str = [f"{m:.3f}" for m in medians]
        means_str[best_mean] = f"\033[1;34;40m{means_str[best_mean]}\033[0m"
        medians_str[best_median] = f"\033[1;34;40m{medians_str[best_median]}\033[0m"
        table.add_row([problem_name, "mean"] + means_str)
        table.add_row(["", "median"] + medians_str)
    print(table)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Visualization of synthetic benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Filenames of the results to be visualized.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the summary and do not show the plots.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Only show the plot and do not print the summary.",
    )
    args = parser.parse_args()

    # load each result and plot
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        data = load_data(filename)
        title = filename if include_title else None
        if not args.no_plot:
            plot_results(data, title)
        if not args.no_summary:
            print_summary(data, title)
    plt.show()
