"""
Visualization and summary of results of benchmarking of myopic and non-myopic Global
Optimization strategies on various problems.
"""

import argparse
import re
from functools import partial
from itertools import zip_longest
from math import ceil
from typing import Any, Literal, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike
from prettytable import PrettyTable
from scipy.stats import sem, t, wilcoxon

from globopt.problems import get_benchmark_problem

plt.style.use("bmh")


METHODS_ORDER = ["random", "ei", "myopic", "myopic-s", "ms"]


def _matches_any_pattern(s: str, patterns: list[str]) -> bool:
    """Returns True if the given string matches any of the given patterns."""
    return any(re.search(p, s) for p in patterns)


def _sort_method(method: str) -> int:
    """Computes sorting rank of given method (takes into account horizon, if any)."""
    parts = method.split(".")
    method = parts[0]
    if method.startswith("ms"):
        method = "ms"
    rank = METHODS_ORDER.index(method)
    return rank if len(parts) == 1 else rank + sum(int(p) for p in parts[1:])


def _compute_all_stats(row: pd.Series) -> pd.Series:
    """Computes all the statistics for the given row of the dataframe."""
    # optimality gaps and final gap
    problem_name = row.name[0]
    f_opt = get_benchmark_problem(problem_name)[0].optimal_value
    bests_so_far = row["best-so-far"]
    init_best = bests_so_far[:, 0, np.newaxis]
    gaps = (bests_so_far - init_best) / (f_opt - init_best)
    final_gap = gaps[:, -1]

    # cumulative return
    return_ = row["stage-reward"].sum(axis=1)

    # average time per iteration and std
    time = row["time"]
    time_avg = time.mean()
    time_std = time.std()

    # pass on data that was not used
    columns_used = ["best-so-far", "stage-reward", "time"]
    other_data = {n: row[n] for n in row.index if n not in columns_used}
    return pd.Series(
        {
            "best-so-far": bests_so_far,
            "gap": gaps,
            "final-gap": final_gap,
            "final-gap-mean": final_gap.mean(),
            "final-gap-median": np.median(final_gap),
            "return": return_,
            "return-mean": return_.mean(),
            "return-median": np.median(return_),
            "time": time,
            "time-mean": time_avg,
            "time-std": time_std,
            **other_data,
        }
    )


def load_data(
    csv_filename: str, include: list[str], exclude: list[str]
) -> pd.DataFrame:
    """Loads the data from the given file into a dataframe."""
    converter = partial(np.fromstring, sep=",")
    df = pd.read_csv(
        csv_filename,
        sep=";",
        dtype={"problem": pd.StringDtype(), "method": pd.StringDtype()},
        converters={s: converter for s in ["stage-reward", "best-so-far", "time"]},
    )
    if include:
        df = df[
            df["method"].apply(_matches_any_pattern, patterns=include)
            | df["problem"].apply(_matches_any_pattern, patterns=include)
        ]
    elif exclude:
        df = df[
            ~(
                df["method"].apply(_matches_any_pattern, patterns=exclude)
                | df["problem"].apply(_matches_any_pattern, patterns=exclude)
            )
        ]

    # manually sort problems alphabetically but methods in a custom order
    df.sort_values(
        ["problem", "method"],
        key=lambda s: s if s.name == "problem" else s.map(_sort_method),
        ignore_index=True,
        inplace=True,
    )

    # group by problem, method, and horizon, and stack all trials into 2d arrays
    df = df.groupby(["problem", "method"], dropna=False, sort=False).aggregate(np.stack)

    # compute all the statistics for visualization and summary
    return df.apply(_compute_all_stats, axis=1)


def plot_converges(df: pd.DataFrame, figtitle: Optional[str], n_cols: int = 5) -> None:
    """Plots the results in the given dataframe. In particular, it plots the
    convergence to the optimum, and the evolution of the optimality gap."""
    # create figure
    problem_names = df.index.unique(level="problem")
    n_rows = ceil(len(problem_names) / n_cols)
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 2.5))

    # create main grid of plots
    plotted_methods = {}
    main_grid = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3)
    for i, problem_name in enumerate(problem_names):
        row, col = i // n_cols, i % n_cols
        df_problem = df.loc[problem_name]

        # in this quadrant, create a subgrid for the convergence plot and the gap plot
        subgrid = main_grid[row, col].subgridspec(2, 1, hspace=0.1)
        ax_opt = fig.add_subplot(subgrid[0])
        ax_gap = fig.add_subplot(subgrid[1], sharex=ax_opt)

        for method, row_data in df_problem.iterrows():
            # plot the best convergence and the optimality gap evolution
            color = None
            evals = np.arange(row_data["best-so-far"].shape[1])
            for ax, col in zip((ax_opt, ax_gap), ("best-so-far", "gap")):
                data = row_data[col]
                avg = data.mean(axis=0)
                scale = sem(data, axis=0) + 1e-12
                ci_lb, ci_ub = t.interval(0.95, data.shape[0] - 1, loc=avg, scale=scale)
                h = ax.step(evals, avg, lw=1.0, color=color)
                if color is None:
                    color = h[0].get_color()
                ax.fill_between(evals, ci_lb, ci_ub, alpha=0.2, color=color, step="pre")
                ax.step(evals, ci_lb, color=color, lw=0.1)
                ax.step(evals, ci_ub, color=color, lw=0.1)
            plotted_methods[method] = color

        # plot also the optimal point in background
        f_opt = get_benchmark_problem(problem_name)[0].optimal_value
        fmin = np.full(evals.size, f_opt)
        ax_opt.plot(evals, fmin, "--", color="grey", zorder=-10000)

        # make axes pretty
        if col == 0:
            ax_opt.set_ylabel(r"$f^\star$")
            ax_gap.set_ylabel(r"$G$")
        if row == n_rows - 1:
            ax_gap.set_xlabel("Evaluations")
        ax_opt.set_title(problem_name, fontsize=11)
        ax_opt.tick_params(labelbottom=False)
        ax_opt.set_xlim(0, evals[-1])
        ax_gap.set_ylim(-0.05, 1.05)
        ax_gap.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # create legend manually
    handles = [Line2D([], [], label=m, color=c) for m, c in plotted_methods.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(figtitle, fontsize=12)


def _custom_violin(
    ax: Axes,
    data: ArrayLike,
    pos: float,
    fc: str = "b",
    ec: str = "k",
    alpha: float = 0.7,
    percentiles: ArrayLike = (25, 50, 75),
    side: Literal["left", "right", "both"] = "both",
    scatter_kwargs: dict[str, Any] = {},
    violin_kwargs: dict[str, Any] = {},
) -> None:
    """Customized violin plot.
    Many thanks to https://stackoverflow.com/a/76184694/19648688."""
    parts = ax.violinplot(data, [pos], **violin_kwargs)
    for pc in parts["bodies"]:
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        if side == "left":
            x_offset, clip_min, clip_max = -0.02, -np.inf, m
        elif side == "right":
            x_offset, clip_min, clip_max = 0.02, m, np.inf
        else:
            x_offset, clip_min, clip_max = 0, -np.inf, np.inf
        points_x = pos + x_offset
        pc.get_paths()[0].vertices[:, 0] = np.clip(
            pc.get_paths()[0].vertices[:, 0], clip_min, clip_max
        )
        pc.set_facecolor(fc)
        pc.set_edgecolor(ec)
        pc.set_alpha(alpha)
    perc = np.percentile(data, percentiles)
    for p in perc:
        ax.scatter(points_x, p, color=ec, zorder=3, **scatter_kwargs)


def plot_gap_reward_violins(df: pd.DataFrame, figtitle: Optional[str]) -> None:
    """Plots the results in the given dataframe. In particular, it plots the
    distribution of the (final) optimality gap versus the cumulative rewards as violins.
    """
    df_ = df[["final-gap", "return"]].stack(0, False).unstack("method", pd.NA)

    # create figure
    problem_names = df.index.unique(level="problem")
    methods = df_.columns.to_list()
    methods_ticks = np.arange(len(methods))
    n_rows = len(problem_names)
    fig, axs = plt.subplots(
        n_rows, 1, constrained_layout=True, figsize=(6, n_rows * 2.5), sharex=True
    )
    if n_rows == 1:
        axs = [axs]

    # for each problem, create two violin plots - one for the optimality gap's
    # distribution and one for the cumulative rewards' distribution
    s_kwargs = {"s": 40, "marker": "_"}
    v_kwargs = {
        "showextrema": False,
        "showmedians": False,
        "showmeans": False,
        "widths": 0.5,
    }
    for ax, problem_name in zip(axs, problem_names):
        ax_twin = ax.twinx()
        for method in methods:
            # plot the two violins
            pos = methods.index(method)
            _custom_violin(
                ax,
                df_.loc[(problem_name, "final-gap")][method],
                pos,
                "C0",
                "C0",
                side="left",
                scatter_kwargs=s_kwargs,
                violin_kwargs=v_kwargs,
            )
            return_ = df_.loc[(problem_name, "return")][method]
            if np.logical_not(np.isnan(return_)).any():
                _custom_violin(
                    ax_twin,
                    return_,
                    pos,
                    "C1",
                    "C1",
                    side="right",
                    scatter_kwargs=s_kwargs,
                    violin_kwargs=v_kwargs,
                )

        # embellish axes
        ax.set_xticks(methods_ticks)
        ax.set_xticklabels(methods)
        ax.set_title(problem_name, fontsize=11)
        ax.set_ylabel(r"$G$")
        ax_twin.set_ylabel(r"$R$")
        ax.tick_params(axis="y", labelcolor="C0")
        ax_twin.tick_params(axis="y", labelcolor="C1")
        ax.grid(which="major", visible=False)
        ax_twin.grid(which="major", visible=False)

    # last embellishments
    ax.set_xlabel("Method")
    fig.suptitle(figtitle, fontsize=12)


def _compute_dispersion(row: pd.Series) -> pd.Series:
    """Computes the dispersion of the given row of the dataframe."""
    # compute the mean and sem of the time per iteration - CI intervals would be too big
    out = {}
    for col in ("final-gap-mean", "time-mean"):
        name = col.split("-")[-2]
        val = np.asarray(row[col])
        mean = val.mean()
        out[name] = mean
        out[f"{name}-err"] = sem(val) if val.size > 1 else 0.0
    return pd.Series(out)


def plot_timings(
    df: pd.DataFrame, figtitle: Optional[str], single_problem: bool = False
) -> None:
    """Plots the average time per iteration versus the optimality gap."""
    if single_problem:
        assert len(df.index.unique(level="problem")) == 1, (
            "Only one problem detected, inter-problem dispersion will be calculated"
            " instead of intra-problem."
        )
        data = {
            "final-gap-mean": df["final-gap"],
            "time-mean": df["time"].apply(partial(np.mean, axis=1), axis=1),
        }
        df_ = pd.DataFrame(data).droplevel("problem").apply(_compute_dispersion, axis=1)
    else:
        df_ = (
            df[["final-gap-mean", "time-mean"]]
            .droplevel("problem")
            .groupby("method", sort=False)
            .aggregate(list)
            .apply(_compute_dispersion, axis=1)
        )
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    options = {
        "random": {"ha": "left", "xytext": (5, 5)},
        "ei": {"ha": "left", "xytext": (5, 5)},
        "myopic": {"ha": "right", "xytext": (-5, 5)},
        "myopic-s": {"ha": "left", "xytext": (5, 5)},
        "ms": {"ha": "left", "xytext": (5, 5)},
    }
    for method, row in df_.iterrows():
        if re.fullmatch(r"ms-mc(\.1)+", method):  # rollout with MC sampling
            color = "C0"
        elif re.fullmatch(r"ms-gh(\.1)+", method):  # rollout with GH sampling
            color = "C1"
        elif method.startswith("ms-mc"):  # multistep with MC sampling
            color = "C2"
        elif method.startswith("ms-gh"):  # multistep with GH sampling
            color = "C3"
        else:  # myopic strategies
            color = "C4"
        opts = (
            options["ms"] if method.startswith("ms") else options[method.split(".")[0]]
        )
        ax.errorbar(
            x=row["time"],
            xerr=row["time-err"],
            y=row["gap"],
            yerr=row["gap-err"],
            ls="none",
            lw=1.5,
            capsize=3,
            capthick=1.5,
            ecolor=color,
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="white",
        )
        ax.annotate(
            method,
            xy=(df_.loc[method, "time"], df_.loc[method, "gap"]),
            textcoords="offset points",
            **opts,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per iteration")
    ax.set_ylabel("Optimality gap")
    fig.suptitle(figtitle, fontsize=12)


def _format_row(
    row: pd.Series,
    src_data: pd.DataFrame,
    order: Literal["min", "max"] = "max",
    prec: int = 3,
    threshold_alpha: float = 5e-2,
) -> list[str]:
    """Formats the given row of the dataframe as a list of (possibly highlighted)
    strings."""
    # first, convert all row entries to strings
    strs = [f"{x:.{prec}f}" for x in row]

    # now, identify the best method and make it bold blue
    methods = row.index.to_list()
    problem = row.name[0]
    best_method_idx = getattr(row, f"arg{order}")()
    strs[best_method_idx] = f"\033[1;34m{strs[best_method_idx]}\033[0m"
    best_method_data = src_data[(problem, methods[best_method_idx])]

    # then, loop over the rest of the methods, and compare them in a pairwise
    # fashion via the Wilcoxon (one-sided) signed-rank test. If the null hypothesis
    # of the best method being better than the other method cannot be rejected,
    # highlight the other method in italic violet
    side = "less" if order == "min" else "greater"
    for other_method_idx in (i for i in range(len(strs)) if i != best_method_idx):
        other_method_data = src_data[(problem, methods[other_method_idx])]
        try:
            _, alpha = wilcoxon(best_method_data, other_method_data, alternative=side)
            if alpha > threshold_alpha:
                strs[other_method_idx] = f"\033[35m{strs[other_method_idx]}\033[0m"
        except ValueError:
            pass
    return strs


def summarize(df: pd.DataFrame, tabletitle: Optional[str]) -> None:
    """Prints the summary of the results in the given dataframe as three tables, one
    containing the (final) optimality gap, one the cumulative rewards, and the last
    the time per iteration."""

    # first, build the dataframe with the statistics for gap, returns, and time
    cols = [
        "final-gap-mean",
        "final-gap-median",
        "return-mean",
        "return-median",
        "time-mean",
        "time-std",
    ]
    df_ = df[cols].stack(0, False).unstack("method", pd.NA)

    # then, instantiate the pretty tables to be filled with the statistics
    field_names = ["Function name", ""] + df.index.unique(level="method").to_list()
    tables = (
        PrettyTable(field_names, title="gap"),
        PrettyTable(field_names, title="return"),
        PrettyTable(field_names[:1] + field_names[2:], title="time"),
    )

    # loop over every problem and fill the pretty tables
    for pname in df_.index.get_level_values("problem").unique():
        # gap (mean and median)
        gap_data = df["final-gap"]
        g_mean = _format_row(df_.loc[(pname, "final-gap-mean")], gap_data, prec=5)
        g_median = _format_row(df_.loc[(pname, "final-gap-median")], gap_data, prec=5)
        tables[0].add_row([pname, "mean"] + g_mean)
        tables[0].add_row(["", "median"] + g_median)

        # return (mean and median)
        return_data = df["return"]
        r_mean = _format_row(df_.loc[(pname, "return-mean")], return_data, prec=3)
        r_median = _format_row(df_.loc[(pname, "return-median")], return_data, prec=3)
        tables[1].add_row([pname, "mean"] + r_mean)
        tables[1].add_row(["", "median"] + r_median)

        # time (mean +/- std)
        time_data = df["time"].map(lambda t: t.mean(axis=1))
        t_mean = _format_row(df_.loc[(pname, "time-mean")], time_data, "min", prec=2)
        t_mean_std = [
            f"{m} +/- {s:.2f}" for m, s in zip(t_mean, df_.loc[(pname, "time-std")])
        ]
        tables[2].add_row([pname] + t_mean_std)
        tables[2].add_row([""] * (len(field_names) - 1))

    # finally, print the tables side by side
    if tabletitle is not None:
        print(tabletitle)
    rows = zip_longest(*(t.get_string().splitlines() for t in tables), fillvalue="")
    print("\n".join("\t".join(group_of_rows) for group_of_rows in rows))


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Visualization of benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Filenames of the results to be visualized.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=[],
        help="List of methods and/or benchmark patterns to plot.",
    )
    group.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="List of methods and/or benchmark patterns not to plot.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the summary and do not show the plots.",
    )
    group.add_argument(
        "--no-summary",
        action="store_true",
        help="Only show the plot and do not print the summary.",
    )
    args = parser.parse_args()

    # load each result and plot
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        title = filename if include_title else None
        dataframe = load_data(filename, args.include, args.exclude)
        if not args.no_plot:
            plot_converges(dataframe, title)
            plot_gap_reward_violins(dataframe, title)
            plot_timings(dataframe, title)
        if not args.no_summary:
            summarize(dataframe, title)
    plt.show()
