"""
Visualization and summary of results of benchmarking of myopic and non-myopic Global
Optimization strategies on various problems.
"""

import argparse
import re
from functools import partial
from math import ceil
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prettytable as pt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import sem, t, wilcoxon

from globopt.problems import get_benchmark_problem

pd.options.mode.copy_on_write = True


ALPHA = 0.95
METHODS_ORDER = ["random", "ei", "myopic", "myopic-s", "ms-gh", "ms-mc"]
METHOD_PATTER = re.compile(r"ms-(mc|gh)((?:\.\d+)+)")
VALID_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def _sort_method(method: str) -> int:
    """Computes sorting rank of given method (takes into account horizon, if any)."""
    parts = method.split(".")
    method = parts[0]
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


def official_method_name_and_type(
    method: str, no_spaces: bool = False, for_filename: bool = False
) -> tuple[str, int]:
    """Utility to get the official name of the method."""
    match = METHOD_PATTER.fullmatch(method)
    if match is not None:  # rollout/multi-step with MC/GH
        sampler = match.group(1).upper()
        fantasies = match.group(2).split(".")[1:]
        horizon = len(fantasies) + 1
        prefix = "R" if all(f == "1" for f in fantasies) else "MS"
        name = f"{prefix}-{horizon} ({sampler})"
        if prefix == "R":
            type_ = 0 if sampler == "MC" else 1
        else:
            type_ = 2 if sampler == "MC" else 3
    else:
        name = method.title()
        type_ = 4
    if no_spaces:
        name = name.replace(" ", r"\,")
    if for_filename:
        name = VALID_PATTERN.sub("", name)
        assert name.isalnum(), f"Invalid file name: {name}"
    return name, type_


def load_data(
    csv_filename: str,
    include_methods: list[str],
    include_problems: list[str],
    exclude_methods: list[str],
    exclude_problems: list[str],
) -> pd.DataFrame:
    """Loads the data from the given file into a dataframe."""
    converter = partial(np.fromstring, sep=",")
    df = pd.read_csv(
        csv_filename,
        sep=";",
        dtype={"problem": pd.StringDtype(), "method": pd.StringDtype()},
        converters={s: converter for s in ["stage-reward", "best-so-far", "time"]},
    )
    for col, include in (("method", include_methods), ("problem", include_problems)):
        if include:
            df = df[df[col].apply(lambda s: any(re.search(p, s) for p in include))]
    for col, exclude in (("method", exclude_methods), ("problem", exclude_problems)):
        if exclude:
            df = df[~df[col].apply(lambda s: any(re.search(p, s) for p in exclude))]

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


def _compute_avg_and_ci(row: pd.Series, column: str) -> pd.Series:
    """Computes the average and conf. interval of the given row of the dataframe."""
    data = row[column]
    avg = data.mean(axis=0)
    scale = sem(data, axis=0) + 1e-12
    ci_lb, ci_ub = t.interval(ALPHA, data.shape[0] - 1, loc=avg, scale=scale)
    return pd.Series({"avg": avg, "lb": ci_lb, "ub": ci_ub})


def optimiser_convergences(
    df: pd.DataFrame,
    plot: bool,
    pgfplotstables: bool,
    column: Literal["best-so-far", "gap"] = "gap",
    title: Optional[str] = None,
    ncols: int = 4,
) -> None:
    """Plots/saves the results in the given dataframe. In particular, it plots the
    convergence to the optimum or the evolution of the optimality gap."""
    df_ = df.apply(_compute_avg_and_ci, args=(column,), axis=1)
    problem_names = df_.index.unique(level="problem")

    if plot:
        nrows = ceil(len(problem_names) / ncols)
        fig, axs = plt.subplots(nrows, ncols, constrained_layout=True)
        plotted_methods = {}
        for i, problem_name in enumerate(problem_names):
            row, col = i // ncols, i % ncols
            ax = axs[row, col]
            df_problem = df_.loc[problem_name]
            for method, row in df_problem.iterrows():
                color = None
                avg = row["avg"]
                lb = row["lb"]
                ub = row["ub"]
                iters = np.arange(avg.size)
                h = ax.step(iters, avg, lw=1.0, color=color)
                if color is None:
                    color = h[0].get_color()
                ax.fill_between(iters, lb, ub, alpha=0.2, color=color, step="pre")
                ax.step(iters, lb, color=color, lw=0.1)
                ax.step(iters, ub, color=color, lw=0.1)
                plotted_methods[method] = color

            # plot also the optimal point in background
            if column == "best-so-far":
                f_opt = get_benchmark_problem(problem_name)[0].optimal_value
                fmin = np.full(iters.size, f_opt)
                ax.plot(iters, fmin, "--", color="grey", zorder=-10000)

            # make axes pretty
            if col == 0:
                ax.set_ylabel(r"$G$" if column == "gap" else r"$f^\star$")
            if row == nrows - 1:
                ax.set_xlabel("Evaluations")
            ax.set_xlim(0, iters[-1])
            if column == "gap":
                ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelbottom=False)
            ax.set_title(problem_name, fontsize=11)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        for j in range(i + 1, nrows * ncols):
            axs[j // ncols, j % ncols].set_axis_off()

        # create legend manually
        handles = [Line2D([], [], label=m, color=c) for m, c in plotted_methods.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        fig.suptitle(title, fontsize=12)

    if pgfplotstables:
        tables_already_written = set()
        column = column.replace("-", "")
        for problem in problem_names:
            for method, row in df_.loc[problem].iterrows():
                method, _ = official_method_name_and_type(method, for_filename=True)
                fn = f"pgfplotstables/{column}_{problem}_{method.lower()}"
                fn += f"_{title}.dat" if title is not None else ".dat"
                pd.DataFrame(row.to_dict()).to_string(fn, index=False)
                if fn in tables_already_written:
                    print(f"WARNING: overwritten `{fn}`")
                else:
                    tables_already_written.add(fn)


def _compute_dispersion(row: pd.Series) -> pd.Series:
    """Computes the dispersion of the given row of the dataframe."""
    # compute the mean and sem of the time per iteration - CI intervals would be too big
    out = {}
    for col in row.index:
        data = np.asarray(row[col])
        mean = data.mean()
        out[col] = mean
        out[f"{col}-err"] = sem(data) if data.size > 1 else 0.0
    return pd.Series(out)


def _compute_official_name_and_type(row: pd.Series) -> pd.Series:
    """Computes the official name and type of the given row of the dataframe."""
    new_row = row.copy()
    new_row["method"], new_row["type"] = official_method_name_and_type(
        row["method"], no_spaces=True
    )
    return new_row


def itertime_vs_gap(
    df: pd.DataFrame, plot: bool, pgfplotstables: bool, title: Optional[str] = None
) -> None:
    """Plots/saves the average time per iteration versus the optimality gap."""
    df_ = (
        df[["final-gap-mean", "time-mean"]]
        .rename(columns={"final-gap-mean": "gap", "time-mean": "time"})
        .droplevel("problem")
        .groupby("method", sort=False)
        .aggregate(list)
        .apply(_compute_dispersion, axis=1)
    )

    if plot:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        opts = {
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
            opt = opts["ms"] if method.startswith("ms") else opts[method.split(".")[0]]
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
                method, xy=(row["time"], row["gap"]), textcoords="offset points", **opt
            )
        ax.set_xscale("log")
        ax.set_xlabel("Seconds per iteration")
        ax.set_ylabel("Optimality gap")
        fig.suptitle(title, fontsize=12)

    if pgfplotstables:
        fn = "pgfplotstables/itertime-vs-gap"
        fn += f"_{title}.dat" if title is not None else ".dat"
        df_.reset_index().apply(_compute_official_name_and_type, axis=1).to_string(
            fn, index=False
        )


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
    for i in filter(lambda i: i != best_method_idx, range(len(strs))):
        other_method_data = src_data[(problem, methods[i])]
        try:
            _, alpha = wilcoxon(best_method_data, other_method_data, alternative=side)
        except ValueError:
            alpha = float("nan")
        if alpha > threshold_alpha:
            strs[i] = f"\033[35m{strs[i]}\033[0m"
    return strs


def summary_tables(
    df: pd.DataFrame, summary: bool, pgfplotstables: bool, title: Optional[str] = None
) -> None:
    """Prints/saves the summary of the results in the given dataframe as three tables,
    one containing the (final) optimality gap, one the cumulative rewards, and the last
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
    field_names = ["Name", ""] + df.index.unique(level="method").to_list()
    tables = (
        pt.PrettyTable(field_names, title="gap"),
        pt.PrettyTable(field_names, title="return"),
        pt.PrettyTable(field_names[:1] + field_names[2:], title="time"),
    )

    # loop over every problem and fill the pretty tables
    for pname in df_.index.get_level_values("problem").unique():
        # gap (mean and median)
        gap_data = df["final-gap"]
        g_mean = _format_row(df_.loc[(pname, "final-gap-mean")], gap_data, prec=6)
        g_median = _format_row(df_.loc[(pname, "final-gap-median")], gap_data, prec=6)
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
    if summary:
        if title is not None:
            print(title)
        for table in tables:
            print(table.get_string())

    # save the first table to a latex-friendly format
    if pgfplotstables:
        table = tables[0].copy()
        table.align = "l"
        table._title = None
        for row in table.rows:
            if problem_name := row[0]:
                row[0] = rf"\multirow{{2}}*{{{problem_name}}}"
            for i, entry in enumerate(row):
                match = re.search(r"0\.\d+", entry)
                if match is not None:
                    num = f"{float(match.group()):.3f}"
                    if entry.startswith("\033[1;34m"):
                        num = rf"{{\color{{blue}}\textbf{{{num}}}}}"
                    elif entry.startswith("\033[35m"):
                        num = rf"{{\color{{purple}}\textit{{{num}}}}}"
                    row[i] = num
        latex = table.get_string(
            border=False, preserve_internal_border=True, hrules=pt.NONE
        )
        latex = latex.replace("|", "&")
        latex = "\n".join(line[:-1] + r"\\" for line in latex.split("\n"))

        fn = "pgfplotstables/summary"
        fn += f"_{title}.tex" if title is not None else ".tex"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(latex)


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
    group = parser.add_argument_group("Include/Exclude options")
    group.add_argument(
        "--include-methods",
        type=str,
        nargs="+",
        default=[],
        help="List of methods patterns to plot.",
    )
    group.add_argument(
        "--include-problems",
        type=str,
        nargs="+",
        default=[],
        help="List of benchmark problems patterns to plot.",
    )
    group.add_argument(
        "--exclude-methods",
        type=str,
        nargs="+",
        default=[],
        help="List of methods patterns not to plot.",
    )
    group.add_argument(
        "--exclude-problems",
        type=str,
        nargs="+",
        default=[],
        help="List of benchmark problems patterns not to plot.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot",
        action="store_true",
        help="Only print the summary and do not show the plots.",
    )
    group.add_argument(
        "--summary",
        action="store_true",
        help="Only show the plot and do not print the summary.",
    )
    group.add_argument(
        "--pgfplotstables",
        action="store_true",
        help="Generates the data files for PGFPLOTS.",
    )
    args = parser.parse_args()
    fplot, fsummary, fpgfplotstables = args.plot, args.summary, args.pgfplotstables

    # load each result and plot
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        stitle = filename if include_title else None
        dataframe = load_data(
            filename,
            args.include_methods,
            args.include_problems,
            args.exclude_methods,
            args.exclude_problems,
        )
        if fplot or fpgfplotstables:
            optimiser_convergences(dataframe, fplot, fpgfplotstables, "gap", stitle)
            itertime_vs_gap(dataframe, fplot, fpgfplotstables, stitle)
        if fsummary or fpgfplotstables:
            summary_tables(dataframe, fsummary, fpgfplotstables, stitle)
    plt.show()
