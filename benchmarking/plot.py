"""
Visualization and summary of results of benchmarking of myopic and non-myopic Global
Optimization strategies on synthetic problems.
"""

import argparse
from functools import partial
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

from globopt.problems import get_benchmark_problem

plt.style.use("bmh")


def load_data(csv_filename: str) -> pd.DataFrame:
    """Loads the data from the given file into a dataframe."""
    converter = partial(np.fromstring, sep=",")
    df = pd.read_csv(
        csv_filename,
        sep=";",
        dtype={"problem": pd.StringDtype(), "method": pd.StringDtype()},
        converters={s: converter for s in ["stage-reward", "best-so-far", "time"]},
    )

    # group by problem, method, and horizon, and stack all trials into 2d arrays
    df = df.groupby(["problem", "method"], dropna=False).aggregate(np.stack)

    # compute the optimality gaps
    def compute_optimality_gaps(row: pd.Series) -> np.ndarray:
        problem_name = row.name[0]
        f_opt = get_benchmark_problem(problem_name)[0].optimal_value
        init_best = row["best-so-far"][:, 0, np.newaxis]
        return (row["best-so-far"] - init_best) / (f_opt - init_best)

    df["gap"] = df.apply(compute_optimality_gaps, axis=1)
    return df


def plot(df: pd.DataFrame, figtitle: Optional[str]) -> None:
    """Plots the results in the given dataframe. In particular, it plots the
    convergence to the optimum, and the evolution of the optimality gap."""
    # create figure
    problem_names = df.index.unique(level="problem")
    n_cols = 5
    n_rows = ceil(len(problem_names) / n_cols)
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 2.5))

    # create main grid of plots
    plotted_methods = set()
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
                worst = data.max(axis=0)
                best = data.min(axis=0)
                h = ax.plot(evals, avg, lw=1.0, color=color)
                if color is None:
                    color = h[0].get_color()
                # ax.plot(evals, worst, color=c, lw=0.25)
                # ax.plot(evals, best, color=c, lw=0.25)
                ax.fill_between(evals, worst, best, alpha=0.2, color=color)
            plotted_methods.add((method, color))

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
        ax_opt.set_xlim(1, evals[-1])
        ax_gap.set_ylim(-0.05, 1.05)
        ax_gap.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    # create legend manually
    handles = [Line2D([], [], label=m, color=c) for m, c in plotted_methods]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(figtitle, fontsize=12)


def custom_violin(
    ax: Axes,
    data: ArrayLike,
    pos: float,
    fc: str = "b",
    ec: str = "k",
    alpha: float = 0.7,
    percentiles: ArrayLike = [25, 50, 75],
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


def plot_violins(df: pd.DataFrame, figtitle: Optional[str]) -> None:
    """Plots the results in the given dataframe. In particular, it plots the
    distribution of the (final) optimality gap versus the cumulative rewards as violins.
    """
    gaps = df["gap"].map(lambda g: g[:, -1])
    returns = df["stage-reward"].map(lambda r: r.sum(axis=1))
    df_: pd.DataFrame = (
        pd.concat((gaps, returns), axis=1, keys=["gap", "return"])
        .stack(level=0, dropna=False)
        .unstack(level="method", fill_value=pd.NA)
    )

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
            custom_violin(
                ax,
                df_.loc[(problem_name, "gap")][method],
                pos,
                "C0",
                "C0",
                side="left",
                scatter_kwargs=s_kwargs,
                violin_kwargs=v_kwargs,
            )
            return_ = df_.loc[(problem_name, "return")][method]
            if np.logical_not(np.isnan(return_)).any():
                custom_violin(
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
    ax.set_xlabel("Horizon")
    fig.suptitle(figtitle, fontsize=12)


def summarize(df: pd.DataFrame, tabletitle: Optional[str]) -> None:
    """Prints the summary of the results in the given dataframe as two tables, one
    containing the (final) optimality gap, and the other the cumulative rewards."""

    precision = 6

    def row2string(row: pd.Series) -> pd.Series:
        strs = row.map(lambda x: f"\033[2;36m{x:.{precision}f}\033[0m")
        best_idx = np.nanargmax(row)
        strs.iloc[best_idx] = f"\033[1;35m{row.iloc[best_idx]:.{precision}f}\033[0m"
        return strs

    tables: list[str] = []
    for col, getter in zip(
        ("gap", "stage-reward"), (lambda g: g[:, -1], lambda r: r.sum(axis=1))
    ):
        mean = df[col].map(lambda g: np.mean(getter(g)))
        median = df[col].map(lambda g: np.median(getter(g)))
        tables.append(
            pd.concat((mean, median), axis=1, keys=["mean", "median"])
            .stack(level=0, dropna=False)
            .unstack(level="method", fill_value=pd.NA)
            .apply(row2string, axis=1)
            .to_string(na_rep="-", justify="left")
        )

    if tabletitle is not None:
        print(tabletitle, "\n")
    rows = zip(*(t.splitlines() for t in tables))
    print("\n".join(row1 + "\t" + row2 for row1, row2 in rows))


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
        title = filename if include_title else None
        dataframe = load_data(filename)
        if not args.no_plot:
            plot(dataframe, title)
            plot_violins(dataframe, title)
        if not args.no_summary:
            summarize(dataframe, title)
    plt.show()
