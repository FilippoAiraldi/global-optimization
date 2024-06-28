import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tune import (
    INIT_ITER,
    MAX_ITER,
    PROBLEM_NAME,
    TIME_STEPS,
    CstrEnv,
    setup_mpc_tuning,
)

sys.path.append(os.getcwd())

from benchmarking.plot import load_data, plot_converges, plot_timings, summarize


def _extract_envdata(row: pd.Series) -> pd.Series:
    """Extracts average and std of the env data from a row of the dataframe."""
    iters = INIT_ITER + MAX_ITER
    S = np.asarray([np.fromstring(o, sep=",") for o in row["env-states"]])
    S = S.reshape(-1, iters, TIME_STEPS + 1, CstrEnv.ns)[..., [1, 2]]
    A = np.asarray([np.fromstring(o, sep=",") for o in row["env-actions"]])
    A = A.reshape(-1, iters, TIME_STEPS, CstrEnv.na)
    R = np.asarray([np.fromstring(o, sep=",") for o in row["env-rewards"]])
    R = R.reshape(-1, iters, TIME_STEPS).sum(-1)
    data = {}
    for n, val in [("S", S), ("A", A), ("R", R)]:
        val = val[:, INIT_ITER:]  # skip the first random iterations, not interesting
        data[n + "-avg"] = val.mean(axis=0)
        data[n + "-std"] = val.std(axis=0)  # sem(arr, axis=0)
    return pd.Series(data)


def plot_envdata(df: pd.DataFrame, figtitle: Optional[str]) -> None:
    """Plots the states, action and rewards of the environment."""
    # TODO: understand first what we really want to plot
    df_ = df.apply(_extract_envdata, axis=1).loc[PROBLEM_NAME]
    time = np.arange(TIME_STEPS + 1) * CstrEnv(float("nan")).tf
    np.arange(INIT_ITER, INIT_ITER + MAX_ITER)

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for i, (method, row) in enumerate(df_.iterrows()):
        c = f"C{i}"

        S_avg, S_std = row["S-avg"], row["S-std"]
        axs[0, 0].plot(time, S_avg[..., 0].T, lw=0.1, color=c)
        axs[0, 1].plot(time, S_avg[..., 1].T, lw=0.1, color=c)

        A_avg, A_std = row["A-avg"], row["A-std"]
        axs[1, 0].plot(time[:-1], A_avg[..., 0].T, lw=0.1, color=c)
        # axs[0, 0].plot(time, S_avg[:, 0], lw=1.0, color=c, label=method)

        # Ravg, Rstd = -row["R-avg"], -row["R-std"]
        # axs[1, 1].fill_between(episodes, Ravg - Rstd, Ravg + Rstd, alpha=0.2, color=c, step="pre")
        # axs[1, 1].step(episodes, Ravg, lw=1.0, color=c, label=method)
        # print("method", method)
        break

    # ax.fill_between(evals, ci_lb, ci_ub, alpha=0.2, color=color, step="pre")

    axs[0, 0].set_xlabel("Time [h]")
    axs[0, 0].set_ylabel("Concentration of B [mol/L]")
    axs[0, 1].set_xlabel("Time [h]")
    axs[0, 1].set_ylabel("Reactor Temperature [°C]")
    axs[1, 0].set_xlabel("Time [h]")
    axs[1, 0].set_ylabel("Feed Flow Rate [1/h]")
    axs[1, 1].set_xlabel("Evaluations")
    axs[1, 1].set_ylabel("Cost")
    fig.suptitle(figtitle, fontsize=12)


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

    setup_mpc_tuning()

    # load each result and plot
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        title = filename if include_title else None
        dataframe = load_data(filename, args.include, args.exclude)
        if not args.no_plot:
            plot_converges(dataframe, title, n_cols=1)
            plot_timings(dataframe, title, single_problem=True)
            plot_envdata(dataframe, title)
        if not args.no_summary:
            summarize(dataframe, title)
    plt.show()
