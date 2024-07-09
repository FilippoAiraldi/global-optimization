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

from benchmarking.plot import (
    itertime_vs_gap,
    load_data,
    optimiser_convergences,
    parse_args,
    summary_tables,
)


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
    args = parse_args("MPC tuning", multiproblem=False)
    fplot, fsummary, fpgfplotstables = args.plot, args.summary, args.pgfplotstables

    # load each result and plot
    setup_mpc_tuning()
    include_title = len(args.filenames) > 1
    for filename in args.filenames:
        stitle = filename if include_title else None
        dataframe = load_data(
            filename, args.include_methods, [], args.exclude_methods, []
        )
        if fplot or fpgfplotstables:
            optimiser_convergences(
                dataframe, fplot, fpgfplotstables, "best-so-far", stitle
            )
            itertime_vs_gap(dataframe, fplot, fpgfplotstables, stitle)
            # plot_envdata(dataframe, stitle)
        if fsummary or fpgfplotstables:
            summary_tables(dataframe, fsummary, fpgfplotstables, stitle)
    plt.show()
