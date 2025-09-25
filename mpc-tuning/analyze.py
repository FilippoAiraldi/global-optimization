"""
Analysis and visualization and summary of results of benchmarking of myopic and
non-myopic Global Optimization strategies on the data-driven MPC tuning problem.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_dir = Path(__file__).resolve().parents[1]
sys.path.extend((str(repo_dir), str(repo_dir / "mpc-tuning")))

from run import INIT_ITER, MAX_ITER, PROBLEM_NAME, TIME_STEPS, CstrEnv, setup_mpc_tuning

from benchmarking.analyze import (
    _compute_avg_and_ci,
    itertime_vs_gap,
    load_data,
    official_method_name_and_type,
    optimiser_convergences,
    parse_args,
    summary_tables,
)


def _extract_reactor_temp(
    row: pd.Series,
    iters: int | list[int] | slice | None = None,
    temp_idx: int = 2,
) -> pd.Series:
    """Extracts the reactor temperature from a row of the dataframe."""
    if iters is None:
        iters = slice(None)
    states = np.asarray([np.fromstring(o, sep=",") for o in row["env-states"]])
    states = states.reshape(-1, INIT_ITER + MAX_ITER, TIME_STEPS + 1, CstrEnv.ns)
    reactor_temp = states[:, iters, :, temp_idx]
    return pd.Series({"Tr": reactor_temp})


def plot_reactor_temp(
    df: pd.DataFrame, plot: bool, pgfplotstables: bool, title: str | None
) -> None:
    """Plots/saves the results on the reactor temperature from the env."""
    iters = [1, 25, 53]
    time = np.arange(TIME_STEPS + 1) * CstrEnv.tf
    df_ = df.apply(_extract_reactor_temp, iters=iters, axis=1).loc[PROBLEM_NAME]
    df_ = df_.map(lambda x: x.transpose((1, 2, 0)))
    df_ = df_.apply(_compute_avg_and_ci, args=("Tr", None), axis=1)

    if plot:
        methods = df_.index
        N = len(methods)
        ncols = int(np.ceil(np.sqrt(N)))
        nrows = int(np.ceil(N / ncols))
        fig, axs = plt.subplots(nrows, ncols, constrained_layout=True, sharex=True)
        axs = np.atleast_1d(axs).flat
        for (method, row_data), ax in zip(df_.iterrows(), axs):
            Tr_avg = row_data["avg"].T
            Tr_lb = row_data["lb"].T
            Tr_ub = row_data["ub"].T
            for i, (Tr_avg_, Tr_lb_, Tr_ub_) in enumerate(zip(Tr_avg, Tr_lb, Tr_ub)):
                color = f"C{i}"
                ax.plot(time, Tr_avg_, lw=1.0, color=color)
                ax.fill_between(time, Tr_lb_, Tr_ub_, alpha=0.2, color=color)
                ax.plot(time, Tr_lb_, color=color, lw=0.1)
                ax.plot(time, Tr_ub_, color=color, lw=0.1)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Reactor Temperature [°C]")
            ax.set_title(method)
        for ax in axs:
            ax.set_axis_off()
        fig.suptitle(title, fontsize=12)

    if pgfplotstables:
        os.makedirs("pgfplotstables", exist_ok=True)
        for method, row_data in df_.iterrows():
            method = official_method_name_and_type(method, for_filename=True)[0].lower()
            Tr_avg = row_data["avg"].T
            Tr_lb = row_data["lb"].T
            Tr_ub = row_data["ub"].T
            columns = ["time", "avg", "lb", "ub"]
            for iter_, Tr_avg_, Tr_lb_, Tr_ub_ in zip(iters, Tr_avg, Tr_lb, Tr_ub):
                data = np.column_stack((time, Tr_avg_, Tr_lb_, Tr_ub_))
                fn = f"pgfplotstables/reactor_temp_{method}_iter{iter_}"
                fn += f"_{title}.dat" if title is not None else ".dat"
                pd.DataFrame(data, columns=columns).to_string(fn, index=False)
                print(f"INFO: written `{fn}`.")


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
            optimiser_convergences(dataframe, fplot, fpgfplotstables, "gap", stitle)
            itertime_vs_gap(dataframe, fplot, fpgfplotstables, stitle)
            plot_reactor_temp(dataframe, fplot, fpgfplotstables, stitle)
        if fsummary or fpgfplotstables:
            summary_tables(dataframe, fsummary, fpgfplotstables, stitle)
    plt.show()
