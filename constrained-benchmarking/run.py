import os
import sys
from pathlib import Path

import torch

from globopt.problems import get_available_constrained_benchmark_problems

# I am lazy so let's import all the helpful functions defined in benchmarking/run.py
# instead of coding them again here
repo_dir = Path(__file__).resolve().parents[1]
sys.path.extend((str(repo_dir), str(repo_dir / "benchmarking")))

from benchmarking.run import create_csv_if_needed, parse_args, run_benchmarks

BENCHMARK_CONSTRAINED_PROBLEMS = get_available_constrained_benchmark_problems()


if __name__ == "__main__":
    args = parse_args("Constrained benchmarking", BENCHMARK_CONSTRAINED_PROBLEMS)
    csv = create_csv_if_needed(args.csv, "problem;method;best-so-far;time")

    # ensure each job runs only on one CPU
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ[
        "OMP_NUM_THREADS"
    ] = "1"
    # torch.set_num_threads(1)  # this must be done inside each job

    run_benchmarks(
        args.methods,
        args.problems,
        args.n_trials,
        args.seed,
        args.n_jobs,
        csv,
        list(map(torch.device, args.devices)),
    )
