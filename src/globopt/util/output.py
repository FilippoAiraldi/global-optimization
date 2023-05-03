"""
Utility classes for better stdout printing of solver's information during minimization
runs.
"""


import contextlib
import sys
from typing import Iterator, Optional, TextIO

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.util.display.single import Column, SingleObjectiveOutput


def pareto_set_if_possible(problem: Problem) -> Optional[np.ndarray]:
    """Gets the pareto set of the problem if possible, otherwise returns `None`."""
    try:
        return problem.pareto_set()
    except Exception:
        return None


class GlobalOptimizationOutput(SingleObjectiveOutput):
    """Display output for global optimization algorithms.

    Adds a column for the distance of the current best minimizer of the objective
    function found so far and the (single) true minimizer.
    """

    def __init__(self) -> None:
        """Instantiate the output object."""
        super().__init__()
        self.x_gap = Column(name="x_gap")

    def initialize(self, algorithm: Algorithm) -> None:
        super().initialize(algorithm)
        problem: Problem = algorithm.problem
        ps = pareto_set_if_possible(problem)
        self.x_opt = None if ps is None else np.reshape(ps, problem.n_var)
        if self.x_opt is not None:
            self.columns.append(self.x_gap)

    def update(self, algorithm: Algorithm) -> None:
        super().update(algorithm)
        if self.x_opt is not None:
            opt = algorithm.opt[0]
            if opt.feas:
                Xopt = opt.X.reshape(algorithm.problem.n_var)
                self.x_gap.set(np.linalg.norm(Xopt - self.x_opt))


class PrefixedStream:
    """Wrapper to prefix a stream with a string every time `write` is called."""

    def __init__(self, prefix: str, original_stream: TextIO) -> None:
        self.prefix = prefix
        self.original_stream = original_stream

    def write(self, message: str) -> int:
        if message != "\n":
            message = self.prefix + message.replace("\n", "\n" + self.prefix)
        return self.original_stream.write(message)

    @classmethod
    @contextlib.contextmanager
    def prefixed_print(cls, prefix: str) -> Iterator[None]:
        old_sys_stdout = sys.stdout  # might be different from sys.__stdout__
        sys.stdout = cls(prefix, sys.stdout)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old_sys_stdout
