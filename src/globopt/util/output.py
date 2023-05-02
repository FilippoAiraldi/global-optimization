import contextlib
import sys
from typing import Iterator, TextIO

from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.util.display.single import Column, SingleObjectiveOutput


class GlobalOptimizationOutput(SingleObjectiveOutput):
    """Display output for global optimization algorithms.

    Adds a column for the minimizer of the objective function found so far.
    """

    def __init__(self, include_x_min: bool = True, problem: Problem = None) -> None:
        """Instantiate the output object.

        Parameters
        ----------
        include_x_min : bool, optional
            Whether to include a column for the current minimizer, by default `True`.
        problem : Problem, optional
            If provided, the width of the `x_min` column will scaled to accommodate the
            number of variables in the problem.
        """
        super().__init__()
        width = 13 * (1 if problem is None else problem.n_var)
        self.x_min = Column(name="x_min", width=width) if include_x_min else None

    def initialize(self, algorithm: Algorithm) -> None:
        super().initialize(algorithm)
        if self.x_min is not None:
            self.columns.append(self.x_min)

    def update(self, algorithm: Algorithm) -> None:
        super().update(algorithm)
        if self.x_min is not None:
            opt = algorithm.opt[0]
            Xopt = opt.X if opt.feas else None
            if algorithm.problem.n_var == 1:
                Xopt = Xopt.item()
            self.x_min.set(Xopt)


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
