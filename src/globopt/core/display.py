"""
Utility classes for better stdout printing of solver's information during minimization
runs.
"""


# TODO: remove this file in its entirety


from typing import Any, Optional

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.util.display.display import Display
from pymoo.util.display.single import Column, Output, SingleObjectiveOutput


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
        ps_ = pareto_set_if_possible(problem)
        if ps_ is None:
            self.x_opt = None
        else:
            ps: np.ndarray = np.atleast_2d(np.squeeze(ps_))
            assert (
                ps.ndim == 2 and ps.shape[1] == problem.n_var
            ), "Pareto set must have shape `(n_points, n_var)`."
            self.x_opt = ps
            self.columns.append(self.x_gap)

    def update(self, algorithm: Algorithm) -> None:
        super().update(algorithm)
        if self.x_opt is not None:
            opt = algorithm.opt[0]
            if opt.feas:
                # compute distance of current best to pareto set and pick closest
                self.x_gap.set(np.linalg.norm(opt.X - self.x_opt, axis=1).min())


class GlobalOptimizationDisplay(Display):
    """
    GO-specific display class that allows to force printing the header at each iteration
    as well as prepend a prefix to the printed message.
    """

    def __init__(
        self,
        *args: Any,
        force_header: bool = True,
        header_prefix: str = "\n",
        line_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        """Instantiates the display object.

        Parameters
        ----------
        force_header : bool, optional
            Whether to force printing the header at each iteration, by default `True`.
        header_prefix : str, optional
            String to prepend to the header of printed message, by default "".
        line_prefix : str, optional
            String to prepend to each line of the printed message, by default "".
        """
        super().__init__(*args, **kwargs)
        self.force_header = force_header
        self.header_prefix = header_prefix
        self.line_prefix = line_prefix

    def update(self, algorithm: Algorithm, **_) -> None:
        """Updates the display, printing the output and updating the progress bar."""
        output, progress = self.output, self.progress
        if self.verbose and output:
            print(
                self._get_text(output, algorithm, self.header_prefix, self.line_prefix)
            )
        if progress:
            perc = algorithm.termination.perc
            progress.set(perc)

    def _get_text(
        self,
        output: Output,
        algorithm: Algorithm,
        header_prefix: str = "",
        line_prefix: str = "",
    ) -> str:
        """Internal method to get the text to print."""
        header = not output.is_initialized or self.force_header
        output(algorithm)
        text = ""
        if header:
            text += output.header(border=True) + "\n"
        text += output.text()
        if line_prefix != "":
            text = line_prefix + text.replace("\n", "\n" + line_prefix)
        if header and header_prefix != "":
            text = header_prefix + text
        return text
