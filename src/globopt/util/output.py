from pymoo.core.algorithm import Algorithm
from pymoo.util.display.single import Column, SingleObjectiveOutput


class GlobalOptimizationOutput(SingleObjectiveOutput):
    """Display output for global optimization algorithms.

    Adds a column for the minimizer of the objective function found so far.
    """

    def __init__(self) -> None:
        super().__init__()
        self.x_min = Column(name="x_min")

    def initialize(self, algorithm: Algorithm) -> None:
        super().initialize(algorithm)
        self.columns.append(self.x_min)

    def update(self, algorithm: Algorithm) -> None:
        super().update(algorithm)
        opt = algorithm.opt[0]
        Xopt = opt.X if opt.feas else None
        if algorithm.problem.n_var == 1:
            Xopt = Xopt.item()
        self.x_min.set(Xopt)
