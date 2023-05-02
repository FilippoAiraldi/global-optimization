"""Utility classes for easier normalization of ranges and values."""


from numbers import Number
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from pymoo.core.problem import Problem
from pymoo.util.normalization import Normalization

from globopt.util.wrapper import Wrapper


class RangeNormalization(Normalization):
    """
    Normalization to and from arbitrary bounds. It is simple in the sense that it does
    not consider any none values and assumes lower as well as upper bounds are given.
    """

    __slots__ = ("xl", "xu", "xl_new", "xu_new")

    def __init__(
        self,
        xl: Union[float, npt.ArrayLike],
        xu: Union[float, npt.ArrayLike],
        xl_new: Union[float, npt.ArrayLike],
        xu_new: Union[float, npt.ArrayLike],
    ) -> None:
        """Constructs the normalization object.

        Parameters
        ----------
        xl : float or array_like
            Old lower bound.
        xu : float or array_like
            Old upper bound.
        xl_new : float or array_like
            New lower bound.
        xu_new : float or array_like
            New upper bound.
        """
        super().__init__()
        self.xl = np.asarray(xl)
        self.xu = np.asarray(xu)
        self.xl_new = np.asarray(xl_new)
        self.xu_new = np.asarray(xu_new)

    def forward(self, X: Union[float, npt.ArrayLike]) -> npt.NDArray[np.floating]:
        """Normalizes `X` from the old bounds to the new bounds.

        Parameters
        ----------
        X : float or array_like
            The array to normalize.

        Returns
        -------
        array
            The normalized array.
        """
        X = np.asarray(X)
        return (X - self.xl) / (self.xu - self.xl) * (
            self.xu_new - self.xl_new
        ) + self.xl_new

    def backward(self, X: Union[float, npt.ArrayLike]) -> npt.NDArray[np.floating]:
        """Denormalizes `X` from the new bounds to the old bounds.

        Parameters
        ----------
        X : float or array_like
            The array to denormalize.

        Returns
        -------
        array
            The denormalized array.
        """
        X = np.asarray(X)
        return (X - self.xl_new) / (self.xu_new - self.xl_new) * (
            self.xu - self.xl
        ) + self.xl

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}[({self.xl}, {self.xu}) -> ({self.xl_new}, {self.xu_new})]"

    def __str__(self) -> str:
        return super().__repr__()


class NormalizedProblemWrapper(Problem, Wrapper[Problem]):
    """A wrapper class for the normalization of problems."""

    def __init__(
        self,
        problem: Problem,
        new_xl: Union[Number, npt.ArrayLike] = -1,
        new_xu: Union[Number, npt.ArrayLike] = +1,
    ) -> None:
        assert problem.has_bounds(), "Can only normalize bounded problems."
        Problem.__init__(
            self,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            n_ieq_constr=problem.n_ieq_constr,
            n_eq_constr=problem.n_eq_constr,
            xl=new_xl,
            xu=new_xu,
            vtype=problem.vtype,
            vars=problem.vars,
            elementwise=problem.elementwise,
            elementwise_func=problem.elementwise_func,
            elementwise_runner=problem.elementwise_runner,
            replace_nan_values_by=problem.replace_nan_values_by,
            exclude_from_serialization=problem.exclude_from_serialization,
            callback=problem.callback,
            strict=problem.strict,
        )
        Wrapper.__init__(self, to_wrap=problem)
        self.normalization = RangeNormalization(*self.wrapped.bounds(), *self.bounds())

    def original_bounds(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Gets the bounds of the wrapped problem."""
        return self.wrapped.bounds()

    def _evaluate(
        self, x: npt.NDArray[np.floating], out: dict[str, Any], *args, **kwargs
    ) -> None:
        x = self.normalization.backward(x)
        self.wrapped._evaluate(x, out, *args, **kwargs)

    def _calc_pareto_set(self) -> Union[None, Number, npt.NDArray[np.floating]]:
        pf = self.wrapped._calc_pareto_set()
        return None if pf is None else self.normalization.forward(pf)
