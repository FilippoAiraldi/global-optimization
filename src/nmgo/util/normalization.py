from typing import Union

import numpy as np
import numpy.typing as npt
from pymoo.util.normalization import Normalization


class SimpleArbitraryNormalization(Normalization):
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
