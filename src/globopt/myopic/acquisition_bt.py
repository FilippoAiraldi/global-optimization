"""
Implementation of the acquisition function for RBF/IDW-based Global Optimization
according to [1].

References
----------
[1] A. Bemporad. Global optimization via inverse distance weighting and radial basis
    functions. Computational Optimization and Applications, 77(2):571–595, 2020
"""

from torch import Tensor


def idw_scale(Y: Tensor, train_Y: Tensor, V: Tensor) -> Tensor:
    """Computes the IDW standard deviation function.

    Parameters
    ----------
    Y : Tensor
        `b x q x 1` estimates of the function values at the candidate points.
    train_Y : Tensor
        `b x m x 1` training data points.
    V : Tensor
        `b x q x m` normalized IDW weights.

    Returns
    -------
    Tensor
        The standard deviation of shape `b x q x 1`.
    """
    sqdiff = (train_Y - Y.transpose(2, 1)).square()
    return V.bmm(sqdiff).diagonal(dim1=1, dim2=2).sqrt().unsqueeze(-1)
