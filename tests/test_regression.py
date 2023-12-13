import unittest

import torch
from scipy.io import loadmat

from globopt.problems import SimpleProblem
from globopt.regression import Idw, Rbf

RESULTS = loadmat("tests/data_test_core.mat")


class TestRegression(unittest.TestCase):
    def test_fit_and_partial_fit(self) -> None:
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
        Y = problem(X)
        n = 3
        mdls = [Idw(X[:n], Y[:n]), Rbf(X[:n], Y[:n], eps=0.5, svd_tol=0.0)]
        for i in range(len(mdls)):
            mdl = mdls[i]
            if isinstance(mdl, Idw):
                mdls[i] = Idw(X, Y)
            else:
                mdls[i] = Rbf(X, Y, mdl.eps, mdl.svd_tol, (mdl.Minv, mdl.coeffs))
        x_hat = torch.linspace(-3, 3, 100, dtype=X.dtype).view(1, -1, 1)
        y_hat = torch.stack([mdl.posterior(x_hat).mean.squeeze() for mdl in mdls])
        y_hat_expected = torch.as_tensor(RESULTS["y_hat"][:2], dtype=y_hat.dtype)
        # NOTE: we only take the first 2 rows of y_hat because the third was computed
        # with a kernel that was later removed.
        torch.testing.assert_close(y_hat, y_hat_expected)


if __name__ == "__main__":
    unittest.main()
