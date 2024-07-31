import pickle
import unittest

import torch
from parameterized import parameterized

from globopt import Idw, Rbf
from globopt.problems import SimpleProblem

with open(r"tests/data_test_regression.pkl", "rb") as f:
    RESULTS = pickle.load(f)


class TestRegression(unittest.TestCase):
    def test_fit__and__partial_fit(self) -> None:
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").view(-1, 1)
        Y = problem(X)
        n = 3

        # fit
        idw_mdl = Idw(X[..., :n, :], Y[..., :n, :])
        rbf_mdl = Rbf(X[..., :n, :], Y[..., :n, :], eps=0.5, svd_tol=0.0)

        # partial fit
        idw_mdl = Idw(X, Y)
        rbf_mdl = Rbf(X, Y, rbf_mdl.eps, rbf_mdl.svd_tol, rbf_mdl.state)

        x_hat = torch.linspace(-3, 3, RESULTS["N"], dtype=X.dtype).view(1, -1, 1)
        y_hat_idw = idw_mdl.posterior(x_hat).mean.squeeze()
        y_hat_rbf = rbf_mdl.posterior(x_hat).mean.squeeze()
        torch.testing.assert_close(
            y_hat_idw, torch.as_tensor(RESULTS["idw"], dtype=y_hat_idw.dtype), msg="idw"
        )
        torch.testing.assert_close(
            y_hat_rbf, torch.as_tensor(RESULTS["rbf"], dtype=y_hat_rbf.dtype), msg="rbf"
        )

    def test_fit__and__partial_fit__with_repeated_datapoint(self) -> None:
        # when a datapoint is repeated, IDW is expected not to be bothered by it, but
        # its output will change. For RBF not to fail instead we have to set svd_tol,
        # and it will ignore the new point and the output will be the same
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2, 2], device="cpu")
        X = X.view(1, -1, 1)  # to avoid issues with torch.script in rbf
        Y = problem(X)
        n = 3

        # fit
        idw_mdl = Idw(X[..., :n, :], Y[..., :n, :])
        rbf_mdl = Rbf(X[..., :n, :], Y[..., :n, :], eps=0.5, svd_tol=1e-8)

        # partial fit
        idw_mdl = Idw(X, Y)
        rbf_mdl = Rbf(X, Y, rbf_mdl.eps, rbf_mdl.svd_tol, rbf_mdl.state)

        x_hat = torch.linspace(-3, 3, RESULTS["N"], dtype=X.dtype).view(1, -1, 1)
        y_hat_idw = idw_mdl.posterior(x_hat).mean.squeeze()
        y_hat_rbf = rbf_mdl.posterior(x_hat).mean.squeeze()
        torch.testing.assert_close(
            y_hat_idw,
            torch.as_tensor(RESULTS["idw_repeated"], dtype=y_hat_idw.dtype),
            msg="idw",
        )
        torch.testing.assert_close(
            y_hat_rbf, torch.as_tensor(RESULTS["rbf"], dtype=y_hat_rbf.dtype), msg="rbf"
        )

    @parameterized.expand([(Rbf,), (Idw,)])
    def test_condition_on_observations__works_as_intended(self, cls: type):
        dim, N, M, fantasies = torch.randint(2, 10, (4,))
        X_train = torch.randn((N, dim))
        Y_train = torch.randn((N, 1))
        mdl = cls(X_train, Y_train)

        X = torch.randn((fantasies, M, dim))
        Y = torch.randn((fantasies, M, 1))
        mdl_new = mdl.condition_on_observations(X, Y)

        self.assertIsInstance(mdl_new, cls)
        self.assertEqual(mdl_new.train_X.shape, (fantasies, N + M, dim))
        self.assertEqual(mdl_new.train_Y.shape, (fantasies, N + M, 1))


if __name__ == "__main__":
    unittest.main()
