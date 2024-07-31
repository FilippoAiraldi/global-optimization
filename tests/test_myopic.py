import pickle
import unittest

import torch

from globopt import (
    GaussHermiteSampler,
    IdwAcquisitionFunction,
    Rbf,
    qIdwAcquisitionFunction,
)
from globopt.myopic_acquisitions import _idw_distance, idw_acquisition_function
from globopt.problems import SimpleProblem

with open(r"tests/data_test_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)


class TestAcquisitionFunction(unittest.TestCase):
    def test_IdwAcquisitionFunction__returns_correct_values(self):
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
        Y = problem(X)

        mdl = Rbf(X, Y, 0.5)
        x = torch.linspace(-3, 3, RESULTS["N"], dtype=X.dtype).view(1, -1, 1)
        MAF = IdwAcquisitionFunction(mdl, 1.0, 0.5)

        # compute the acquisition function with N-batches 1x1 input
        acqfun1 = MAF(x.transpose(1, 0))

        # compute the same acquisition function with a single Nx1 input
        y_hat, scale, W_sum_recipr, _ = mdl(x)
        dym = Y.amax(-2) - Y.amin(-2)
        dist = _idw_distance(W_sum_recipr)
        acqfun2 = idw_acquisition_function(
            y_hat, scale, dym, W_sum_recipr, MAF.c1, MAF.c2
        )

        torch.testing.assert_close(
            acqfun1.flatten(), acqfun2.flatten(), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            acqfun1.flatten(),
            torch.as_tensor(RESULTS["acquisition"], dtype=acqfun1.dtype),
        )
        torch.testing.assert_close(
            scale.flatten(), torch.as_tensor(RESULTS["scale"], dtype=scale.dtype)
        )
        torch.testing.assert_close(
            dist.flatten(), torch.as_tensor(RESULTS["distance"], dtype=dist.dtype)
        )

    def test_qIdwAcquisitionFunction__returns_correct_values(self):
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
        Y = problem(X)

        mdl = Rbf(X, Y, 0.5)
        x = torch.linspace(-3, 3, RESULTS["N"], dtype=X.dtype).view(1, -1, 1)
        sampler = GaussHermiteSampler(torch.Size([RESULTS["gh_samples"]]))
        MAF = qIdwAcquisitionFunction(mdl, 1.0, 0.5, sampler)
        acqfun = MAF(x.transpose(1, 0)).squeeze().neg()

        torch.testing.assert_close(
            acqfun.flatten(),
            torch.as_tensor(RESULTS["qacquisition"], dtype=acqfun.dtype),
        )


if __name__ == "__main__":
    unittest.main()
