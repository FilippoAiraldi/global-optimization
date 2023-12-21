import pickle
import unittest

import torch

from globopt.myopic_acquisitions import (
    IdwAcquisitionFunction,
    _idw_distance,
    acquisition_function,
)
from globopt.problems import SimpleProblem
from globopt.regression import Rbf

with open(r"tests/data_test_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)


class TestAcquisitionFunction(unittest.TestCase):
    def test__methods__returns_correct_values(self):
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
        Y = problem(X)

        mdl = Rbf(X, Y, 0.5)
        x = torch.linspace(-3, 3, 1000, dtype=X.dtype).view(1, -1, 1)
        MAF = IdwAcquisitionFunction(mdl, 1.0, 0.5)
        a1 = MAF(x.transpose(1, 0)).squeeze().neg()

        y_hat, s, W_sum_recipr, _ = mdl(x)
        dym = Y.amax(-2) - Y.amin(-2)
        z = _idw_distance(W_sum_recipr)
        a2 = acquisition_function(y_hat, s, dym, W_sum_recipr, MAF.c1, MAF.c2).neg()

        out = torch.stack((s.squeeze(), z.squeeze(), a2.squeeze()))
        out_expected = torch.as_tensor(RESULTS["acquisitions"], dtype=a1.dtype)
        torch.testing.assert_close(a1, a2.squeeze())
        torch.testing.assert_close(a1, out_expected[-1], rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(out, out_expected, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
