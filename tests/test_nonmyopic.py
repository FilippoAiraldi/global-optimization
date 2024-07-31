import unittest

import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement

from globopt import GaussHermiteSampler, Ms, Rbf, make_idw_acq_factory
from globopt.problems import SimpleProblem


class TestNonMyopicAcquisitionFunction(unittest.TestCase):
    def test_make_idw_acq_factory(self):
        c1, c2, span_Y_min = 1.0, 0.5, 1e-3
        factory = make_idw_acq_factory(c1, c2, span_Y_min)
        self.assertTrue(callable(factory))
        self.assertDictEqual(factory(), {"c1": c1, "c2": c2, "span_Y_min": span_Y_min})

    def test_init__overrides_default_samplers__with_base_MC_acq_func(self):
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
        Y = problem(X)
        mdl = Rbf(X, Y, 0.5)
        fantasies_samplers = [GaussHermiteSampler(torch.Size([1]))]
        valfunc_sampler = GaussHermiteSampler(torch.Size([16]))

        acqfun = Ms(
            mdl,
            fantasies_samplers,
            qExpectedImprovement,
            valfunc_sampler=valfunc_sampler,
        )

        self.assertTrue(all(s is valfunc_sampler for s in acqfun.inner_samplers))

    def test_init__does_not_override_default_samplers__with_base_analytical_acq_func(
        self,
    ):
        problem = SimpleProblem()
        X = torch.as_tensor([-2.61, -1.92, -0.63, 0.38, 2], device="cpu").unsqueeze(-1)
        Y = problem(X)
        mdl = Rbf(X, Y, 0.5)
        fantasies_samplers = [GaussHermiteSampler(torch.Size([1]))]
        valfunc_sampler = GaussHermiteSampler(torch.Size([16]))

        acqfun = Ms(
            mdl,
            fantasies_samplers,
            ExpectedImprovement,
            valfunc_sampler=valfunc_sampler,
        )

        self.assertTrue(all(s is None for s in acqfun.inner_samplers))


if __name__ == "__main__":
    unittest.main()
