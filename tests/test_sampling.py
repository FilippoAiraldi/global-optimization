import unittest
from math import pi, sqrt

import torch
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from globopt import GaussHermiteSampler


class TestSampling(unittest.TestCase):
    def test_GH__supports_only_single_dim(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, "Only a single dimension is supported."
        ):
            GaussHermiteSampler(torch.Size([1000, 100]))

    def test_GH__returns_correct_base_samples(self) -> None:
        # with such posterior, the base samples are returned as they are, i.e., 1*s + 0
        distribution = MultivariateNormal(torch.zeros((1,)), torch.eye(1))
        posterior = GPyTorchPosterior(distribution)
        SQRT2 = sqrt(2.0)
        SQRTPI = sqrt(pi)
        EXPECTED = {
            2: ([-0.707107, 0.707107], [0.886227, 0.886227]),
            3: ([-1.22474, 0, 1.22474], [0.295409, 1.18164, 0.295409]),
            4: (
                [-1.65068, -0.524648, 0.524648, 1.65068],
                [0.0813128, 0.804914, 0.804914, 0.0813128],
            ),
            5: (
                [-2.02018, -0.958572, 0, 0.958572, 2.02018],
                [0.0199532, 0.393619, 0.945309, 0.393619, 0.0199532],
            ),
        }

        for n, (samples, weights) in EXPECTED.items():
            sampler = GaussHermiteSampler(torch.Size([n]))
            actual_samples = sampler(posterior)
            torch.testing.assert_close(
                actual_samples.flatten(), torch.tensor(samples) * SQRT2, msg=f"{n}"
            )
            torch.testing.assert_close(
                sampler.base_weights.flatten(),
                torch.tensor(weights) / SQRTPI,
                msg=f"{n}",
            )


if __name__ == "__main__":
    unittest.main()
