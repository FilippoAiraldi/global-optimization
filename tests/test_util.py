import unittest

import numpy as np
from pymoo.util.normalization import ZeroToOneNormalization

from nmgo.util.normalization import SimpleArbitraryNormalization


class TestNormalization(unittest.TestCase):
    def test_simple_arbitrary_normalization__normalizes_correctly(self) -> None:
        xl = np.random.randn()
        xu = xl**2 + np.random.randn() ** 2
        xl_new = np.random.randn()
        xu_new = xl_new**2 + np.random.randn() ** 2

        normalizations = (
            SimpleArbitraryNormalization(xl, xu, xl_new, xu_new),
            ZeroToOneNormalization(xl, xu),
        )
        x = np.random.randn(1000)
        x_normalized = (
            normalizations[0].forward(x),
            normalizations[1].forward(x) * (xu_new - xl_new) + xl_new,
        )
        x_denormalized = (
            normalizations[0].backward(x),
            normalizations[1].backward((x - xl_new) / (xu_new - xl_new)),
        )

        np.testing.assert_allclose(x_normalized[0], x_normalized[1])
        np.testing.assert_allclose(x_denormalized[0], x_denormalized[1])


if __name__ == "__main__":
    unittest.main()
