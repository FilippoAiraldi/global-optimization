import pickle
import unittest
from itertools import product

import numpy as np
from parameterized import parameterized

from globopt.core.regression import Idw, fit
from globopt.nonmyopic.acquisition import acquisition

with open(r"tests/data_test_non_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)


class TestAcquisition(unittest.TestCase):
    @parameterized.expand(product((0, 2**3), (False, True)))
    def test__returns_correct_values(self, mc_iters: int, rollout: bool):
        np_random = np.random.default_rng(1909)
        n_samples = 7
        horizon = 5
        dim = 3
        X = np_random.standard_normal((1, n_samples, dim))
        y = np_random.standard_normal((1, n_samples, 1))
        mdl = fit(Idw(), X, y)
        discount = np_random.random()
        c1 = np_random.random() * 2 + 1
        c2 = np_random.random() * 2 + 1
        x = np_random.standard_normal(
            (n_samples * 2, 1, dim) if rollout else (n_samples * 2, horizon, dim)
        )

        a = acquisition(
            x=x,
            mdl=mdl,
            horizon=horizon,
            discount=discount,
            c1=c1,
            c2=c2,
            rollout=rollout,
            lb=X.min(1).squeeze() - 0.1,
            ub=X.max(1).squeeze() + 0.1,
            mc_iters=mc_iters,
            seed=np_random,
            parallel={"n_jobs": -1, "verbose": 0},
            return_iters=True,
            pso_kwargs={"maxiter": 2000, "ftol": 1e-12, "xtol": 1e-12},
        )

        expected = RESULTS[(mc_iters, rollout)]
        np.testing.assert_allclose(a, expected, atol=1e-6, rtol=1e-6)


# class TestAlgorithm(unittest.TestCase):
#     def test_example1__returns_correct_result(self):
#         seed = 17
#         problem = Simple1DProblem()
#         x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
#         algorithm = NonMyopicGO(
#             regression=Rbf("thinplatespline", 0.01),
#             init_points=x0,
#             c1=1,
#             c2=0.5,
#             horizon=2,
#             discount=1.0,
#         )
#         res = minimize(
#             problem,
#             algorithm,
#             termination=("n_iter", 4),
#             seed=1,
#             save_history=True,
#             copy_algorithm=False,
#         )

#         x = np.linspace(*problem.bounds(), 300).reshape(-1, 1)
#         out: dict[int, tuple[np.ndarray, ...]] = {}
#         with Parallel(n_jobs=-1, batch_size=8) as parallel:
#             for i, algo in enumerate(res.history, start=1):
#                 y_hat = predict(algo.regression, x)
#                 Xm = algo.pop.get("X")
#                 ym = algo.pop.get("F").reshape(-1)
#                 a = acquisition(
#                     x,
#                     algo.regression,
#                     algo.horizon,
#                     algo.discount,
#                     algo.c1,
#                     algo.c2,
#                     None,
#                     problem.xl,
#                     problem.xu,
#                     parallel,
#                     seed,
#                 )
#                 acq_min = (
#                     algo.acquisition_min_res.X
#                     if hasattr(algo, "acquisition_min_res")
#                     else np.nan
#                 )
#                 out[i] = (y_hat, Xm, ym, a, acq_min)

#         for key in out:
#             for actual, expected in zip(out[key], RESULTS[key]):
#                 np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
