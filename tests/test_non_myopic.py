import pickle
import unittest

import numpy as np
from joblib import Parallel

from globopt.core.problems import Simple1DProblem
from globopt.core.regression import Idw, fit
from globopt.nonmyopic.acquisition import acquisition

with open(r"tests/data_test_non_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)


xl, xu = -3, +3
f = Simple1DProblem.f


class TestAcquisition(unittest.TestCase):
    def test__returns_correct_values(self):
        seed = 17
        np.random.seed(seed)
        n_var = 3
        n_samples = 10
        X = np.random.randn(n_samples, n_var)
        y = np.random.randn(n_samples)
        mdl = fit(Idw(), X, y)

        h = 5
        discount = np.random.rand()
        c1 = np.random.rand() * 2 + 1
        c2 = np.random.rand() * 2 + 1
        x = np.random.randn(n_samples * 2, n_var)

        with Parallel(n_jobs=-1, batch_size=8, verbose=0) as parallel:
            a = acquisition(x, mdl, h, discount, c1, c2, None, xl, xu, parallel, seed)

        np.testing.assert_allclose(a, RESULTS["acquisition"], atol=1e-4, rtol=1e-4)


# class TestAlgorithm(unittest.TestCase):
#     def test__returns_correct_result(self):
#         problem = Simple1DProblem()
#         x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
#         algorithm = NonMyopicGO(
#             horizon=2,
#             regression=Rbf("thinplatespline", 0.01, svd_tol=0),
#             init_points=x0,
#             acquisition_fun_kwargs={"c1": 1, "c2": 0.5},
#         )

#         res = minimize(
#             problem,
#             algorithm,
#             termination=("n_iter", 4),
#             seed=1,
#             save_history=True,
#         )

#         x = np.linspace(*problem.bounds(), 500).reshape(-1, 1)
#         out: dict[int, tuple[np.ndarray, ...]] = {}
#         for i, algo in enumerate(res.history, start=1):
#             y_hat = predict(algo.regression, x[np.newaxis])
#             Xm = algo.pop.get("X").reshape(-1, 1)
#             ym = algo.pop.get("F").reshape(-1)
#             a = acquisition(
#                 x.reshape(-1, 1, 1), algo.regression, **algo.acquisition_fun_kwargs
#             )
#             acq_min = (
#                 algo.acquisition_min_res.opt.item().X
#                 if hasattr(algo, "acquisition_min_res")
#                 else np.nan
#             )
#             out[i] = (y_hat.squeeze(), Xm, ym, a, acq_min)

#         for key in out:
#             for actual, expected in zip(out[key], RESULTS[key]):
#                 np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
