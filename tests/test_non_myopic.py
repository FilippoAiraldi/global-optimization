import pickle
import unittest

import numpy as np
from joblib import Parallel
from pymoo.optimize import minimize

from globopt.core.problems import Simple1DProblem
from globopt.core.regression import Idw, Rbf, fit, predict
from globopt.nonmyopic.acquisition import acquisition
from globopt.nonmyopic.algorithm import NonMyopicGO

with open(r"tests/data_test_non_myopic.pkl", "rb") as f:
    RESULTS = pickle.load(f)


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

        with Parallel(n_jobs=1, batch_size=8, verbose=0) as parallel:
            a = acquisition(x, mdl, h, discount, c1, c2, None, -3, +3, parallel, seed)

        np.testing.assert_allclose(a, RESULTS["acquisition"], atol=1e-4, rtol=1e-4)


class TestAlgorithm(unittest.TestCase):
    def test_example1__returns_correct_result(self):
        seed = 17
        problem = Simple1DProblem()
        x0 = [-2.62, -1.2, 0.14, 1.1, 2.82]
        algorithm = NonMyopicGO(
            regression=Rbf("thinplatespline", 0.01),
            init_points=x0,
            c1=1,
            c2=0.5,
            horizon=2,
            discount=1.0,
        )
        res = minimize(
            problem,
            algorithm,
            termination=("n_iter", 4),
            seed=1,
            save_history=True,
            copy_algorithm=False,
        )

        x = np.linspace(*problem.bounds(), 300).reshape(-1, 1)
        out: dict[int, tuple[np.ndarray, ...]] = {}
        with Parallel(n_jobs=-1, batch_size=8) as parallel:
            for i, algo in enumerate(res.history, start=1):
                y_hat = predict(algo.regression, x)
                Xm = algo.pop.get("X")
                ym = algo.pop.get("F").reshape(-1)
                a = acquisition(
                    x,
                    algo.regression,
                    algo.horizon,
                    algo.discount,
                    algo.c1,
                    algo.c2,
                    None,
                    problem.xl,
                    problem.xu,
                    parallel,
                    seed,
                )
                acq_min = (
                    algo.acquisition_min_res.X
                    if hasattr(algo, "acquisition_min_res")
                    else np.nan
                )
                out[i] = (y_hat, Xm, ym, a, acq_min)

        for key in out:
            for actual, expected in zip(out[key], RESULTS[key]):
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
