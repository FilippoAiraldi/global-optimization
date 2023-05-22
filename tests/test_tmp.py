import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import unittest

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from globopt.core.problems import get_benchmark_problem
from globopt.myopic.algorithm import GO, Rbf
from globopt.util.callback import BestSoFarCallback

EXPECTED_RES = np.load("out1_res.npy")
EXPECTED_COEF = np.load("out1_coef.npy")
EXPECTED_MINV = np.load("out1_Minv.npy")


class TestNormalization(unittest.TestCase):
    def test(self) -> None:
        name = "hartman6"
        problem = get_benchmark_problem(name)[0]
        n_var = problem.n_var
        algorithm = GO(
            regression=Rbf(eps=1.0775 / n_var, svd_tol=0),
            init_points=2 * n_var,
            acquisition_min_algorithm=PSO(pop_size=10),
            acquisition_min_kwargs={
                "termination": DefaultSingleObjectiveTermination(
                    ftol=1e-4, n_max_gen=300, period=10
                )
            },
            acquisition_fun_kwargs={"c1": 1.5078 / n_var, "c2": 1.4246 / n_var},
        )
        callback = BestSoFarCallback()
        minimize(
            problem,
            algorithm,
            termination=("n_iter", EXPECTED_RES.size),
            callback=callback,
            copy_algorithm=False,
            verbose=True,
            seed=2088275051,
        )

        ACTUAL_RES = np.array(callback.data["best"])
        ACTUAL_COEF = algorithm.regression.coef_
        ACTUAL_MINV = algorithm.regression.Minv_
        np.testing.assert_allclose(ACTUAL_RES.flatten(), EXPECTED_RES.flatten())
        np.testing.assert_allclose(ACTUAL_COEF.flatten(), EXPECTED_COEF.flatten())
        np.testing.assert_allclose(ACTUAL_MINV.flatten(), EXPECTED_MINV.flatten())


if __name__ == "__main__":
    unittest.main()
