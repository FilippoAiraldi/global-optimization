# Global Optimization

**global-optimization** is a Python package for **Glob**al **Opt**imization of expensive
black-box functions via Inverse Distance Weighting (IDW) and Radial Basis Function (RBF)
approximation.

> |                   |                                                                 |
> | ----------------- | --------------------------------------------------------------- |
> | **Download**      | <https://pypi.python.org/pypi/globopt/>                         |
> | **Source code**   | <https://github.com/FilippoAiraldi/global-optimization/>        |
> | **Report issues** | <https://github.com/FilippoAiraldi/global-optimization/issues/> |

[![PyPI version](https://badge.fury.io/py/globopt.svg)](https://badge.fury.io/py/globopt)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/global-optimization/blob/botorch/LICENSE)
![Python 3.9](https://img.shields.io/badge/python->=3.9-green.svg)

[![Tests](https://github.com/FilippoAiraldi/global-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/FilippoAiraldi/global-optimization/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/globopt)](https://www.pepy.tech/projects/globopt)
[![Maintainability](https://api.codeclimate.com/v1/badges/6847f2c2c04b20a909fe/maintainability)](https://codeclimate.com/github/FilippoAiraldi/global-optimization/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/6847f2c2c04b20a909fe/test_coverage)](https://codeclimate.com/github/FilippoAiraldi/global-optimization/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Features

**globopt** builds on top of [BoTorch](https://botorch.org/) [[1]](#1), a powerful
framework for Bayesian optimization (that leverages
[PyTorch](https://pytorch.org/) [[2]](#2) and its computational benefits), and extends
it to the more generic field of global optimization via IDW [[3]](#3) and RBF [[4]](#4) deterministic surrogate models. This expansion is achieved in two ways:

1. appropriate surrogate models based on IDW and RBF are implemented to approximate the
   black-box function, which support partially fitting new datapoints without the need
   to retrain the model from scratch on the whole new dataset

2. acquisition functions are designed to guide the search strategy towards the minimum
   (or one of) of the black-box function. These acquisition functions are available both
   as myopic versions, or as nonmyopic formulations that consider future evaluations and
   evolution of the surrogate models to predict the best point to query next.

The repository of this package includes also the source code for the following paper:

```bibtex
@article{airaldi2024nonmyopic,
  title = {Nonmyopic Global Optimisation via Approximate Dynamic Programming},
  year = {2024},
  author = {Filippo Airaldi and Bart De Schutter and Azita Dabiri},
  journal = {arXiv preprint arXiv:2412.04882},
}
```

More information is available in the
[section Paper](https://github.com/FilippoAiraldi/global-optimization/#paper)
below.

---

## Installation

### Using `pip`

You can use `pip` to install **globopt** with the command

```bash
pip install globopt
```

**globopt** has the following dependencies

- Python 3.9 or higher
- [PyTorch](https://pytorch.org/)
- [BoTorch](https://botorch.org/).

### Using source code

If you'd like to play around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/global-optimization.git
```

The main branch is `botorch`, and the other branches contain previous or experimental
versions of the package. You can then install the package to edit it
as you wish as

```bash
pip install -e /path/to/global-optimization
```

---

## Getting started

Here we provide a compact example on how **globopt** can be used to optimize a custom
black-box function. First of all, we need to implement this function as a subclass of
`SyntheticTestFunction`.

```python
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class CustomProblem(SyntheticTestFunction):
    r"""Custom optimization problem:

        f(x) = (1 + x sin(2x) cos(3x) / (1 + x^2))^2 + x^2 / 12 + x / 10

    x is bounded [-3, +3], and f in has a global minimum at `x_opt = -0.959769`
    with `f_opt = 0.2795`.
    """

    dim = 1
    _optimal_value = 0.279504
    _optimizers = [(-0.959769,)]
    _bounds = [(-3.0, +3.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        X2 = X.square()
        return (
            (1 + X * (2 * X).sin() * (3 * X).cos() / (1 + X2)).square()
            + X2 / 12
            + X / 10
        )
```

Then, we can draw some random points to initialize the surrogate model (in this case,
IDW), and define some other constants.

```python
import torch

# instantiate problem and create starting training data
N_ITERS = ...
problem = CustomProblem()
lb, ub = problem._bounds[0]
bounds = torch.as_tensor([[lb], [ub]])
train_X = torch.as_tensor([[-2.62, -1.2, 0.14, 1.1, 2.82]]).T
train_Y = problem(train_X)
c1, c2 = 0.5, 1.0, 0.5
```

Finally, we can loop over the optimization iterations, optimizing the acquisition
function (in this case, the myopic one) and updating the surrogate model with the
newly queried point at each iteration.

```python
from botorch.optim import optimize_acqf
from globopt import IdwAcquisitionFunction, Rbf

# run optimization loop
for iteration in range(N_ITERS):
    # instantiate model and acquisition function
    mdl = Idw(train_X, train_Y, init_state=rbf_state)
    MAF = IdwAcquisitionFunction(mdl, c1, c2)

    # minimize acquisition function
    X_opt, acq_opt = optimize_acqf(MAF, bounds, 1, 8, 16, options={"seed": iteration})

    # evaluate objective function at the new point, and append it to training data
    Y_opt = problem(X_opt)
    train_X = torch.cat((train_X, X_opt))
    train_Y = torch.cat((train_Y, Y_opt))
```

Assuming a sufficiently large number of iterations is carried out, the optimization
process will converge to the global minimum of the black-box function, which can be
retrieved, in theory, as the last queried point `train_Y[-1]`, but for technical
reasons it is more convenient to retrieved the best-so-far `train_Y.min()`.

---

## Examples

Our
[examples](https://github.com/FilippoAiraldi/global-optimization/tree/botorch/examples)
subdirectory contains example applications of this package showing how to build the
IDW and RBF surrogate models, evaluate myopic and nonmyopic acquistion functions, and
use them to optimize custom black-box functions.

---

## Paper

As aforementioned, this package was used as source code of the following paper:

```bibtex
@article{airaldi2024nonmyopic,
  title = {Nonmyopic Global Optimisation via Approximate Dynamic Programming},
  year = {2024},
  author = {Filippo Airaldi and Bart De Schutter and Azita Dabiri},
  journal = {arXiv preprint arXiv:2412.04882},
}
```

Below the details on how to run the experiments and reproduce the results of the paper
are reported. Note that, while the package is available for Python >= 3.9, the results
of the paper, and thus the commands below, are based on Python 3.11.3.

### Synthetic and real problems

To reproduce the results of the paper on the collection of synthetic and real benchmark
problems, first make sure the Python version and the correct packages are installed

```bash
python --version  # 3.11.3 in our case
pip install -r benchmarking/requirements-benchmarking.txt
```

Then, you can run all the experiments (which are a lot) by executing the following
command

```bash
python benchmarking/run.py --methods myopic ms-mc.1 ms-mc.1.1 ms-mc.1.1.1 ms-mc.1.1.1.1 ms-gh.1 ms-gh.1.1 ms-gh.1.1.1 ms-gh.1.1.1.1 ms-mc.10 ms-mc.10.5 ms-gh.10 ms-gh.10.5 --n-jobs={number-of-jobs} --devices {list-of-available-devices} --csv={filename}
```

where `{number-of-jobs}`, `{list-of-available-devices}` and `{filename}` are
placeholders and should be replaced with the desired values. Be aware that this command
will take several days to run, depending on the number of jobs and the devices at your
disposal. However, the results are incrementally saved to the CSV file, so you can stop
and start the script at any time without throwing partial results away. Additionally,
you can also plot the ongoing results. To fetch the status of the simulation, you can
run

```bash
python benchmarking/status.py {filename}
```

which will print a dataframe with the number of trials already completed for each
problem-method pair. Once the results are ready (or partially ready), you can analyze
them by running the `benchmarking/analyze.py` script. Three different modes are
available: `summary`, `plot`, and `pgfplotstables`. To get the results reported in the
paper and simulated by us, run

```bash
python benchmarking/analyze.py benchmarking/results.csv --summary --exclude-methods random ei myopic-s
python benchmarking/analyze.py benchmarking/results.csv --plot --exclude-methods random ei myopic-s
python benchmarking/analyze.py benchmarking/results.csv --pgfplotstables --exclude-methods random ei myopic-s
```

In turn, these command will report a textual summary of the results (of which the table
of gaps is the primary interest), plot the results in a crude way, and generate in the
`pgfplotstables/` folder the `.dat` tables with all the data to be later plotted in
LaTeX with PGFPlots.

### Data-driven tuning of an MPC controller

Similarly to the previous results (since it is based on the same scripts), to reproduce
the results on the second numerical experiment, the tuning of a Model Predictive Control
controller, first install the requirements

```bash
pip install -r mpc-tuning/requirements-mpc-tuning.txt
```

Then, to launch the simulations, run

```bash
python mpc-tuning/tune.py --methods myopic ms-gh.1.1.1 ms-gh.10.5 --n-jobs={number-of-jobs} --devices {list-of-available-devices} --csv={filename} --n-trials=30
```

You can monitor the progress of the simulation with the same `benchmarking/status.py`
script as before. To analyze the results obtained by us, run

```bash
python mpc-tuning/analyze.py mpc-tuning/results.csv {--summary,--plot,--pgfplotstables} --include-methods myopic ms-gh.1.1.1$ ms-gh.10.5
```

---

## License

The repository is provided under the MIT License. See the LICENSE file included with
this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate
[f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/)
> in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2024 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest
in the program “globopt” (Global Optimization) written by the Author(s). Prof. Dr. Ir.
Fred van Keulen, Dean of ME.

---

## References

<a id="1">[1]</a>
Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., Bakshy, E. (2020).
[BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html).
Advances in Neural Information Processing Systems, 33, 21524-21538.

<a id="2">[2]</a>
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S. (2019).
[PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703).
Advances in Neural Information Processing Systems, 33, 21524-21538.

<a id="3">[3]</a>
Joseph, V.R., Kang, L. (2011).
[Regression-based inverse distance weighting with applications to computer experiments](https://www.jstor.org/stable/23210401).
Technometrics 53(3), 254–265.

<a id="4">[4]</a>
McDonald, D.B., Grantham, W.J., Tabor, W.L., Murphy, M.J. (2007).
[Global and local optimization using radial basis function response surface models](https://www.sciencedirect.com/science/article/pii/S0307904X06002009).
Applied Mathematical Modelling 31(10), 2095–2110.
