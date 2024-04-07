import argparse
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from itertools import cycle, product
from math import prod
from pathlib import Path
from typing import Any, Optional
from warnings import filterwarnings

import casadi as cs
import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from csnlp.multistart import (
    ParallelMultistartNlp,
    RandomStartPoint,
    RandomStartPoints,
    StructuredStartPoint,
    StructuredStartPoints,
)
from csnlp.wrappers import Mpc
from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
from joblib import Parallel, delayed
from mpcrl import Agent, MpcSolverWarning, WarmStartStrategy
from mpcrl.wrappers.envs import MonitorEpisodes
from numpy import typing as npt
from torch import Tensor

sys.path.append(os.path.join(os.getcwd(), "benchmarking"))

# I am lazy so let's import all the helpful functions defined in benchmarking/run.py
# instead of coding them again here
from run import check_methods_arg, fnv1a_64, lock_write, run_problem
from status import filter_tasks_by_status

PROBLEM = "cstr-mpc-tuning"
MAX_ITER = 40
REGRESSION_TYPE = "rbf"
TUNABLE_PARS = ("narx_weights", "backoff")


class CstrEnv(Env[npt.NDArray[np.floating], float]):
    """
    ## Description

    Continuously stirred tank reactor environment. The ongoing reaction is
                    A -> B -> C, 2A -> D.

    ## Action Space

    The action is an array of shape `(1,)`, where the action is the normalized inflow
    rate of the tank. It is bounded in the range `[5, 35]`.

    ## Observation Space

    The state space is an array of shape `(4,)`, containing the concentrations of
    reagents A and B (mol/L), and the temperatures of the reactor and the coolant (°C).
    The first two states must be positive, while the latter two (the temperatures)
    should be bounded (but not forced) below `100` and `150`, respectively.

    The observation (a.k.a., measurable states) space is an array of shape `(2,)`,
    containing the concentration of reagent B and the temperature of the reactor. The
    former is unconstrained, while the latter should be bounded (but not forced) in the
    range `[100, 150]` (See Rewards section).

    The internal, non-observable states are the concentrations of reagent A and B, and
    the temperatures of the reactor and the coolant, for a total of 4 states.

    ## Rewards

    The reward to be maximized here is to be intended as the number of moles of
    production of component B. However, penalties are incurred for violating bounds on
    the temperature of the reactor.

    ## Starting State

    The initial state is set to `[1, 1, 100, 100]`

    ## Episode End

    The episode does not have an end, so wrapping it in, e.g., `TimeLimit`, is strongly
    suggested.

    References
    ----------
    [1] Sorourifar, F., Makrygirgos, G., Mesbah, A. and Paulson, J.A., 2021. A
        data-driven automatic tuning method for MPC under uncertainty using constrained
        Bayesian optimization. IFAC-PapersOnLine, 54(3), pp.243-250.
    """

    ns = 4  # number of states
    na = 1  # number of inputs
    reactor_temperature_bound = (100, 150)
    inflow_bound = (5, 35)
    x0 = np.asarray([1.0, 1.0, 100.0, 100.0])  # initial state

    def __init__(self, constraint_violation_penalty: float) -> None:
        """Creates a CSTR environment.

        Parameters
        ----------
        constraint_violation_penalty : float
            Reward penalty for violating soft constraints on the reactor temperature.
        """
        super().__init__()
        self.constraint_violation_penalty = constraint_violation_penalty
        self.observation_space = Box(
            np.array([0.0, 0.0, -273.15, -273.15]), np.inf, (self.ns,), np.float64
        )
        self.action_space = Box(*self.inflow_bound, (self.na,), np.float64)

        # set the nonlinear dynamics parameters (see [1, Table 1] for these values)
        k01 = k02 = (1.287, 12)
        k03 = (9.043, 9)
        EA1R = EA2R = 9758.3
        EA3R = 7704.0
        DHAB = 4.2
        DHBC = 4.2
        DHAD = 4.2
        rho = 0.9342
        cP = 3.01
        cPK = 2.0
        A = 0.215
        self.VR = VR = 10.01
        mK = 5.0
        Tin = 130.0
        kW = 4032
        QK = -4500

        # instantiate states and control action
        x = cs.SX.sym("x", self.ns)
        cA, cB, TR, TK = cs.vertsplit_n(x, self.ns)
        F = cs.SX.sym("u", self.na)

        # define the states' PDEs
        k1 = k01[0] * cs.exp(k01[1] * np.log(10) - EA1R / (TR + 273.15))
        k2 = k02[0] * cs.exp(k02[1] * np.log(10) - EA2R / (TR + 273.15))
        k3 = k03[0] * cs.exp(k03[1] * np.log(10) - EA3R / (TR + 273.15))
        cA_dot = F * (self.x0[0] - cA) - k1 * cA - k3 * cA**2
        cB_dot = -F * cB + k1 * cA - k2 * cB
        TR_dot = (
            F * (Tin - TR)
            + kW * A / (rho * cP * VR) * (TK - TR)
            - (k1 * cA * DHAB + k2 * cB * DHBC + k3 * cA**2 * DHAD) / (rho * cP)
        )
        TK_dot = (QK + kW * A * (TR - TK)) / (mK * cPK)
        x_dot = cs.vertcat(cA_dot, cB_dot, TR_dot, TK_dot)

        # define the reward function, i.e., moles of B + constraint penalties
        lb_TR, ub_TR = self.reactor_temperature_bound
        reward = VR * F * cB - constraint_violation_penalty * (
            cs.fmax(0, lb_TR - TR) + cs.fmax(0, TR - ub_TR)
        )

        # build the casadi integrator
        dae = {"x": x, "p": F, "ode": cs.cse(cs.simplify(x_dot)), "quad": reward}
        self.tf = 0.2 / 40  # 0.2 hours / 40 steps
        self.dynamics = cs.integrator("cstr_dynamics", "cvodes", dae, 0.0, self.tf)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the CSTR env."""
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self._state = self.x0
        assert self.observation_space.contains(
            self._state
        ), f"invalid reset state {self._state}"
        return self._state.copy(), {}

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the CSTR env."""
        action = np.reshape(action, self.action_space.shape)
        assert self.action_space.contains(action), f"invalid step action {action}"
        integration = self.dynamics(x0=self._state, p=action)
        self._state = np.asarray(integration["xf"].elements())
        assert self.observation_space.contains(
            self._state
        ), f"invalid step next state {self._state}"
        return self._state.copy(), float(integration["qf"]), False, False, {}


class NoisyFilterObservation(ObservationWrapper):
    """Wrapper for filtering the env's (internal) states to the subset of measurable
    ones. Moreover, it can corrupt the measurements with additive zero-mean gaussian
    noise."""

    def __init__(
        self,
        env: Env[npt.NDArray[np.floating], float],
        measurable_states: Iterable[int],
        measurement_noise_std: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Instantiates the wrapper.

        Parameters
        ----------
        env : gymnasium Env
            The env to wrap.
        measurable_states : iterable of int
            The indices of the states that are measurables.
        measurement_noise_std : array-like, optional
            The standard deviation of the measurement noise to be applied to the
            measurements. If specified, must have the same length as the indices. If
            `None`, no noise is applied.
        """
        assert isinstance(env.observation_space, Box), "only Box spaces are supported."
        super().__init__(env)
        self.measurable_states = list(map(int, measurable_states))
        self.measurement_noise_std = measurement_noise_std
        low = env.observation_space.low[self.measurable_states]
        high = env.observation_space.high[self.measurable_states]
        self.observation_space = Box(low, high, low.shape, env.observation_space.dtype)

    def observation(
        self, observation: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        measurable = observation[self.measurable_states]
        if self.measurement_noise_std is not None:
            obs_space: Box = self.observation_space
            noise = self.np_random.normal(scale=self.measurement_noise_std)
            measurable = np.clip(measurable + noise, obs_space.low, obs_space.high)
        assert self.observation_space.contains(measurable), "Invalid measurable state."
        return measurable


def get_cstr_mpc(
    env: CstrEnv, horizon: int, multistarts: int, n_jobs: int
) -> Mpc[cs.SX]:
    """Returns an MPC controller for the given CSTR env."""
    nlp = ParallelMultistartNlp[cs.SX]("SX", starts=multistarts, n_jobs=n_jobs)
    mpc = Mpc[cs.SX](nlp, horizon)

    # variables (state, action)
    y_space, u_space = env.observation_space, env.action_space
    ny, nu = y_space.shape[0], u_space.shape[0]
    y, _ = mpc.state(
        "y",
        ny,
        lb=y_space.low[:, None],
        ub=[[1e2], [1e3]],  # just some high numbers to bound the state domain
        bound_initial=False,
    )
    u, _ = mpc.action("u", nu, lb=u_space.low[:, None], ub=u_space.high[:, None])

    # set the dynamics based on the NARX model - but first scale approximately to [0, 1]
    lb = np.concatenate([[0.0, 100.0], u_space.low])
    ub = np.concatenate([[10.0, 150.0], u_space.high])
    n_weights = 1 + 2 * (ny + nu)
    narx_weights = (
        mpc.parameter(TUNABLE_PARS[0], (n_weights * ny, 1)).reshape((-1, ny)).T
    )

    def narx_dynamics(y: cs.SX, u: cs.SX) -> cs.SX:
        yu = cs.vertcat(y, u)
        yu_scaled = (yu - lb) / (ub - lb)
        basis = cs.vertcat(cs.SX.ones(1), yu_scaled, yu_scaled**2)
        y_next_scaled = cs.mtimes(narx_weights, basis)
        y_next = y_next_scaled * (ub[:ny] - lb[:ny]) + lb[:ny]
        return cs.cse(cs.simplify(y_next))

    mpc.set_dynamics(narx_dynamics, n_in=2, n_out=1)

    # add constraints on the reactor temperature (soft and with backoff)
    b = mpc.parameter(TUNABLE_PARS[1])
    _, _, slack_lb = mpc.constraint("TR_lb", y[1, :], ">=", 100.0 + b, soft=True)
    _, _, slack_ub = mpc.constraint("TR_ub", y[1, :], "<=", 150.0 - b, soft=True)

    # objective  is the production of moles of B with penalties for violations
    VR = env.get_wrapper_attr("VR")
    cv = env.get_wrapper_attr("constraint_violation_penalty")
    tf = env.get_wrapper_attr("tf")
    moles_B = VR * cs.sum2(u * y[0, :-1])
    constr_viol = cv * cs.sum2(slack_lb + slack_ub)
    mpc.minimize((constr_viol - moles_B) * tf)

    # solver
    opts = {
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "calc_lam_x": False,
        "calc_lam_p": False,
        "calc_multipliers": False,
        "ipopt": {"max_iter": 1000, "sb": "yes", "print_level": 0},
    }
    mpc.init_solver(opts, solver="ipopt")
    return mpc


class CstrMpcControllerTuning(SyntheticTestFunction):
    def __init__(
        self,
        negate: bool = True,  # because the inner env returns rewards instead of costs
        bounds_dict: Optional[dict[str, tuple[float, float]]] = None,
        env_constraint_violation_penalty: float = 4e3,
        env_measurement_noise_std: Optional[npt.ArrayLike] = (0.2, 10.0),
        mpc_horizon: int = 10,
        mpc_multistarts: int = 10,
        mpc_n_jobs: int = 1,
        mc_repeats: int = 1,  # a.k.a., M in [1]
        seed: Optional[int] = None,
    ) -> None:
        if bounds_dict is None:
            bounds_dict = {"narx_weights": (-2, 2), "backoff": (0, 10)}
        assert all(n in bounds_dict for n in TUNABLE_PARS), "missing bounds"

        # create the env
        measurable_states = [1, 2]
        env = CstrEnv(env_constraint_violation_penalty)
        env = NoisyFilterObservation(
            MonitorEpisodes(TimeLimit(env, max_episode_steps=40)),
            measurable_states=measurable_states,
            measurement_noise_std=env_measurement_noise_std,
        )

        # create the mpc and the dict of adjustable parameters - the initial values we
        # give here do not really matter
        mpc = get_cstr_mpc(env, mpc_horizon, mpc_multistarts, mpc_n_jobs)
        pars = {n: np.empty(mpc.parameters[n].shape) for n in bounds_dict}

        # since the MPC is highly nonlinear due to the NARX model, set up a warmstart
        # strategy in order to automatically try different initial conditions at each
        # call
        Y = mpc.variables["y"].shape
        U = mpc.variables["u"].shape
        act_space = env.action_space
        self._mpc_multistarts_struct = (mpc_multistarts - 1) // 2
        self._mpc_multistarts_rand = mpc_multistarts - 1 - self._mpc_multistarts_struct
        warmstart = WarmStartStrategy(
            warmstart="last-successful",
            structured_points=StructuredStartPoints(
                {
                    "y": StructuredStartPoint(
                        np.full(Y, [[0.0], [50.0]]), np.full(Y, [[20.0], [150.0]])
                    ),
                    "u": StructuredStartPoint(
                        np.full(U, act_space.low), np.full(U, act_space.high)
                    ),
                },
                multistarts=self._mpc_multistarts_struct,
            ),
            random_points=RandomStartPoints(
                {
                    "y": RandomStartPoint("normal", scale=[[1.0], [20.0]], size=Y),
                    "u": RandomStartPoint("normal", scale=5.0, size=U),
                },
                multistarts=self._mpc_multistarts_rand,
                biases={
                    "y": CstrEnv.x0[measurable_states].reshape(-1, 1),
                    "u": sum(CstrEnv.inflow_bound) / 2,
                },
            ),
        )

        # finally create an MPC agent
        agent = Agent[cs.SX](mpc, pars, warmstart)
        self._np_random = np.random.default_rng(seed)
        agent.reset(self._np_random)

        # set dim and bounds - required by the parent class
        bounds: list[tuple[float, float]] = []
        self._sizes: list[int] = []
        for par in TUNABLE_PARS:
            par_bounds = bounds_dict[par]
            par_size = prod(mpc.parameters[par].shape)
            bounds.extend(par_bounds for _ in range(par_size))
            self._sizes.append(par_size)
        self.dim = sum(self._sizes)

        # instantiate the parent class
        super().__init__(None, negate, bounds)
        self._env = env
        self._agent = agent
        self._mc_repeats = mc_repeats

    def evaluate_true(self, X: Tensor) -> Tensor:
        agent = self._agent
        env = self._env
        repeats = self._mc_repeats

        # evaluate the MPC for each batch element
        batch = X.shape[0]
        J = torch.empty(batch, dtype=X.dtype, device=X.device)
        for b in range(batch):
            # convert X[b] to a dictionary of numpy arrays and use it to update the
            # agent's parameters
            pars = {
                n: val.detach().clone().cpu().numpy().astype(np.float64)
                for n, val in zip(TUNABLE_PARS, X[b].split_with_sizes(self._sizes))
            }
            agent.fixed_parameters.update(pars)

            # evaluate the MPC with the current parameters for `repeats` times
            evals = agent.evaluate(env, repeats, seed=self._np_random, raises=False)
            J[b] = evals.mean().item()
        return J


def callback(problem: CstrMpcControllerTuning) -> str:
    """A callback that gets called at the end of the optimization for saving additional
    custom information to the csv."""
    env: MonitorEpisodes = problem._env.env
    states = ",".join(map(str, np.asarray(env.observations).flat))
    actions = ",".join(map(str, np.asarray(env.actions).flat))
    rewards = ",".join(map(str, np.asarray(env.rewards).flat))
    return f"{states};{actions};{rewards}"


def run_benchmark(method: str, seed: int, csv: str, device: str) -> None:
    """See `benchmarking/run.py/run_benchmark`."""
    filterwarnings("ignore", "Optimization failed", RuntimeWarning, "botorch")
    filterwarnings("ignore", "Mpc failure", MpcSolverWarning, "mpcrl")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    problem = CstrMpcControllerTuning(seed=seed)
    run_problem(
        problem_name=PROBLEM,
        problem=problem,
        regression_type=REGRESSION_TYPE,
        method=method,
        maxiter=MAX_ITER,
        seed=seed,
        csv=csv,
        device=device,
        n_init=2,
        callback=callback,
    )


def run_benchmarks(
    methods: Iterable[str],
    n_trials: int,
    seed: int,
    n_jobs: int,
    csv: str,
    devices: list[torch.device],
) -> None:
    """See `benchmarking/run.py/run_benchmarks`."""
    seeds = np.random.SeedSequence(fnv1a_64(PROBLEM, seed)).generate_state(n_trials)
    tasks = filter_tasks_by_status(product(range(n_trials), (PROBLEM,), methods), csv)
    Parallel(n_jobs=n_jobs, verbose=100, backend="loky")(
        delayed(run_benchmark)(method, seeds[trial], csv, device)
        for (trial, _, method), device in zip(tasks, cycle(devices))
    )


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="Benchmarking of Global Optimization strategies on synthetic "
        "benchmark problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Benchmarking options")
    group.add_argument(
        "--methods",
        type=check_methods_arg,
        nargs="+",
        help="Methods to run the benchmarking on. Greedy algorithms include `ei` and "
        " `myopic`. Non-myopic multi-step algorithms have the following semantic: "
        "`ms-sampler.m1.m2. ...`, where `ms` stands for multi-step, `sampler` is either"
        "`gh` or `mc` (for Gauss Hermite and Monte Carlo, respectively), while `m1`, "
        "`m2` and so on are the number of fantasies at each stage. The overall horizon "
        "of an `ms` method is the number of fantasies plus one.",
        required=True,
    )
    group.add_argument(
        "--n-trials", type=int, default=30, help="Number  of trials to run per problem."
    )
    group = parser.add_argument_group("Simulation options")
    group.add_argument(
        "--n-jobs", type=int, default=2, help="Number (positive) of parallel processes."
    )
    group.add_argument("--seed", type=int, default=0, help="RNG seed.")
    group.add_argument("--csv", type=str, default="", help="Output csv filename.")
    group.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu"],
        help="List of torch devices to use, e.g., `cpu`, `cuda:0`, etc..",
    )
    args = parser.parse_args()

    # if the output csv is not specified, create it, and write header if anew
    if args.csv is None or args.csv == "":
        args.csv = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    elif not args.csv.endswith(".csv"):
        args.csv += ".csv"
    if not Path(args.csv).is_file():
        lock_write(
            args.csv,
            "problem;method;stage-reward;best-so-far;time;states;actions;rewards",
        )

    # run the benchmarks
    run_benchmarks(
        args.methods,
        args.n_trials,
        args.seed,
        args.n_jobs,
        args.csv,
        args.devices,
    )