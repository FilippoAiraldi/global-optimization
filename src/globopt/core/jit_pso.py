import numpy as np
from numba import njit
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.individual import Individual
from pymoo.operators.mutation.pm import PM


@njit
def set_to_bounds_if_outside(X, xl, xu):
    xl = np.broadcast_to(xl, X.shape)
    X = np.where(X < xl, xl, X)
    xu = np.broadcast_to(xu, X.shape)
    X = np.where(X > xu, xu, X)
    return X


@njit
def pso_equation(X, P_X, S_X, V, V_max, w, c1, c2):
    n_particles, n_var = X.shape
    r1 = np.random.random((n_particles, n_var))
    r2 = np.random.random((n_particles, n_var))
    inerta = w * V
    cognitive = c1 * r1 * (P_X - X)
    social = c2 * r2 * (S_X - X)
    Vp = inerta + cognitive + social
    Vp = set_to_bounds_if_outside(Vp, -V_max, V_max)
    Xp = X + Vp
    return Xp, Vp


@njit
def repair_random_init(Xp, X, xl, xu):
    n = len(Xp)
    XL = xl.repeat(n).reshape(-1, n).T
    XU = xu.repeat(n).reshape(-1, n).T
    Xp = np.where(Xp < XL, XL + np.random.random(Xp.shape) * (X - XL), Xp)
    Xp = np.where(Xp > XU, XU - np.random.random(Xp.shape) * (XU - X), Xp)
    return Xp


@njit
def correct_infeasible_points(X, xl, xu, P_X, S_X, V, V_max, w, c1, c2):
    Xp, Vp = pso_equation(X, P_X, S_X, V, V_max, w, c1, c2)
    for _ in range(20):
        mask = np.logical_or(X < xl, X > xu)
        if not mask.any():
            break
        Xp_new, Vp_new = pso_equation(X, P_X, S_X, V, V_max, w, c1, c2)
        Xp = np.where(mask, Xp_new, Xp)
        Vp = np.where(mask, Vp_new, Vp)
    Xp = repair_random_init(Xp, X, xl, xu)
    return Xp, Vp


@njit
def get_S(f):
    if f <= 0.4:
        s1 = 0
    elif 0.4 < f <= 0.6:
        s1 = 5 * f - 2
    elif 0.6 < f <= 0.7:
        s1 = 1
    elif 0.7 < f <= 0.8:
        s1 = -10 * f + 8
    else:
        s1 = 0
    #
    if f <= 0.2:
        s2 = 0
    elif 0.2 < f <= 0.3:
        s2 = 10 * f - 2
    elif 0.3 < f <= 0.4:
        s2 = 1
    elif 0.4 < f <= 0.6:
        s2 = -5 * f + 3
    else:
        s2 = 0
    #
    if f <= 0.1:
        s3 = 1
    elif 0.1 < f <= 0.3:
        s3 = -5 * f + 1.5
    else:
        s3 = 0
    #
    if f <= 0.7:
        s4 = 0
    elif 0.7 < f <= 0.9:
        s4 = 5 * f - 3.5
    else:
        s4 = 1
    return np.asarray([s1, s2, s3, s4])


@njit
def norm_eucl_dist(A, B, xl, xu):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(1, *B.shape)
    return np.sqrt((((A - B) / (xu - xl)) ** 2).sum(axis=-1))


@njit
def compute_adaptation(X, xl, xu, S_X, c1, c2, len_pop):
    # get the average distance from one to another for normalization
    D = norm_eucl_dist(X, X, xl, xu)
    mD = D.sum(axis=1) / (len_pop - 1)
    _min, _max = mD.min(), mD.max()

    # get the average distance to the best
    g_D = norm_eucl_dist(S_X, X, xl, xu).mean()
    f = (g_D - _min) / (_max - _min + 1e-32)
    strategy = get_S(f).argmax() + 1

    # aplly the strategy
    delta = 0.05 + (np.random.random() * 0.05)
    if strategy == 1:
        c1 += delta
        c2 -= delta
    elif strategy == 2:
        c1 += 0.5 * delta
        c2 -= 0.5 * delta
    elif strategy == 3:
        c1 += 0.5 * delta
        c2 += 0.5 * delta
    elif strategy == 4:
        c1 -= delta
        c2 += delta
    c1 = max(1.5, min(2.5, c1))
    c2 = max(1.5, min(2.5, c2))
    if c1 + c2 > 4.0:
        c1 = 4.0 * (c1 / (c1 + c2))
        c2 = 4.0 * (c2 / (c1 + c2))
    w = 1 / (1 + 1.5 * np.exp(-2.6 * f))
    return f, strategy, c1, c2, w


class JitPSO(PSO):
    def _infill(self):
        problem, particles, pbest = self.problem, self.particles, self.pop
        (X, V) = particles.get("X", "V")
        P_X = pbest.get("X")
        sbest = self._social_best()
        S_X = sbest.get("X")
        xl, xu = problem.bounds()

        # correct infeasible points
        Xp, Vp = correct_infeasible_points(
            X, xl, xu, P_X, S_X, V, self.V_max, self.w, self.c1, self.c2
        )

        # create the offspring population
        off = Population.new(X=Xp, V=Vp)

        # try to improve the current best with a pertubation
        if self.pertube_best:
            k = FitnessSurvival().do(problem, pbest, n_survive=1, return_indices=True)[
                0
            ]
            mut = PM(prob=0.9, eta=np.random.uniform(5, 30), at_least_once=False)
            mutant = mut(problem, Population(Individual(X=pbest[k].X)))[0]
            off[k].set("X", mutant.X)
        self.repair(problem, off)
        self.sbest = sbest
        return off

    def _adapt(self):
        pop = self.pop
        X = pop.get("X")
        S_X = self.sbest.get("X")
        c1, c2, = self.c1, self.c2
        xl, xu = self.problem.bounds()
        self.f, self.strategy, self.c1, self.c2, self.w = compute_adaptation(
            X, xl, xu, S_X, c1, c2, len(pop)
        )
