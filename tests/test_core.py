import numpy as np

from ppiplan import plan, realized_power


def test_exact_plan_gives_target_power():
    assert abs(realized_power(0.4, 0.4) - 0.80) < 1e-9


def test_plan_saves_humans():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    P = 0.6 * y[:, None] + rng.normal(size=(200, 5))
    p = plan(y, P, effect=0.2)
    assert 0 < p.n_human < p.n_classical
    assert 0 < p.share_saved < 1


def test_arms_quadruple_classical_sample():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    P = 0.6 * y[:, None] + rng.normal(size=(200, 5))
    arm = np.repeat([0, 1], 100)
    assert abs(plan(y, P, effect=0.2, sd=1.0, arm=arm).n_classical - 4 * plan(y, P, effect=0.2, sd=1.0).n_classical) <= 4
