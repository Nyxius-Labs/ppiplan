from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Plan:
    rho_pilot: float
    rho_plan: float
    n_classical: int
    n_human: int
    share_saved: float
    power_if_rho_is: dict


def _demean(x, arm):
    x = np.asarray(x, float).copy()
    if arm is None:
        return x - x.mean(0)
    arm = np.asarray(arm)
    for a in np.unique(arm):
        x[arm == a] -= x[arm == a].mean(0)
    return x


def average_predictor(predictions, arm=None):
    p = np.asarray(predictions, float)
    if p.ndim == 1:
        p = p[:, None]
    p = _demean(p, arm)
    return (p / p.std(0, ddof=1)).mean(1)


def pilot_rho(y, predictions, arm=None):
    a = _demean(average_predictor(predictions, arm), arm)
    yc = _demean(y, arm)
    return float(a @ yc / np.sqrt((a @ a) * (yc @ yc)))


def realized_power(rho_plan, rho_true, alpha=0.05, power=0.80):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    rp = np.clip(rho_plan, -0.999, 0.999)
    return float(norm.cdf(np.sqrt((1 - rp ** 2) / (1 - rho_true ** 2)) * (za + zb) - za))


def plan(y_pilot, predictions_pilot, effect, sd=None, arm=None, alpha=0.05, power=0.80, lower_bound=False):
    y = np.asarray(y_pilot, float)
    r = pilot_rho(y, predictions_pilot, arm)
    rp = np.tanh(np.arctanh(np.clip(r, -0.999, 0.999)) - norm.ppf(power) / np.sqrt(len(y) - 3)) if lower_bound else r
    rp = max(float(rp), 0.0)
    s = float(np.std(_demean(y, arm), ddof=1)) if sd is None else float(sd)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    k = 4 if arm is not None else 1
    n_cl = int(np.ceil(k * ((za + zb) * s / effect) ** 2))
    n_h = int(np.ceil(n_cl * (1 - rp ** 2)))
    grid = {round(t, 2): realized_power(rp, t, alpha, power) for t in (rp - 0.15, rp - 0.10, rp - 0.05, rp) if t > -1}
    return Plan(r, rp, n_cl, n_h, 1 - n_h / n_cl, grid)
