import numpy as np

from ppiplan import pilot_rho, plan, realized_power

rng = np.random.default_rng(1)
N, K = 20000, 11
y = rng.normal(size=N)
common = rng.normal(size=N)
P = 0.35 * y[:, None] + 0.6 * common[:, None] + rng.normal(size=(N, K))
true_avg = pilot_rho(y, P)
true_single = np.array([np.corrcoef(P[:, k], y)[0, 1] for k in range(K)])

avg, pick = [], []
for _ in range(2000):
    i = rng.choice(N, 50, replace=False)
    avg.append(realized_power(plan(y[i], P[i], effect=0.2).rho_plan, true_avg))
    r = np.array([np.corrcoef(P[i, k], y[i])[0, 1] for k in range(K)])
    k = r.argmax()
    pick.append(realized_power(max(r[k], 0), true_single[k]))

print(f"average of fixed configurations: realized power {np.mean(avg):.3f}")
print(f"best configuration on the pilot: realized power {np.mean(pick):.3f}")
