# baseline.py - the classical comparison: Monte Carlo sampling of the same weekend model,
# and how uncertain the breach probability is given only 86 weekends of data.
import numpy as np

import data

CALL_BUDGETS = [1800, 16000, 160000]  # model calls used by the quantum runs (ideal, repeat study, IonQ)
REPS = 2000
SEED = 11


def monte_carlo_se(probs, calls, reps=REPS, seed=SEED):
    """Return the empirical standard error of a Monte Carlo breach estimate using `calls` samples."""
    rng = np.random.default_rng(seed)
    p_breach = float(probs[2] + probs[3])
    estimates = rng.binomial(calls, p_breach, size=reps) / calls
    return float(np.std(estimates, ddof=1))


def data_bootstrap_se(drawdowns, buffer=0.10, reps=REPS, seed=SEED):
    """Return the bootstrap standard error of the breach probability from resampling the weekends."""
    rng = np.random.default_rng(seed)
    x = np.asarray(drawdowns, dtype=float)
    est = [np.mean(rng.choice(x, size=len(x), replace=True) > buffer) for _ in range(reps)]
    return float(np.std(est, ddof=1))


def baseline_table(buffer=0.10):
    """Return a dict: Monte Carlo error at each call budget, analytic check, and data uncertainty."""
    dd = data.weekend_drawdowns(data.load_balances())["drawdown"].to_numpy()
    probs, p = data.bin_probabilities(dd, buffer)
    rows = []
    for n in CALL_BUDGETS:
        rows.append({"model_calls": n,
                     "se_mc_empirical": round(monte_carlo_se(probs, n), 6),
                     "se_mc_analytic": round(float(np.sqrt(p * (1 - p) / n)), 6)})
    return {"p_breach": round(float(p), 6), "weekends": int(len(dd)), "monte_carlo": rows,
            "data_se_bootstrap": round(data_bootstrap_se(dd, buffer), 4),
            "data_se_analytic": round(float(np.sqrt(p * (1 - p) / len(dd))), 4)}