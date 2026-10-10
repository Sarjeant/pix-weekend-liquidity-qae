# estimate.py - turns RESULT lines into the breach-probability estimate and the advantage over Monte Carlo.
# Method: maximum-likelihood amplitude estimation (Suzuki et al. 2020) with a per-step signal-loss term
# (depth-limited noise model, cf. Giurgica-Tiron et al. 2022) and readout correction from calibration runs.
import numpy as np
from scipy.optimize import minimize

import results

THETA_GRID = np.linspace(1e-4, np.pi / 2 - 1e-4, 2000)
ETA_GRID = np.linspace(0.6, 1.0, 41)


def load_results(backend=None):
    """Return the list of result entries from results.py, optionally only those from one backend."""
    rows = list(results.RESULTS)
    if backend:
        rows = [r for r in rows if r.get("backend") == backend]
    return rows


def readout_rates(rows):
    """Return (e0, e1): P(read 1 | prepared 0) and P(read 0 | prepared 1); zeros if no calibration."""
    cal = {r["prepared"]: r for r in rows if r.get("kind") == "cal"}
    e0 = cal[0]["ones"] / cal[0]["shots"] if 0 in cal else 0.0
    e1 = 1 - cal[1]["ones"] / cal[1]["shots"] if 1 in cal else 0.0
    return e0, e1


def model(theta, eta, m, e0=0.0, e1=0.0):
    """Return the probability of reading flag = 1 after m steps.

    Ideal: sin^2((2m+1) theta). Each step keeps a fraction eta of the signal and
    the rest relaxes to 1/2. Readout then flips 0->1 with e0 and 1->0 with e1.
    """
    m = np.asarray(m, dtype=float)
    p = 0.5 + (eta ** m) * (np.sin((2 * m + 1) * theta) ** 2 - 0.5)
    return e0 + (1 - e0 - e1) * p


def _nll(params, m, ones, shots, e0, e1):
    """Return the negative binomial log-likelihood of the depth data."""
    theta, eta = params
    p = np.clip(model(theta, eta, m, e0, e1), 1e-9, 1 - 1e-9)
    return -np.sum(ones * np.log(p) + (shots - ones) * np.log(1 - p))


def _fit_arrays(m, ones, shots, e0, e1):
    """Return (theta, eta) maximising the likelihood: grid search, then local refinement."""
    t = THETA_GRID[:, None, None]
    e = ETA_GRID[None, :, None]
    p = np.clip(model(t, e, m[None, None, :], e0, e1), 1e-9, 1 - 1e-9)
    nll = -np.sum(ones * np.log(p) + (shots - ones) * np.log(1 - p), axis=2)
    i, j = np.unravel_index(np.argmin(nll), nll.shape)
    res = minimize(_nll, x0=[THETA_GRID[i], ETA_GRID[j]], args=(m, ones, shots, e0, e1),
                   bounds=[(1e-4, np.pi / 2 - 1e-4), (0.5, 1.0)], method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def fit(rows):
    """Return a dict with the estimate, signal kept per step, readout rates and model calls."""
    depth = [r for r in rows if r.get("kind") == "depth"]
    m = np.array([r["m"] for r in depth], dtype=float)
    ones = np.array([r["ones"] for r in depth], dtype=float)
    shots = np.array([r["shots"] for r in depth], dtype=float)
    e0, e1 = readout_rates(rows)
    theta, eta = _fit_arrays(m, ones, shots, e0, e1)
    calls = int(np.sum(shots * (2 * m + 1)))
    return {"p_hat": float(np.sin(theta) ** 2), "theta": theta, "eta": eta,
            "e0": e0, "e1": e1, "depths": [int(x) for x in m], "model_calls": calls,
            "_arrays": (m, ones, shots)}


def bootstrap_se(fitted, n_boot=200, seed=7):
    """Return the parametric-bootstrap standard error of p_hat (fixed seed)."""
    rng = np.random.default_rng(seed)
    m, _, shots = fitted["_arrays"]
    probs = model(fitted["theta"], fitted["eta"], m, fitted["e0"], fitted["e1"])
    est = []
    for _ in range(n_boot):
        ones = rng.binomial(shots.astype(int), probs)
        t, _eta = _fit_arrays(m, ones, shots, fitted["e0"], fitted["e1"])
        est.append(np.sin(t) ** 2)
    return float(np.std(est, ddof=1))


def mc_se(p, calls):
    """Return the classical Monte Carlo standard error with the same number of model calls."""
    return float(np.sqrt(p * (1 - p) / calls))


def advantage(rows, n_boot=200, seed=7):
    """Return the full estimate dict: p_hat, its standard error, Monte Carlo error, and their ratio."""
    f = fit(rows)
    se = bootstrap_se(f, n_boot, seed)
    mc = mc_se(f["p_hat"], f["model_calls"])
    out = {k: v for k, v in f.items() if not k.startswith("_")}
    out.update({"se_qae": se, "se_mc_equal_calls": mc, "advantage_ratio": mc / se if se > 0 else None})
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}