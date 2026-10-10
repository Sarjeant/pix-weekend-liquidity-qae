# rotation.py - how much coherent rotation error per Grover step the method can tolerate,
# and how much the measured runs actually show.
#   measure: hold the breach probability at its true value (from the data), fit an extra
#            rotation delta per step plus the per-step signal kept, and give a 95% range.
#   threshold: the delta at which an uncorrected over-rotation biases the standard estimator
#              by as much as Monte Carlo's own error at the same number of model calls.
import numpy as np
from scipy.optimize import brentq, minimize

import estimate

SCHEDULE = [0, 1, 2, 3, 4, 6, 8, 12]  # planned depth schedule
SHOTS = 2000                          # shots per depth in the planned schedule
ETA = 0.99745                         # per-step signal kept, measured on the IonQ noise model


def model(theta, delta, eta, m):
    """Return P(flag = 1) after m steps with an extra rotation delta per step and signal loss eta."""
    m = np.asarray(m, dtype=float)
    return 0.5 + eta ** m * (np.sin((2 * m + 1) * theta + m * delta) ** 2 - 0.5)


def _profile_nll(theta, delta, m, ones, shots):
    """Return the negative log-likelihood at this delta, minimised over eta."""
    def f(x):
        q = np.clip(model(theta, delta, x[0], m), 1e-9, 1 - 1e-9)
        return -np.sum(ones * np.log(q) + (shots - ones) * np.log(1 - q))
    return float(minimize(f, [0.997], bounds=[(0.9, 1.0)], method="L-BFGS-B").fun)


def measure(rows, p_true):
    """Return the fitted rotation error per step (degrees) and its 95% range from the depth runs."""
    depth = [r for r in rows if r.get("kind") == "depth"]
    m = np.array([r["m"] for r in depth], dtype=float)
    ones = np.array([r["ones"] for r in depth], dtype=float)
    shots = np.array([r["shots"] for r in depth], dtype=float)
    theta = float(np.arcsin(np.sqrt(p_true)))
    grid = np.linspace(-0.03, 0.03, 241)
    nll = np.array([_profile_nll(theta, d, m, ones, shots) for d in grid])
    inside = grid[nll <= nll.min() + 1.92]  # 95% profile-likelihood range
    return {"delta_deg": round(float(np.degrees(grid[nll.argmin()])), 3),
            "delta_95_low_deg": round(float(np.degrees(inside.min())), 3),
            "delta_95_high_deg": round(float(np.degrees(inside.max())), 3)}


def bias_from_rotation(p_true, delta_deg, eta=ETA, schedule=SCHEDULE, shots=SHOTS):
    """Return the bias of the standard estimator when each step over-rotates by delta_deg."""
    theta = float(np.arcsin(np.sqrt(p_true)))
    m = np.array(schedule, dtype=float)
    n = np.full(len(m), float(shots))
    ones = model(theta, np.radians(delta_deg), eta, m) * n
    t, _ = estimate._fit_arrays(m, ones, n, 0.0, 0.0)
    return float(np.sin(t) ** 2 - p_true)


def threshold(p_true, eta=ETA, schedule=SCHEDULE, shots=SHOTS):
    """Return the rotation error per step (degrees) at which bias equals Monte Carlo's error."""
    calls = int(np.sum(shots * (2 * np.array(schedule) + 1)))
    mc = estimate.mc_se(p_true, calls)
    deg = brentq(lambda d: abs(bias_from_rotation(p_true, d, eta, schedule, shots)) - mc, 1e-3, 5.0)
    table = [{"delta_deg": d, "bias": round(bias_from_rotation(p_true, d, eta, schedule, shots), 6)}
             for d in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)]
    return {"threshold_deg": round(float(deg), 3), "model_calls": calls,
            "se_mc_equal_calls": round(mc, 6), "bias_table": table}