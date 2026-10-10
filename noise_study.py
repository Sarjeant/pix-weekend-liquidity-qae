# noise_study.py - repeated-runs study on the Built-in simulator with a stated noise model.
# Each run: the real circuits are sampled on the simulator, then the stated noise channel is
# applied shot by shot (seeded), and the estimator is refit. Five runs give the real spread
# of the estimate, which is compared with the error the estimator claims.
import numpy as np

import circuits
import estimate

# Stated noise model (applied to every shot after the simulator samples the ideal circuit):
#   each Grover step keeps a fraction ETA of the signal; otherwise the flag is a fair coin
#   (depolarising), then readout flips 0->1 with E0 and 1->0 with E1.
ETA = 0.9975  # per-step signal kept, measured on the IonQ Forte Enterprise noise model (99.75%)
E0 = 0.003    # readout 0->1, assumed (typical trapped-ion value)
E1 = 0.005    # readout 1->0, assumed (typical trapped-ion value)
DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12]
RUNS = 5
SHOTS = 200
LABEL = "basic_simulator+noise"


def apply_noise(ones, shots, m, rng, eta=ETA, e0=E0, e1=E1):
    """Return the number of 1s after applying the stated noise channel to `shots` ideal samples."""
    bits = np.zeros(shots, dtype=int)
    bits[:ones] = 1
    rng.shuffle(bits)
    scrambled = rng.random(shots) > eta ** m
    bits[scrambled] = rng.integers(0, 2, scrambled.sum())
    flip0 = (bits == 0) & (rng.random(shots) < e0)
    flip1 = (bits == 1) & (rng.random(shots) < e1)
    bits[flip0], bits[flip1] = 1, 0
    return int(bits.sum())


def one_run(probs, run_and_wait, run, depths=DEPTHS, shots=SHOTS):
    """Return the result rows of one run: calibration pair plus one row per depth."""
    from qiskit import transpile
    rng = np.random.default_rng(1000 + run)
    rows = []
    for prepared, circ in enumerate(circuits.calibration_circuits()):
        ideal = run_and_wait(transpile(circ, basis_gates=circuits.BASIS), shots,
                             seed=1000 * run + 900 + prepared).get("1", 0)
        rows.append({"kind": "cal", "prepared": prepared, "shots": shots,
                     "ones": apply_noise(ideal, shots, 0, rng), "backend": LABEL, "run": run})
    for m in depths:
        qc = transpile(circuits.build_circuit(probs, m), basis_gates=circuits.BASIS)
        ideal = run_and_wait(qc, shots, seed=1000 * run + m).get("1", 0)
        rows.append({"kind": "depth", "m": m, "shots": shots,
                     "ones": apply_noise(ideal, shots, m, rng), "backend": LABEL, "run": run})
    return rows


def study(probs, p_true, run_and_wait, runs=RUNS):
    """Return (per-run estimates, summary) for the repeated-runs study."""
    per_run = []
    for r in range(runs):
        rows = one_run(probs, run_and_wait, r)
        est = estimate.advantage(rows, n_boot=100, seed=r)
        per_run.append({"run": r, "p_hat": est["p_hat"], "se_claimed": est["se_qae"],
                        "eta_hat": est["eta"], "model_calls": est["model_calls"],
                        "ones": [x["ones"] for x in rows if x["kind"] == "depth"]})
    p = np.array([x["p_hat"] for x in per_run])
    claimed = float(np.mean([x["se_claimed"] for x in per_run]))
    spread = float(np.std(p, ddof=1))
    calls = per_run[0]["model_calls"]
    mc = estimate.mc_se(p_true, calls)
    summary = {"p_true": round(p_true, 6), "mean_p_hat": round(float(p.mean()), 6),
               "bias": round(float(p.mean() - p_true), 6), "spread": round(spread, 6),
               "se_claimed_mean": round(claimed, 6), "model_calls_per_run": calls,
               "se_mc_equal_calls": round(mc, 6),
               "empirical_advantage": round(mc / spread, 3) if spread > 0 else None,
               "noise": {"eta": ETA, "e0": E0, "e1": E1}, "runs": runs, "shots": SHOTS,
               "depths": DEPTHS}
    return per_run, summary