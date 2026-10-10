# main.py - conductor. Pick a STAGE, press Run.
# NOTE: 'backend' is pre-created by the platform as a global variable (the QPU chosen in the Run dialog).
import json
import time

from qiskit import transpile
from qiskit.providers.jobstatus import JobStatus

import circuits
import data

STAGE = "judge"   # default for judges. Others: "data", "theory", "circuit", "calibration", "estimate", "repeat", "baseline", "rotation", "charts"
M = 0             # depth for "circuit"; for "calibration", 0 or 1 = which state the flag is prepared in
BUFFER = 0.10     # share of Friday's prefunded liquidity a weekend may consume
BACKEND_FILTER = "ionq_forte_ent_sim"  # which results "estimate" uses; "" = all
BACKEND_LABEL = "ionq_forte_ent_sim"  # tag for IonQ runs; set "" to use the platform's name


def run_and_wait(circ, shots, seed=None):
    """Submit a circuit to the selected backend, poll until done, return counts.

    A seed fixes the simulator's sampling when the backend supports it (Built-in simulator).
    """
    job = None
    if seed is not None:
        try:
            job = backend.run(circ, shots=shots, seed_simulator=seed)
        except Exception:
            job = None
    if job is None:
        job = backend.run(circ, shots=shots)
    while True:
        status = job.status()
        if status == JobStatus.DONE:
            break
        if status == JobStatus.ERROR:
            raise Exception(job.result())
        print(f"Job status is {status}")
        time.sleep(3)
    return job.result().get_counts()


def emit(payload):
    """Print one RESULT line as JSON, ready to paste into results.py."""
    print("RESULT " + json.dumps(payload))


def backend_name():
    """Return the selected backend's name as a string."""
    if BACKEND_LABEL and "ionq" in str(getattr(backend, "name", "")):
        return BACKEND_LABEL
    name = getattr(backend, "name", backend)
    return name() if callable(name) else str(name)


def distribution():
    """Return (four bin probabilities, breach probability) from the BCB data."""
    w = data.weekend_drawdowns(data.load_balances())
    return data.bin_probabilities(w["drawdown"].to_numpy(), BUFFER)


def judge_run(shots=200):
    """Default run for judges: a live circuit check, then every headline result, about a minute."""
    import baseline
    import estimate
    import noise_study
    import rotation
    probs, p = distribution()
    print("== 1. Data: Central Bank of Brazil Conta PI balances ==")
    d = data.summary(BUFFER)
    print(f"{d['weekends']} weekends since January 2025; P(weekend drains > {BUFFER:.0%}) = {d['p_breach']:.3f}")
    print("\n== 2. Live quantum circuit on the selected simulator ==")
    for m in (0, 1, 2, 4):
        qc = transpile(circuits.build_circuit(probs, m), basis_gates=circuits.BASIS)
        counts = run_and_wait(qc, shots, seed=500 + m)
        print(f"m = {m}: flag = 1 in {counts.get('1', 0) / shots:.3f} of {shots} shots "
              f"(theory {circuits.theory(p, m):.3f})")
    print("\n== 3. Repeated-runs study (5 seeded runs, stated noise model) ==")
    _, rep = noise_study.study(probs, p, run_and_wait)
    print(f"mean estimate {rep['mean_p_hat']:.4f} vs true {p:.4f}; spread {rep['spread']:.5f}; "
          f"Monte Carlo at equal calls {rep['se_mc_equal_calls']:.5f}: {rep['empirical_advantage']:.1f}x smaller error")
    print("\n== 4. Saved IonQ Forte Enterprise 1 noise-model runs (results.py) ==")
    ionq = estimate.load_results("ionq_forte_ent_sim")
    est = estimate.advantage(ionq)
    print(f"estimate {est['p_hat']:.4f}; signal kept per Grover step {est['eta']:.4f}; "
          f"{est['se_mc_equal_calls'] / est['se_qae']:.1f}x smaller error than Monte Carlo")
    rot = rotation.measure(ionq, p)
    thr = rotation.threshold(p)
    print(f"coherent rotation error per step: {rot['delta_deg']} deg "
          f"(95% range {rot['delta_95_low_deg']} to {rot['delta_95_high_deg']}); "
          f"threshold for any advantage: {thr['threshold_deg']} deg")
    print("\n== 5. Classical context ==")
    base = baseline.baseline_table(BUFFER)
    print(f"uncertainty from only {base['weekends']} weekends of data: +/-{base['data_se_bootstrap']} "
          "(the advantage is in computing a model's tail probability, not in the data)")
    emit({"stage": "judge", "p_breach": d["p_breach"], "repeat_advantage": rep["empirical_advantage"],
          "ionq_p_hat": est["p_hat"], "ionq_eta": est["eta"], "threshold_deg": thr["threshold_deg"]})


def main(shots: int = 200, flag: bool = True, value: float = 0.05):
    """Entry point called by the platform's Run button."""
    if STAGE == "data":
        emit({"stage": "data", **data.summary(BUFFER)})

    elif STAGE == "theory":
        probs, p = distribution()
        for row in circuits.check_theory(probs, p):
            print(row)
        print("two-qubit gates per Grover step:", circuits.two_qubit_gates_per_step(probs))

    elif STAGE == "circuit":
        probs, p = distribution()
        qc = transpile(circuits.build_circuit(probs, M), basis_gates=circuits.BASIS)
        print(qc.count_ops())
        counts = run_and_wait(qc, shots)
        emit({"kind": "depth", "m": M, "shots": shots, "ones": int(counts.get("1", 0)),
              "counts": counts, "theory": round(circuits.theory(p, M), 6),
              "backend": backend_name(), "time": time.strftime("%Y-%m-%d %H:%M")})

    elif STAGE == "calibration":
        qc = transpile(circuits.calibration_circuits()[M], basis_gates=circuits.BASIS)
        counts = run_and_wait(qc, shots)
        emit({"kind": "cal", "prepared": M, "shots": shots, "ones": int(counts.get("1", 0)),
              "counts": counts, "backend": backend_name(), "time": time.strftime("%Y-%m-%d %H:%M")})

    elif STAGE == "estimate":
        import estimate
        rows = estimate.load_results(BACKEND_FILTER or None)
        emit({"stage": "estimate", "backend": BACKEND_FILTER or "all", **estimate.advantage(rows)})

    elif STAGE == "repeat":
        import noise_study
        probs, p = distribution()
        per_run, summary = noise_study.study(probs, p, run_and_wait)
        for row in per_run:
            emit({"kind": "repeat_run", **row})
        emit({"stage": "repeat", **summary})

    elif STAGE == "baseline":
        import baseline
        emit({"stage": "baseline", **baseline.baseline_table(BUFFER)})

    elif STAGE == "rotation":
        import estimate
        import rotation
        _, p = distribution()
        rows = estimate.load_results(BACKEND_FILTER or None)
        emit({"stage": "rotation", "backend": BACKEND_FILTER or "all",
              **rotation.measure(rows, p), **rotation.threshold(p)})

    elif STAGE == "charts":
        import baseline
        import charts
        import estimate
        import noise_study
        probs, p = distribution()
        ionq = estimate.load_results("ionq_forte_ent_sim")
        ideal = estimate.advantage(estimate.load_results("basic_simulator"))
        on_ionq = estimate.advantage(ionq)
        _, rep = noise_study.study(probs, p, run_and_wait)
        points = [("Ideal simulator", ideal["model_calls"], ideal["se_qae"]),
                  ("Repeated runs, stated noise", rep["model_calls_per_run"], rep["spread"]),
                  ("IonQ noise model", on_ionq["model_calls"], on_ionq["se_qae"])]
        base = baseline.baseline_table(BUFFER)
        for path in (charts.chart1_balances(), charts.chart2_drawdowns(BUFFER),
                     charts.chart3_amplification(ionq, p),
                     charts.chart4_precision(points, p, base["data_se_bootstrap"])):
            print("saved", path)
        emit({"stage": "charts", "record_weekend_drawdown": round(charts.record_weekend_drawdown(), 4)})

    elif STAGE == "judge":
        judge_run(shots)

    else:
        print(f"STAGE '{STAGE}' not written yet")