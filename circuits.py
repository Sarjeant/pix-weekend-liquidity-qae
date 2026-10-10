# circuits.py - 2-qubit amplitude-estimation circuits for the weekend breach probability.
# Bin index = q0 + 2*q1, so q1 (the top qubit) is 1 exactly when the weekend breaches the buffer.
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, grover_operator
from qiskit.quantum_info import Statevector

FLAG = 1                          # index of the breach-flag qubit
BASIS = ["rx", "ry", "rz", "cx"]  # gate set every circuit is transpiled to


def state_prep(probs):
    """Return A: a 2-qubit circuit whose measurement probabilities are the four bin probabilities."""
    amps = np.sqrt(np.asarray(probs, dtype=float))
    amps = amps / np.linalg.norm(amps)
    a = QuantumCircuit(2, name="A")
    a.append(StatePreparation(amps), [0, 1])
    return transpile(a, basis_gates=BASIS, optimization_level=1)


def oracle():
    """Return the oracle: a phase flip on the breach flag (Z on q1)."""
    o = QuantumCircuit(2, name="oracle")
    o.z(FLAG)
    return o


def build_circuit(probs, m, measure=True):
    """Return A followed by m Grover steps, barriers between steps, flag qubit measured."""
    a = state_prep(probs)
    q = transpile(grover_operator(oracle(), state_preparation=a),
                  basis_gates=BASIS, optimization_level=1)
    qc = QuantumCircuit(2, 1) if measure else QuantumCircuit(2)
    qc.compose(a, inplace=True)
    for _ in range(m):
        qc.barrier()
        qc.compose(q, inplace=True)
    if measure:
        qc.barrier()
        qc.measure(FLAG, 0)
    return qc


def calibration_circuits():
    """Return two 1-bit circuits that prepare the flag in 0 and in 1, for readout error rates."""
    c0 = QuantumCircuit(2, 1, name="cal0")
    c0.measure(FLAG, 0)
    c1 = QuantumCircuit(2, 1, name="cal1")
    c1.x(FLAG)
    c1.measure(FLAG, 0)
    return [c0, c1]


def flag_probability(probs, m):
    """Return the exact probability of measuring flag = 1 after m steps (Statevector)."""
    sv = Statevector(build_circuit(probs, m, measure=False))
    return float(sv.probabilities([FLAG])[1])


def theory(p_breach, m):
    """Return sin^2((2m+1) theta) with sin^2(theta) = p_breach."""
    theta = np.arcsin(np.sqrt(p_breach))
    return float(np.sin((2 * m + 1) * theta) ** 2)


def two_qubit_gates_per_step(probs):
    """Return the number of CX gates in one transpiled Grover step."""
    a = state_prep(probs)
    q = transpile(grover_operator(oracle(), state_preparation=a),
                  basis_gates=BASIS, optimization_level=1)
    return int(q.count_ops().get("cx", 0))


def check_theory(probs, p_breach, depths=(0, 1, 2, 3)):
    """Return a list of {m, statevector, theory, diff} rows comparing circuit and formula."""
    rows = []
    for m in depths:
        sv, th = flag_probability(probs, m), theory(p_breach, m)
        rows.append({"m": m, "statevector": round(sv, 6), "theory": round(th, 6),
                     "diff": round(abs(sv - th), 8)})
    return rows