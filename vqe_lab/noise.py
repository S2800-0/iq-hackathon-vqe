"""Shot-based energy estimation on an ideal or noisy simulator, with readout mitigation.

Each Pauli term is measured with its own circuit (fine for a handful of terms).
Noise comes from a fake IBM backend's calibration data via Aer. Readout
mitigation inverts per-qubit assignment matrices measured on the same qubits.
"""

from dataclasses import dataclass
from functools import reduce

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


@dataclass
class _Data:
    evs: float


@dataclass
class _PubResult:
    data: _Data


class _Job:
    def __init__(self, results):
        self._results = results

    def result(self):
        return self._results


def _simulator(noisy: bool, backend):
    return AerSimulator.from_backend(backend) if noisy else AerSimulator()


def _run_counts(circuits, sim, shots, seed, layout):
    tc = transpile(circuits, sim, initial_layout=layout, optimization_level=1, seed_transpiler=0)
    result = sim.run(tc, shots=shots, seed_simulator=seed).result()
    return [result.get_counts(i) for i in range(len(circuits))]


def _probabilities(counts, num_qubits):
    p = np.zeros(2**num_qubits)
    for key, n in counts.items():
        p[int(key.replace(" ", ""), 2)] += n
    return p / p.sum()


def postselect(p: np.ndarray, forbidden) -> np.ndarray:
    """Zero the forbidden basis states and renormalize (symmetry verification)."""
    p = np.array(p, dtype=float)
    p[list(forbidden)] = 0.0
    return p / p.sum()


def assignment_matrices(num_qubits: int, shots: int = 8000, seed: int = 0, backend=None):
    """Per-qubit matrices A[measured, prepared], from all-0 and all-1 calibration runs."""
    backend = backend or FakeManilaV2()
    sim = _simulator(True, backend)
    zeros = QuantumCircuit(num_qubits)
    zeros.measure_all()
    ones = QuantumCircuit(num_qubits)
    ones.x(range(num_qubits))
    ones.measure_all()
    c0, c1 = _run_counts([zeros, ones], sim, shots, seed, list(range(num_qubits)))
    mats = []
    for q in range(num_qubits):
        p1_given0 = sum(n for k, n in c0.items() if k[::-1][q] == "1") / shots
        p0_given1 = sum(n for k, n in c1.items() if k[::-1][q] == "0") / shots
        mats.append(np.array([[1 - p1_given0, p0_given1], [p1_given0, 1 - p0_given1]]))
    return mats


class SamplingEstimator:
    """Estimator-like object usable with ``vqe_lab.vqe.energy`` and ``run_vqe``."""

    def __init__(self, noisy: bool = True, shots: int = 4000, seed: int | None = None,
                 mitigate_readout: bool = False, backend=None, forbidden_states=None):
        if mitigate_readout and not noisy:
            raise ValueError("readout mitigation only makes sense with noisy=True")
        self.noisy, self.shots, self.mitigate = noisy, shots, mitigate_readout
        # Basis states the exact ground state never occupies; dropped from all-Z measurements.
        self.forbidden = list(forbidden_states or [])
        self.backend = backend or FakeManilaV2()
        self.sim = _simulator(noisy, self.backend)
        self._seed = seed
        self._calls = 0
        self._inverse = None

    def _next_seed(self):
        self._calls += 1
        return None if self._seed is None else self._seed + self._calls

    def _readout_inverse(self, num_qubits):
        if self._inverse is None:
            mats = assignment_matrices(num_qubits, self.shots, self._seed or 0, self.backend)
            # Kron order A_{n-1} x ... x A_0 matches Qiskit's little-endian bitstring index.
            self._inverse = np.linalg.inv(reduce(np.kron, mats[::-1]))
        return self._inverse

    def _expectation(self, circuit: QuantumCircuit, operator: SparsePauliOp) -> float:
        n = operator.num_qubits
        total, circuits, terms = 0.0, [], []
        for pauli, coeff in zip(operator.paulis, operator.coeffs):
            mask = sum(1 << q for q in range(n) if pauli.x[q] or pauli.z[q])
            if mask == 0:
                total += coeff.real
                continue
            qc = circuit.copy()
            for q in range(n):
                if pauli.x[q] and pauli.z[q]:
                    qc.sdg(q)
                if pauli.x[q]:
                    qc.h(q)
            qc.measure_all()
            circuits.append(qc)
            terms.append((mask, coeff.real, not pauli.x.any()))
        counts = _run_counts(circuits, self.sim, self.shots, self._next_seed(), list(range(n)))
        for (mask, coeff, z_only), c in zip(terms, counts):
            p = _probabilities(c, n)
            if self.mitigate:
                p = self._readout_inverse(n) @ p
            if self.forbidden and z_only:
                p = postselect(p, self.forbidden)
            signs = np.array([1 - 2 * (bin(i & mask).count("1") % 2) for i in range(2**n)])
            total += coeff * float(signs @ p)
        return total

    def run(self, pubs):
        results = []
        for circuit, operator, params in pubs:
            bound = circuit.assign_parameters(np.asarray(params, dtype=float))
            results.append(_PubResult(_Data(self._expectation(bound, operator))))
        return _Job(results)
