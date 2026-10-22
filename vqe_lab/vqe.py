"""Minimal VQE loop: an ansatz, an Estimator, and a SciPy optimizer."""

from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

OPTIMIZERS = ("COBYLA", "SLSQP", "L-BFGS-B", "Nelder-Mead")


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    nfev: int
    optimizer: str
    history: list[float] = field(default_factory=list)


def prototype_ansatz() -> QuantumCircuit:
    """One-parameter H2 ansatz from the training notebook; theta=0 is Hartree-Fock."""
    theta = Parameter("θ")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.x(0)
    return qc


def efficient_ansatz(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    """General hardware-efficient ansatz (RY/RZ layers with linear CNOTs)."""
    return efficient_su2(num_qubits, reps=reps, entanglement="linear")


def energy(operator: SparsePauliOp, ansatz: QuantumCircuit, params, estimator=None) -> float:
    estimator = estimator or StatevectorEstimator()
    job = estimator.run([(ansatz, operator, np.asarray(params, dtype=float))])
    return float(job.result()[0].data.evs)


def run_vqe(
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer: str = "COBYLA",
    initial_point=None,
    estimator=None,
    seed: int | None = None,
    maxiter: int = 200,
) -> VQEResult:
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
    if initial_point is None:
        rng = np.random.default_rng(seed)
        initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

    history: list[float] = []
    tried: list[np.ndarray] = []

    def cost(params):
        value = energy(operator, ansatz, params, estimator)
        history.append(value)
        tried.append(np.array(params))
        return value

    minimize(
        cost, np.asarray(initial_point, dtype=float), method=optimizer,
        options={"maxiter": maxiter},
    )
    best = int(np.argmin(history))
    return VQEResult(
        energy=history[best],
        params=tried[best],
        nfev=len(history),
        optimizer=optimizer,
        history=history,
    )
