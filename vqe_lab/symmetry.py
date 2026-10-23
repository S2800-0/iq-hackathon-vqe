"""Symmetry verification: use basis states the exact ground state never occupies to discard errors.

For H2, mirror symmetry keeps |00> and |11> out of the ground state (section 3), so any
Z-basis shot landing there is an error. For HeH+ that symmetry is absent, and the same
filter becomes invalid.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from vqe_lab.hamiltonians import ground_state_weights
from vqe_lab.noise import postselect


def forbidden_states(operator: SparsePauliOp, tol: float = 1e-8) -> list[int]:
    """Basis-state indices with (numerically) zero weight in the exact ground state."""
    return [i for i, w in enumerate(ground_state_weights(operator)) if w < tol]


def exact_filtered_energy(operator: SparsePauliOp, circuit: QuantumCircuit, forbidden) -> float:
    """Energy with postselection on all-Z terms, from exact (shot-free) probabilities."""
    state = Statevector(circuit)
    pz = postselect(state.probabilities(), forbidden) if forbidden else state.probabilities()
    n = operator.num_qubits
    total = 0.0
    for pauli, coeff in zip(operator.paulis, operator.coeffs):
        if pauli.x.any():
            total += coeff.real * float(state.expectation_value(SparsePauliOp(pauli)).real)
            continue
        mask = sum(1 << q for q in range(n) if pauli.z[q])
        signs = np.array([1 - 2 * (bin(i & mask).count("1") % 2) for i in range(2**n)])
        total += coeff.real * float(signs @ pz)
    return total
