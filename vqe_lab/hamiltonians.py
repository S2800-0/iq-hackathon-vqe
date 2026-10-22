"""Molecular qubit Hamiltonians and exact reference energies.

Energies are electronic (nuclear repulsion excluded), matching the provided
training notebook. Use ``Molecule.total`` to add nuclear repulsion back.
"""

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.units import DistanceUnit


@dataclass(frozen=True)
class Molecule:
    atom: str
    operator: SparsePauliOp
    num_particles: tuple[int, int]
    num_spatial_orbitals: int
    nuclear_repulsion: float
    hf_energy: float

    def total(self, electronic_energy: float) -> float:
        return electronic_energy + self.nuclear_repulsion


def build_molecule(atom: str, charge: int = 0, basis: str = "sto3g") -> Molecule:
    """Build a parity-mapped, two-qubit-reduced Hamiltonian with PySCF."""
    problem = PySCFDriver(
        atom=atom, basis=basis, charge=charge, unit=DistanceUnit.ANGSTROM
    ).run()
    mapper = ParityMapper(num_particles=problem.num_particles)
    operator = mapper.map(problem.hamiltonian.second_q_op())
    nuclear = problem.nuclear_repulsion_energy
    return Molecule(
        atom=atom,
        operator=operator,
        num_particles=problem.num_particles,
        num_spatial_orbitals=problem.num_spatial_orbitals,
        nuclear_repulsion=nuclear,
        hf_energy=problem.reference_energy - nuclear,
    )


def h2(bond_length: float) -> Molecule:
    return build_molecule(f"H 0 0 0; H 0 0 {bond_length}")


def exact_energy(operator: SparsePauliOp) -> float:
    """Lowest eigenvalue by dense diagonalization (fine for a few qubits)."""
    return float(np.linalg.eigvalsh(operator.to_matrix()).min())
