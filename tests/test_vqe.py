import numpy as np
import pytest

from vqe_lab.hamiltonians import exact_energy, h2
from vqe_lab.vqe import efficient_ansatz, energy, prototype_ansatz, run_vqe

CHEM_ACC = 1.6e-3


def test_prototype_at_zero_angle_is_hartree_fock():
    mol = h2(0.735)
    assert energy(mol.operator, prototype_ansatz(), [0.0]) == pytest.approx(
        mol.hf_energy, abs=1e-8
    )


def test_prototype_reaches_exact_at_equilibrium():
    mol = h2(0.735)
    result = run_vqe(mol.operator, prototype_ansatz(), initial_point=[0.0])
    assert result.energy == pytest.approx(exact_energy(mol.operator), abs=1e-5)


def test_efficient_ansatz_reaches_chemical_accuracy_stretched():
    mol = h2(2.0)
    result = run_vqe(mol.operator, efficient_ansatz(2, reps=2), seed=1, maxiter=500)
    assert result.energy - exact_energy(mol.operator) < CHEM_ACC


@pytest.mark.parametrize("optimizer", ["COBYLA", "SLSQP", "L-BFGS-B", "Nelder-Mead"])
def test_supported_optimizers_converge(optimizer):
    mol = h2(0.735)
    result = run_vqe(mol.operator, prototype_ansatz(), optimizer=optimizer, initial_point=[0.1])
    assert result.energy - exact_energy(mol.operator) < CHEM_ACC
    assert result.optimizer == optimizer


def test_history_records_every_evaluation():
    result = run_vqe(h2(0.735).operator, prototype_ansatz(), initial_point=[0.0])
    assert len(result.history) == result.nfev
    assert result.energy == pytest.approx(min(result.history))


def test_warm_start_needs_no_more_evaluations_than_cold():
    first = run_vqe(h2(1.0).operator, prototype_ansatz(), initial_point=[0.0])
    nxt = h2(1.1).operator
    cold = run_vqe(nxt, prototype_ansatz(), initial_point=[0.0])
    warm = run_vqe(nxt, prototype_ansatz(), initial_point=first.params)
    assert warm.nfev <= cold.nfev


def test_seed_makes_random_start_reproducible():
    op = h2(0.735).operator
    a = run_vqe(op, efficient_ansatz(2), seed=7, maxiter=50)
    b = run_vqe(op, efficient_ansatz(2), seed=7, maxiter=50)
    np.testing.assert_allclose(a.params, b.params)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        run_vqe(h2(0.735).operator, prototype_ansatz(), optimizer="Adam")
