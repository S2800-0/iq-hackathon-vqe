import numpy as np
import pytest

from vqe_lab.hamiltonians import exact_energy, h2
from vqe_lab.noise import SamplingEstimator, assignment_matrices
from vqe_lab.vqe import energy, prototype_ansatz, run_vqe

SHOTS = 20_000


@pytest.fixture(scope="module")
def mol():
    return h2(0.735)


@pytest.fixture(scope="module")
def best_theta(mol):
    return run_vqe(mol.operator, prototype_ansatz(), initial_point=[0.0]).params


def test_ideal_sampler_matches_exact_within_shot_noise(mol, best_theta):
    est = SamplingEstimator(noisy=False, shots=SHOTS, seed=1)
    e = energy(mol.operator, prototype_ansatz(), best_theta, est)
    assert e == pytest.approx(exact_energy(mol.operator), abs=0.02)


def test_noise_makes_energy_worse(mol, best_theta):
    exact = exact_energy(mol.operator)
    ideal = energy(mol.operator, prototype_ansatz(), best_theta,
                   SamplingEstimator(noisy=False, shots=SHOTS, seed=1))
    noisy = energy(mol.operator, prototype_ansatz(), best_theta,
                   SamplingEstimator(noisy=True, shots=SHOTS, seed=1))
    assert noisy - exact > abs(ideal - exact)


def test_readout_mitigation_reduces_error(mol, best_theta):
    exact = exact_energy(mol.operator)
    raw = energy(mol.operator, prototype_ansatz(), best_theta,
                 SamplingEstimator(noisy=True, shots=SHOTS, seed=2))
    mit = energy(mol.operator, prototype_ansatz(), best_theta,
                 SamplingEstimator(noisy=True, shots=SHOTS, seed=2, mitigate_readout=True))
    assert abs(mit - exact) < abs(raw - exact)


def test_assignment_matrices_are_column_stochastic():
    mats = assignment_matrices(num_qubits=2, shots=4000, seed=0)
    assert len(mats) == 2
    for a in mats:
        np.testing.assert_allclose(a.sum(axis=0), [1.0, 1.0])
        assert a[0, 0] > 0.9 and a[1, 1] > 0.9


def test_same_seed_is_reproducible(mol, best_theta):
    a = energy(mol.operator, prototype_ansatz(), best_theta,
               SamplingEstimator(noisy=True, shots=2000, seed=5))
    b = energy(mol.operator, prototype_ansatz(), best_theta,
               SamplingEstimator(noisy=True, shots=2000, seed=5))
    assert a == b


def test_mitigation_requires_noise():
    with pytest.raises(ValueError):
        SamplingEstimator(noisy=False, mitigate_readout=True)
