import numpy as np
import pytest

from vqe_lab.hamiltonians import build_molecule, exact_energy, h2
from vqe_lab.noise import SamplingEstimator, postselect
from vqe_lab.symmetry import exact_filtered_energy, forbidden_states
from vqe_lab.vqe import efficient_ansatz, energy, prototype_ansatz, run_vqe

FORBIDDEN = [0, 3]  # |00>, |11>


def test_postselect_drops_and_renormalizes():
    p = postselect(np.array([0.1, 0.5, 0.3, 0.1]), FORBIDDEN)
    np.testing.assert_allclose(p, [0.0, 0.625, 0.375, 0.0])


@pytest.mark.parametrize("d", [0.5, 1.5, 3.0])
def test_h2_forbidden_states_are_00_and_11(d):
    assert forbidden_states(h2(d).operator) == FORBIDDEN


def test_heh_plus_has_no_forbidden_states():
    op = build_molecule("He 0 0 0; H 0 0 0.9", charge=1).operator
    assert forbidden_states(op) == []


def test_filter_is_harmless_when_symmetry_holds():
    op = h2(0.735).operator
    theta = run_vqe(op, prototype_ansatz(), initial_point=[0.0]).params
    filtered = exact_filtered_energy(op, prototype_ansatz().assign_parameters(theta), FORBIDDEN)
    assert filtered == pytest.approx(exact_energy(op), abs=1e-8)


def test_invalid_filter_on_heh_plus_goes_below_true_minimum():
    op = build_molecule("He 0 0 0; H 0 0 0.9", charge=1).operator
    r = run_vqe(op, efficient_ansatz(2, reps=2), seed=0, maxiter=800)
    filtered = exact_filtered_energy(op, efficient_ansatz(2, reps=2).assign_parameters(r.params), FORBIDDEN)
    assert filtered < exact_energy(op) - 1e-4


def test_symmetry_filter_reduces_noisy_h2_error():
    op = h2(0.735).operator
    ex = exact_energy(op)
    theta = run_vqe(op, prototype_ansatz(), initial_point=[0.0]).params

    def mean_error(**kw):
        return np.mean([abs(energy(op, prototype_ansatz(), theta,
                                   SamplingEstimator(noisy=True, mitigate_readout=True, shots=20_000, seed=s, **kw)) - ex)
                        for s in range(3)])

    assert mean_error(forbidden_states=FORBIDDEN) < mean_error()
