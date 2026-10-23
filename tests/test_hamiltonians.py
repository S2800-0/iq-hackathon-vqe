import pytest

from vqe_lab.hamiltonians import build_molecule, exact_energy, h2

# Electronic energies (nuclear repulsion excluded), matching the provided notebook.
H2_EQ_EXACT = -1.857275
H2_EQ_HF = -1.836968
CHEM_ACC = 1.6e-3


def test_h2_equilibrium_maps_to_two_qubits():
    mol = h2(0.735)
    assert mol.operator.num_qubits == 2
    assert len(mol.operator) == 5


def test_h2_equilibrium_exact_matches_prototype():
    assert exact_energy(h2(0.735).operator) == pytest.approx(H2_EQ_EXACT, abs=1e-5)


def test_h2_hartree_fock_is_above_exact():
    mol = h2(0.735)
    assert mol.hf_energy == pytest.approx(H2_EQ_HF, abs=1e-5)
    assert mol.hf_energy > exact_energy(mol.operator)


def test_correlation_grows_when_stretched():
    near, far = h2(0.735), h2(2.0)
    corr_near = near.hf_energy - exact_energy(near.operator)
    corr_far = far.hf_energy - exact_energy(far.operator)
    assert corr_far > corr_near > CHEM_ACC


def test_total_energy_adds_nuclear_repulsion():
    mol = h2(0.735)
    assert mol.total(exact_energy(mol.operator)) == pytest.approx(
        H2_EQ_EXACT + mol.nuclear_repulsion
    )


def test_heh_plus_builds_with_charge():
    mol = build_molecule("He 0 0 0; H 0 0 0.772", charge=1)
    assert mol.operator.num_qubits == 2
    assert mol.hf_energy > exact_energy(mol.operator)


def test_h2_ground_state_lives_on_two_basis_states():
    from vqe_lab.hamiltonians import ground_state_weights
    for d in (0.5, 1.5, 3.0):
        w = ground_state_weights(h2(d).operator)
        assert w[0] + w[3] < 1e-10
        assert w.sum() == pytest.approx(1.0)


def test_heh_plus_ground_state_uses_all_four_basis_states():
    from vqe_lab.hamiltonians import ground_state_weights
    w = ground_state_weights(build_molecule("He 0 0 0; H 0 0 0.772", charge=1).operator)
    assert w[0] + w[3] > 1e-4
