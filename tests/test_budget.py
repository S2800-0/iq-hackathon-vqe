import pytest

from vqe_lab.budget import error_budget

# Energies in Ha, chosen so each component is easy to read in mHa.
E = dict(exact=-1.000, vqe_noiseless=-0.999, optimizer_noiseless=-0.997,
         ideal_shots=-0.9985, mitigated=-0.995, noisy=-0.950)


def test_components_in_mha():
    b = error_budget(**E)
    assert b["ansatz"] == pytest.approx(1.0)
    assert b["optimizer"] == pytest.approx(2.0)
    assert b["shots"] == pytest.approx(0.5)
    assert b["gate_noise"] == pytest.approx(3.5)
    assert b["readout"] == pytest.approx(45.0)


def test_totals_are_sums_of_components():
    b = error_budget(**E)
    parts = ["ansatz", "optimizer", "shots", "gate_noise", "readout"]
    assert b["total_raw"] == pytest.approx(sum(b[k] for k in parts))
    assert b["total_mitigated"] == pytest.approx(b["total_raw"] - b["readout"])


def test_mitigation_recovery_percent():
    b = error_budget(**E)
    assert b["readout_recovered_pct"] == pytest.approx(100 * 45.0 / b["total_raw"])
