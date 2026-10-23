import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from vqe_lab.hamiltonians import exact_energy, h2
from vqe_lab.hardware import analyze_curve, build_curve, run_curve
from vqe_lab.vqe import prototype_ansatz

BONDS = [0.735, 1.5]


def test_build_curve_batches_two_bases_per_bond_plus_calibration():
    plan = build_curve(BONDS)
    assert len(plan["circuits"]) == 2 * len(BONDS) + 2
    assert plan["bonds"] == BONDS


def _ideal_counts(circuit, shots=10**6):
    probs = Statevector(circuit.remove_final_measurements(inplace=False)).probabilities()
    return {format(i, "02b"): int(round(p * shots)) for i, p in enumerate(probs) if p > 0}


def test_analyze_perfect_counts_recovers_exact_energy():
    plan = build_curve(BONDS)
    counts = [_ideal_counts(c) for c in plan["circuits"]]
    result = analyze_curve(plan, counts)
    for d in BONDS:
        ex = exact_energy(h2(d).operator)
        for mode in ("raw", "readout", "symmetry"):
            assert result[d][mode] == pytest.approx(ex, abs=1e-5)


def test_run_curve_on_fake_backend_mitigation_helps():
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    backend = FakeManilaV2()
    out = run_curve(SamplerV2(mode=backend), backend, BONDS, shots=8000, seed=0)
    for d in BONDS:
        ex = exact_energy(h2(d).operator)
        assert abs(out["energies"][d]["readout"] - ex) < abs(out["energies"][d]["raw"] - ex)
    assert len(out["physical_qubits"]) == 2
