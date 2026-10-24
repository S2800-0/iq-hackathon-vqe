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


def test_perfect_counts_have_no_negative_mass_and_clipped_matches_exact():
    plan = build_curve(BONDS)
    result = analyze_curve(plan, [_ideal_counts(c) for c in plan["circuits"]])
    for d in BONDS:
        assert result[d]["negative_mass"] == pytest.approx(0.0, abs=1e-9)
        assert result[d]["symmetry_clipped"] == pytest.approx(exact_energy(h2(d).operator), abs=1e-5)


def test_overcorrected_readout_shows_negative_mass():
    plan = build_curve([0.735])
    counts = [_ideal_counts(c) for c in plan["circuits"]]
    # Calibration claims 10% readout error on both qubits, but the data had none:
    # the inverse then over-corrects and pushes |00>/|11> negative.
    for i, key in enumerate(plan["index"]):
        if key[0] == "cal":
            ok = "00" if key[1] == 0 else "11"
            flips = ["01", "10"]
            counts[i] = {ok: 810_000, flips[0]: 90_000, flips[1]: 90_000, ("11" if ok == "00" else "00"): 10_000}
    r = analyze_curve(plan, counts)[0.735]
    assert r["negative_mass"] < -1e-3


def test_run_curve_returns_raw_counts_for_later_analysis():
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    backend = FakeManilaV2()
    out = run_curve(SamplerV2(mode=backend), backend, [0.735], shots=2000, seed=0)
    assert len(out["counts"]) == len(out["index"]) == 4
    assert all(sum(c.values()) == 2000 for c in out["counts"])


def _pair_score(backend, pair):
    t = backend.target
    g2 = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
    gate = t[g2].get(tuple(pair)) or t[g2].get(tuple(pair[::-1]))
    return gate.error + t["measure"][(pair[0],)].error + t["measure"][(pair[1],)].error


def test_best_qubit_pair_beats_qubits_0_1_on_kingston():
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    from vqe_lab.hardware import best_qubit_pair

    backend = FakeKingston()
    pair = best_qubit_pair(backend)
    assert _pair_score(backend, pair) < _pair_score(backend, [0, 1]) / 5


def test_run_curve_uses_best_pair_by_default():
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from vqe_lab.hardware import best_qubit_pair

    backend = FakeManilaV2()
    out = run_curve(SamplerV2(mode=backend), backend, [0.735], shots=1000, seed=0)
    assert sorted(out["physical_qubits"]) == sorted(best_qubit_pair(backend))


def test_seeded_local_rehearsal_is_reproducible():
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    from vqe_lab.hardware import local_sampler

    backend = FakeManilaV2()
    a = run_curve(local_sampler(backend, seed=3), backend, [0.735], shots=2000, seed=3)
    b = run_curve(local_sampler(backend, seed=3), backend, [0.735], shots=2000, seed=3)
    assert a["counts"] == b["counts"]
