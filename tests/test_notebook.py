"""Run the notebook's code cells once (skipping %pip lines) and check what each section produces."""
import json
import pathlib

import matplotlib
import pytest

matplotlib.use("Agg")

NB = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "vqe_study.ipynb"


@pytest.fixture(scope="module")
def ns():
    namespace = {}
    for cell in json.loads(NB.read_text())["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(l for l in cell["source"] if not l.lstrip().startswith("%"))
            exec(compile(src, str(NB), "exec"), namespace)
    return namespace


def test_notebook_first_cell_installs_packages():
    cells = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    assert "".join(cells[0]["source"]).lstrip().startswith("%pip install")


def test_notebook_code_cells_execute(ns):
    assert "mols" in ns


def test_noise_section_shows_mitigation_helps(ns):
    res = ns["noise_results"]
    assert set(res) == {"ideal", "noisy", "mitigated"}
    for d in res["noisy"]:
        assert abs(res["mitigated"][d]["mean"]) < abs(res["noisy"][d]["mean"])


def test_optimizer_section_covers_every_optimizer(ns):
    res = ns["optimizer_results"]
    assert set(res) == set(ns["OPTIMIZERS"])
    for runs in res.values():
        assert all({"error_mha", "nfev"} <= set(r) for r in runs)


def test_noisy_optimizer_run_compares_cobyla_and_spsa(ns):
    assert set(ns["noisy_opt_results"]) == {"COBYLA", "SPSA"}


def test_warm_start_section_reports_savings(ns):
    res = ns["warm_results"]
    assert set(res) == {"cold", "warm"}
    assert sum(r.nfev for r in res["warm"]) < sum(r.nfev for r in res["cold"])


def test_error_budget_section(ns):
    b = ns["budget"]
    assert b["readout"] > 0
    assert b["total_mitigated"] < b["total_raw"]


def test_summary_table_lists_headline_numbers(ns):
    import math
    rows = ns["summary"]
    names = {r["quantity"] for r in rows}
    for needed in ["Baseline VQE energy, H₂ 0.735 Å", "Warm start evaluations saved",
                   "Raw noisy error, 0.735 Å", "Mitigated error, 0.735 Å",
                   "Readout share of raw error"]:
        assert needed in names
    for r in rows:
        assert {"quantity", "value", "unit", "section"} <= set(r)
        assert math.isfinite(r["value"])
    assert ns["summary_markdown"].startswith("| Quantity |")


def test_section3_shows_h2_ground_state_support(ns):
    w = ns["h2_ground_weights"]
    assert len(w) == len(ns["bonds"])
    assert all(row[0] + row[3] < 1e-10 for row in w)


def test_section8_heh_plus_shows_prototype_limit(ns):
    r = ns["heh_results"]
    assert len(r["proto_mha"]) == len(r["bonds"]) == len(r["eff_mha"])
    # Near equilibrium the prototype misses states it cannot reach; the general ansatz does not.
    assert r["proto_mha"][0] > 10 * max(r["eff_mha"][0], 1e-6)


def test_section8_hardware_cell_skips_without_token(ns):
    import os
    if not os.environ.get("IBM_QUANTUM_TOKEN"):
        assert ns["hardware_result"] is None


def test_install_cell_always_refreshes_vqe_lab():
    cells = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    line = next(l for l in cells[0]["source"] if "git+" in l)
    assert "--force-reinstall" in line and "--no-deps" in line


def test_recorded_hardware_run_is_in_notebook_and_summary(ns):
    rec = ns["recorded_hardware"]
    assert rec["backend"] == "ibm_marrakesh"
    assert rec["job_id"] and rec["date"]
    assert rec["error_mha"] == pytest.approx((rec["energy"] - ns["exact_energy"](ns["mol"].operator)) * 1e3, abs=1e-6)
    names = {r["quantity"] for r in ns["summary"]}
    assert "Real hardware error, 0.735 Å (ibm_marrakesh, recorded)" in names


def test_symmetry_section_helps_h2_and_flags_heh_plus(ns):
    sym = ns["symmetry_results"]
    for d in ns["NOISE_BONDS"]:
        assert abs(sym[d]["mean"]) < abs(ns["noise_results"]["noisy"][d]["mean"])
    bias = ns["heh_filter_bias_mha"]
    assert all(b <= 1e-6 for b in bias)
    assert min(bias) < -0.1


def test_hardware_curve_section_rehearses_on_fake_backend(ns):
    import os
    rehearsal = ns["fake_curve"]["energies"]
    assert len(rehearsal) == len(ns["CURVE_BONDS"]) >= 6
    for e in rehearsal.values():
        assert abs(e["readout"] - e["exact"]) < abs(e["raw"] - e["exact"])
    if not os.environ.get("IBM_QUANTUM_TOKEN"):
        assert ns["hardware_curve"] is None


def test_single_point_hardware_cell_uses_best_qubit_pair():
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    cell = next(c for c in cells if "RuntimeEstimator(mode=backend" in c)
    assert "initial_layout=best_qubit_pair(backend)" in cell
