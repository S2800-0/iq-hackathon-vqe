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
