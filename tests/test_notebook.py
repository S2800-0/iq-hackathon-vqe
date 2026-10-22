"""Run the notebook's code cells (skipping %pip lines and plt.show) so it cannot silently break."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")

NB = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "vqe_study.ipynb"


def test_notebook_first_cell_installs_packages():
    cells = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
    assert "".join(cells[0]["source"]).lstrip().startswith("%pip install")


def test_notebook_code_cells_execute():
    ns = {}
    for cell in json.loads(NB.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(l for l in cell["source"] if not l.lstrip().startswith("%"))
        exec(compile(src, str(NB), "exec"), ns)


def test_noise_section_shows_mitigation_helps():
    ns = {}
    for cell in json.loads(NB.read_text())["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(l for l in cell["source"] if not l.lstrip().startswith("%"))
            exec(compile(src, str(NB), "exec"), ns)
    res = ns["noise_results"]
    assert set(res) == {"ideal", "noisy", "mitigated"}
    for d in res["noisy"]:
        assert abs(res["mitigated"][d]["mean"]) < abs(res["noisy"][d]["mean"])
