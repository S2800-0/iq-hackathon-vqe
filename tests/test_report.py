"""The report must contain the organizers' required sections and only link figures that exist."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "report.md"

REQUIRED = [  # launch slides, "Report must haves"
    "Group name", "Affiliation", "Department", "Career level",
    "Submission category", "Summary of what I did", "Motivation", "Workflow", "Results",
]


def test_report_has_required_sections():
    text = REPORT.read_text()
    missing = [h for h in REQUIRED if h.lower() not in text.lower()]
    assert not missing, missing


def test_report_figures_exist():
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", REPORT.read_text())
    assert len(links) >= 6
    for link in links:
        assert (ROOT / link).exists(), link


def test_best_pair_hardware_run_is_recorded():
    import json
    runs = json.loads((ROOT / "results" / "hardware_runs.json").read_text())
    best = [c for c in runs["curves"] if c["physical_qubits"] == [148, 149]]
    assert len(best) == 1 and best[0]["job_id"] == "dbe0scfgdj9s73fl0cm0"
    text = REPORT.read_text()
    assert "dbe0scfgdj9s73fl0cm0" in text and "TODO if the best-qubit rerun" not in text
