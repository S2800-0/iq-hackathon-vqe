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
