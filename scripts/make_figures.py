"""Run the notebook headless (no IBM key) and save each plot to figures/, plus hardware plots from results/."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = ["02_dissociation", "03_ground_state_weights", "03_ansatz_error", "04a_optimizers", "04b_noisy_optimizers",
         "05_warm_start", "06_noise_mitigation", "07_error_budget", "08a_heh_plus", "08b_symmetry_filter",
         "08b_falsification", "08d_rehearsal", "08d_recorded_best_pair"]
saved = []


def _save(*_, **__):
    name = NAMES[len(saved)]
    plt.gcf().savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    saved.append(name)


plt.show = _save
ns = {}
for cell in json.loads((ROOT / "notebooks" / "vqe_study.ipynb").read_text())["cells"]:
    if cell["cell_type"] == "code":
        exec("".join(l for l in cell["source"] if not l.lstrip().startswith("%")), ns)
assert saved == NAMES, saved

runs = json.loads((ROOT / "results" / "hardware_runs.json").read_text())
fig, axes = plt.subplots(1, len(runs["curves"]), figsize=(5.5 * len(runs["curves"]), 4))
for ax, run in zip(axes, runs["curves"]):
    for mode, mk in [("raw", "x-"), ("readout", "s-"), ("symmetry", "o-")]:
        ax.plot(run["bonds"], run[mode], mk, label=mode)
    ax.axhline(1.6, color="gray", ls=":", label="chemical accuracy")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="H–H bond length (Å)", ylabel="error vs exact (mHa)",
           title=f"{run['backend']}, qubits {run['physical_qubits']} (real hardware)")
    ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIG / "08d_hardware_curves.png", dpi=150, bbox_inches="tight")
print("saved", len(saved) + 1, "figures to", FIG)
