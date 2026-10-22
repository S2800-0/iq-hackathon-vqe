"""Generate notebooks/vqe_study.ipynb from the cell list below (keeps diffs readable)."""
import json, pathlib

REPO = "https://github.com/S2800-0/iq-hackathon-vqe"
cells = []
md = lambda s: cells.append(("markdown", s))
code = lambda s: cells.append(("code", s))

md("""# Diagnosing the VQE Prototype: H₂ under stretching, optimizers and noise
**IQ Hackathon 2026 · VQE track · Rising Star**

Question: where does the provided VQE prototype lose accuracy, and how much does each fix recover?""")
md("## 0. Setup\nInstalls every package needed (submission rule). The `vqe_lab` helpers install from this project's GitHub repo, which has to be public for this cell to work in Colab.")
code(f"""%pip install -q numpy scipy matplotlib pylatexenc "qiskit>=2.0" qiskit-aer "qiskit-nature>=0.7" qiskit-ibm-runtime pyscf
%pip install -q git+{REPO}""")
code("""import numpy as np, matplotlib.pyplot as plt
import qiskit, qiskit_nature, pyscf
from vqe_lab.hamiltonians import h2, build_molecule, exact_energy
from vqe_lab.vqe import prototype_ansatz, efficient_ansatz, run_vqe, energy, OPTIMIZERS
print("qiskit", qiskit.__version__, "| qiskit-nature", qiskit_nature.__version__, "| pyscf", pyscf.__version__)
CHEM_ACC_MHA = 1.6
SEEDS = [0, 1, 2]""")
md("## 1. Baseline: reproduce the prototype at 0.735 Å\nShould match the training notebook: −1.857275 Ha (electronic energy).")
code("""mol = h2(0.735)
base = run_vqe(mol.operator, prototype_ansatz(), initial_point=[0.0])
print(f"VQE  {base.energy:.6f} Ha  ({base.nfev} evaluations)")
print(f"exact {exact_energy(mol.operator):.6f} Ha   HF {mol.hf_energy:.6f} Ha")""")
md("## 2. Energy curve: exact and Hartree–Fock references")
code("""bonds = np.linspace(0.5, 2.5, 11)
mols = [h2(d) for d in bonds]
exact = np.array([exact_energy(m.operator) for m in mols])
hf = np.array([m.hf_energy for m in mols])
nuc = np.array([m.nuclear_repulsion for m in mols])

fig, ax = plt.subplots()
ax.plot(bonds, exact + nuc, "k-", label="exact")
ax.plot(bonds, hf + nuc, "--", label="Hartree–Fock")
ax.set(xlabel="H–H bond length (Å)", ylabel="total energy (Ha)", title="H₂ dissociation curve")
ax.legend(); plt.show()""")
md("## 3. Ansatz comparison (noiseless)\nFinding from testing: on 2-qubit H₂ the 1-parameter prototype is already exact at every bond length. Without noise, the ansatz isn't what limits accuracy here. TODO: write up why (symmetry: one parameter spans the whole relevant subspace).")
code("""proto = [run_vqe(m.operator, prototype_ansatz(), initial_point=[0.0]) for m in mols]
eff = [run_vqe(m.operator, efficient_ansatz(2, reps=2), seed=0, maxiter=500) for m in mols]

fig, ax = plt.subplots()
ax.semilogy(bonds, [max((r.energy - e) * 1e3, 1e-7) for r, e in zip(proto, exact)], "o-", label="prototype (1 param)")
ax.semilogy(bonds, [max((r.energy - e) * 1e3, 1e-7) for r, e in zip(eff, exact)], "s-", label="efficient_su2 (reps=2)")
ax.axhline(CHEM_ACC_MHA, color="gray", ls=":", label="chemical accuracy")
ax.set(xlabel="H–H bond length (Å)", ylabel="error vs exact (mHa)", title="Ansatz error, noiseless")
ax.legend(); plt.show()""")
md("## 4. Optimizer comparison\nTODO: for 2–3 bond lengths and each optimizer: final error (mHa) and number of evaluations, over SEEDS. Add SPSA once it's implemented.")
code("""# TODO: loop over OPTIMIZERS x [0.735, 1.5, 2.5] x SEEDS, collect error and nfev, bar chart""")
md("## 5. Warm start along the curve\nTODO: pass each geometry's best params to the next; compare total evaluations cold vs warm, mean and range over SEEDS.")
code("""# TODO""")
md("## 6. Noise and readout mitigation\nTODO: Aer noisy estimator built from a fake IBM backend → readout mitigation. State the backend name and number of shots.")
code("""# TODO""")
md("## 7. Error budget\nTODO: stacked bars at one geometry: ansatz | optimizer | shots | hardware noise.")
code("""# TODO""")
md("## 8. (Optional) HeH⁺ and real IBM hardware")
code("""# TODO""")
md("## 9. Summary table\nTODO: every number quoted in the report, printed here.")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "vqe_study.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out)
