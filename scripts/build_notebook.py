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
from vqe_lab.noise import SamplingEstimator
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
md("""## 4. Optimizer comparison
**4a. Noiseless.** The 1-parameter prototype is too easy to separate optimizers, so this uses the general `efficient_su2` ansatz (8 parameters) from random starting points, one per seed. Each optimizer gets `maxiter=200`. That limits iterations, not evaluations, so the evaluation counts differ. SPSA uses textbook default gains, untuned.""")
code("""OPT_BONDS = [0.735, 1.5, 2.5]
optimizer_results = {opt: [] for opt in OPTIMIZERS}
for opt in OPTIMIZERS:
    for d in OPT_BONDS:
        m = h2(d); ex = exact_energy(m.operator)
        for s in SEEDS:
            r = run_vqe(m.operator, efficient_ansatz(2, reps=1), optimizer=opt, seed=s, maxiter=200)
            optimizer_results[opt].append({"bond": d, "seed": s, "error_mha": (r.energy - ex) * 1e3, "nfev": r.nfev})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for opt, runs in optimizer_results.items():
    ax1.scatter([r["nfev"] for r in runs], [max(r["error_mha"], 1e-7) for r in runs], label=opt, alpha=0.8)
ax1.set_yscale("log"); ax1.axhline(CHEM_ACC_MHA, color="gray", ls=":")
ax1.set(xlabel="energy evaluations", ylabel="error vs exact (mHa)", title="Accuracy vs cost (noiseless, all bonds × seeds)")
ax1.legend(fontsize=8)
ok = [np.mean([r["error_mha"] < CHEM_ACC_MHA for r in optimizer_results[o]]) * 100 for o in OPTIMIZERS]
ax2.bar(OPTIMIZERS, ok)
ax2.set(xlabel="optimizer", ylabel="runs within chemical accuracy (%)", title="Success rate", ylim=(0, 105))
plt.tight_layout(); plt.show()

for opt, runs in optimizer_results.items():
    e = [r["error_mha"] for r in runs]; n = [r["nfev"] for r in runs]
    print(f"{opt:12s} median error {np.median(e):9.4f} mHa   median evals {int(np.median(n)):4d}")""")
md("""**4b. Under noise.** Full VQE optimization on the noisy, readout-mitigated simulator at 0.735 Å with the prototype ansatz. Gradient-based optimizers (SLSQP, L-BFGS-B) rely on finite differences, which shot noise wrecks, so only COBYLA and SPSA are compared here.""")
code("""NOISY_SHOTS = 2000
m = h2(0.735); ex = exact_energy(m.operator)
noisy_opt_results = {}
for opt in ["COBYLA", "SPSA"]:
    est = SamplingEstimator(noisy=True, mitigate_readout=True, shots=NOISY_SHOTS, seed=0)
    r = run_vqe(m.operator, prototype_ansatz(), optimizer=opt, initial_point=[0.5], estimator=est, seed=0, maxiter=40)
    noisy_opt_results[opt] = r

fig, ax = plt.subplots()
for opt, r in noisy_opt_results.items():
    ax.plot(np.array(r.history) - ex, label=f"{opt} ({r.nfev} evals)", alpha=0.8)
ax.axhline(CHEM_ACC_MHA / 1e3, color="gray", ls=":", label="chemical accuracy")
ax.set(xlabel="energy evaluation #", ylabel="measured energy − exact (Ha)",
       title=f"Noisy VQE at 0.735 Å (FakeManilaV2 + readout mitigation, {NOISY_SHOTS} shots)")
ax.legend(); plt.show()
# The lowest noisy value seen is biased low (it picks the luckiest shot noise), so
# re-measure the chosen parameters with fresh shots and the exact simulator.
for opt, r in noisy_opt_results.items():
    fresh = energy(m.operator, prototype_ansatz(), r.params,
                   SamplingEstimator(noisy=True, mitigate_readout=True, shots=8000, seed=99))
    ideal = energy(m.operator, prototype_ansatz(), r.params)
    print(f"{opt:7s} θ = {r.params[0]:.4f}   re-measured error {(fresh - ex) * 1e3:6.2f} mHa"
          f"   noiseless error at that θ {(ideal - ex) * 1e3:6.3f} mHa")""")
md("""## 5. Warm start along the curve
The prototype starts every geometry from θ = 0 (Hartree–Fock), a **cold** start. A **warm** start begins each geometry at the previous geometry's optimum, since neighbouring bond lengths have similar answers.

COBYLA's default first step (1 rad) jumps away from a good start, so both runs use `rhobeg=0.05`. Finding from testing: with the default step, warm start saved only ~5%.""")
code("""from vqe_lab.vqe import scan_curve

ops = [m.operator for m in mols]
opts = {"rhobeg": 0.05}
warm_results = {
    "cold": scan_curve(ops, prototype_ansatz(), warm=False, initial_point=[0.0], options=opts),
    "warm": scan_curve(ops, prototype_ansatz(), warm=True, initial_point=[0.0], options=opts),
}
cold_n = sum(r.nfev for r in warm_results["cold"]); warm_n = sum(r.nfev for r in warm_results["warm"])

fig, ax = plt.subplots()
for name, rs in warm_results.items():
    ax.plot(bonds, [r.nfev for r in rs], "o-", label=f"{name} (total {sum(r.nfev for r in rs)})")
ax.set(xlabel="H–H bond length (Å)", ylabel="energy evaluations to converge",
       title=f"Warm start: {100 * (1 - warm_n / cold_n):.0f}% fewer evaluations (prototype ansatz, COBYLA)")
ax.legend(); plt.show()

print(f"cold {cold_n} evals, warm {warm_n} evals, saved {100 * (1 - warm_n / cold_n):.1f}%")
print("max warm error:", f"{max((r.energy - e) * 1e3 for r, e in zip(warm_results['warm'], exact)):.2e} mHa")""")
md("""## 6. Noise and readout mitigation
Same circuit, same optimal angle, three ways of measuring it:
- **ideal**: real shots, no hardware noise (shot noise only)
- **noisy**: Aer simulator built from IBM's `FakeManilaV2` calibration (gate + readout errors)
- **mitigated**: noisy, plus readout-error mitigation (inverse of measured per-qubit assignment matrices)

The angle is the noiseless VQE optimum, so this isolates the measurement error from the optimizer.""")
code("""NOISE_BONDS = [0.735, 1.5, 2.5]
SHOTS = 8000
settings = {"ideal": dict(noisy=False), "noisy": dict(noisy=True),
            "mitigated": dict(noisy=True, mitigate_readout=True)}
noise_results = {k: {} for k in settings}
for d in NOISE_BONDS:
    m = h2(d)
    theta = run_vqe(m.operator, prototype_ansatz(), initial_point=[0.0]).params
    ex = exact_energy(m.operator)
    for name, kw in settings.items():
        errs = [(energy(m.operator, prototype_ansatz(), theta,
                        SamplingEstimator(shots=SHOTS, seed=s, **kw)) - ex) * 1e3 for s in SEEDS]
        noise_results[name][d] = {"mean": float(np.mean(errs)), "min": min(errs), "max": max(errs)}

x = np.arange(len(NOISE_BONDS)); w = 0.27
fig, ax = plt.subplots()
for i, name in enumerate(settings):
    r = noise_results[name]
    mean = np.array([r[d]["mean"] for d in NOISE_BONDS])
    lo = mean - np.array([r[d]["min"] for d in NOISE_BONDS])
    hi = np.array([r[d]["max"] for d in NOISE_BONDS]) - mean
    ax.bar(x + (i - 1) * w, mean, w, yerr=[lo, hi], capsize=3, label=name)
ax.axhline(CHEM_ACC_MHA, color="gray", ls=":", label="chemical accuracy")
ax.set_xticks(x, [f"{d} Å" for d in NOISE_BONDS])
ax.set(xlabel="H–H bond length", ylabel="energy error vs exact (mHa)",
       title=f"Noise and readout mitigation (FakeManilaV2, {SHOTS} shots, {len(SEEDS)} seeds)")
ax.legend(); plt.show()

for name in settings:
    print(name.ljust(10), "  ".join(f"{d} Å: {noise_results[name][d]['mean']:6.2f} mHa" for d in NOISE_BONDS))""")
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
