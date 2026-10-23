"""Generate notebooks/vqe_study.ipynb from the cell list below (keeps diffs readable)."""
import json, pathlib

REPO = "https://github.com/S2800-0/iq-hackathon-vqe"
cells = []
md = lambda s: cells.append(("markdown", s))
code = lambda s: cells.append(("code", s))

md("""# Diagnosing the VQE Prototype: H₂ under stretching, optimizers and noise
**IQ Hackathon 2026 · VQE track · Rising Star**

Question: where does the provided VQE prototype lose accuracy, and how much does each fix recover?""")
md("## 0. Setup\nInstalls every package needed (submission rule). The `vqe_lab` helpers install from this project's GitHub repo, which has to be public for this cell to work in Colab. `--force-reinstall` makes a reused Colab session pick up the latest helpers.")
code(f"""%pip install -q numpy scipy matplotlib pylatexenc "qiskit>=2.0" qiskit-aer "qiskit-nature>=0.7" qiskit-ibm-runtime pyscf
%pip install -q --force-reinstall --no-deps git+{REPO}""")
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
md("""## 3. Ansatz comparison (noiseless)
**Finding:** on 2-qubit H₂ the 1-parameter prototype is exact at every bond length. Without noise, the ansatz isn't what limits accuracy.

**Why:** after parity mapping, H₂ has 4 basis states: |00⟩, |01⟩, |10⟩, |11⟩. Two of them are the Hartree–Fock state (both electrons in the bonding orbital) and the doubly excited state (both in the antibonding orbital). The other two are single excitations. Those have the wrong symmetry: H₂ is mirror-symmetric, and a single excitation flips that symmetry, so the Hamiltonian never mixes them into the ground state. The ground state is therefore always cos(θ/2)·|HF⟩ + sin(θ/2)·|doubly excited⟩, exactly the family the prototype circuit prepares. One angle is enough. The cell below checks this numerically.""")
code("""from vqe_lab.hamiltonians import ground_state_weights

h2_ground_weights = [ground_state_weights(m.operator) for m in mols]
labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
fig, ax = plt.subplots()
bottom = np.zeros(len(bonds))
for k in range(4):
    vals = np.array([w[k] for w in h2_ground_weights])
    ax.bar(bonds, vals, width=0.15, bottom=bottom, label=labels[k]); bottom += vals
ax.set(xlabel="H–H bond length (Å)", ylabel="weight in exact ground state",
       title="H₂ ground state uses only two basis states")
ax.legend(); plt.show()
print("max weight outside the prototype's two states:", f"{max(w[0] + w[3] for w in h2_ground_weights):.1e}")""")
md("The general `efficient_su2` circuit can reach all 4 states, but its optimizer has 12 parameters to tune. Its error grows as the bond stretches (still under chemical accuracy), while the 1-parameter circuit stays exact. For this molecule, the circuit built from the chemistry beats the generic one.")
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

COBYLA's default first step (1 rad) jumps away from a good start, so both runs use `rhobeg=0.05`. Finding from testing: with the default step, warm start saved only ~5%.

Reproducibility note: across repeated runs (local and Colab) the saving varies between about 21% and 23%. One geometry occasionally takes 2 more evaluations, most likely from tiny floating-point differences in the PySCF Hamiltonian coefficients that shift where COBYLA stops. Report it as about 21–23%.""")
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
md("""## 7. Error budget at 0.735 Å
Where does the error come from? Combines results from sections 1, 4b and 6 (see `vqe_lab/budget.py` for the definitions):
- **ansatz**: best noiseless VQE vs exact
- **optimizer**: how much worse the θ found by noisy COBYLA is, evaluated noiselessly
- **shots**: finite-shot bias on a noise-free simulator
- **gate noise**: what remains after readout mitigation
- **readout**: what readout mitigation removes

This is a first-order additive model, not an exact decomposition.""")
code("""from vqe_lab.budget import error_budget

d0 = 0.735
m0 = h2(d0); ex0 = exact_energy(m0.operator)
at = lambda name: ex0 + noise_results[name][d0]["mean"] / 1e3   # section 6 stores mHa errors
budget = error_budget(
    exact=ex0,
    vqe_noiseless=base.energy,
    optimizer_noiseless=energy(m0.operator, prototype_ansatz(), noisy_opt_results["COBYLA"].params),
    ideal_shots=at("ideal"), mitigated=at("mitigated"), noisy=at("noisy"),
)

parts = ["ansatz", "optimizer", "shots", "gate_noise", "readout"]
fig, ax = plt.subplots(figsize=(9, 3))
for row, (label, keep) in enumerate([("raw noisy", parts), ("with readout mitigation", parts[:-1])]):
    left = 0.0
    for i, k in enumerate(parts):
        width = max(budget[k], 0.0) if k in keep else 0.0
        ax.barh(row, width, left=left, color=f"C{i}", label=k if row == 0 else None)
        left += width
ax.axvline(CHEM_ACC_MHA, color="gray", ls=":", label="chemical accuracy")
ax.set_yticks([0, 1], ["raw noisy", "with readout\\nmitigation"])
ax.set(xlabel="energy error vs exact (mHa)", title="Error budget, H₂ at 0.735 Å (FakeManilaV2)")
ax.legend(ncol=3, fontsize=8, loc="lower right"); plt.tight_layout(); plt.show()

for k in parts + ["total_raw", "total_mitigated"]:
    print(f"{k:16s} {budget[k]:8.2f} mHa")
print(f"readout mitigation removes {budget['readout_recovered_pct']:.0f}% of the raw error")""")
md("""## 8. HeH⁺: where the prototype ansatz falls short
HeH⁺ is also 2 electrons in 2 orbitals (2 qubits), but He and H are different atoms, so the molecule has no mirror symmetry. Single excitations are no longer forbidden and mix into the ground state. The 1-parameter prototype can't produce them, so it should miss a little. The general ansatz can.""")
code("""heh_bonds = np.linspace(0.6, 3.0, 9)
heh_mols = [build_molecule(f"He 0 0 0; H 0 0 {d}", charge=1) for d in heh_bonds]
heh_results = {"bonds": heh_bonds, "proto_mha": [], "eff_mha": [], "outside_weight": []}
for m in heh_mols:
    ex = exact_energy(m.operator)
    p = run_vqe(m.operator, prototype_ansatz(), initial_point=[0.0], options={"rhobeg": 0.5})
    e = run_vqe(m.operator, efficient_ansatz(2, reps=2), seed=0, maxiter=800)
    heh_results["proto_mha"].append((p.energy - ex) * 1e3)
    heh_results["eff_mha"].append((e.energy - ex) * 1e3)
    w = ground_state_weights(m.operator)
    heh_results["outside_weight"].append(w[0] + w[3])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.semilogy(heh_bonds, np.maximum(heh_results["proto_mha"], 1e-7), "o-", label="prototype (1 param)")
ax1.semilogy(heh_bonds, np.maximum(heh_results["eff_mha"], 1e-7), "s-", label="efficient_su2 (reps=2)")
ax1.axhline(CHEM_ACC_MHA, color="gray", ls=":", label="chemical accuracy")
ax1.set(xlabel="He–H bond length (Å)", ylabel="error vs exact (mHa)", title="HeH⁺ ansatz error, noiseless")
ax1.legend()
ax2.semilogy(heh_bonds, np.maximum(heh_results["outside_weight"], 1e-12), "o-", color="C2")
ax2.set(xlabel="He–H bond length (Å)", ylabel="ground-state weight on |00⟩ + |11⟩",
        title="Weight the prototype cannot reach")
plt.tight_layout(); plt.show()
print(f"HeH⁺ at {heh_bonds[0]:.2f} Å: prototype {heh_results['proto_mha'][0]:.3f} mHa, general {heh_results['eff_mha'][0]:.1e} mHa")""")
md("""## 8b. Symmetry as an error filter
Section 3 found that H₂'s exact ground state **never** occupies |00⟩ or |11⟩. So any Z-basis shot that lands there must be an error, and it can be thrown away. This is **symmetry verification** (Bonet-Monroig et al., 2018; Sagastizabal et al., 2019). The filter applies to the Z-basis terms (IZ, ZI, ZZ). The XX term is measured in another basis and stays unfiltered.

The same reasoning predicts that the filter is **invalid for HeH⁺**, which section 8 showed does occupy |00⟩ and |11⟩.""")
code("""from vqe_lab.symmetry import forbidden_states, exact_filtered_energy

H2_FORBIDDEN = forbidden_states(h2(0.735).operator)
print("H₂ forbidden basis states:", [labels[i] for i in H2_FORBIDDEN],
      "| HeH⁺ forbidden:", [labels[i] for i in forbidden_states(heh_mols[0].operator)] or "none")

symmetry_results = {}
for d in NOISE_BONDS:
    m = h2(d); ex = exact_energy(m.operator)
    theta = run_vqe(m.operator, prototype_ansatz(), initial_point=[0.0]).params
    errs = [(energy(m.operator, prototype_ansatz(), theta,
                    SamplingEstimator(noisy=True, mitigate_readout=True, forbidden_states=H2_FORBIDDEN,
                                      shots=SHOTS, seed=s)) - ex) * 1e3 for s in SEEDS]
    symmetry_results[d] = {"mean": float(np.mean(errs)), "min": min(errs), "max": max(errs)}

x = np.arange(len(NOISE_BONDS)); w = 0.38
fig, ax = plt.subplots()
for i, (name, r) in enumerate([("readout mitigation", noise_results["mitigated"]), ("+ symmetry filter", symmetry_results)]):
    mean = np.array([r[d]["mean"] for d in NOISE_BONDS])
    err = [mean - np.array([r[d]["min"] for d in NOISE_BONDS]), np.array([r[d]["max"] for d in NOISE_BONDS]) - mean]
    ax.bar(x + (i - 0.5) * w, mean, w, yerr=err, capsize=3, label=name, color=f"C{i + 2}")
ax.axhline(CHEM_ACC_MHA, color="gray", ls=":", label="chemical accuracy")
ax.set_xticks(x, [f"{d} Å" for d in NOISE_BONDS])
ax.set(xlabel="H–H bond length", ylabel="energy error vs exact (mHa)",
       title=f"Symmetry filter on top of readout mitigation (FakeManilaV2, {SHOTS} shots)")
ax.legend(); plt.show()
for d in NOISE_BONDS:
    print(f"{d} Å: noisy {noise_results['noisy'][d]['mean']:6.2f} → readout {noise_results['mitigated'][d]['mean']:5.2f}"
          f" → + symmetry {symmetry_results[d]['mean']:5.2f} mHa")""")
md("""**Honest reading:** the error bars (3 seeds) overlap. In simulation the filter's effect on H₂ is **within shot noise**, and across seeds it sometimes makes the Z-term error worse. On real hardware (section 8d) the filter clearly *hurt* at short bond lengths. See the discussion there.""")
md("""**Falsification test.** Apply the H₂ filter to the *exact* HeH⁺ ground state, using shot-free probabilities, because the effect is smaller than shot noise. A valid filter would leave the energy unchanged. An invalid one throws away real parts of the state and can push the energy **below the true ground-state energy**, which no physical state can do. That impossible result is the warning sign.""")
code("""heh_filter_bias_mha = []
for m in heh_mols:
    r = run_vqe(m.operator, efficient_ansatz(2, reps=2), seed=0, maxiter=800)
    circ = efficient_ansatz(2, reps=2).assign_parameters(r.params)
    heh_filter_bias_mha.append((exact_filtered_energy(m.operator, circ, H2_FORBIDDEN) - exact_energy(m.operator)) * 1e3)

fig, ax = plt.subplots()
ax.plot(heh_bonds, heh_filter_bias_mha, "o-", color="C3")
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="He–H bond length (Å)", ylabel="filtered energy − exact ground energy (mHa)",
       title="H₂'s filter applied to HeH⁺ goes below the true minimum")
plt.show()
print(f"most negative: {min(heh_filter_bias_mha):.3f} mHa (impossible for a valid state)")""")
md("""## 8c. (Optional) Real IBM hardware
Runs the prototype at its noiseless optimum for H₂ at 0.735 Å on the least busy real IBM backend, using the Runtime Estimator in job mode (free plan) with `resilience_level=1` (IBM's built-in readout mitigation).

To enable it, add your IBM Quantum API key as a Colab secret named `IBM_QUANTUM_TOKEN`, or set it as an environment variable. Without a key, the cell skips. The author's recorded run is stored in the cell after it, so the summary works without a key.""")
code("""import os
token = os.environ.get("IBM_QUANTUM_TOKEN")
if token is None:
    try:
        from google.colab import userdata
        token = userdata.get("IBM_QUANTUM_TOKEN")
    except Exception:
        token = None

hardware_result = None
if token:
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimator
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.least_busy(operational=True, simulator=False)
    circ = transpile(prototype_ansatz().assign_parameters(base.params), backend, optimization_level=1)
    obs = mol.operator.apply_layout(circ.layout)
    est = RuntimeEstimator(mode=backend, options={"resilience_level": 1, "default_shots": 4000})
    job = est.run([(circ, obs)])
    print("job id:", job.job_id(), "on", backend.name, "(waiting in queue...)")
    hw = float(job.result()[0].data.evs)
    hardware_result = {"backend": backend.name, "job_id": job.job_id(), "energy": hw,
                       "error_mha": (hw - exact_energy(mol.operator)) * 1e3}
    print(hardware_result)
else:
    print("No IBM_QUANTUM_TOKEN found: skipping the real-hardware run.")""")
md("""**Recorded run (author, 2026-10-23).** Output of the cell above: H₂ at 0.735 Å with the prototype's noiseless optimum on `ibm_marrakesh` (156-qubit Heron r2), 4000 shots, `resilience_level=1`, Open plan. The job ID can be checked on the author's IBM Quantum dashboard.

Real hardware error (7.0 mHa) is close to the readout-mitigated simulator from section 6 (5.9 mHa), and far below the unmitigated simulator (54.8 mHa).""")
code("""recorded_hardware = {"backend": "ibm_marrakesh", "job_id": "dbdat51gv2cc738hoe60", "date": "2026-10-23",
                     "shots": 4000, "resilience_level": 1, "energy": -1.85026823746133}
recorded_hardware["error_mha"] = (recorded_hardware["energy"] - exact_energy(mol.operator)) * 1e3
print(f"ibm_marrakesh (recorded): {recorded_hardware['energy']:.6f} Ha, error {recorded_hardware['error_mha']:.2f} mHa")""")
md("""## 8d. H₂ dissociation curve on real hardware (one batched job)
Eight bond lengths in a **single** Sampler job: two measurement bases per bond (Z for IZ/ZI/ZZ, X for XX) at the prototype's noiseless optimum, plus two readout-calibration circuits on the same two physical qubits. That's 18 circuits × 4000 shots.

From the raw counts, the analysis computes three curves: **raw**, **readout-mitigated** (our own calibration, not IBM's built-in setting) and **readout + symmetry filter** (section 8b).

First a **rehearsal** of the identical job on the fake backend (local, free), then the real run if `IBM_QUANTUM_TOKEN` is set (same key as 8c). Estimated QPU use: well under a minute for 18 small circuits, but this is unverified, so check usage on the IBM dashboard afterwards.""")
code("""from vqe_lab.hardware import run_curve
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

CURVE_BONDS = [0.5, 0.735, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
CURVE_SHOTS = 4000

def plot_curve(out, title):
    E = out["energies"]; ds = sorted(E)
    nuc_of = {d: h2(d).nuclear_repulsion for d in ds}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fine = np.linspace(min(ds), max(ds), 40)
    ax1.plot(fine, [exact_energy(h2(d).operator) + h2(d).nuclear_repulsion for d in fine], "k-", label="exact")
    for mode, mk in [("raw", "x"), ("readout", "s"), ("symmetry", "o")]:
        ax1.plot(ds, [E[d][mode] + nuc_of[d] for d in ds], mk, label=mode)
        ax2.plot(ds, [(E[d][mode] - E[d]["exact"]) * 1e3 for d in ds], mk + "-", label=mode)
    ax2.axhline(CHEM_ACC_MHA, color="gray", ls=":", label="chemical accuracy"); ax2.axhline(0, color="k", lw=0.6)
    ax1.set(xlabel="H–H bond length (Å)", ylabel="total energy (Ha)", title=title)
    ax2.set(xlabel="H–H bond length (Å)", ylabel="error vs exact (mHa)", title="Error by mitigation level")
    ax1.legend(); ax2.legend(fontsize=8); plt.tight_layout(); plt.show()
    for d in ds:
        print(f"{d:5.3f} Å  raw {(E[d]['raw'] - E[d]['exact']) * 1e3:7.2f}  readout {(E[d]['readout'] - E[d]['exact']) * 1e3:6.2f}"
              f"  +symmetry {(E[d]['symmetry'] - E[d]['exact']) * 1e3:6.2f} mHa")

fake_backend = FakeManilaV2()
fake_curve = run_curve(SamplerV2(mode=fake_backend), fake_backend, CURVE_BONDS, shots=CURVE_SHOTS, seed=0)
plot_curve(fake_curve, "Rehearsal: H₂ curve on FakeManilaV2 (simulated)")""")
code("""hardware_curve = None
if token:
    backend = service.least_busy(operational=True, simulator=False)
    print("submitting 18 circuits to", backend.name, "...")
    hardware_curve = run_curve(SamplerV2(mode=backend), backend, CURVE_BONDS, shots=CURVE_SHOTS, seed=0)
    hardware_curve["backend"] = backend.name
    print("job id:", hardware_curve["job_id"], "| physical qubits:", hardware_curve["physical_qubits"])
    plot_curve(hardware_curve, f"H₂ curve on {backend.name} (real hardware)")
    print({d: {k: float(v) for k, v in e.items()} for d, e in hardware_curve["energies"].items()})
else:
    print("No IBM_QUANTUM_TOKEN found: skipping the real-hardware curve.")""")
md("## 9. Summary table\nEvery number quoted in the report comes from this table. The Markdown printout can be pasted straight into the report.")
code("""best_noiseless_opt = min(OPTIMIZERS, key=lambda o: np.median([r["error_mha"] for r in optimizer_results[o]]))
summary = [
    {"quantity": "Baseline VQE energy, H₂ 0.735 Å", "value": base.energy, "unit": "Ha", "section": 1},
    {"quantity": "Exact energy, H₂ 0.735 Å", "value": exact_energy(h2(0.735).operator), "unit": "Ha", "section": 1},
    {"quantity": "Prototype max error, 0.5–2.5 Å (noiseless)",
     "value": max((r.energy - e) * 1e3 for r, e in zip(proto, exact)), "unit": "mHa", "section": 3},
    {"quantity": f"Best noiseless optimizer median error ({best_noiseless_opt})",
     "value": float(np.median([r["error_mha"] for r in optimizer_results[best_noiseless_opt]])), "unit": "mHa", "section": "4a"},
    {"quantity": "Noisy COBYLA: noiseless error at found θ", "value": budget["optimizer"], "unit": "mHa", "section": "4b"},
    {"quantity": "Warm start evaluations saved", "value": 100 * (1 - warm_n / cold_n), "unit": "%", "section": 5},
    {"quantity": "Raw noisy error, 0.735 Å", "value": noise_results["noisy"][0.735]["mean"], "unit": "mHa", "section": 6},
    {"quantity": "Mitigated error, 0.735 Å", "value": noise_results["mitigated"][0.735]["mean"], "unit": "mHa", "section": 6},
    {"quantity": "Readout share of raw error", "value": budget["readout_recovered_pct"], "unit": "%", "section": 7},
    {"quantity": "Error budget total, mitigated", "value": budget["total_mitigated"], "unit": "mHa", "section": 7},
    {"quantity": "Readout + symmetry filter error, 0.735 Å", "value": symmetry_results[0.735]["mean"], "unit": "mHa", "section": "8b"},
    {"quantity": "H₂ filter on HeH⁺: most negative bias (invalid filter)", "value": min(heh_filter_bias_mha), "unit": "mHa", "section": "8b"},
    {"quantity": "HeH⁺ prototype error, 0.60 Å", "value": heh_results["proto_mha"][0], "unit": "mHa", "section": "8a"},
    {"quantity": "Real hardware error, 0.735 Å (ibm_marrakesh, recorded)", "value": recorded_hardware["error_mha"], "unit": "mHa", "section": "8c"},
]
fmt = lambda v, u: f"{v:.6f}" if u == "Ha" else (f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.1e}")
summary_markdown = "| Quantity | Value | Unit | Section |\\n|---|---|---|---|\\n" + "\\n".join(
    f"| {r['quantity']} | {fmt(r['value'], r['unit'])} | {r['unit']} | {r['section']} |" for r in summary)
print(summary_markdown)""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "vqe_study.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out)
