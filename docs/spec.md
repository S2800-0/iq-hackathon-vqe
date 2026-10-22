# Spec: Diagnosing the VQE Prototype (Solo, Rising Star)

**Track:** VQE · **Level:** Rising Star · **Team:** solo · **Build day:** Sat Oct 24, 9 AM–4 PM CDT
**Deliverables:** one Colab notebook + one written report, both on GitHub.

## One-sentence promise
Starting from the provided H₂ VQE notebook, I show where its energy drifts from the exact answer as the molecule is pulled apart and under realistic noise, and measure how much each of three fixes recovers.

## Scope (solo-sized)
| Tier | Content | Status |
|---|---|---|
| Must | H₂ energy curve, 0.5–2.5 Å (~10 points): exact vs Hartree–Fock vs provided 1-param ansatz vs a general ansatz (`TwoLocal` or UCCSD) on a noiseless simulator | Core |
| Must | Optimizer comparison (COBYLA, SLSQP, SPSA, L-BFGS-B) at 2–3 geometries: final error and evaluation count. Mirrors the organizers' "check effects of optimizers" example | Core |
| Must | Warm start along the curve: optimizer evaluations, cold vs warm, 3 seeds | Core |
| Must | Noise rung on 2–3 bond lengths: noiseless → fake-backend noise → + readout mitigation | Core |
| Should | Error-budget bar chart (ansatz / shots / hardware noise) at one geometry | Story |
| Could | Repeat for HeH⁺ (already in notebook, Part 5) | Second molecule |
| Could | One real IBM hardware run at 0.735 Å (bonus) | Bonus |
| Won't | LiH, ZNE, custom UI | Out of scope solo |

## Notebook structure
0. Setup: installs, pinned versions printed, seeds fixed
1. Baseline: reproduce the given prototype at 0.735 Å (must match the notebook's number)
2. Energy curve: exact (NumPy eigensolver) and HF reference across geometries
3. Ansatz comparison: 1-param vs general ansatz, error in mHa vs chemical-accuracy band (1.6 mHa)
4. Warm start: evaluation counts, cold vs warm, mean ± range over 3 seeds
5. Noise: `FakeBackend` estimator, then readout mitigation; shots stated
6. Error budget figure
7. (Could) HeH⁺ / hardware
8. Summary table of results

Rule: every number in the report is printed by a notebook cell.

## Acceptance tests
- [ ] Section 1 reproduces the prototype's H₂ energy within 1e-6 Ha
- [ ] Exact energies are consistent (nuclear repulsion included or excluded everywhere, stated once)
- [ ] Curve plot has all methods + chemical-accuracy band, axes labeled with units
- [ ] Warm-start result reports seeds and a range, not a single number
- [ ] Noise section states backend name, shots, and mitigation method
- [ ] Notebook runs top-to-bottom in fresh Colab in under ~15 min (hardware cells optional/skippable)
- [ ] At least one honest negative result recorded (what didn't help)

## Kill tests (Friday, ~2 h)
1. qiskit-nature builds H₂ at 2.0 Å in Colab without API errors.
2. Does the 1-param ansatz lose accuracy at 2.0 Å? If not (likely exact for 2-qubit H₂), the story shifts to noise + warm start, and HeH⁺ becomes Must.
3. Noisy fake-backend run at one geometry finishes in < 2 min; readout mitigation measurably changes the error.
4. (Bonus) Submit one tiny IBM job; note queue time.

## Saturday timeline
| Time | Work |
|---|---|
| 9:00–10:00 | Sections 0–2 |
| 10:00–11:30 | Section 3 |
| 11:30–12:30 | Section 4 |
| 12:30–1:00 | Break |
| 1:00–2:15 | Section 5–6 (submit hardware job at 1:00 if attempting) |
| 2:15–3:15 | Write report from notebook outputs |
| 3:15–3:45 | Fresh-runtime rerun, push to GitHub |
| 3:45 | Buffer, submit |

## Submission rules (launch slides, Oct 22)
- Judged on code (notebook) + project report, both in a **public GitHub repository**.
- First cell: `%pip install ...` of every package needed; notebook must run as is.
- Report must include: group name, members' names/affiliation/department/career level, category, bullet summary, motivation, brief workflow, results with labeled plots.
- When done, email the repo link to the organizer (address on the slide).
- Category is self-selected but final placement is decided by the grading committee. Two winners per category.
- Winners give a short presentation at the closing ceremony, Oct 30, 5 PM CDT.
- Organizers' guidance: explore as much as you can within your topic (their QAOA example: optimizers, system size, mixers).

## Open questions for organizers
Pre-existing code rule · AI-tool policy · report length/format (PDF vs Markdown).
