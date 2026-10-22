# Report Outline: Diagnosing the VQE Prototype

*Sections marked **[Required]** are the organizers' "report must haves" (launch slides, Oct 22). Every figure and number comes from the notebook. All plots need labeled axes with units.*

## 0. Title block [Required]
- Group name: ___
- Member: Shahesta Mohamed S___ · Affiliation: ___ · Department: ___ · Career level: ___
- **Submission category: Rising Star**
- Track: VQE
- Public GitHub repo: ___

## 1. Summary of what I did [Required: bullet list]
- Reproduced the provided H₂ VQE prototype at 0.735 Å.
- Swept bond length 0.5–2.5 Å and compared prototype ansatz vs a general ansatz against the exact answer.
- Compared classical optimizers (COBYLA, SLSQP, SPSA, ...) on accuracy and cost.
- Tested warm-starting along the curve.
- Added a noisy simulator + readout mitigation and built an error budget.
- (If done) HeH⁺ and/or one real IBM hardware run.

## 2. Motivation for track and problem [Required]
- Why VQE / chemistry interests me.
- Why bond-breaking is a meaningful stress test (prototype only checked one geometry).
- What question the project answers: where does the prototype lose accuracy, and what fixes it?

## 3. Workflow explaining the code [Required: brief]
Short diagram or numbered list matching notebook sections:
install → build Hamiltonian (PySCF + parity mapper) → exact reference → VQE (ansatz + optimizer + estimator) → sweep geometries → noise + mitigation → plots.
Mention package install cell at top and software versions.

## 4. Results [Required: most important]
- **Fig 1:** energy vs bond length (exact, HF, prototype, new ansatz); lower panel error in mHa with 1.6 mHa band.
- **Fig 2:** optimizer comparison: final error and number of evaluations per optimizer.
- **Fig 3:** cold vs warm start evaluation counts (mean and range, 3 seeds).
- **Fig 4:** error budget at one geometry (ideal → noisy → mitigated).
- **Table:** all key numbers.
- One paragraph per figure: what it shows, what it means.

## 5. What improved, what didn't, limitations
- Improvements with numbers.
- At least one honest negative result.
- No quantum advantage claimed: classical exact solution is trivial at this size.

## 6. Next steps and references
LiH, ZNE, more hardware. Cite the training notebook, Qiskit/Qiskit Nature docs, Peruzzo et al. 2014. Verify citations.
