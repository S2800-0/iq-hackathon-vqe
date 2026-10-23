"""Split the gap between a noisy VQE energy and the exact answer into causes.

An additive model with two parts:

- Parameter quality, in noiseless energies:
    ansatz     = best noiseless VQE - exact
    optimizer  = noiseless energy at the θ a noisy run found - best noiseless VQE
- Measurement error, all at the best noiseless θ:
    shots      = ideal simulator with shots - best noiseless VQE
    gate_noise = noisy + readout mitigation - ideal with shots (what mitigation can't fix)
    readout    = noisy - noisy + readout mitigation

The total treats these as independent and adds them, which is a first-order
approximation (measurement error is evaluated at the best θ, not the noisy-run θ).
Inputs in Ha, outputs in mHa. A component can be slightly negative from shot noise.
"""


def error_budget(exact, vqe_noiseless, optimizer_noiseless, ideal_shots, mitigated, noisy):
    mha = 1e3
    b = {
        "ansatz": (vqe_noiseless - exact) * mha,
        "optimizer": (optimizer_noiseless - vqe_noiseless) * mha,
        "shots": (ideal_shots - vqe_noiseless) * mha,
        "gate_noise": (mitigated - ideal_shots) * mha,
        "readout": (noisy - mitigated) * mha,
    }
    b["total_raw"] = sum(b.values())
    b["total_mitigated"] = b["total_raw"] - b["readout"]
    b["readout_recovered_pct"] = 100 * b["readout"] / b["total_raw"]
    return b
