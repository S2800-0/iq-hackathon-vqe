"""H2 dissociation curve on a real backend in one batched Sampler job.

Each bond length needs two measurement bases (Z for IZ/ZI/ZZ, X for XX), at the
prototype's noiseless optimum. Two calibration circuits (all-0, all-1) on the same
physical qubits give per-qubit readout matrices. Analysis then reports three
energies per bond: raw, readout-mitigated, and readout + symmetry filter.
"""

from functools import reduce

import numpy as np
from qiskit import QuantumCircuit, transpile

from vqe_lab.hamiltonians import exact_energy, h2
from vqe_lab.noise import postselect
from vqe_lab.symmetry import forbidden_states
from vqe_lab.vqe import prototype_ansatz, run_vqe


def _basis_label(pauli) -> str:
    # One character per qubit (qubit 0 first): X if the term needs an X measurement there.
    return "".join("X" if pauli.x[q] else "Z" for q in range(pauli.num_qubits))


def build_curve(bonds) -> dict:
    circuits, index, entries = [], [], {}
    for d in bonds:
        op = h2(d).operator
        theta = run_vqe(op, prototype_ansatz(), initial_point=[0.0]).params
        bases = sorted({_basis_label(p) for p in op.paulis if p.x.any() or p.z.any()})
        entries[d] = {"operator": op, "theta": theta, "forbidden": forbidden_states(op)}
        for basis in bases:
            qc = prototype_ansatz().assign_parameters(theta)
            for q, b in enumerate(basis):
                if b == "X":
                    qc.h(q)
            qc.measure_all()
            circuits.append(qc)
            index.append(("bond", d, basis))
    for bit in (0, 1):
        qc = QuantumCircuit(2)
        if bit:
            qc.x([0, 1])
        qc.measure_all()
        circuits.append(qc)
        index.append(("cal", bit, None))
    return {"bonds": list(bonds), "circuits": circuits, "index": index, "entries": entries}


def _probs(counts, n=2):
    p = np.zeros(2**n)
    for key, c in counts.items():
        p[int(key.replace(" ", ""), 2)] += c
    return p / p.sum()


def _readout_inverse(c0, c1, n=2):
    mats = []
    for q in range(n):
        t0, t1 = sum(c0.values()), sum(c1.values())
        p1_given0 = sum(c for k, c in c0.items() if k[::-1][q] == "1") / t0
        p0_given1 = sum(c for k, c in c1.items() if k[::-1][q] == "0") / t1
        mats.append(np.array([[1 - p1_given0, p0_given1], [p1_given0, 1 - p0_given1]]))
    return np.linalg.inv(reduce(np.kron, mats[::-1]))


def _energy(op, probs_by_basis, inverse=None, forbidden=None, clip=False):
    n = op.num_qubits
    total = 0.0
    for pauli, coeff in zip(op.paulis, op.coeffs):
        mask = sum(1 << q for q in range(n) if pauli.x[q] or pauli.z[q])
        if mask == 0:
            total += coeff.real
            continue
        p = probs_by_basis[_basis_label(pauli)]
        if inverse is not None:
            p = inverse @ p
        if clip:
            # Readout inversion can leave small negative "probabilities"; project them away first.
            p = np.clip(p, 0.0, None)
            p = p / p.sum()
        if forbidden and not pauli.x.any():
            p = postselect(p, forbidden)
        signs = np.array([1 - 2 * (bin(i & mask).count("1") % 2) for i in range(2**n)])
        total += coeff.real * float(signs @ p)
    return total


def analyze_curve(plan: dict, counts: list[dict]) -> dict:
    cal = {key[1]: c for key, c in zip(plan["index"], counts) if key[0] == "cal"}
    inverse = _readout_inverse(cal[0], cal[1])
    probs = {}
    for key, c in zip(plan["index"], counts):
        if key[0] == "bond":
            probs.setdefault(key[1], {})[key[2]] = _probs(c)
    out = {}
    for d in plan["bonds"]:
        e = plan["entries"][d]
        out[d] = {
            "exact": exact_energy(e["operator"]),
            "raw": _energy(e["operator"], probs[d]),
            "readout": _energy(e["operator"], probs[d], inverse),
            "symmetry": _energy(e["operator"], probs[d], inverse, e["forbidden"]),
            "symmetry_clipped": _energy(e["operator"], probs[d], inverse, e["forbidden"], clip=True),
            # Sum of negative entries in the readout-corrected Z-basis distribution.
            "negative_mass": float(np.minimum(inverse @ probs[d]["ZZ"], 0.0).sum()),
        }
    return out


def best_qubit_pair(backend) -> list[int]:
    """Coupled pair with the lowest two-qubit gate error plus both readout errors.

    Left to itself, the transpiler tends to place a 2-qubit circuit on qubits 0 and 1,
    whatever their quality that day.
    """
    t = backend.target
    g2 = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)

    def err(props):
        return props.error if props is not None and props.error is not None else 1.0

    best, best_score = None, float("inf")
    for (a, b), props in t[g2].items():
        score = err(props) + err(t["measure"].get((a,))) + err(t["measure"].get((b,)))
        if score < best_score:
            best, best_score = [a, b], score
    return best


def local_sampler(backend, seed: int = 0):
    """Seeded Aer sampler with the fake backend's noise, for a rehearsal that reruns identically.

    The runtime's local mode (``SamplerV2(mode=fake_backend)``) did not honour a simulator
    seed across processes, so this uses Aer's own sampler, which takes the seed directly.
    """
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.primitives import SamplerV2

    return SamplerV2(seed=seed, options={"backend_options": {"noise_model": NoiseModel.from_backend(backend)}})


def run_curve(sampler, backend, bonds, shots: int = 4000, seed: int = 0, physical=None) -> dict:
    """Transpile everything onto the same two physical qubits and submit one job."""
    plan = build_curve(bonds)
    physical = list(physical or best_qubit_pair(backend))
    tcs = transpile(plan["circuits"], backend, initial_layout=physical,
                    optimization_level=1, seed_transpiler=seed)
    job = sampler.run(tcs, shots=shots)
    result = job.result()
    counts = [r.data.meas.get_counts() for r in result]
    return {"job_id": job.job_id(), "physical_qubits": physical, "shots": shots,
            "index": plan["index"], "counts": counts, "energies": analyze_curve(plan, counts)}
