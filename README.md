# IQ Hackathon 2026: Diagnosing the VQE Prototype

Rising Star submission, VQE track.

Starting from the provided H₂ VQE notebook, this project measures where its energy drifts from the exact answer as the molecule is stretched and under realistic noise, and how much each fix recovers.

## Layout
- `vqe_lab/` reusable helpers (Hamiltonians, VQE runner), covered by tests
- `tests/` pytest suite
- `notebooks/` submission notebook
- `docs/` spec and report outline

## Run tests
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```
