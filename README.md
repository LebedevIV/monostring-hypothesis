# The Monostring Hypothesis

**Seven Computational Experiments That Killed One Path to Emergent Spacetime — and Opened Three Others**

## Overview

This repository contains the complete computational study of the 
Sole Oscillator Hypothesis (SOH) — the conjecture that all of 
observable reality emerges from the discrete state transitions of 
a single vibrating entity (informally dubbed "the Monostring").

The study was conducted by a human researcher in collaboration 
with two AI systems:
- **Google Gemini 3.1 Pro** — initial mathematical implementation (v0)
- **Anthropic Claude** — critical analysis, falsifying tests (v1–v7)

## The Story in 30 Seconds

1. **The idea:** One vibrating entity with 6 internal phases. 
   Phase resonances fold the 1D timeline into multi-dimensional space.
2. **v0 (Gemini):** Built a 150K-node graph. Got D≈6, high clustering, 
   "mass spectrum." Looked amazing.
3. **v1–v4 (Claude):** Introduced E₆ nonlinear dynamics, Coxeter 
   frequencies, proper null models. Got **D = 4.025 ± 0.040** — 
   tantalizingly close to our 4D spacetime.
4. **v5–v6:** Discovered D≈4 is not unique to E₆ — ALL Lie algebras 
   of rank 6 produce it. It follows from the intermediate value theorem.
5. **v7 (fatal):** The symplectic (Hamiltonian) version gives D = 2r 
   identically. **The 4D result was an artifact of dissipative dynamics.**
6. **Next:** Causal sets, quantum walks, stochastic quantization.

## Key Result

The Sole Oscillator Hypothesis, as formulated through coupled 
standard maps on tori, is **falsified**. The dimensional reduction 
mechanism (Lyapunov compactification) does not survive the transition 
to physically correct (Hamiltonian) dynamics.

The philosophical core — monistic, discrete, relational reality — 
remains viable and is being reformulated through alternative 
mathematical frameworks.

## Secondary Finding

A universal D_KY ≈ 4 plateau was observed in dissipative coupled 
standard maps with Cartan-matrix coupling across ALL simple Lie 
algebras of ranks 4–8. This is a result of independent interest 
in dynamical systems theory.

## Repository Structure

paper/ Full paper (English)
scripts/ All Python scripts v0–v7
v0_gemini_original.py Initial model (Gemini 3.1 Pro)
v1_claude_first_test.py E₆ coupling, auto-calibration
v2_claude_null_model.py Null model, Ollivier-Ricci curvature
v3_claude_coxeter.py Coxeter frequencies, Bell null control
v4_claude_lyapunov.py Full Lyapunov spectrum, D_KY, stability
v5_claude_all_algebras.py All rank-6 Lie algebras
v6_claude_rank_analysis.py Ranks 3–12, D(rank) dependence
v7_claude_symplectic.py Symplectic test (decisive falsification)
results/ Raw output logs from each experiment
figures/ Generated plots
requirements.txt Python dependencies

## Running the Experiments

### Requirements

```bash
pip install -r requirements.txt
```

### Quick Start (v7 — the decisive experiment)
```
cd scripts
python v7_claude_symplectic.py
```

Expected runtime: ~60 minutes on a modern CPU (Core i5 or better).
The symplectic test (Part 3) is the critical result.

### Full Reproduction

Run scripts in order v0 → v7. Each script is self-contained.
Expected total runtime: ~4–6 hours.

### Requirements

- Python 3.8+
- NumPy
- SciPy
- NetworkX (v0 only)
- Matplotlib

### Citation

If you reference this work, please cite:
"Lebedev, I. (2025). The Monostring Hypothesis: Seven Computational 
Experiments That Killed One Path to Emergent Spacetime — and Opened 
Three Others. GitHub/Zenodo. https://github.com/LebedevIV/monostring-hypothesis"

## 📄 Documentation & Papers

[![DOI: Paper](https://img.shields.io/badge/DOI-Paper%20(PDF)-blue.svg)](https://doi.org/10.5281/zenodo.18886047)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.18890266.svg)](https://doi.org/10.5281/zenodo.18890266)

*The full academic paper (PDF) is officially published and available on [Zenodo](https://doi.org/10.5281/zenodo.18886047).*

This repository contains the source code, raw data, and markdown drafts associated with the research. You can read the materials directly on GitHub:

*   **[The Main Paper (English)](paper/monostring_paper_en.md)** — The complete write-up of the 7 computational experiments, dimensional reduction, and ultimate falsification of the classical model.
*   **[Philosophical Foundations (English)](paper/monostring_philosophy_en.md)** | **[(Русский)](paper/monostring_philosophy_ru.md)** — The ontological framework and motivation behind the Monostring conjecture.
*   *[Historical Artifact: Original Raw Draft (Russian)](paper/draft_v0_raw_ru.md)*


## Part II: Emergent Gauge Higgs Mechanism (NEW)

After the falsification of the dimensional reduction mechanism (Part I),
we investigated the gauge structure of the Monostring's phase differences.

**Key discovery:** Treating edge phase differences Δφ_d as lattice gauge
connections, we found that the Kuramoto synchronization transition
produces a **gauge Higgs mechanism** with:

- **Edge variance ratio = 12.5** between synchronized (massive) and
  unsynchronized (massless) gauge directions
- **Spectral gap = 0.012** for synchronized dimensions, **zero** for
  unsynchronized — directly analogous to W/Z (massive) vs photon (massless)
- **Correlation with anisotropy = 0.773**

This does NOT rescue the dimensional reduction mechanism, but shows that
the Monostring's phase structure contains non-trivial gauge physics.

📄 **[Read the full Gauge Higgs paper](paper/monostring_gauge_higgs.md)**

### New Scripts

| Script | Description |
|---|---|
| `scripts/higgs_v1_thermal.py` | Thermal model, clustering as VEV |
| `scripts/higgs_v2_kuramoto.py` | Kuramoto order parameter, null model |
| `scripts/higgs_v3_three_masses.py` | Three mass definitions, Goldstone modes |
| `scripts/higgs_v4_anisotropic.py` | Anisotropic Higgs field |
| `scripts/higgs_v5_directional.py` | Directional mass (synced vs unsynced) |
| `scripts/higgs_v6_metric.py` | Weighted metric approach |
| `scripts/higgs_v7_dispersion.py` | Dispersion relation E²=m²+c²p² |
| `scripts/higgs_v8_three_measures.py` | Three independent mass measures |
| `scripts/higgs_v9_scaling.py` | Scaling analysis (ratio vs N, κ) |
| `scripts/gauge_v1_plaquette.py` | Plaquette action, Wilson loops |
| **`scripts/gauge_v2_edge_variance.py`** | **Edge variance — main result** |

## Discussion

Habr (Russian): [link pending]
Reddit r/HypotheticalPhysics: [link pending]

## License

This work is licensed under CC-BY 4.0 for the paper
and MIT License for the code.

## Acknowledgments

This research was conducted as an exercise in AI-assisted theoretical
physics. The most important contribution of the AI collaborators was
not building the theory — it was designing the experiments that
destroyed it.
