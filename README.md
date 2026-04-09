# The Monostring Hypothesis

**Seven Computational Experiments That Killed One Path to Emergent Spacetime — and Opened Three Others**

[![DOI: Paper](https://img.shields.io/badge/DOI-Paper%20(PDF)-blue.svg)](https://doi.org/10.5281/zenodo.18886047)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.18890266.svg)](https://doi.org/10.5281/zenodo.18890266)

---

## Project Status

| Part | Topic | Status |
|------|-------|--------|
| I | Dimensional reduction (Lyapunov compactification) | ❌ **Falsified** (symplectic test) |
| II | Gauge Higgs mechanism + causal sets | ❌ **Trivial** (null model) / ⚠️ parameter-dependent |
| III | Spectral dimension (Weyl law) | ✅ Real effect, but d_s ≠ 4.0 and not tested for size dependence |
| IV | Independent verification (graph cosmology v1–v7) | ❌ d_s depends on N; dark energy claim is circular |

### What survives

- Anisotropic Kuramoto synchronization transition (2/6 dims, T_c ≈ 1.4) — non-trivial, reproducible, absent in null model
- Spectral dimension reduction of 37–51% vs random phases — confirmed by two independent methods (Weyl law + heat kernel)
- Frequencies ω dominate over the Cartan matrix K in controlling graph topology (ANOVA: 66% vs 3%)
- Universal D_KY ≈ 4 plateau in dissipative coupled standard maps across all simple Lie algebras of ranks 4–8
- GUE spectral statistics in graph Laplacian (⟨r⟩ = 0.529 vs GUE = 0.531)
- Number-theoretic resonances in d_s(β) — sharp peaks driven by quasi-periodic recurrence on T⁶

### What is definitively ruled out

- Emergent 4D spacetime from the original SOH formulation
- E₆ uniqueness for D ≈ 4 (all rank-6 algebras produce it; B₆ is closer to d_s = 4 than E₆)
- Gauge Higgs interpretation (null model with artificial sync gives higher ratio)
- Yukawa mechanism (6 independent definitions all anti-correlate with VEV)
- Bell test (null model also violates)
- d_s = 4.0 as a fixed spatial dimension (d_s grows linearly with graph size N)
- Dark energy as geometric inevitability (λ_decay was hand-coded, not emergent)
- Compactification of synchronized dimensions (synced and unsynced give comparable d_s)

---

## The Story in 60 Seconds

1. **The idea:** One vibrating entity with 6 internal phases.
   Phase resonances fold the 1D timeline into multi-dimensional space.
2. **v0 (Gemini):** Built a 150K-node graph. Got D ≈ 6, high clustering,
   "mass spectrum." Looked amazing.
3. **v1–v4 (Claude):** Introduced E₆ nonlinear dynamics, Coxeter
   frequencies, proper null models. Got **D = 4.025 ± 0.040** —
   tantalizingly close to our 4D spacetime.
4. **v5–v6:** Discovered D ≈ 4 is not unique to E₆ — ALL Lie algebras
   of rank 6 produce it. Follows from the intermediate value theorem.
5. **v7 (fatal):** The symplectic (Hamiltonian) version gives D = 2r
   identically. **The 4D result was an artifact of dissipative dynamics.**
6. **Part II:** Gauge Higgs mechanism (edge variance ratio = 12.5) —
   falsified by null model (ratio = 22.2 with artificial sync).
7. **Part III:** Spectral dimension reduced by 37–51% vs null — real effect,
   but d_s ≠ 4.0 at any configuration.
8. **Part IV (new):** Independent verification showed d_s ∝ N (not a fixed
   dimension), dark energy model was circular, and B₆ beats E₆ for d_s ≈ 4.

---

## Recommended Reading Order

| # | Document | What you learn |
|---|----------|----------------|
| 1 | [Part I — Main Paper](paper/monostring_paper_en.md) | The full story of falsification (v0–v7) |
| 2 | [Part II — Gauge & Causal Sets](paper/monostring_part2_gauge_causal.md) | Gauge Higgs search + causal set exploration |
| 3 | [Part III — Spectral Dimension](paper/monostring_part3_spectral.md) | Weyl law, algebra comparison, d_s reduction |
| 4 | [Part IV — Independent Verification](paper/monostring_part4_independent_verification.md) | Graph cosmology v1–v7, d_s(N) test, ANOVA |
| 5 | [Gauge Higgs Paper](paper/monostring_gauge_higgs.md) | Detailed gauge analysis (pre-falsification snapshot) |
| 6 | [Philosophical Foundations](paper/monostring_philosophy_en.md) | Speculative ontological context (optional) |

---

## Repository Structure

```
monostring-hypothesis/
├── paper/                                    # Manuscripts and theory documents
│   ├── monostring_paper_en.md                # Part I: Main paper (falsification)
│   ├── monostring_part2_gauge_causal.md      # Part II: Gauge + causal sets
│   ├── monostring_part3_spectral.md          # Part III: Spectral dimension
│   ├── monostring_part4_independent_verification.md  # Part IV: Graph cosmology v1-v7
│   ├── monostring_gauge_higgs.md             # Gauge Higgs detailed analysis
│   ├── monostring_philosophy_en.md           # Philosophical foundations (EN)
│   ├── monostring_philosophy_ru.md           # Philosophical foundations (RU)
│   └── draft_v0_raw_ru.md                    # Historical artifact: original draft
│
├── scripts/
│   ├── part1/                                # Dimensional reduction (v0-v7)
│   │   ├── v0_gemini_original.py
│   │   ├── v1_claude_first_test.py
│   │   ├── v2_claude_null_model.py
│   │   ├── v3_claude_coxeter.py
│   │   ├── v4_claude_lyapunov.py
│   │   ├── v5_claude_all_algebras.py
│   │   ├── v6_claude_rank_analysis.py
│   │   └── v7_claude_symplectic.py
│   │
│   ├── part2/                                # Gauge Higgs + causal sets
│   │   ├── higgs_v1_thermal.py
│   │   ├── higgs_v2_kuramoto.py
│   │   ├── higgs_v3_three_masses.py
│   │   ├── higgs_v4_anisotropic.py
│   │   ├── higgs_v5_directional.py
│   │   ├── higgs_v6_metric.py
│   │   ├── higgs_v7_dispersion.py
│   │   ├── higgs_v8_three_measures.py
│   │   ├── higgs_v9_scaling.py
│   │   ├── gauge_v1_plaquette.py
│   │   ├── gauge_v2_edge_variance.py
│   │   ├── gauge_v3_algebra_scan.py
│   │   ├── causal_v1_basic.py
│   │   ├── causal_v2_corrected.py
│   │   ├── causal_v3_nonlinear.py
│   │   └── causal_v4_light_cone.py
│   │
│   ├── part3/                                # Spectral dimension
│   │   ├── qwalk_v1_spectral.py
│   │   ├── qwalk_v2_weyl.py
│   │   ├── qwalk_v3_nphases.py
│   │   └── qwalk_v4_clarification.py
│   │
│   └── part4/                                # Graph cosmology (independent verification)
│       ├── graph_cosmology_v1.py             # Original dark energy model
│       ├── graph_cosmology_v2.py             # First fix attempt (bug found)
│       ├── graph_cosmology_v3.py             # Three-model comparison
│       ├── graph_cosmology_v4.py             # Matrix scan + ANOVA
│       ├── graph_cosmology_v5.py             # Full spectrum (broken d_s)
│       ├── graph_cosmology_v6.py             # Fixed d_s + benchmarks
│       └── graph_cosmology_v7.py             # Fine scan + size dependence (KEY)
│
├── figures/
│   ├── part1/                                # Lyapunov, symplectic, plateau plots
│   ├── part2/                                # Kuramoto, edge variance, causal sets
│   ├── part3/                                # Weyl d_s, algebra comparison
│   └── part4/                                # Benchmarks, ANOVA, d_s(N)
│       ├── benchmark_ds.png
│       ├── rescue_experiment.png
│       ├── rescue_v5_omega_scan.png
│       ├── rescue_v6.png
│       └── rescue_v7.png
│
├── results/
│   ├── part1/                                # Lyapunov spectra, D_KY tables
│   ├── part2/                                # Edge variance, causal D tables
│   ├── part3/                                # Weyl d_s tables
│   └── part4/                                # Graph cosmology logs and tables
│       ├── v1_dark_energy_log.txt
│       ├── v4_matrix_scan.csv
│       ├── v4_anova_results.csv
│       ├── v6_benchmark_results.csv
│       ├── v6_algebra_comparison.csv
│       ├── v7_fine_scan.csv
│       ├── v7_algebra_scan.csv
│       ├── v7_precision_measurement.csv
│       ├── v7_size_dependence.csv
│       └── v7_final_verdict.md
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Key Results by Part

### Part I: Dimensional Reduction — Falsified

The E₆-coupled standard map with Coxeter frequencies at κ = 0.25 produces
D_corr = 4.025 ± 0.040. However, the symplectic (Hamiltonian) version gives
D_KY = 2r identically for all Lie algebras. The dimensional reduction was
an artifact of dissipative dynamics.

**Consolation prize:** Universal D_KY ≈ 4 plateau in dissipative maps across
all 13/13 tested Lie algebras (ranks 4–8).

📄 [Full paper](paper/monostring_paper_en.md)

### Part II: Gauge Higgs + Causal Sets — Falsified / Parameter-Dependent

Edge variance ratio = 12.5 between synchronized and unsynchronized gauge
directions. But null model (artificial sync, no E₆) gives ratio = 22.2.
The effect is trivial synchronization geometry.

Causal set D = 4.01 at c = 0.20, but any D from 1 to 7 is achievable by
tuning the speed-of-light parameter.

**Surviving:** Kuramoto transition (2+4 anisotropic breaking), 3 Goldstone modes.

📄 [Full paper](paper/monostring_part2_gauge_causal.md)

### Part III: Spectral Dimension — Real Effect, Not d_s = 4

E₆ synchronization reduces spectral dimension by 37–51% vs null model.
D₆ = SO(12) gives d_s = 3.92 — closest to 4.0. But d_s = 4.0 is excluded
by 95% CI at all configurations.

📄 [Full paper](paper/monostring_part3_spectral.md)

### Part IV: Independent Verification — d_s Depends on N

Seven iterations of graph cosmology experiments (v1–v7) with independent
critical analysis. Key findings:

| Finding | Experiment | Key number |
|---------|-----------|------------|
| Dark energy claim is circular | v1–v3 | λ_decay = f(epoch) is input |
| ω dominates over K | v4 (ANOVA) | 66% vs 3% |
| d_s measurement was broken | v5 vs v6 | Fixed t-range → d_s = 0.49 |
| Benchmarks pass after fix | v6 | Path→1.07, Grid→2.03/3.00 |
| **d_s scales with graph size** | **v7** | **d_s ≈ 2.16 + 0.002·N** |
| B₆ closer to d_s = 4 than E₆ | v7 | B₆ in 95% CI, E₆ not |

**The key result:** d_s depends linearly on graph size N. This means the
graph does not possess a fixed dimensionality. The value d_s = 4.0 at
N = 1000 is a coincidence with graph size, not a property of E₆.

📄 [Full paper](paper/monostring_part4_independent_verification.md)

---

## Complete Scorecard

### Confirmed

| Finding | Part | Evidence |
|---------|------|----------|
| Kuramoto transition T_c ≈ 1.4 (2+4 anisotropic) | II | 20+ runs, null control |
| Spectral dimension reduction 37–51% vs null | III, IV | Two methods (Weyl + heat kernel) |
| ω dominates K for graph topology | IV | ANOVA: 66% vs 3% |
| Universal D ≈ 4 plateau (dissipative maps) | I | 13/13 algebras |
| GUE spectral statistics | III | ⟨r⟩ = 0.529 |
| Heat-kernel benchmarks (path→1, grid→2, 3) | IV | Error < 7% |

### Falsified

| Claim | Part | How |
|-------|------|-----|
| 6D → 4D via Lyapunov | I | Symplectic: D_KY = 2r always |
| E₆ uniqueness | I, IV | All algebras; B₆ closer to d_s = 4 |
| Gauge Higgs mechanism | II | Null ratio = 22.2 > E₆ ratio = 12.5 |
| Yukawa mechanism | II | 6 definitions anti-correlate |
| Bell test validity | I | Null also violates |
| d_s = 4.0 as fixed dimension | III, IV | 95% CI excludes; d_s ∝ N |
| Dark energy = graph geometry | IV | λ(t) is circular logic |
| Compactification of synced dims | III, IV | d_s(sync) ≈ d_s(unsync) |

### Open

| Direction | Status | Key question |
|-----------|--------|-------------|
| Quantum walks → Dirac equation | Not implemented | Unitarity avoids dissipation |
| Number-theoretic resonances in d_s(β) | Observed (IV) | Connection to KAM theory? |
| Universal D ≈ 4 plateau | Confirmed (I) | Why 4 specifically? |
| D₆ / B₆ vs E₆ for spectral dimension | Partial (III, IV) | Does d_s converge at large N? |

---

## Running the Experiments

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: Python 3.8+, NumPy, SciPy, NetworkX, Matplotlib

### Quick Start

**Part I (decisive falsification):**
```bash
cd scripts/part1
python v7_claude_symplectic.py
```
Expected runtime: ~60 minutes.

**Part IV (key new result — d_s depends on N):**
```bash
cd scripts/part4
python graph_cosmology_v7.py
```
Expected runtime: ~10 minutes.

### Full Reproduction

Run scripts within each part in numerical order. Each script is
self-contained with explicit random seeds for reproducibility.

| Part | Scripts | Total runtime |
|------|---------|---------------|
| I | v0–v7 | ~4–6 hours |
| II | higgs_v1–v9, gauge_v1–v3, causal_v1–v4 | ~2–3 hours |
| III | qwalk_v1–v4 | ~1–2 hours |
| IV | graph_cosmology_v1–v7 | ~1–2 hours |

---

## Citation

```bibtex
@misc{lebedev2025monostring,
  author       = {Lebedev, Igor},
  title        = {The Monostring Hypothesis: Seven Computational Experiments
                  That Killed One Path to Emergent Spacetime ---
                  and Opened Three Others},
  year         = {2025},
  publisher    = {GitHub / Zenodo},
  url          = {https://github.com/LebedevIV/monostring-hypothesis},
  doi          = {10.5281/zenodo.18886047}
}
```

## Discussion

- Habr (Russian): [link pending]
- Reddit r/HypotheticalPhysics: [link pending]

## License

- Paper and documentation: CC-BY 4.0
- Code: MIT License

## Acknowledgments

This research was conducted as an exercise in AI-assisted theoretical
physics. The human author provided the hypothesis and direction; AI
collaborators provided implementations, critical analysis, and
falsifying tests.

The most important contribution of the AI collaborators was not
building the theory — it was designing the experiments that destroyed it.

**AI collaborators:**
- Google Gemini 3.1 Pro — initial mathematical implementation (Part I, v0)
- Anthropic Claude — critical analysis, falsifying tests (Parts I–IV)
