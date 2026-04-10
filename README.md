# The Monostring Hypothesis

**Eight Computational Experiments That Killed One Path to Emergent Spacetime — and Closed Three Others**

[![DOI: Paper](https://img.shields.io/badge/DOI-Paper%20(PDF)-blue.svg)](https://doi.org/10.5281/zenodo.18886047)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.18890266.svg)](https://doi.org/10.5281/zenodo.18890266)

---

## Project Status

| Part | Topic | Status |
|------|-------|--------|
| I | Dimensional reduction (Lyapunov compactification) | ❌ **Falsified** (symplectic test) |
| II | Gauge Higgs mechanism + causal sets | ❌ **Trivial** (null model) / ⚠️ parameter-dependent |
| III | Spectral dimension (Weyl law) | ⚠️ Real effect, but d_s ≠ 4.0 and not tested for size dependence |
| IV | Independent verification (graph cosmology v1–v7) | ❌ d_s depends on N; dark energy claim is circular |
| **V** | **Resolution (Part IV+ v1–v8)** | ❌ **D_corr(E₆) ≈ 3.02; d_s ≈ 4 is a 3D k-NN graph effect** |

### What survives

- Anisotropic Kuramoto synchronization transition (2/6 dims, T_c ≈ 1.4) — non-trivial, reproducible, absent in null model
- Spectral dimension reduction of 37–51% vs random phases — confirmed by two independent methods (Weyl law + heat kernel)
- Frequencies ω dominate over the Cartan matrix K in controlling graph topology (ANOVA: 66% vs 3%)
- Universal D_KY ≈ 4 plateau in dissipative coupled standard maps across all simple Lie algebras of ranks 4–8
- GUE spectral statistics in graph Laplacian (⟨r⟩ = 0.529 vs GUE = 0.531)
- Number-theoretic resonances in d_s(β) — sharp peaks driven by quasi-periodic recurrence on T⁶
- **D_corr(E₆) ≈ 3.02: the E₆ Coxeter orbit occupies a quasi-3D subset of T⁶** (confirmed, N=1000, 15 seeds)

### What is definitively ruled out

- Emergent 4D spacetime from the original SOH formulation
- E₆ uniqueness for D ≈ 4 (all rank-6 algebras give D_corr ≈ 2.9–3.0)
- Gauge Higgs interpretation (null model with artificial sync gives higher ratio)
- Yukawa mechanism (6 independent definitions all anti-correlate with VEV)
- Bell test (null model also violates)
- d_s = 4.0 as a fixed spatial dimension (d_s grows with N in chain graphs; grows with k in k-NN graphs)
- **d_s(k-NN) as a manifold dimension** (T³ → d_s=4.2, T⁴ → d_s=1.1 at k=20)
- **d_s ≈ 4 as evidence of 4D spacetime** (it identifies 3D structures, not 4D)
- Dark energy as geometric inevitability (λ_decay was hand-coded; E₆ irrelevant, p=0.90)
- Compactification of synchronized dimensions (synced and unsynced give comparable d_s)

---

## The Story in 90 Seconds

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
8. **Part IV (graph cosmology v1–v7):** d_s ∝ N (not a fixed dimension);
   dark energy model was circular.
9. **Part V (rescue attempt, v1–v8):** Fixed graph construction (k-NN),
   fixed d_s measurement (adaptive t-range, benchmarks), measured
   correlation dimension. **D_corr(E₆) = 3.02 ≈ D_corr(T³) = 3.00.**
   The orbit is quasi-3D. d_s ≈ 4 at k=20 is what any 3D structure
   produces — it identifies the orbit as 3D, not as 4D spacetime.

---

## Recommended Reading Order

| # | Document | What you learn |
|---|----------|----------------|
| 1 | [Part I — Main Paper](paper/monostring_paper_en.md) | The full story of falsification (v0–v7) |
| 2 | [Part II — Gauge & Causal Sets](paper/monostring_part2_gauge_causal.md) | Gauge Higgs search + causal set exploration |
| 3 | [Part III — Spectral Dimension](paper/monostring_part3_spectral.md) | Weyl law, algebra comparison, d_s reduction |
| 4 | [Part IV — Independent Verification](paper/monostring_part4_independent_verification.md) | Graph cosmology v1–v7, d_s(N) test, ANOVA |
| 5 | [Part V — Resolution](paper/monostring_part5_resolution.md) | D_corr(E₆)≈3, d_s≈4 is 3D k-NN effect, final verdict |
| 6 | [Gauge Higgs Paper](paper/monostring_gauge_higgs.md) | Detailed gauge analysis (pre-falsification snapshot) |
| 7 | [Philosophical Foundations](paper/monostring_philosophy_en.md) | Speculative ontological context (optional) |

---

## Repository Structure

```
monostring-hypothesis/
├── paper/
│   ├── monostring_paper_en.md
│   ├── monostring_part2_gauge_causal.md
│   ├── monostring_part3_spectral.md
│   ├── monostring_part4_independent_verification.md
│   ├── monostring_part5_resolution.md           ← NEW
│   ├── monostring_gauge_higgs.md
│   ├── monostring_philosophy_en.md
│   ├── monostring_philosophy_ru.md
│   └── draft_v0_raw_ru.md
│
├── scripts/
│   ├── part1/                    # v0–v7 (Lyapunov, symplectic)
│   ├── part2/                    # Gauge Higgs, causal sets
│   ├── part3/                    # Spectral dimension (Weyl)
│   ├── part4/                    # Graph cosmology v1–v7
│   └── part5/                    ← NEW
│       ├── part4plus_v1.py       # Dark energy (circular logic found)
│       ├── part4plus_v2.py       # k-NN attempt (broken d_s)
│       ├── part4plus_v3.py       # Normalized Laplacian + benchmarks
│       ├── part4plus_v4.py       # k-scan: k*=20 for E₆
│       ├── part4plus_v5.py       # D_corr≈2.9, d_eff≈3.5
│       ├── part4plus_v6.py       # Manifold comparison (buggy T⁴)
│       ├── part4plus_v7.py       # Correct manifold distances
│       └── part4plus_v8.py       # Final: D_corr(E₆)=3.02
│
├── figures/
│   ├── part1/
│   ├── part2/
│   ├── part3/
│   ├── part4/
│   └── part5/                    ← NEW
│       ├── benchmarks_v3.png
│       ├── chain_diagnostic.png
│       ├── chain_collapse_v5.png
│       ├── part4plus_v3.png
│       ├── part4plus_v4.png
│       ├── part4plus_v5.png
│       ├── part4plus_v6.png
│       ├── part4plus_v7_final.png
│       └── part4plus_v8_FINAL.png
│
├── results/
│   ├── part1/
│   ├── part2/
│   ├── part3/
│   ├── part4/
│   └── part5/                    ← NEW
│       ├── dcorr_precision.csv
│       ├── knn_kscan.csv
│       ├── manifold_comparison.csv
│       └── final_verdict.md
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Key Results by Part

### Part I: Dimensional Reduction — Falsified

E₆-coupled standard map with Coxeter frequencies at κ = 0.25
produces D_corr = 4.025 ± 0.040. However, the symplectic
(Hamiltonian) version gives D_KY = 2r identically for all
Lie algebras. The dimensional reduction was an artifact of
dissipative dynamics.

**Consolation prize:** Universal D_KY ≈ 4 plateau in dissipative
maps across all 13/13 tested Lie algebras (ranks 4–8).

📄 [Full paper](paper/monostring_paper_en.md)

### Part II: Gauge Higgs + Causal Sets — Falsified

Edge variance ratio = 12.5 between synchronized and unsynchronized
gauge directions. But null model gives ratio = 22.2.

**Surviving:** Kuramoto transition (2+4 anisotropic breaking),
3 Goldstone modes.

📄 [Full paper](paper/monostring_part2_gauge_causal.md)

### Part III: Spectral Dimension — Real Effect, Not d_s = 4

E₆ synchronization reduces spectral dimension by 37–51% vs null.
D₆ = SO(12) gives d_s = 3.92 — closest to 4.0. But d_s = 4.0
is excluded by 95% CI at all configurations.

📄 [Full paper](paper/monostring_part3_spectral.md)

### Part IV: Independent Verification — d_s Depends on N

Seven iterations of graph cosmology experiments. Key finding:
d_s scales linearly with graph size N in chain graphs.
Dark energy model is circular.

📄 [Full paper](paper/monostring_part4_independent_verification.md)

### Part V: Resolution — D_corr(E₆) ≈ 3, Not 4 ← NEW

Eight versions of rescue experiments (Part IV+ v1–v8).
Central result: the E₆ Coxeter orbit on T⁶ has correlation
dimension D_corr = 3.021 ± 0.005 — statistically
indistinguishable from a 3D flat torus (D_corr = 2.996).

d_s ≈ 4 at k=20 is a k-NN graph effect: any 3D structure
at this connectivity produces d_s_cal ≈ 4.2. The "4D"
result identifies the orbit as **3D**, not as 4D spacetime.

| Method | E₆ | T³ (reference) | T⁴ | T⁶ / null |
|--------|-----|----------------|-----|-----------|
| D_corr | 3.021 | 2.996 | 3.930 | 5.453 |
| d_s_cal (k=20) | 4.32 | 4.22 | 1.09 | 1.08 |
| d_s/D_corr | 1.42 | 1.40 | 0.28 | 0.20 |

📄 [Full paper](paper/monostring_part5_resolution.md)

---

## Complete Scorecard

### Confirmed

| Finding | Part | Evidence |
|---------|------|----------|
| Kuramoto transition T_c ≈ 1.4 (2+4 anisotropic) | II | 20+ runs, null control |
| Spectral dimension reduction 37–51% vs null | III, IV | Two methods |
| ω dominates K for graph topology | IV | ANOVA: 66% vs 3% |
| Universal D ≈ 4 plateau (dissipative maps) | I | 13/13 algebras |
| GUE spectral statistics | III | ⟨r⟩ = 0.529 |
| Heat-kernel benchmarks validated | V | Path→1.07, Grid→2.03/3.00 |
| **D_corr(E₆) ≈ 3.02 (quasi-3D orbit)** | **V** | **N=1000, 15 seeds** |

### Falsified

| Claim | Part | How |
|-------|------|-----|
| 6D → 4D via Lyapunov | I | Symplectic: D_KY = 2r always |
| E₆ uniqueness for D ≈ 4 | I, IV, V | All rank-6 algebras D_corr ≈ 3 |
| Gauge Higgs mechanism | II | Null ratio > E₆ ratio |
| Yukawa mechanism | II | 6 definitions anti-correlate |
| Bell test validity | I | Null also violates |
| d_s = 4.0 as fixed dimension | IV | d_s ∝ N (chain) |
| d_s(k-NN) measures manifold dim | V | T³→4.2, T⁴→1.1 at k=20 |
| **d_s ≈ 4 → emergent 4D spacetime** | **V** | **It identifies 3D structures** |
| Dark energy = graph geometry | IV, V | λ(t) circular; E₆ irrelevant p=0.90 |

### Open

| Direction | Status | Key question |
|-----------|--------|-------------|
| Why D_corr ≈ 3 for all rank-6 orbits? | Observed | KAM theory |
| Analytic derivation of D_corr from ω | Not attempted | Coxeter + quasi-periodic flows |
| Quantum walks → Dirac equation | Not implemented | Unitarity avoids dissipation |
| Number-theoretic resonances | Observed (IV, V) | Diophantine approximation |
| Universal D_KY ≈ 4: why 4 specifically? | Confirmed, unexplained | Unknown |

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
python scripts/part1/v7_claude_symplectic.py
```

**Part V (final result — D_corr of E₆ orbit):**
```bash
python scripts/part5/part4plus_v8.py
```
Expected runtime: ~70 minutes.

### Full Reproduction

| Part | Scripts | Runtime |
|------|---------|---------|
| I | v0–v7 | ~4–6 hours |
| II | higgs_v1–v9, gauge_v1–v3, causal_v1–v4 | ~2–3 hours |
| III | qwalk_v1–v4 | ~1–2 hours |
| IV | graph_cosmology_v1–v7 | ~1–2 hours |
| V | part4plus_v1–v8 | ~8–10 hours total |

---

## Citation

```bibtex
@misc{lebedev2025monostring,
  author       = {Lebedev, Igor},
  title        = {The Monostring Hypothesis: Eight Computational Experiments
                  That Killed One Path to Emergent Spacetime ---
                  and Closed Three Others},
  year         = {2025},
  publisher    = {GitHub / Zenodo},
  url          = {https://github.com/LebedevIV/monostring-hypothesis},
  doi          = {10.5281/zenodo.18886047}
}
```

---

## License

- Paper and documentation: CC-BY 4.0
- Code: MIT License

## Acknowledgments

This research was conducted as an exercise in AI-assisted
theoretical physics. The human author provided the hypothesis
and direction; AI collaborators provided implementations,
critical analysis, and falsifying tests.

**The most important contribution of the AI collaborators was
not building the theory — it was designing the experiments
that destroyed it, and then destroying the replacements too.**

**AI collaborators:**
- Google Gemini 3.1 Pro — initial mathematical implementation (Part I, v0)
- Anthropic Claude — critical analysis, falsifying tests (Parts I–V)
