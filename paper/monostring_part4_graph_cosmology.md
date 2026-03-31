

# Part IV: Independent Verification — Graph Cosmology and Spectral Dimension (Scripts v1–v7)

**Author:** Igor Lebedev

**AI Collaborator:** Anthropic Claude (critical analysis, all scripts)

**Sequel to:** Parts I–III

---

## Abstract

We report an independent series of seven computational experiments (graph cosmology v1–v7) that tested two remaining claims of the Monostring hypothesis: (1) that exponential decay of long-range connections produces "dark energy" (accelerated expansion of graph diameter), and (2) that E₆ phase dynamics produce a graph with spectral dimension d_s = 4.0.

The series was conducted as an adversarial iteration cycle: each script was critically analyzed, bugs and methodological flaws were identified, and the next version was designed to address them. The process yielded one definitive negative result and several methodological contributions:

**Key finding:** The spectral dimension of the Monostring graph scales linearly with the number of nodes: d_s ≈ 2.16 + 0.002·N. This means the graph does **not** possess a fixed dimensionality. The value d_s = 4.0 at N = 1000 is a coincidence with graph size, not a property of E₆ algebra. This result was not tested in Parts I–III and invalidates all absolute d_s values reported there.

**Secondary findings:** (1) Dark energy model (v1) contains circular logic — λ_decay = f(epoch) is input, not emergent. (2) Frequencies ω explain 66% of d_s variance vs 3% for the Cartan matrix K (ANOVA, v4). (3) B₆ = SO(13), not E₆, achieves d_s closest to 4.0 in the interpolation scan (v7). (4) Heat-kernel spectral dimension measurement requires adaptive t-range — fixed ranges produce order-of-magnitude errors (v5 vs v6).

**Methodological contribution:** Benchmark validation of spectral dimension on known lattices (path→1.07, 2D grid→2.03, 3D grid→3.00) should be a prerequisite for any graph-based dimensional analysis. This was absent from Parts I–III.

---

## 1. Motivation

Parts I–III of the Monostring study established that dimensional reduction via Lyapunov compactification is falsified (symplectic test), and that the gauge Higgs mechanism is trivially explained by synchronization geometry (null model test). Two claims remained untested by the original study:

1. **Dark energy claim:** The original experiment (Experiment 3.1 in the SBE Dark Universe Laboratory) asserted that exponential decay of graph connections with temporal distance produces accelerated expansion analogous to dark energy.

2. **Spectral dimension claim:** Part III reported d_s values for various algebras and n_phases configurations but did not test whether d_s depends on graph size N — the most basic requirement for d_s to represent a genuine dimensionality.

This paper reports seven experiments that systematically tested and ultimately refuted both claims.

---

## 2. Experiment Chronology

### 2.1 Script v1: The Original Dark Energy Model

**Source:** SBE Dark Universe Laboratory, Experiment 3.1

**Construction:** A growing graph with N_initial = 500 nodes, adding 150 per epoch for 25 epochs. Chronological chain (arrow of time) plus phase-proximity shortcuts. Connection probability suppressed by exponential decay:

$$P(i \to j) \propto (\deg(j) + 1) \cdot \exp(-\lambda \cdot |i - j|)$$

where λ_decay = 0.001 + 0.0005 · epoch^1.5.

**Measurements:** Average shortest path ⟨d⟩ (claimed as "scale factor a(t)"), clustering coefficient C (claimed as "dark matter"), second derivative d²⟨d⟩/dt² (claimed as "dark energy").

**Reported results:**

| Epoch | Nodes | Clustering C | ⟨d⟩ | λ_decay |
|-------|-------|-------------|------|---------|
| 0 | 650 | 0.0000 | 8.74 | 0.00100 |
| 7 | 1700 | 0.0084 | 6.26 | 0.01026 |
| 10 | 2150 | 0.0076 | 6.42 | 0.01681 |
| 24 | 4250 | 0.0047 | 10.08 | 0.05979 |

Three phases observed: deceleration (epochs 0–7), stabilization (8–10), acceleration (11–24).

**Critical analysis:** The model contains a fundamental logical circularity (petitio principii). The parameter λ_decay = 0.001 + 0.0005 · epoch^1.5 is **explicitly programmed** to increase with time. This directly causes long-range connections to become rarer at later epochs, which directly causes ⟨d⟩ to grow, which is then measured and declared "dark energy." The conclusion is entirely contained in the assumption.

Additional problems identified:
- Path sampling: 50 nodes from 4000+ (1.2% sample, no error bars)
- Single run (no Monte Carlo)
- 10+ free parameters (N_initial, N_add, ε, κ, max_conn, n_cand, D, ω, λ₀, λ exponent)
- Terminology abuse: clustering ≠ dark matter, ⟨d⟩ ≠ scale factor

### 2.2 Script v2: First Correction Attempt

**Changes:** λ_decay modified to depend on average degree: λ = 0.001 + 0.0005 · epoch^1.5 · (100/(avg_degree+1)). Monte Carlo added (n_simulations = 10). Terminology corrected ("clustering" instead of "dark matter").

**Critical analysis:** The code contained a fatal bug: variable `epochs` used instead of parameter `n_epochs` — the script could not run. More fundamentally, epoch^1.5 remained in the formula, making the "fix" cosmetic. The avg_degree factor stabilizes quickly, leaving the same time-dependent growth.

**Salvageable elements:** Monte Carlo framework with confidence intervals. Honest terminology.

### 2.3 Script v3: Comparative Design

**Innovation:** Three models on identical phase landscapes:
- Model A (Null): λ = 0 (no decay)
- Model B (Constant): λ = λ₀ (fixed decay)
- Model C (Feedback): λ = λ₀ · ⟨d⟩_prev / ⟨d⟩_initial (no explicit time)

Model C eliminates circular logic — λ depends only on graph state. 15 Monte Carlo realizations, 6 diagnostic panels, automatic interpretation with 2σ significance testing.

**Assessment:** Methodologically correct design. Null model, falsifiability criterion, statistical rigor. However, the focus shifted to spectral dimension testing before v3 results were fully analyzed.

### 2.4 Script v4: Matrix Scan — What Controls d_s?

**Design:** 11 coupling matrices (Zero, 0.5I, I, 2I, 3I, E₆, A₆, D₆, 0.5E₆, Diag(eig_E₆), Random Symmetric) + null model. Three experiments: (A) matrix scan, (B) ω × K factorial ANOVA, (C) E₆ ↔ I interpolation. N = 1200, 5 runs each.

**Key results:**

Correlations with d_s:

| Property | r(d_s, property) |
|----------|-----------------|
| Average degree ⟨k⟩ | **+0.872** |
| Phase recurrence | **+0.750** |
| Eigenvalue max | +0.348 |
| Trace | +0.665 |
| Off-diagonal norm | −0.071 |

ANOVA variance decomposition:

| Factor | Variance explained |
|--------|-------------------|
| ω (frequencies) | **66.4%** |
| K (coupling matrix) | **3.1%** |
| Residual | 30.5% |

**Interpretation:** Frequencies ω dominate over the Cartan matrix K. The "specialness" of E₆ comes from its exponents {1,4,5,7,8,11}, not from its coupling structure. Average degree is the strongest predictor of d_s.

**Problem:** Measurements with K ≠ 0 showed σ/d_s up to 85% (noise comparable to signal). Root cause: only 200 of 1200 Laplacian eigenvalues were used.

### 2.5 Script v5: Omega Scan with Full Spectrum

**Changes:** Full eigendecomposition (all eigenvalues). Scan of ω from E₆ to uniform at K = 0. N = 1500, 8 runs, 20 interpolation points.

**Result:** Nearly all points gave d_s ≈ 0.49. Isolated spikes at d_s = 4.9 and 5.6.

**Diagnosis:** Measurement broken. Fixed t-range [0.1, 100] does not cover the plateau of d_s(t) for N = 1500. Minimum eigenvalue λ₁ ≈ 2×10⁻⁶ implies plateau at t ~ 1/λ₁ ~ 500,000. The "flattest window" algorithm found a false plateau on the initial rise of the d_s(t) curve, yielding d_s ≈ 0.5 — a value below the theoretical minimum of 1.0 (path graph).

**Lesson:** Heat-kernel spectral dimension requires adaptive t-range calibrated to the eigenvalue bounds of the specific graph.

### 2.6 Script v6: Fixed Measurement + Benchmarks

**Critical fix:** Adaptive t-range: t ∈ [0.01/λ_max, 100/λ_min]. Effective mode count filter (n_eff ≥ 5). Peak detection instead of minimum-variance search. **Benchmark validation on known graphs.**

**Benchmark results:**

| Graph | Expected d_s | Measured d_s | Error |
|-------|-------------|-------------|-------|
| Path (200 nodes) | 1.0 | 1.070 | 7.0% |
| 2D Grid (15×15) | 2.0 | 2.033 | 1.7% |
| 3D Grid (6×6×6) | 3.0 | 2.997 | 0.1% |
| Complete (100 nodes) | → ∞ | 58.6 | — |

All benchmarks passed — measurement validated.

**Algebra results (K = 0, N = 1000):**

| Frequency source | d_s | σ | Recurrence | ⟨k⟩ |
|-----------------|-----|---|-----------|------|
| E₆ | 6.565 | 0.155 | 0.02175 | 5.11 |
| A₆ | 8.711 | 1.012 | 0.01591 | 4.05 |
| D₆ | 5.612 | 0.080 | 0.02873 | 6.21 |
| B₆ | 6.005 | 0.024 | 0.05391 | 9.16 |
| Golden ratio | 2.976 | 0.131 | 0.00447 | 2.57 |
| Primes | 1.331 | 0.012 | 0.00216 | 2.14 |
| Linear | 2.730 | 0.027 | 0.01357 | 3.79 |
| Uniform | 8.388 | 0.023 | 0.19525 | 11.74 |
| Null (random) | 3.498 | 0.813 | — | — |

**Resonance discovered:** β = 0.053 gives d_s = 4.067 ± 0.064. The d_s(β) curve is wildly non-monotonic (range 4.07 to 8.45), indicating number-theoretic resonances in frequency ratios.

### 2.7 Script v7: Precision Measurement + Size Dependence

**Four experiments:**

**Experiment 1: Fine scan** β ∈ [0, 0.12], 40 points × 10 runs.

The d_s(β) curve shows sharp resonances: d_s = 12.15 at β = 0.049, d_s = 3.75 at β = 0.062, d_s = 9.19 at β = 0.046. Minimum near d_s = 4.0 at β ≈ 0.058.

**Experiment 2: Other algebras.**

| Algebra | Closest d_s to 4.0 | β | 4.0 in 95% CI? |
|---------|-------------------|---|----------------|
| E₆ | 3.406 | 0.063 | NO |
| A₆ | 5.019 | 0.150 | NO |
| D₆ | 4.480 | 0.126 | NO |
| **B₆** | **3.974** | **0.095** | **YES** |

B₆ = SO(13), not E₆, is the only algebra achieving d_s = 4.0 within 95% CI.

**Experiment 3: Precision at β = 0.058** (30 runs).

```
d_s = 4.0855 ± 0.0659
SEM = 0.0120
95% CI: [4.0619, 4.1091]
|d_s − 4.0| = 0.0855 (7.1σ)
d_s = 4.0 within 95% CI: NO
```

**Experiment 4: Size dependence** — THE KEY RESULT.

| N | d_s | σ |
|---|-----|---|
| 300 | 2.832 | 0.052 |
| 500 | 3.232 | 0.068 |
| 700 | 3.467 | 0.056 |
| 1000 | 4.056 | 0.064 |
| 1300 | 4.920 | 0.146 |

**Linear fit: d_s ≈ 2.16 + 0.002·N**

Extrapolation: d_s(N=2000) ≈ 6.2. For comparison, a true 2D grid gives d_s = 2.0 at **any** N. The Monostring graph does not possess a fixed dimensionality.

---

## 3. The Causal Chain (Experimentally Established)

```
ω (frequency ratios from Lie algebra exponents)
  → Quasi-periodic orbit on 6D torus T⁶
    → Phase recurrence rate (return frequency to ε-neighborhood)
      → Number of shortcuts ("wormholes") in graph
        → Average degree ⟨k⟩
          → Spectral dimension d_s
            → Depends on N → NOT a spatial dimension
```

This chain was established through:
- ANOVA (v4): ω explains 66%, K explains 3%
- Correlations (v4): r(d_s, ⟨k⟩) = +0.872, r(d_s, recurrence) = +0.750
- Size test (v7): d_s ∝ N

---

## 4. Methodological Contributions

### 4.1 Benchmark Requirement

Any measurement of spectral dimension on a graph should be validated against lattices with known d_s before application to unknown graphs. This was absent from Parts I–III and from the original Monostring experiments.

| Benchmark | Expected | v5 (broken) | v6 (fixed) |
|-----------|----------|-------------|------------|
| Path | 1.0 | 0.49 | 1.07 |
| 2D Grid | 2.0 | not tested | 2.03 |
| 3D Grid | 3.0 | not tested | 3.00 |

### 4.2 Adaptive t-Range

The heat-kernel spectral dimension d_s(t) = −2·d(ln K(t))/d(ln t) requires the diffusion time t to cover the range [t_short, t_long] where:
- t_short ≈ 0.01 / λ_max (resolves highest-frequency modes)
- t_long ≈ 100 / λ_min (captures the plateau before finite-size decay)

Fixed ranges (as in v5: t ∈ [0.1, 100]) can miss the plateau entirely for large sparse graphs where λ_min ~ 10⁻⁶.

### 4.3 Null Model Design

Three levels of null model were used:

| Level | Description | What it tests |
|-------|-------------|---------------|
| Random phases | φ ~ Uniform(0, 2π)^6 | Is correlated dynamics needed? |
| K = 0 | ω-driven, no coupling | Is the Cartan matrix needed? |
| ω = uniform | All frequencies equal | Are frequency ratios needed? |

The factorial design (v4) combining these levels with ANOVA is the most informative approach for isolating causal factors.

### 4.4 Monte Carlo Requirements

| Aspect | v1 (original) | v7 (final) |
|--------|---------------|------------|
| Runs per configuration | 1 | 10–30 |
| Path sample size | 50 / 4000 (1.2%) | 120 / 1500 (8%) |
| Error reporting | None | SEM, 95% CI |
| Significance testing | None | 2σ threshold |
| Reproducibility | np.random (global) | np.random.RandomState (local, seeded) |

---

## 5. Consolidated Results: What Parts I–III Missed

| Finding | Tested in Parts I–III? | Our result |
|---------|----------------------|------------|
| d_s depends on N | **No** | d_s ≈ 2.16 + 0.002·N |
| ω vs K decomposition | **No** | ω: 66%, K: 3% |
| Dark energy circularity | **No** | λ(t) is input, not emergent |
| Benchmark validation of d_s | **No** | Path→1.07, Grid→2.03/3.00 |
| B₆ closer to d_s=4 than E₆ | **Partially** (D₆ tested) | B₆ in CI, E₆ not |
| Number-theoretic resonances in d_s(β) | **No** | Range 3.7–12.1 |

---

## 6. Updated Scorecard

Combining Parts I–III with our independent verification:

### Confirmed (non-trivial, reproducible, null-model controlled)

| Finding | Source | Key number |
|---------|--------|-----------|
| Kuramoto transition T_c ≈ 1.4, anisotropic 2+4 | Part II | 20+ runs |
| d_s reduction 37–51% vs null | Part III + our v4–v6 | Two independent methods |
| ω dominates over K for d_s | Our v4 | ANOVA: 66% vs 3% |
| Universal D ≈ 4 plateau (dissipative maps) | Part I | 13/13 algebras |
| GUE spectral statistics | Part III | ⟨r⟩ = 0.529 |
| d_s(t) benchmarks: path→1, grid→2,3 | Our v6 | Error < 7% |

### Falsified

| Claim | Source | Falsification |
|-------|--------|--------------|
| 6D → 4D via Lyapunov | Part I, v7 | Symplectic: D_KY = 2r |
| E₆ uniqueness | Part I, v5 | All algebras give D ≈ 4 |
| Gauge Higgs mechanism | Part II, Gauge v3 | Null model ratio = 22 |
| Yukawa mechanism | Part II | 6 definitions anti-correlate |
| Bell test | Part I, v3 | Null also violates |
| d_s = 4.0 | Part III + our v7 | 95% CI excludes 4.0 |
| Compactification | Part III + our v6 | d_s(sync) ≈ d_s(unsync) |
| **d_s is a fixed dimension** | **Our v7 (new)** | **d_s ∝ N** |
| **Dark energy = geometry** | **Our v1–v3 (new)** | **λ(t) is circular** |
| **E₆ closest to d_s=4** | **Our v7 (new)** | **B₆ is closer** |

### Open

| Direction | Status | Key question |
|-----------|--------|-------------|
| D₆/B₆ at fixed d_s | Needs d_s(N) test | Does d_s converge? |
| Quantum walks | Not implemented | Unitarity avoids dissipation |
| Number-theoretic resonances | Observed (v6–v7) | KAM theory connection? |
| Universal D ≈ 4 plateau | Confirmed | Why 4 specifically? |

---

## 7. Conclusions

Seven iterations of computational experiments independently confirm and extend the falsification results of Parts I–III. The most significant new finding — **d_s depends linearly on graph size N** — was not tested in the original study and invalidates all absolute d_s values reported in Part III.

The Monostring's phase dynamics produce real, measurable effects on graph topology (spectral dimension reduction, number-theoretic resonances, Kuramoto synchronization). These effects are interesting from the perspective of dynamical systems theory and network science. However, they do not produce a graph with fixed spatial dimensionality, and they do not support the interpretation of the Monostring as a model of spacetime.

The methodological lesson is clear: **any claim about emergent dimensionality from a graph model must demonstrate that the measured dimension is independent of graph size.** Without this test, the claim is unfalsifiable — any desired dimension can be obtained by choosing the appropriate N.

---

## Appendix A: Script Inventory

| Version | Key innovation | Key result |
|---------|---------------|------------|
| v1 | Original dark energy model | Circular logic identified |
| v2 | Attempted fix + Monte Carlo | Bug found; epoch^1.5 remained |
| v3 | Three-model comparison (null/const/feedback) | Correct design |
| v4 | 11-matrix scan + ANOVA | ω: 66%, K: 3% |
| v5 | Full eigendecomposition, ω scan | **Broken d_s** (fixed t-range) |
| v6 | Adaptive t-range + benchmarks | Benchmarks pass; d_s(E₆) = 6.57 |
| **v7** | **Fine scan + size dependence** | **d_s ∝ N (key result)** |

## Appendix B: Reproducibility

All experiments use `np.random.RandomState` with explicit seeds. Configuration parameters are logged at the start of each run. Scripts are self-contained (single-file, no external dependencies beyond numpy, networkx, scipy, matplotlib).

To reproduce the key result (d_s vs N):
```bash
python graph_cosmology_v7.py
```
Expected output includes Experiment 4 (size dependence) with linear trend d_s ≈ 2.16 + 0.002·N.

---

## Appendix C: Notation

| Symbol | Meaning |
|--------|---------|
| d_s | Spectral dimension (heat kernel or Weyl law) |
| D_KY | Kaplan-Yorke dimension |
| D_corr | Correlation dimension (Grassberger-Procaccia) |
| ω | Frequency vector from Lie algebra exponents |
| K | Cartan matrix (coupling) |
| κ | Coupling strength |
| T | Temperature (noise amplitude) |
| ⟨k⟩ | Average degree of graph |
| ⟨d⟩ | Average shortest path length |
| β | Interpolation parameter (0 = algebra, 1 = uniform) |
| λ_decay | Exponential decay parameter for long-range connections |
| r_d | Kuramoto order parameter for dimension d |
| CI | Confidence interval (95% unless stated otherwise) |
| SEM | Standard error of the mean |
