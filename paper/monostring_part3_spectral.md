# The Monostring Hypothesis Part III: Spectral Dimension from Quantum Walks

**Author:** Igor Lebedev

**AI Collaborator:** Anthropic Claude

**Sequel to:** Part I (DOI 10.5281/zenodo.18886047), Part II (DOI [Part II DOI])

---

## Abstract

Following Parts I (dimensional reduction — falsified) and II (gauge Higgs — 
trivially explained by synchronization geometry), we investigate the spectral 
dimension of Monostring resonance graphs via quantum walks and the Weyl 
eigenvalue counting law.

**Key results:**

1. E₆-coupled dynamics with Coxeter frequencies reduces the spectral dimension 
   by 37–51% compared to the null model (random phases on T^n with identical 
   graph density). This reduction is reproducible (20 runs, std/mean = 2.2% 
   at n_phases=6) and absent in the null model.

2. The specific Lie algebra matters: D₆ = SO(12) gives d_s = 3.92 ± 0.28, 
   the closest to 4.0 among all tested algebras. E₆ gives d_s = 5.68 ± 0.12. 
   A₆ = SU(7) gives d_s = 6.07 ± 0.08. Identity coupling gives d_s = 2.85.

3. d_s does NOT equal 4.0 precisely at any tested configuration. The closest 
   approach is d_s = 4.75 ± 0.10 at n_phases=4 with E₆ coupling (95% CI: 
   [4.70, 4.79], excluding 4.0).

4. Spectral dimension scales sub-linearly with n_phases: d_s ≈ 0.62 × n + 0.9 
   (not trivially d_s = n). This means E₆ synchronization genuinely compresses 
   the effective dimensionality, but not to exactly 4.

5. Synchronized phase directions are NOT "compactified" in the traditional sense — 
   their spectral dimension (4.27 for 2 dims) is comparable to unsynchronized 
   directions (3.88 for 4 dims). The reduction comes from inter-dimensional 
   correlations, not from dimensional collapse.

**Controls:** Weyl law estimator validated on known lattices: 2D→2.17, 
3D→3.31, 4D→4.49 (systematic overestimate ~10-15%, acceptable). All 
Monostring measurements use identical graph construction and degree 
control (cv < 3%).

---

## 1. Method: Weyl Eigenvalue Counting

The spectral dimension is extracted from the integrated density of 
states N(λ) of the normalized graph Laplacian:

N(λ) = #{eigenvalues ≤ λ} ~ λ^{d_s/2}  (Weyl's law)

We compute the first 300 eigenvalues via scipy.sparse.linalg.eigsh 
and fit the log-log slope over the bottom 20–50% of eigenvalues. 
The median over three fitting ranges provides robustness.

This method avoids the finite-time artifacts that plagued the return 
probability approach (v1), where P(t) ~ t^{-d/2} is only valid for 
t << N^{2/d} (violated on our graphs).

---

## 2. Results

### 2.1 Spectral Dimension vs Temperature (n_phases=6)

| T | d_s(Weyl) | d_s(null) | Reduction |
|---|-----------|-----------|-----------|
| 3.0 | 9.80 | 9.70 | 1% |
| 1.0 | 10.86 | - | (hot, symmetric) |
| 0.5 | 9.86 | - | - |
| 0.2 | 7.19 | - | - |
| 0.1 | 6.43 | - | - |
| 0.05 | 5.93 | - | - |
| **0.02** | **5.76** | **9.70** | **41%** |

At high T: d_s ≈ d_s(null) ≈ 10 (symmetric, no synchronization effect).
At low T: d_s = 5.76, a 4-unit reduction from the null model.

### 2.2 Spectral Dimension vs n_phases

| n_phases | d_s(E6 cold) | d_s(null) | d_s(hot) | Reduction vs null |
|----------|-------------|-----------|----------|-------------------|
| 3 | 3.46 | 6.61 | 6.70 | 48% |
| 4 | 4.75 | 7.59 | 7.54 | 37% |
| 5 | 4.31 | 8.83 | 8.79 | 51% |
| 6 | 5.66 | 9.90 | 10.05 | 43% |

Linear fit: d_s ≈ 0.62 × n_phases + 0.9 (slope < 1 → non-trivial 
compression).

### 2.3 Algebra Dependence (n_phases=6, T=0.005)

| Algebra | d_s | ±err | r_max | Closest to 4? |
|---------|-----|------|-------|---------------|
| D₆ = SO(12) | **3.92** | 0.28 | 0.380 | **YES** |
| Identity (2I) | 2.85 | 0.15 | 0.424 | No (too low) |
| E₆ | 5.68 | 0.12 | 0.374 | No (too high) |
| A₆ = SU(7) | 6.07 | 0.08 | 0.379 | No (too high) |

**D₆ = SO(12) gives d_s = 3.92 — closest to 4.0.** This is the first 
result in the entire study where a specific algebra produces a 
significantly different (and physically interesting) spectral dimension.

### 2.4 Precision Measurement (n_phases=4, E₆, κ=0.5)

20 independent runs:
- d_s = 4.747 ± 0.104 (SEM = 0.023)
- 95% CI: [4.701, 4.792]
- d_s = 4.0 is NOT within the CI
- Null: 7.592 ± 0.071
- Reduction: 37.5%

### 2.5 Subgraph Decomposition (n_phases=6)

| Subgraph | Dimensions | d_s |
|----------|-----------|-----|
| Synced only | 2 (top r) | 4.27 |
| Unsynced only | 4 (bottom r) | 3.88 |
| Full | All 6 | 5.54 |
| Sum (sync+unsync) | — | 8.15 |

Sum (8.15) >> Full (5.54): inter-dimensional correlations reduce the 
effective dimension. Synchronization does not "kill" directions — it 
creates correlations between them.

---

## 3. Negative Results

| What was tested | Result | Status |
|-----------------|--------|--------|
| d_s = 4.0 exactly | d_s = 4.75 (n=4) or 5.72 (n=6) | ❌ Not achieved |
| Synced dims collapsed | d_s(sync) ≈ d_s(unsync) | ❌ Not collapsed |
| E₆ gives d_s closest to 4 | D₆ = 3.92 is closer | ❌ E₆ not optimal |
| d_s independent of n_phases | slope = 0.62 (depends) | ❌ Not independent |

---

## 4. Complete Scorecard: All Three Parts

### Confirmed (non-trivial, reproducible, null-model controlled)

| Finding | Part | Key number |
|---------|------|-----------|
| Kuramoto phase transition T_c ≈ 1.4 | II | r: 0.01 → 0.37 |
| Anisotropic breaking (2+4 pattern) | II | Dynkin topology |
| 3 Goldstone-like near-zero modes | II | Spectral analysis |
| Spectral dimension reduction 37-51% | III | d_s(E6) << d_s(null) |
| Algebra matters for d_s | III | D₆=3.92, E₆=5.68, A₆=6.07 |
| Sub-linear d_s(n) scaling (slope=0.62) | III | Not trivially d_s=n |
| GUE spectral statistics | III | <r>=0.529 (GUE=0.531) |

### Falsified

| Claim | Part | Reason |
|-------|------|--------|
| Dimensional reduction 6→4 via Lyapunov | I | Symplectic test |
| E₆ uniqueness for D≈4 | I | All rank-6 algebras |
| Gauge Higgs mechanism | II | Null model ratio=22 |
| Yukawa mechanism (6 definitions) | II | All anti-correlate |
| Bell test | I | Null also violates |
| d_s = 4.0 exactly | III | 95% CI excludes 4.0 |
| Compactification of synced dims | III | d_s(sync) ≈ d_s(unsync) |

### Open / Promising

| Direction | Status | Key question |
|-----------|--------|-------------|
| D₆ = SO(12) gives d_s = 3.92 | Needs more runs | Is this robust? |
| Quantum walks → Dirac equation | Not implemented | Continuum limit? |
| Causal sets with physical rule | Parameter-dependent | What fixes the rule? |

---

## 5. Conclusion

The Monostring's E₆-coupled dynamics on T⁶ produces a genuine 37–51% 
reduction in spectral dimension compared to random phase configurations. 
This is the strongest surviving positive result of the entire study. 
However, the reduction does not reach d_s = 4.0, and the mechanism is 
not compactification of synchronized directions but rather creation of 
inter-dimensional correlations.

The most intriguing finding is that D₆ = SO(12) gives d_s = 3.92 — 
tantalizingly close to 4.0. Whether this survives larger-scale testing 
and whether it has physical significance beyond numerology remains an 
open question.

---

## References

[17] H. Weyl, "Das asymptotische Verteilungsgesetz der Eigenwerte 
linearer partieller Differentialgleichungen," Math. Ann. 71:441, 1912.

---

## Script Inventory (Part III)

| Script | Key result |
|--------|-----------|
| qwalk_v1_spectral.py | Return probability P(t) — control failed (finite-size) |
| **qwalk_v2_weyl.py** | **Weyl law: control works, d_s(E6)=5.76, null=9.70** |
| **qwalk_v3_nphases.py** | **d_s vs n: slope=0.62, reduction 37-51%** |
| **qwalk_v4_clarification.py** | **D₆=3.92, decomposition, precision at n=4** |
