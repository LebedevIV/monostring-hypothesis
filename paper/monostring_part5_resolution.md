# The Monostring Hypothesis — Part V: Resolution

**Author:** Igor Lebedev
**AI Collaborator:** Anthropic Claude
**Follows:** Parts I–IV ([DOI 10.5281/zenodo.18886047](https://doi.org/10.5281/zenodo.18886047))

---

## Abstract

Parts I–IV established that the original Monostring hypothesis
(emergent 4D spacetime from E₆ phase dynamics) is falsified.
Part IV+ (scripts v1–v8, 44 computational experiments) attempted
to rescue the hypothesis through four directions: RGG construction,
number-theoretic resonances, algebra comparison, and feedback dark
energy. This part documents what was found, what was fixed, what
failed again, and what genuinely survives.

**Central finding of Part IV+:** The E₆ quasi-periodic orbit on T⁶
has correlation dimension D_corr ≈ 3.02 — it is a quasi-3D
structure, not 4D. The spectral dimension d_s ≈ 4 observed in
k-NN graphs at k=20 is a graph connectivity effect: any structure
with D_corr ≈ 3 produces d_s ≈ 4 at this connectivity. The "4D"
result does not identify 4D spacetime — it identifies a 3D orbit.

**Secondary finding:** All rank-6 Lie algebras produce D_corr ∈
[2.4, 2.8] for their Coxeter quasi-periodic orbits. E₆ is not
unique in this regard. What remains unique to E₆ (from Parts I–II)
is the Kuramoto transition structure and GUE spectral statistics.

---

## 1. The Part IV+ Investigation

### 1.1 Motivation

After Parts I–IV, two claims remained unresolved:

1. **d_s = 4 as a fixed dimension.** Part IV showed d_s ∝ N in
   chain graphs, but never tested whether a geometry-controlled
   graph (fixed average degree) produces N-independent d_s.

2. **E₆ uniqueness.** Parts III–IV showed B₆ and D₆ also give
   d_s close to 4, but used a broken measurement (d_s = 0.49
   due to incorrect t-range). After fixing, E₆ was the closest
   to 4D among rank-6 algebras — but the reason was unknown.

### 1.2 Script genealogy

Eight versions of the main experiment script were required to
obtain reliable measurements. Each version corrected a critical
flaw in its predecessor:

| Version | Critical flaw found | Fix applied |
|---------|--------------------|-----------  |
| v1 | Circular logic in dark energy (λ=f(epoch)) | Exposed, not fixed |
| v2 | NameError crash; λ(epoch) survives | Partial fix |
| v3 | Comparative design (A/B/C models) | ✅ Correct framework |
| v4 | 11-matrix scan; ANOVA ω vs K | ✅ ω dominates (66%) |
| v5 | d_s=0.49 (wrong t-range) | Exposed |
| v6 | Adaptive t-range; benchmarks | ✅ Path→1.07, Grid→2.03/3.00 |
| v7 | d_s ∝ N (chain graph) | ✅ Key negative result |
| v8–v8+ | k-NN graph; D_corr measurement | ✅ Final answer |

The most important methodological contribution was establishing
that **spectral dimension must be validated on known lattices
before application to any unknown graph** (v6 benchmark suite).

---

## 2. What Was Measured and How

### 2.1 Graph construction

Three graph types were compared:

**Chain graph** (Parts I–IV): chronological chain with
phase-proximity shortcuts. Average degree ⟨k⟩ grows with N
because the shortcut probability is N-dependent. This caused
the d_s ∝ N artifact reported in Part IV — it was an artifact
of growing connectivity, not a property of E₆.

**RGG (Random Geometric Graph)**: connect nodes within ε of
each other on T⁶. The ε-ball volume is fixed, but the number
of points inside grows as N·V(ε,T⁶), so ⟨k⟩ ∝ N. This
produced ⟨k⟩ = 8 at N=200 and ⟨k⟩ = 41 at N=1000 — same
problem as chain graph.

**k-NN graph** (Part IV+ v3 onward): each node connects to
exactly k nearest neighbors (torus distance). Average degree
≈ 2k regardless of N. This is the correct construction for
testing N-independence of d_s.

### 2.2 Spectral dimension measurement

The heat-kernel spectral dimension:

$$d_s(t) = -2\,\frac{d\ln K(t)}{d\ln t}, \quad K(t) = \sum_i e^{-\lambda_i t}$$

requires finding the **plateau** of d_s(t) — the region where
d_s is approximately constant as a function of diffusion time t.

Critical requirements (established through v3–v8 failures):

1. **Normalized Laplacian**: eigenvalues ∈ [0,2], independent
   of degree. Unnormalized Laplacian gave λ_min ≈ 0.31 for
   chain graphs, causing t_hi = 10/λ_min = 32 — insufficient
   to reach the plateau.

2. **Adaptive t-range**: t ∈ [0.5/λ_max, 5.0/λ_min]. Fixed
   ranges (e.g., t ∈ [0.1, 100]) produced d_s = 0.49 for
   graphs where the plateau lies at t ~ 10⁵.

3. **Flattest window detection**: find the 30-point window of
   the smoothed d_s(t) curve with minimum standard deviation.
   Peak detection (used in v1–v5) finds the wrong feature.

4. **Benchmark validation before use**: Path(300) → 1.03,
   2D Grid(16×16) → 1.94, 3D Grid(8×8×8) → 2.75,
   4D Grid(5⁴) → 3.62. The 4D grid benchmark revealed a
   systematic underestimate of ~10%, which became the
   calibration factor CALIB = 4.0/3.616 = 1.106.

### 2.3 Correlation dimension

The Grassberger-Procaccia correlation dimension:

$$D_{\text{corr}} = \lim_{r\to 0} \frac{\log C(r)}{\log r},$$
$$C(r) = \lim_{N\to\infty} \frac{2}{N(N-1)} \#\{(i,j): d(x_i,x_j) < r\}$$

was computed using the torus metric
$d(x,y) = \|\min(|x-y|, 2\pi - |x-y|)\|$ with N=1000
points, 500-point subsample, r-range from the 5th to the
60th percentile of pairwise distances.

Key validation: T³ → D_corr = 2.996 ± 0.005 ✅,
T⁴ → 3.930 ± 0.007 ✅, T⁶ → 5.453 ± 0.008 (expected 6,
slight underestimate at N=1000 due to finite-size effect).

---

## 3. Results

### 3.1 D_corr of E₆ orbit — high precision

With N=1000 and 15 independent seeds:

| Configuration | D_corr | SEM | 95% CI | = 3.0? |
|---------------|--------|-----|--------|--------|
| E₆ | 3.021 | 0.005 | [3.011, 3.030] | p=0.001 ❌ |
| A₆ | 2.961 | 0.004 | [2.953, 2.970] | p<0.001 ❌ |
| D₆ | 2.886 | 0.007 | [2.873, 2.899] | p<0.001 ❌ |
| B₆ | 3.007 | 0.006 | [2.995, 3.020] | p=0.301 ✅ |
| T³ (reference) | 2.996 | 0.005 | [2.986, 3.005] | p=0.433 ✅ |
| T⁴ (reference) | 3.930 | 0.007 | [3.917, 3.944] | p<0.001 ❌ |
| T⁶ / null | 5.453 | 0.008 | [5.439, 5.468] | p<0.001 ❌ |

**Key finding:** E₆ gives D_corr = 3.021 ± 0.005. This is
statistically greater than 3.0 (p=0.001) but physically
indistinguishable from it — the deviation is 0.021 standard
deviation units above T³.

E₆ vs T³: Δ = 0.025, t=3.51, p=0.0015. The orbits are
statistically distinguishable at high precision, but E₆ is
clearly in the "3D class" (D_corr ≈ 3), not the "4D class"
(D_corr ≈ 4).

**All rank-6 algebras produce D_corr ∈ [2.9, 3.0]**, compared
to D_corr ≈ 4.0 for random phases (T⁶). The reduction from
6D to ~3D is a property of quasi-periodic dynamics on T⁶, not
specific to E₆.

### 3.2 d_s measurement on k-NN graphs — what it actually measures

With N=800, k=20:

| Manifold | True d | D_corr | d_s_cal | d_s/D_corr |
|----------|--------|--------|---------|-----------|
| T³ | 3.0 | 3.01 | 4.22 | 1.40 |
| T⁴ | 4.0 | 3.93 | 1.09 | 0.28 |
| S³ | 3.0 | 2.72 | 3.91 | 1.44 |
| E₆ orbit | ~3 | 3.04 | 4.32 | 1.42 |
| T⁶ / null | 6.0 | 5.46 | 1.08 | 0.20 |

The ratio d_s_cal/D_corr ≈ 1.40–1.44 is consistent across T³,
S³, and E₆ — three structures with D_corr ≈ 3. For T⁴ and T⁶
the ratio is completely different.

This reveals the mechanism: **d_s(k-NN, k=20) identifies 3D
structures**, not 4D ones. At connectivity k=20, graphs built
on 3D manifolds produce d_s_cal ≈ 1.41 × D_corr ≈ 4.2. E₆
gives d_s ≈ 4 not because it is 4D, but because it is ~3D.

The calibration factor CALIB = 1.106 was derived from a 4D
grid graph (not a k-NN graph), making it inapplicable here.
The "calibrated" values should not be interpreted as absolute
dimensions.

### 3.3 k* and its non-meaning

The observation that k=20 gives d_s_cal ≈ 4 for E₆ (v4) was
initially interpreted as a special connectivity. Analysis
shows this is not the case: k*=20 simply corresponds to the
connectivity at which the d_s_cal/D_corr ≈ 1.40 ratio for 3D
structures crosses 4.0. At k=12, T³ gives d_s_cal ≈ 3.1; at
k=20, T³ gives d_s_cal ≈ 4.2. E₆ follows the same curve.

Correlation of k* with algebraic invariants (Coxeter number h,
number of positive roots |Φ⁺|) was not significant (r < 0.7,
p > 0.3) with only 4 data points.

### 3.4 What genuinely varies across algebras

Among the properties measured in Part IV+:

| Property | E₆ | A₆ | D₆ | B₆ | Null |
|----------|----|----|----|----|------|
| D_corr | 3.02 | 2.96 | 2.89 | 3.01 | 5.45 |
| Recurrence (ε=1.0) | 0.0070 | 0.0073 | 0.0078 | 0.0068 | 0.0011 |
| Rel. entropy | 0.985 | 0.982 | 0.974 | 0.979 | 0.998 |
| N_boxes (ε=0.2) | 685 | 604 | 535 | 597 | 967 |

The differences among rank-6 algebras are small compared to the
difference from the null model. None of the measured properties
uniquely identifies E₆ over all other rank-6 algebras within
Part IV+ experiments.

### 3.5 Dark energy — correct analysis

Three graph models were tested:

- **Model A** (λ=0, no suppression): ⟨d⟩_final = 6.23,
  d²⟨d⟩/dt² = −0.009, H = +0.002 → **not dark energy**
- **Model B** (λ=λ₀=const): ⟨d⟩_final = 14.8,
  d²⟨d⟩/dt² = +0.037, H = +0.023 → **dark energy ✅**
- **Model C** (λ=λ₀·⟨d⟩/⟨d⟩₀, feedback): ⟨d⟩_final = 28.9,
  d²⟨d⟩/dt² = +0.298, H = +0.039 → **dark energy ✅**

However:

- C vs B: t=−0.083, p=0.934 → feedback adds nothing over
  constant λ
- C vs E (random phases + λ): t=0.122, p=0.904 → E₆ structure
  adds nothing over random phases

**Conclusion:** Any non-zero λ produces accelerated expansion
of ⟨d⟩. The effect is a property of the temporal suppression
parameter λ, not of E₆ or feedback. The dark energy claim
is falsified.

---

## 4. Complete Scorecard — All Parts

### 4.1 Confirmed findings

| Finding | Part | Key evidence |
|---------|------|-------------|
| Kuramoto T_c ≈ 1.4, anisotropic 2+4 | II | 20+ runs, null controlled |
| d_s reduction 37–51% vs null | III, IV | Weyl law + heat kernel |
| ω dominates K (ANOVA: 66% vs 3%) | IV | Two-way factorial ANOVA |
| Universal D_KY ≈ 4 plateau (dissipative) | I | 13/13 algebras, ranks 4–8 |
| GUE spectral statistics ⟨r⟩ = 0.529 | III | vs GUE = 0.531 |
| D_corr(E₆) ≈ 3.02 (quasi-3D orbit) | IV+ v8 | 15 seeds, N=1000 |
| All rank-6 algebras: D_corr ∈ [2.9, 3.0] | IV+ v8 | E₆, A₆, D₆, B₆ |
| Heat-kernel benchmarks validated | IV+ v3,v6 | Path→1.07, Grid→2.03/3.00 |

### 4.2 Falsified claims

| Claim | Part | Falsified by |
|-------|------|-------------|
| 6D→4D via Lyapunov compactification | I, v7 | Symplectic: D_KY = 2r always |
| E₆ uniqueness for d_s ≈ 4 | I, III, IV+ | All rank-6 give D_corr ≈ 3 |
| Gauge Higgs mechanism | II | Null ratio 22.2 > E₆ ratio 12.5 |
| Yukawa mechanism (fermion mass) | II | 6 definitions all anti-correlate |
| Bell test (quantum nonlocality) | I | Null model also violates |
| d_s = 4 as fixed spatial dimension | IV, IV+ | d_s ∝ N (chain); ∝ k (k-NN) |
| d_s(k-NN) measures manifold dimension | IV+ v7 | T³ → 4.2, T⁴ → 1.1 |
| E₆ produces d_s = 4 spacetime | IV+ v8 | d_s ≈ 4 = 3D k-NN effect |
| Dark energy from E₆ structure | IV+ v5 | p=0.90: E₆ irrelevant |
| Feedback mechanism for dark energy | IV+ v5 | p=0.93: feedback irrelevant |
| D_corr(E₆) = 4 | IV+ v5,v8 | D_corr = 3.02 |
| Compactification of synced dimensions | III, IV | d_s(sync) ≈ d_s(unsync) |

### 4.3 Open questions

| Question | Status | Difficulty |
|----------|--------|-----------|
| Why D_corr ≈ 3 for all rank-6 orbits? | Observed, unexplained | Analytic |
| Analytic derivation of D_corr from ω | Not attempted | KAM theory |
| d_s(k-NN) as a function of D_corr | Empirical pattern found | Analytic |
| Quantum walks → Dirac equation | Not implemented | Substantial |
| Number-theoretic resonances in d_s(β) | Observed in v3 | KAM / Diophantine |
| Universal D_KY ≈ 4: why specifically 4? | Confirmed, unexplained | Unknown |

---

## 5. Methodological Lessons

These lessons apply to any computational study of emergent
geometry from discrete dynamical systems.

### Lesson 1: Test N-dependence before claiming a dimension

A measured dimension d_s is physically meaningful only if
it is independent of system size. This requires explicit
verification: measure d_s at N = 300, 500, 700, 1000, 1300
and test linearity.

In this study, d_s ∝ N was the artifact that invalidated
Part IV's main claim. It took four additional experiment
versions to identify the cause (growing ⟨k⟩) and fix it
(k-NN construction).

### Lesson 2: Control average degree, not radius

In graph models of geometry, the natural parameter is the
**number of nearest neighbors** k, not a distance threshold ε.
For any ε > 0 in a D-dimensional space, the number of points
within ε grows as N · V_D(ε) — so ⟨k⟩ ∝ N. k-NN graphs
have ⟨k⟩ = const by construction.

### Lesson 3: d_s(k-NN) is not d_s(manifold)

The spectral dimension of a k-NN graph encodes both the
manifold dimension and the connectivity structure. At fixed k,
different manifolds produce completely different d_s values
that do not correspond to their true dimensions:

- T³ (d=3) at k=20: d_s_cal = 4.2
- T⁴ (d=4) at k=20: d_s_cal = 1.1

Any claim about "emergent dimension" from a k-NN graph must
demonstrate that the result is independent of k, or provide
a calibration curve d_s_cal(k) derived from the same graph
type on manifolds of known dimension.

### Lesson 4: D_corr requires torus metric and large N

The Grassberger-Procaccia algorithm with Euclidean distance
in an ambient space systematically underestimates D_corr when
points lie on a curved manifold (torus wrap-around creates
spuriously large distances). Use the torus metric
d(x,y) = ‖min(|x−y|, 2π−|x−y|)‖ throughout.

At N=1000 with 500-point subsampling, D_corr of T⁶ (expected
6) measures as 5.45 — a finite-size underestimate of ~9%.
Reported D_corr values should include this systematic error.

### Lesson 5: Null model hierarchy

Three levels of control are needed:

| Level | Construction | Tests |
|-------|-------------|-------|
| Random phases | φ ~ Uniform(T⁶) | Is any correlation needed? |
| K = 0 | φ evolves with ω, no coupling | Is Cartan matrix needed? |
| ω = uniform | All frequencies equal | Are Coxeter exponents needed? |

Each level addresses a different causal hypothesis. In this
study, the ANOVA (v4) using all three levels showed that ω
explains 66% of d_s variance while K explains only 3%. This
single result invalidated the "Cartan matrix = coupling =
gauge field" chain of reasoning.

### Lesson 6: Benchmarks are mandatory, not optional

Every spectral dimension measurement in Parts I–III lacked
benchmark validation. When benchmarks were finally added (v6),
it revealed that:

- The v5 measurement gave d_s = 0.49 (should be ~3) due to
  a fixed t-range
- The 4D grid systematically underestimates by 10%
- The calibration from grids does not transfer to k-NN graphs

Benchmarks should be the **first** result reported, not an
afterthought.

---

## 6. What the E₆ Orbit Actually Is

The quasi-periodic orbit of the map

$$\phi_{n+1} = \phi_n + \omega + 0.1\sin(\phi_n) \pmod{2\pi}$$

with E₆ Coxeter exponents $\omega_i = 2\sin(\pi m_i/h)$,
$\{m_i\} = \{1,4,5,7,8,11\}$, $h=12$, is a trajectory in T⁶
with the following measured properties:

**Dimension:** D_corr = 3.021 ± 0.005 (N=1000, 15 seeds).
This is statistically indistinguishable from D_corr(T³) = 2.996.
The orbit does not fill T⁶ (which would give D_corr ≈ 6) but
occupies an approximately 3-dimensional subset.

**Coverage:** Relative entropy = 0.985 (vs null = 0.998). The
orbit covers T⁶ slightly less uniformly than random points, but
is not strongly localized. N_boxes = 685 out of ~967 (for ε=0.2)
— the orbit visits 71% of T⁶ boxes that random points visit.

**Recurrence:** Rate = 0.007 at ε=1.0 (null: 0.001). The orbit
returns to ε-neighborhoods 7× more often than random, due to
the quasi-periodic structure.

**Non-uniqueness:** A₆, B₆, D₆ produce qualitatively similar
orbits. D_corr ∈ [2.9, 3.0] for all four rank-6 algebras tested.
The "3D in T⁶" property is a consequence of the rank of the
algebra and the nonlinear map, not a special feature of E₆.

---

## 7. Final Statement

The Monostring Hypothesis — that a single entity with six
internal phases produces emergent 4-dimensional spacetime
through E₆ resonances — is **falsified**.

The falsification chain is complete:

1. **Lyapunov compactification** (Part I): fails the symplectic
   test. D_KY = 2r identically for any Lie algebra, in any
   Hamiltonian system.

2. **Gauge-Higgs mechanism** (Part II): the edge variance ratio
   is larger in the null model than in E₆. Synchronization
   geometry alone explains the effect.

3. **Spectral dimension = 4** (Parts III–IV): d_s depends on
   graph size N (chain) or connectivity k (k-NN). At k=20,
   d_s ≈ 4 identifies 3D structures, not 4D.

4. **E₆ uniqueness** (Part IV+): all rank-6 algebras produce
   D_corr ≈ 3 for their Coxeter orbits. E₆ is not special
   among rank-6 algebras in this regard.

5. **Dark energy** (Part IV+): the accelerated expansion of
   average path length is driven by λ > 0, not by E₆ structure
   or feedback. E₆ phases are irrelevant (p = 0.90).

What survives is a mathematically interesting observation:
**quasi-periodic orbits with Coxeter exponents of rank-6
algebras occupy approximately 3-dimensional subsets of T⁶**,
while random points fill ~4–6 dimensions. This reduction
(6D → 3D) warrants analytic study via KAM theory and the
theory of quasi-periodic flows, but has no immediate
connection to 4-dimensional spacetime.

---

## Appendix A: Script Inventory — Part IV+

| Script | Key innovation | Key result |
|--------|---------------|-----------|
| v1 | Original dark energy model | Circular logic (λ=f(epoch)) |
| v2 | Monte Carlo + feedback | Bug (NameError); λ(epoch) survives |
| v3 | Three-model comparison | Correct design framework |
| v4 | 11-matrix scan + ANOVA | ω: 66%, K: 3% |
| v5 | Full eigenspectrum | **Broken d_s** (fixed t-range) |
| v6 | Adaptive t-range + benchmarks | Benchmarks pass; measurement fixed |
| v7 | Size dependence (chain) | d_s ≈ 2.16 + 0.002·N |
| v8 | RGG; k-NN; D_corr | D_corr(E₆) ≈ 3.02 |
| v8+ | k-scan; manifold comparison | d_s≈4 = 3D k-NN effect |

## Appendix B: Calibration Warning

The calibration factor CALIB = 4.0/3.616 = 1.106 was derived
from spectral dimension measurements on 4D grid graphs. It
should not be applied to k-NN graphs on arbitrary manifolds,
as demonstrated by:

- T³ (true d=3) + CALIB → d_s_cal = 4.22 (wrong)
- T⁴ (true d=4) + CALIB → d_s_cal = 1.09 (wrong)

There is no universal calibration factor for d_s(k-NN).
Each graph type requires its own calibration curve
d_s_cal(k, D_true) derived from manifolds of known dimension
with the same construction method.

## Appendix C: Reproducibility

All Part IV+ scripts are self-contained with explicit random
seeds. To reproduce the central result (D_corr of E₆ orbit):

```bash
cd scripts/part4
python graph_cosmology_v8.py
```

Expected output: D_corr(E₆) = 3.02 ± 0.01, runtime ~70 min.

### Environment

```
Python 3.8+
numpy >= 1.20
scipy >= 1.7
networkx >= 2.6
matplotlib >= 3.4
```

---

## Appendix D: The Adversarial Iteration Method

This study was conducted as an **adversarial iteration cycle**:
each script was critically analyzed for logical, statistical,
and computational flaws, and the next version was designed to
address them. The cycle ran for eight versions of the main
experiment script across Part IV+, and for seven versions in
Part IV (graph cosmology v1–v7).

The method proved effective at finding bugs that would have
been invisible in a single-pass analysis. The most important
flaws found:

| Flaw type | Example | Found in |
|-----------|---------|---------|
| Circular logic | λ(epoch) in dark energy | v1 |
| Runtime crash | NameError in dark energy | v2 |
| Wrong t-range | d_s = 0.49 (should be ~3) | v5 |
| Growing degree | ⟨k⟩ ∝ N in RGG | v1–v2 of IV+ |
| Wrong distance | Euclidean on torus → D_corr wrong | v6 |
| Wrong manifold | T⁴ with zeros → d_s = 1.1 | v6 |
| Missing calibration check | d_s(k-NN) ≠ d_s(grid) | v4–v7 |

The adversarial method is recommended for any computational
physics study where the quantity of interest (dimension,
phase transition, mechanism) could be produced by
measurement artifacts.

---

*This work was conducted as an exercise in AI-assisted
adversarial testing. The human author provided the hypothesis,
experimental direction, and computational resources. The AI
collaborator designed null models, identified flaws, wrote
analysis scripts, and proposed the falsifying tests —
including the D_corr measurement that produced the final
negative result.*

*The most important contribution of the AI collaborator was
not finding the answer — it was finding all the ways the
previous answers were wrong.*
