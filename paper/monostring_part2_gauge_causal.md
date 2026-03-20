# The Monostring Hypothesis Part II: Gauge Higgs Mechanism and Causal Set Explorations

**Author:** Igor Lebedev

**AI Collaborator:** Anthropic Claude (critical analysis, scripts)

**Sequel to:** [Part I — DOI 10.5281/zenodo.18886047](https://doi.org/10.5281/zenodo.18886047)

---

## Abstract

Following the falsification of the dimensional reduction mechanism in Part I, 
we investigate two new directions for the Sole Oscillator Hypothesis (SOH / 
Monostring): (1) gauge field structure via lattice gauge theory interpretation 
of phase differences, and (2) causal set construction from the Monostring's 
phase trajectory.

**Gauge results:** Treating edge phase differences as discrete U(1) gauge 
connections, we discover that the anisotropic Kuramoto synchronization 
transition produces an edge variance ratio of 12.5 between synchronized 
(massive) and unsynchronized (massless) gauge directions, with a nonzero 
gauge spectral gap (0.0175) for synchronized dimensions and zero gap for 
unsynchronized ones — passing 6/6 tests. However, a null model with 
artificially synchronized phases produces an even higher ratio (22.2), 
demonstrating that the effect is a trivial geometric consequence of 
synchronization, not a dynamical gauge Higgs mechanism.

**Causal set results:** Three approaches to defining causal order from 
the Monostring's phases were tested. The resonance rule (original SBE 
concept) produces D ≈ 1 for all trajectories and all parameter values. 
The light-cone rule produces D ≈ 4 at specific parameter values, but D 
depends on the parameter choice and the null model (random phases) gives 
comparable results. The Myrheim-Meyer dimension estimator shows systematic 
bias of 20-50% at N = 1000.

**Conclusion:** Neither direction produces results that are simultaneously 
non-trivial, robust, and specific to E₆ dynamics. The Monostring's phase 
structure on T⁶ appears to be exhausted as a source of emergent physics. 
Quantum walks on graphs remain the most promising unexplored direction.

---

## 1. Context

Part I of this study (scripts v0–v7) established that:

- The Monostring's E₆-coupled standard map with Coxeter frequencies 
  produces a strange attractor with D_corr = 4.025 ± 0.040
- This result is an **artifact of dissipative dynamics** — the symplectic 
  version gives D_KY = 2r identically
- The dimensional reduction mechanism is **falsified**

Two questions remained open from Part I:

1. Does the phase structure contain gauge field physics (Ollivier-Ricci 
   curvature correlation with density was r = 0.47)?
2. Can causal set theory provide an alternative path to emergent spacetime?

This paper addresses both questions through 15 additional computational 
experiments (Higgs Lab v1–v9, Gauge Lab v1–v3, Causal Set v1–v4).

---

## 2. Part IIa: Search for the Gauge Higgs Mechanism

### 2.1 Confirmed: Kuramoto Phase Transition

The E₆-coupled map with thermal noise exhibits a Kuramoto synchronization 
transition at T_c ≈ 1.4 with anisotropic breaking: dimensions φ₁ and φ₆ 
synchronize (r > 0.95) while φ₂–φ₅ remain disordered (r ≈ 0.2). This 
2+4 pattern is stable across 20+ runs and specific to E₆ dynamics (null 
model shows r ≈ 0.01).

### 2.2 Six Failed Attempts to Define Mass

| Version | Mass definition | corr(mass, VEV) | Verdict |
|---------|----------------|-----------------|---------|
| Higgs v2 | Spectral gap of graph Laplacian | −0.854 | Anti-Yukawa |
| Higgs v3 | Correlation length (1/ξ) | −0.532 | Anti-Yukawa |
| Higgs v3 | Effective potential curvature V''(v) | −0.435 | Anti-Yukawa |
| Higgs v5 | Wavepacket spreading rate (directional) | Inverted | Anti-Yukawa |
| Higgs v6 | Weighted edge metric | −0.657 | Anti-Yukawa |
| Higgs v8 | Stiffness (π² − ⟨(Δφ)²⟩) | +0.979 | Ratio=1.043 (constant) |

The stiffness measure showed perfect correlation but a fixed ratio of 1.043 
independent of N (tested up to N=20,000). This is a geometric constant, not 
a dynamical mass.

### 2.3 Gauge Field Interpretation: Initial Success

Treating phase differences Δφ_d(u,v) along graph edges as gauge connections:

| T | Var(sync) | Var(unsync) | Ratio |
|---|-----------|-------------|-------|
| 3.0 | 0.637 | 0.636 | 1.00 |
| 1.0 | 0.410 | 0.585 | 1.43 |
| 0.2 | 0.091 | 0.421 | 4.64 |
| 0.02 | 0.027 | 0.341 | **12.50** |

Gauge spectral gap: 0.0175 (synced) vs 0.0000 (unsynced). Score: **6/6 tests passed.**

### 2.4 Falsification: Null Model

Testing all rank-6 Lie algebras and a null model:

| Model | Ratio | Pattern |
|-------|-------|---------|
| E₆ | 12.62 | 2+4 |
| A₆ (SU7) | 18.88 | 2+4 |
| D₆ (SO12) | 15.98 | 2+4 |
| B₆ (SO13) | 9.50 | 2+4 |
| C₆ (Sp12) | 4.77 | 2+4 |
| **Null (artificial sync)** | **22.20** | 2+4 |
| Null (no sync) | 1.00 | 0+6 |

The null model with artificially synchronized phases (no E₆ dynamics, 
just φ₁ = φ₆ ≈ const) gives ratio = 22.2 — **higher** than any algebra. 
The edge variance ratio measures **the degree of synchronization**, not a 
gauge condensation mechanism. The result is **trivially explained** by the 
geometry of phase differences.

All algebras produce identical 2+4 pattern with synced dims = [0, 5] — 
this is a property of Dynkin diagram topology (end nodes synchronize first), 
not of specific Lie algebra structure.

### 2.5 One Surviving Observation: Weinberg Angle Analog

For E₆: Var(φ₁)/Var(φ₆) = 0.922. The Standard Model Weinberg angle gives 
cos(θ_W) ≈ 0.877. The resemblance (0.922 vs 0.877) may be coincidental — 
other algebras give very different values (0.30 to 1.39).

---

## 3. Part IIb: Causal Set Approach

### 3.1 Resonance Rule: D ≈ 1 (Failure)

The original SBE resonance concept (connect causally if phases match) 
produces D ≈ 1.0 for all trajectories (linear, chaotic E₆, stochastic, 
random). Phase resonance on a 1D timeline creates chain structures, not 
multidimensional causal sets.

### 3.2 Light-Cone Rule: D ≈ 4 (But Parameter-Dependent)

The rule m ≺ n if phase_distance(m,n) < c × (n−m) gives:

| c_speed | D | f | Interpretation |
|---------|---|---|---------------|
| 0.02 | 5.50 | 0.014 | Too sparse |
| 0.05 | 5.02 | 0.021 | Still sparse |
| **0.20** | **4.06** | **0.048** | Target! |
| 0.50 | 1.54 | 0.350 | Too dense |
| 1.00 | 1.00 | 0.614 | Saturated |

D = 4.06 at c = 0.20, but D depends continuously on c — any D from 1 to 7 
is achievable by tuning c.

### 3.3 Trajectory Independence Test: Mixed Results

At the best parameters (mixed cone: c_short=1.0, c_long=0.02, cross=20):

| Trajectory | D |
|-----------|---|
| E6 κ=0.5 | 5.18 |
| Stochastic T=0.1 | 3.75 |
| Pure random | 3.87 |
| E6 κ=1.0 | 1.04 |
| Linear | 0.00 |

D depends on trajectory type. Pure random phases give D ≈ 3.87 — comparable 
to the E₆ result. E₆ dynamics is not essential.

### 3.4 Estimator Reliability

At N = 1000, the Myrheim-Meyer estimator systematically underestimates:
- True D=2 → Estimated D=1.0 (−50%)
- True D=3 → Estimated D=2.0 (−33%)
- True D=4 → Estimated D=2.9 (−28%)
- True D=5 → Estimated D=4.0 (−20%)

The "D = 4.01" result at c=0.20 likely corresponds to true D ≈ 5–6.

---

## 4. Summary: Complete Scorecard After Parts I and II

| Claim | Method | Result | Status |
|-------|--------|--------|--------|
| 6 phases → 4D space | Lyapunov/symplectic | D_KY = 2r always | ❌ Falsified |
| E₆ is special | All algebras tested | All give same results | ❌ Falsified |
| Gauge Higgs mechanism | Edge variance | Null model ratio = 22 | ❌ Trivial |
| Yukawa mechanism | 6 mass definitions | All anti-correlate | ❌ Failed |
| Kuramoto transition | Order parameter | 2+4 breaking, T_c=1.4 | ✓ Confirmed |
| Causal D=4 (resonance) | Myrheim-Meyer | D ≈ 1 always | ❌ Failed |
| Causal D=4 (light cone) | Myrheim-Meyer | D=4 at specific c | ⚠️ Parameter-dependent |
| E₆ needed for D=4 | Null model | Random gives D≈4 too | ❌ Not needed |

### What is genuinely established:

1. E₆-coupled maps with Coxeter frequencies exhibit anisotropic Kuramoto 
   synchronization (2/6 directions) — non-trivial, reproducible, absent in 
   null model
2. The synchronization pattern follows Dynkin diagram topology (end nodes first)
3. Three Goldstone-like near-zero modes in broken phase

### What is definitively ruled out:

1. Dimensional reduction via Lyapunov compactification (symplectic test)
2. Gauge Higgs mechanism via edge variance (null model test)
3. Six independent definitions of particle mass
4. Causal dimension from resonance rule

---

## 5. Future Direction: Quantum Walks

The remaining unexplored path from Part I's roadmap is quantum walks on 
graphs. Unlike classical trajectories (which give D=1 in causal sets) and 
unlike ad hoc causal rules (which give tunable D), quantum walks have 
**theorems** connecting them to continuum physics:

- Strauch (2006): quantum walks on lattices reproduce the Dirac equation
- The evolution operator is unitary — no dissipation, no symplectic problem
- The concept of D_KY is inapplicable — dimension comes from spectral properties

This is the last direction that has not been tested and has mathematical 
guarantees of producing non-trivial physics.

---

## References

[14] K. Wilson, "Confinement of quarks," Phys. Rev. D 10:2445, 1974.

[15] Y. Kuramoto, "Self-entrainment of a population of coupled non-linear 
oscillators," Lecture Notes in Physics 39, Springer, 1975.

[16] F.W. Strauch, "Relativistic quantum walks," Phys. Rev. A 73:054302, 2006.

---

## Appendix: Script Inventory (Part II)

| Script | Key result |
|--------|-----------|
| higgs_v1_thermal.py | Clustering as VEV — artifact |
| higgs_v2_kuramoto.py | Phase transition confirmed, mass anti-Yukawa |
| higgs_v3_three_masses.py | All three anti-Yukawa, 3 Goldstones found |
| higgs_v4_anisotropic.py | Anisotropic field confirmed |
| higgs_v5_directional.py | Directional mass inverted |
| higgs_v6_metric.py | Weighted metric also inverted |
| higgs_v7_dispersion.py | Weak signal ratio=1.05, corr=0.78 |
| higgs_v8_three_measures.py | Stiffness corr=0.979, ratio=1.043 constant |
| higgs_v9_scaling.py | Ratio=1.043 independent of N (geometric) |
| gauge_v1_plaquette.py | F²=0 numerical, Wilson W=1 |
| **gauge_v2_edge_variance.py** | **6/6 PASS, ratio=12.5** |
| **gauge_v3_algebra_scan.py** | **Null model ratio=22 → trivial** |
| causal_v1_basic.py | D≈1 for all rules (N=500 too small) |
| causal_v2_corrected.py | D≈1 confirmed at N=1500 |
| causal_v3_nonlinear.py | E6 chaos doesn't help, light_cone D=6.6 |
| **causal_v4_light_cone.py** | **D=4.01 but parameter-dependent** |
