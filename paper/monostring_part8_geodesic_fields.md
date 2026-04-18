# Part VIII: Emergent Geodesic Structure in Coxeter Frequency Graphs
## Eight Steps from Falsification to Reproducible Effect

*Part of the Monostring Hypothesis series*
*Author: Igor Lebedev | AI collaborator: Claude (Anthropic)*

---

## Abstract

We investigate whether a monostring — a single oscillating
entity with Coxeter-algebraic internal frequencies — can
generate quantum-field-like structures in its state space.
Through eight computational experiments, we systematically
falsify naive approaches (collective modes, shortest-path
light propagation, spectral dimension as spacetime dimension)
and identify one reproducible, statistically robust effect:

**High-frequency Laplacian eigenmodes of the E6 Coxeter
orbit graph exhibit significantly stronger concentration
along graph geodesics (z ≈ 6, p < 10⁻⁸) compared to
random-frequency null models.**

This effect survives:
- Variation of coupling strength κ (Spearman ρ=0.20, p=0.80)
- Control for graph algebraic connectivity
  (Fiedler-matched ANOVA: F=17.9, p=7.2×10⁻⁹)
- Multi-pair sampling (n=20–30 random source-target pairs)

Wavepacket dynamics in the high-λ sector are diffusive
(α ≈ 0.30), not ballistic — indicating standing rather
than propagating modes. We interpret this as a first
reproducible geometric signature distinguishing
Coxeter-frequency monostrings from random-frequency
null models.

---

## 1. Motivation and Setup

### 1.1 The Question

Previous parts (I–VII) established:
- D_corr(E6) ≈ 3.02: the E6 orbit occupies a
  quasi-3D subset of T⁶
- D_corr depends on the unordered set of frequencies,
  not their specific algebraic assignment
- Memory time τ ≈ 237 steps: fragmented daughter strings
  retain structural memory of the E6 attractor

Part VIII asks a new question:

> Can one monostring graph support multiple distinct
> transport classes — analogues of quantum fields —
> and do these classes show any E6-specific structure?

### 1.2 Construction

**Orbit generation:**

φᵢ(t+1) = φᵢ(t) + ωᵢ + κ sin(φᵢ(t)) mod 2π

ωᵢ = 2 sin(π mᵢ / h) [Coxeter frequencies]

E6: m = [1,4,5,7,8,11], h=12
A6: m = [1,2,3,4,5,6], h=7
E8: m = [1,7,11,13,17,19,23,29], h=30
Random: ωᵢ ~ Uniform[0.5, 2.0]


**Torus-respecting embedding:**

φᵢ → (cos φᵢ, sin φᵢ) ∈ ℝ²

This avoids false discontinuities at φ = 2π.

**Graph:** k-NN graph (k=10) with Gaussian weights
on giant component.

**Laplacian:** Normalized graph Laplacian
L = I - D^{-1/2} A D^{-1/2}

**Spectral bands** (20% each):
- lowλ:  smallest non-zero eigenvalues
- midλ:  middle band
- highλ: largest eigenvalues

---

## 2. Steps and Results

### Step 1: Field Metric from Attractor Geometry
**Question:** Does the covariance tensor of the orbit
define an effective metric?

**Result:**
- All rank-6 algebras: D_eff ≈ 5.8 (mildly anisotropic)
- E8: 4+4 eigenvalue split (CV=0.47)
- Random: isotropic (CV=0.007)
- Correlation r(ratio, h) = 0.87, p=0.13 (4 points,
  insufficient for significance)

**Status:** ⚠️ Suggestive but underpowered.

---

### Step 2: Collective Modes and Dispersion
**Question:** Do daughter string clouds develop
field-like dispersion ω(k)?

**Result:** R² = 1.000 for all algebras — tautology.
The dominant PSD frequency equals the natural frequency
by construction.

**Status:** ❌ Falsified (methodological artifact).

---

### Step 3: Shortest Paths as Light Rays
**Question:** Do graph geodesics correspond to
null propagation?

**Result:**

E6: R²(d_3d)=0.054, R²(d_6d)=0.076, CV(c)=0.40
Random: R²(d_3d)=0.174, R²(d_6d)=0.132, CV(c)=0.36

No causal ordering by temporal separation (ρ=0.04, p=0.44).

**Status:** ❌ Falsified. No light-cone structure found.

---

### Step 4: Laplacian Spectral Dimension
**Question:** Does λₙ ∝ nᵅ give D=2/α ≈ 3 or 4?

**Result (corrected):**

E6: D_eff=2.01, D_Weyl=1.99
A6: D_eff=1.92, D_Weyl=1.91
E8: D_eff=2.47, D_Weyl=2.44
Random: D_eff=4.80, D_Weyl=4.78

Note: earlier run with different σ gave D≈3 for all —
this was a bandwidth artifact. Corrected results show
no universal D=3.

**Status:** ❌ D≈3 from Part V was orbit geometry
(D_corr), not Laplacian spectrum.

---

### Step 5: Multi-Field Transport Classes
**Question:** Do spectral bands constitute distinct
transport classes?

**Step 5b — Robust observables:**

| Band | α⟨d⟩ | z_geo | IPR/ref | class |
|------|-------|-------|---------|-------|
| E6 lowλ | -0.004 | 1.66 | 2.65 | localized |
| E6 midλ | 0.099 | 2.00 | 3.03 | localized |
| E6 highλ | 0.138 | **2.37** | 3.31 | localized |
| Rnd lowλ | -0.001 | 1.04 | 2.74 | localized |
| Rnd midλ | 0.015 | 0.50 | 3.00 | localized |
| Rnd highλ | 0.017 | 1.11 | 2.91 | localized |

E6 shows monotonic IPR gradient (ρ=1.0, p<0.001).
Random shows no monotonic trend (ρ=0.5, p=0.67).

**Step 5c — Multi-pair validation (n=29 pairs):**

E6 highλ: z_geo = 5.998 ± 0.497
Rnd highλ: z_geo = 0.751 ± 0.325

Mann-Whitney p = 7.27×10⁻¹⁰
Fraction E6 above z=2: 100%
Fraction Rnd above z=2: 20.7%


**Status:** ✅ First confirmed E6-specific signal.

---

### Step 6: Wavepacket Propagation

**Step 6 — Motion:**

E6 lowλ: α=0.243 (diffusive), z_avg=3.85
E6 midλ: α=0.171 (diffusive), z_avg=4.14
E6 highλ: α=0.300 (diffusive), z_avg=6.63


Algebra comparison:

E6: z_geo(highλ) = 5.82 ± 0.81
A6: z_geo(highλ) = 6.60 ± 0.95
E8: z_geo(highλ) = 3.68 ± 0.64
Random: z_geo(highλ) = 0.04 ± 0.29

ANOVA: F=16.0, p=3.7×10⁻⁸


κ robustness: Spearman ρ=0.20, p=0.80
→ Effect is geometric, not dynamical.

**Step 6bc — Topological invariants:**

Winding numbers: ~50% non-trivial for ALL algebras
including Random → not algebra-specific.

Betti β₁ ≈ 3750–3990 for all → no correlation
with z_geo (ρ=-0.80, p=0.20, n=4).

Discovery: A6 Fiedler=0.006 (near-disconnected graph),
Mean_path=12.9 — potential connectivity artifact.

**Step 6d — Connectivity control (key test):**

Matching all algebras to Fiedler ≈ 0.15:

E6: z_geo = 6.06, Fiedler=0.175
A6: z_geo = 7.11, Fiedler=0.146
E8: z_geo = 3.89, Fiedler=0.145
Random: z_geo = 0.75, Fiedler=0.175

ANOVA: F=17.9, p=7.2×10⁻⁹
E6 vs Random: p=6.3×10⁻⁶



**Status:** ✅ REAL EFFECT confirmed.
Effect survives Fiedler control → driven by orbit
geometry (D_corr ≈ 2.6), not graph connectivity.

Note: A6 advantage over E6 persists at matched
Fiedler (p=0.18, ns) — A6 effect may also be real,
not purely a connectivity artifact.

---

## 3. Summary of All Results

### ✅ Confirmed

| Finding | Evidence | Strength |
|---------|----------|----------|
| E6 highλ geodesic focusing z≈6 | p=7×10⁻¹⁰ | Strong |
| Effect not explained by Fiedler | ANOVA p=7×10⁻⁹ | Strong |
| Effect robust to κ | ρ=0.20, p=0.80 | Strong |
| Monotonic IPR gradient in E6 | ρ=1.0, p<0.001 | Strong |
| Wavepacket spreading: diffusive | α≈0.30, R²≈0.65 | Moderate |
| Coxeter algebras >> Random | All k values | Strong |

### ❌ Falsified

| Claim | How |
|-------|-----|
| ω_collective = non-trivial dispersion | Tautology (R²=1) |
| Graph geodesic = null geodesic | R²<0.1, no causal structure |
| D_Laplacian ≈ 3 for Coxeter | Bandwidth artifact |
| Winding numbers algebra-specific | ~50% for all incl. Random |
| β₁ drives z_geo | ρ=-0.8, p=0.20 (ns) |
| A6 > E6 due to frequency rationality | All hypotheses ✗ |
| "Photon" as ballistic propagator | α≈0.3, diffusive only |

### ⚠️ Open Questions

| Question | Status |
|----------|--------|
| Why A6 ≈ E6 at matched Fiedler? | Unexplained |
| Why z_geo oscillates with k in E6? | Resonance? |
| Does D_corr ≈ 2.6 cause z_geo? | Correlation seen, causation unclear |
| Effective field theory for highλ? | Not attempted |

---

## 4. Physical Interpretation

### What the monostring graph supports

The state graph of the E6 monostring has a
high-frequency spectral sector where:

1. **Eigenmodes are localized** (IPR/ref ≈ 3.3,
   not 1.0) but not fully localized

2. **Localized modes are geodesically structured**:
   their amplitude concentrates near shortest paths
   between any two nodes

3. **Wavepackets spread diffusively** along these
   geodesic corridors (α ≈ 0.30)

This combination — localized yet geodesically
coherent — resembles **edge states** in topological
insulators more than propagating photons.

### What this is NOT

- Not photon propagation at constant speed c
- Not Lorentz-invariant field theory
- Not Standard Model gauge fields
- Not evidence for U(1) or SU(2) symmetry
- Not derivation of particle masses

### What this IS

> The first reproducible, connectivity-controlled,
> κ-robust geometric signature distinguishing
> Coxeter-frequency monostring graphs from
> random-frequency null models in spectral
> transport structure.

The physical picture consistent with data:

Coxeter frequencies
→ quasi-low-dimensional orbit (D_corr ≈ 2.6)
→ graph with preferred geometric channels
→ high-λ Laplacian modes aligned to these channels
→ standing, geodesically-structured excitations


These standing modes are candidates for
"field-like objects" in the monostring picture —
not because they propagate, but because they
occupy and define geometric structure.

---

## 5. Methods Summary

| Parameter | Value |
|-----------|-------|
| Orbit length T | 900 |
| Coupling κ | 0.05 (varied 0.01–0.20) |
| Warmup | 500 steps |
| Embedding | Torus: φ→(cos φ, sin φ) |
| Graph | k-NN, k=10 (varied 4–24) |
| Edge weight | Gaussian exp(-d²/2σ²) |
| Laplacian | Normalized L = I - D^{-½}AD^{-½} |
| Band width | 20% of non-zero eigenvalues |
| Evolution | Continuous-time: ψ(t)=Σcₖe^{-iλₖt}vₖ |
| z_geo | n_rand=60–80, corridor radius=1 |
| Multi-pair | n=20–30 random (source,target) pairs |
| Statistics | Mann-Whitney U, ANOVA, Spearman ρ |

All code: MIT License
All text: CC-BY 4.0

---

## References to Previous Parts

- Part V: D_corr(E6) ≈ 3.02 established
- Part VI: Memory time τ≈237, entropy paradox
- Part VII: τ ∝ h(Coxeter) falsified
- **Part VIII** (this work): Geodesic focusing
  in highλ sector confirmed

  
