# Part XXI: Stochastic Realization of the Monostring Hypothesis

*Author: Igor Lebedev | AI collaborator: Anthropic Claude Sonnet 4*

---

## Summary

| Step | Finding | Status |
|------|---------|--------|
| 1 | Simple traversal → white noise (slope=-0.01) | ❌ No correlations |
| 2 | Local KG coupling → slope=-1.4 | ⚠️ Partial |
| 3 | Metropolis MC → slope=-1.83 (random best) | ⚠️ Near-KG |
| 4 | Convergence study (N=64..1024) | ⚠️ Non-monotone (noise) |
| 5 | Exact KG configurations: slope=-1.72..-1.81 | ❌ Not -2.0 |
| 6 | UV regime analysis: slope=-1.652 stable | ❌ Lattice artifact |
| 7 | Diagnostic: sin(πf)≠πf at f>0.08 | ✓ Explained |

**Bottom line:** BЭ via Metropolis MC ≡ Euclidean KG QFT (theorem).
Zero new physical predictions. CMB spectrum unreachable via KG statistics.

---

## XXI.1 Motivation

Parts I–XX falsified all classical and semi-classical realizations
of the monostring hypothesis. One path remained unexplored:

> *If the Basic Element (BЭ) sequentially visits all quantum field
> states within one Planck-time cycle, does the time-averaged
> statistics reproduce quantum field theory?*

This is the stochastic interpretation: BЭ as a sequential automaton
traversing N states per Planck cycle, with the observed field being
the statistical footprint of these visits.

### Connection to known physics

The idea maps onto two established frameworks:

**Wheeler–Feynman absorber theory** (1945): retarded and advanced
fields from a single source reproduce observed electrodynamics.

**Stochastic mechanics** (Nelson 1966; Guerra & Morato 1983):
quantum mechanics derived from Brownian motion in configuration space.

Our question: does a *sequential deterministic/stochastic traversal*
of field configurations reproduce quantum statistics?

---

## XXI.2 Model: BЭ as Sequential Automaton

We model the 1D Klein-Gordon field on N sites with periodic
boundary conditions:

$$S[\phi] = \frac{1}{2}\sum_x \left[(\phi_{x+1}-\phi_x)^2 + m^2\phi_x^2\right]$$

**BЭ traversal rule:** In each cycle T = t_P, BЭ visits all N
sites in some order and updates φ(x) according to a local rule.

### Three traversal orders tested

| Order | Rule | Physical interpretation |
|-------|------|------------------------|
| Random | permutation each cycle | quantum randomness as consequence |
| Sequential | x=0,1,2,...,N-1 | preferred spatial direction |
| Checkerboard | even sites, then odd | causal structure |

### Three update rules tested (Steps 1→3)

**Step 1** (no coupling):
```
φ(x) ← 0.99·φ(x) + noise·ξ
```
Result: white noise, slope = -0.01.

**Step 2** (discrete KG):
```
φ(x) ← φ(x) + α·[φ(x+1)+φ(x-1)-2φ(x)] - m²·φ(x) + noise·ξ
```
Result: slope = -1.14 to -1.46 (below KG).

**Step 3** (Metropolis / detailed balance):
```
Propose φ'(x) = φ(x) + δ·ξ
Accept with P = min(1, exp(-ΔS/T))
```
Result: slope = -1.83 (random order, best match).

---

## XXI.3 Results

### Step 3: Metropolis (detailed balance)

| Order | Acc. rate | slope | l_corr | χ²/dof |
|-------|-----------|-------|--------|--------|
| random | 0.783 | **-1.828** | 8.0 | 0.632 |
| sequential | 0.783 | -1.580 | 6.0 | 0.718 |
| checkerboard | 0.783 | -1.561 | 7.0 | 1.426 |
| **KG theory** | — | **-2.000** | **10.0** | **0** |

**Key finding:** Random traversal order gives best agreement
with KG vacuum. Correlation length l_corr stable across all T.

**Theorem (standard):** Metropolis MC with detailed balance
converges to the stationary distribution P[φ] ~ exp(-S[φ]/T).
This is the Euclidean path integral weight. Therefore:

$$\text{BЭ (Metropolis)} \equiv \text{Euclidean KG QFT}$$

This is a mathematical identity, not an empirical discovery.

### Step 5: Exact KG configurations

Generating exact KG configurations via mode decomposition:

$$\tilde{\phi}(k) \sim \mathcal{N}\!\left(0,\ \frac{T}{\lambda_k}\right), \quad
\lambda_k = 2 - 2\cos\!\left(\frac{2\pi k}{N}\right) + m^2$$

| N | slope (exact KG) | ±stderr |
|---|-----------------|---------|
| 64 | -1.809 | 0.023 |
| 128 | -1.782 | 0.012 |
| 256 | -1.732 | 0.013 |
| 512 | -1.669 | 0.015 |
| 1024 | -1.596 | 0.014 |
| ∞ (KG) | **-2.000** | exact |

**Finding:** slope of *exact* KG configurations measured in
range f ∈ [0.02, 0.30] gives -1.60 to -1.81, not -2.00.
This is **not** a Monte Carlo artifact.

### Step 7: Lattice artifact diagnosis

The discrepancy is explained analytically.

The exact dispersion relation on a 1D lattice:

$$\lambda_k = 4\sin^2\!\left(\frac{\pi k}{N}\right) + m^2$$

The continuum approximation (valid only for k/N ≪ 1):

$$\lambda_k \approx \left(\frac{2\pi k}{N}\right)^2 + m^2$$

The local slope of P(k) = 1/λ_k is:

$$\text{slope}(f) = \frac{d\log P}{d\log f} =
-\frac{2\pi f \cos(\pi f)}{\sin(\pi f)} \cdot
\frac{1}{1 + \left(\frac{m}{2\sin(\pi f)}\right)^2}$$

| f | Local slope | Regime |
|---|-------------|--------|
| 0.008 | -0.18 | IR plateau |
| 0.020 | -1.22 | transition |
| 0.050 | -1.80 | transition |
| 0.080 | **-1.88** | near-UV |
| 0.100 | **-1.88** | near-UV maximum |
| 0.150 | -1.83 | lattice UV |
| 0.200 | -1.72 | lattice UV |
| 0.300 | -1.36 | lattice UV |
| 0.400 | -0.81 | Brillouin zone edge |

**Maximum achievable slope on lattice: -1.885**
at f ≈ 0.08..0.10 (one decade above f_cross = m/2π).

The continuum limit slope = -2.000 is reached only as a → 0
(lattice spacing → 0), which corresponds to N → ∞ with
physical box size fixed — not simply larger N at fixed a.

---

## XXI.4 Comparison with CMB

The primordial power spectrum observed by Planck:

$$P(k) \propto k^{n_s - 1}, \quad n_s = 0.9649 \pm 0.0042$$

$$\text{slope}_{\text{CMB}} = n_s - 1 = -0.035$$

| Spectrum | slope | Source |
|----------|-------|--------|
| KG vacuum (continuum) | -2.000 | analytic |
| KG vacuum (lattice, best) | -1.885 | this work |
| BЭ (Metropolis, random) | -1.828 | this work |
| CMB (Planck 2018) | **-0.035** | observation |

**Gap:** |slope_KG − slope_CMB| ≈ 1.85

This gap is **fundamental**, not a lattice artifact.
The Harrison–Zel'dovich scale-invariant spectrum (slope = 0)
arises from de Sitter quantum fluctuations of a slow-roll
inflaton, not from the KG vacuum statistics.

The KG vacuum is the ground state of a field in **Minkowski**
spacetime. The CMB spectrum encodes quantum fluctuations
generated during inflation in **de Sitter** spacetime,
where the mode functions differ fundamentally:

$$u_k(\tau) = \frac{H}{\sqrt{2k^3}}\left(1 + ik\tau\right)e^{-ik\tau}$$

vs. flat-space KG:

$$u_k(t) = \frac{1}{\sqrt{2\omega_k}}e^{-i\omega_k t}$$

Reproducing n_s ≈ 0.965 requires slow-roll dynamics in
de Sitter space — not achievable by KG statistics alone.

---

## XXI.5 Physical Interpretation

### What BЭ(Metropolis) is

The Metropolis algorithm with action S[φ] samples from:

$$P[\phi] \propto e^{-S[\phi]/T}$$

This is the **Euclidean path integral** measure. Therefore:

> BЭ modeled as a sequential automaton with detailed balance
> is mathematically identical to Euclidean quantum field theory.

This is a *reinterpretation*, not a new theory. The same
mathematics appears in:
- Lattice QCD (Wilson 1974)
- Stochastic quantization (Parisi & Wu 1981)
- Path integral Monte Carlo

### Physical meaning of the parameters

| BЭ parameter | QFT meaning |
|-------------|-------------|
| Traversal cycle T | Planck time t_P |
| Temperature T | ℏ (quantum of action) |
| Update rule | Local field equation |
| Detailed balance | Principle of least action |
| Random order | Quantum randomness |

### What is new (ontological, not physical)

The monostring hypothesis provides a *sequential* ontology
for QFT: instead of fields existing simultaneously everywhere,
a single object visits all configurations in sequence.

This is analogous to the difference between:
- Boltzmann: gas = N molecules at positions {xᵢ}
- Maxwell: gas = distribution function f(x,p)

Both are correct. Neither is more "fundamental."
The sequential BЭ interpretation adds no new predictions.

---

## XXI.6 Thermalization Diagnostic

Metropolis autocorrelation time for N=256, m²=0.01:

$$\tau_{\text{AC}} = 8 \text{ sweeps} \ll N_{\text{sweeps}} = 5000$$

System is well-thermalized. The non-monotone slope(N) in
Step 4 was statistical noise from single-seed runs,
not thermalization failure.

**Temperature independence of l_corr:**

| T | l_corr (measured) | l_corr (theory = 1/m) |
|---|------------------|----------------------|
| 0.1 | 10.0 | 10.0 |
| 0.5 | 10.0 | 10.0 |
| 1.0 | 10.0 | 10.0 |
| 2.0 | 10.0 | 10.0 |
| 5.0 | 10.0 | 10.0 |

Correlation length is T-independent (as expected from
the mass parameter alone). This confirms correct sampling.

---

## XXI.7 Conclusions

### What Part XXI established

1. **BЭ ≡ Euclidean KG QFT** (theorem via detailed balance).
   The monostring sequential traversal, if governed by
   Metropolis acceptance, is mathematically identical to
   the Euclidean path integral. This is a known result
   in a new interpretive framework.

2. **Lattice slope = -1.885**, not -2.000.
   The gap arises from sin(πf) ≠ πf at f > 0.08.
   This is a property of the discrete Laplacian, not
   a Monte Carlo artifact.

3. **CMB gap is fundamental.** The KG vacuum gives
   slope ≈ -1.9; Planck observes n_s-1 = -0.035.
   Closing this gap requires slow-roll inflation in
   de Sitter space, not KG statistics.

4. **Zero new predictions.** The stochastic BЭ
   reinterpretation reproduces known QFT results
   without adding falsifiable content beyond QFT.

### What this means for the hypothesis

The monostring hypothesis in its stochastic form is
**interpretively valid but physically empty**: it
provides an alternative ontology for QFT without
new predictions. This is the strongest possible
form of the hypothesis that survives.

### Open directions (require different frameworks)

- **Quantum BЭ:** |ψ_BЭ⟩ ∈ ℋ with unitary evolution U.
  Requires QFT on curved spacetime.
- **de Sitter BЭ:** BЭ traversal in expanding background.
  Might reproduce n_s ≈ 0.965 via Bunch-Davies vacuum.
- **Holographic BЭ:** BЭ as boundary CFT; bulk =
  emergent spacetime via AdS/CFT.

These require mathematical frameworks beyond the scope
of this computational project.

---

## Artifact documented in Part XXI

| # | Name | Step | Description | Fix |
|---|------|------|-------------|-----|
| 12 | slope=-2 claim | 4 | Non-monotone slope(N) from single-seed noise | N≥50 independent runs |
| 13 | UV range error | 6 | fit [0.08,0.40] includes lattice UV → slope=-1.65 | fit [0.08,0.10] only |

---

*Part XXI closes the stochastic framework investigation.
See [README](../README.md) for complete project scorecard.*
