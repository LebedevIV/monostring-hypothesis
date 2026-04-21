# Parts XII–XVI: Inflation Search
## Monostring Hypothesis — Classical Coxeter Dynamics vs Cosmological Inflation

**Authors:** Igor Lebedev, Claude (Anthropic)
**Date:** 2025
**Status:** All experiments falsified (H₀ not rejected)

---

## Summary

Five experiments tested whether classical dynamics on
Coxeter algebras can produce inflationary cosmology
consistent with Planck 2018 (n_s=0.9649±0.0042, r<0.036).

**Result:** No experiment produced a statistically
significant signal. Three mathematical theorems were
discovered as byproducts.

---

## Part XII: Emergent Inflaton Potential V_eff(φ₁)

**H₀:** V_eff flatness for Coxeter algebras ≤ random controls

**Method:** Standard map for E8/E6/E7/A6 (T=100,000,
warmup=10,000). Mode φ₁ treated as inflaton; remaining
modes as environment. V_eff(φ₁) = −κ⟨Σⱼcos(φ₁−φⱼ)⟩.
Slow-roll: ε=½(V'/V)², flat_frac = fraction where ε<0.01.

**Key artifact discovered (Step 4 vs Step 5):**
Single random control (seed=999) gave ff=0.500,
suggesting E8 ff=0.990 was significant. N=200 controls
gave true mean ff=0.184. Single-seed inflation: 3x.

**Results (N=200 rank-matched controls):**

| κ | E8 ff | Random ff | p-value |
|---|-------|-----------|---------|
| 0.005 | 0.210 | 0.274±0.178 | 0.580 |
| 0.010 | 0.170 | 0.184±0.116 | 0.425 |
| 0.020 | 0.100 | 0.131±0.048 | 0.945 |
| 0.050 | 0.110 | 0.108±0.043 | 0.345 |

n_s values: 0.14–0.57 (Planck: 0.9649). **H₀ holds.**

---

## Part XIII: KAM Threshold κ_c(algebra)

**H₀:** κ_c does not depend on Coxeter algebra

**Method:** Maximum Lyapunov exponent λ_max(κ) via
Benettin algorithm. κ_c = first κ where λ_max > 0.001.
N=200 rank-matched controls for rank 6, 7, 8.

**Results:**

| Algebra | h | κ_c | Random ref | p |
|---------|---|-----|-----------|---|
| E8 | 30 | 1.027 | 1.045±0.132 | 0.430 |
| E6 | 12 | 1.077 | 1.057±0.118 | 0.190 |
| E7 | 18 | 0.987 | 1.058±0.135 | 0.910 |
| G2 | 6  | 2.000 | 1.045±0.132 | 0.015 |

r(h, κ_c) = −0.474 (weak, not significant).

**G2 anomaly:** ω₁=ω₂=2sin(π/6)=1.000 exactly.
Two identical frequencies → trivial resonance →
integrable subsystem → κ_c→∞. Not Coxeter physics.

**H₀ holds.**

---

## Part XIV: Complex Coxeter Frequencies ω+iΓ

**H₀:** Im(ω) > 0 gives inflation; E8 maximises growth rate

**Method:** ωᵢ → 2sin(π(mᵢ+iε)/h) gives
Γᵢ = 2cos(πmᵢ/h)·sinh(πε/h).
Total growth rate: Γ_total = 2sinh(πε/h)·Σcos(πmᵢ/h).

**Theorem (discovered analytically):**
```
Σᵢ cos(πmᵢ/h) = 0  for ALL Coxeter algebras.
```
Proof: Weyl involution mᵢ + m_{r+1−i} = h implies
cos(πmᵢ/h) = −cos(πm_{r+1−i}/h). Terms cancel pairwise.

Verified numerically (|Σcos| < 10⁻¹⁰ for E8,E6,E7,A6,F4,G2).

**Consequence:** Γ_total = 0 identically. Growing modes
exactly cancelled by decaying modes. No net inflation
possible from complex-frequency Coxeter dynamics.

**Closed analytically. No simulation needed.**

---

## Part XV: Blind Search — Inflation Potentials V(φ)

**H₀:** Coxeter potentials V=Σwᵢφ^(2mᵢ/h) do not
produce Planck-compatible (n_s, r) more often than random

**Method:** 50,000 Dirichlet-random weight vectors per
algebra. Slow-roll parameters from numerical derivatives.
Target: n_s∈[0.960,0.969], r<0.036.

**Step 1 results (pure power-law):**

| Algebra | n_s mean | n_s std | Winners |
|---------|----------|---------|---------|
| E8 | 0.9748 | 0.0042 | 0/50000 |
| E6 | 0.9764 | 0.0043 | 0/50000 |
| E7 | 0.9755 | 0.0043 | 0/50000 |
| A6 | 0.9781 | 0.0033 | 0/50000 |
| Random | 0.9921 | 0.0020 | 0/50000 |

**Systematic bias:** n_s(E8) = 0.9748 vs Planck 0.9649.
Offset = +2.4σ. Explanation: V~φ^α with α<2 gives
n_s → 1 as α → 0. E8 has minimum exponent
2m₁/h = 2/30 ≈ 0.067 → n_s near 1.

**Step 2 (modified forms, N=30,000 each):**

| Form | E8 | Random | Signal? |
|------|----|--------|---------|
| Starobinsky×Coxeter | 0.16% | 0.68% | NO (E8 worse) |
| Hilltop | 0.01% | n/a | NO |
| Axion-like | 0.04% | 0.03% | NO |

Note: A6 Form C = 0.98% vs random 0.03%,
but rank=6 random control not computed (unfair comparison).
E8 shows no signal in any form.

**H₀ holds.**

---

## Part XVI: Cartan Matrix Hamiltonian

**H₀:** Dynamics with H=Σpᵢ²/2+ΣCᵢⱼcos(φᵢ−φⱼ)
does not produce inflation-like growth for E8

**Cartan matrix properties:**

| Algebra | det | min_eig | trace |
|---------|-----|---------|-------|
| E8 | 1.0 | 0.011 | 16 |
| E7 | 2.0 | 0.030 | 14 |
| E6 | 3.0 | 0.068 | 12 |
| A6 | 7.0 | 0.198 | 12 |

**Key mathematical fact:** det(C_E8)=1 (unimodular).
min_eig(C_E8)=0.011 — E8 is near the stability boundary.
This is a theorem, not an observation.

**Dynamical results (N=20 random SPD matrices, κ=1.0):**

```
E8 spread_rate = 0.0020  (85th pct, p=0.150)
E6 spread_rate = 0.0002  (10th pct, p=0.900)
Random: 0.0015 ± 0.0023
```

**Inflation search (PC1 growth):**
```
E8 κ=0.5: 3.85 e-folds per 10⁵ steps
E8 κ=1.0: −0.79 e-folds (contracting!)
Required: >60 e-folds
```

**H₀ holds.** E8 Cartan dynamics indistinguishable
from random positive-definite matrix dynamics.

---

## Discussion

### Why classical Coxeter dynamics fails for inflation

1. **Weyl symmetry is too constraining.** Every signal
   that appears (complex frequencies, V_eff flatness,
   KAM thresholds) is cancelled by the Weyl involution
   mᵢ+m_{r+1−i}=h. This symmetry is a feature of the
   mathematics, not of physics.

2. **Power-law potentials have systematic bias.**
   V~φ^(2m/h) with small exponents (E8 has 2m₁/h≈0.07)
   always produces n_s→1, not n_s=0.965.

3. **Cartan matrix is near-singular.** det(C_E8)=1 and
   min_eig=0.011 make E8 dynamics nearly integrable,
   not chaotic enough for inflation.

4. **Coupling κ cannot be fixed from first principles.**
   Every test required choosing κ, which is a free
   parameter with no physical motivation.

### Mathematical theorems found

These results are valid mathematics independent of
the monostring hypothesis:

- Σcos(πmᵢ/h) = 0 for all finite Coxeter groups
- det(C_E8) = 1 (unimodular Cartan matrix)
- G2 KAM anomaly from ω₁=ω₂ resonance
- n_s systematic bias in power-law potentials

---

## Conclusion

Classical monostring dynamics on Coxeter algebras
(standard map, Cartan Hamiltonian, complex frequencies,
power-law potentials) does not reproduce inflationary
cosmology. After 16 experiments across 6 mathematical
frameworks, H₀ is not rejected.

**Remaining open directions** require qualitatively
different approaches: quantum field theory on E8 root
lattice, mutual information between Coxeter modes,
or connection to existing E8 unification proposals
(Lisi 2007).

The negative result is the result.
