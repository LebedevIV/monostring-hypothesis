# Part X: Cayley Graphs and XXZ Spin Chains

**Authors:** Igor Lebedev, Claude (Anthropic)
**Date:** 2025
**Status:** ❌ Falsified — both approaches yield negative results

---

## Motivation

Parts I–IX falsified all standard-map-based formulations
of the monostring hypothesis. Part X tests two
mathematically distinct frameworks:

- **Plan A:** Cayley graph of the Weyl group W(E6)
- **Plan B:** XXZ spin chain at Coxeter-specific parameters

---

## Plan A: Cayley Graphs of Coxeter Groups

### Setup

Cayley graph Cay(W, S) where:
- W = Weyl group of a Lie algebra
- S = simple reflections (Coxeter generators)
- Edges: w ~ ws_i for each s_i ∈ S

### Results (Steps 1–3)

**Universal property (theorem):**
```
λ_max = 2 for ALL Coxeter groups, ALL generator sets.
Proof: sign representation χ_sign(s) = −1 for all s
→ λ_L(sign) = 1 − (−1) = 2. □
```

**Spectral gap scaling:**
```
Adjacent generators: λ₁ ~ |W|^{−0.46} → 0
All transpositions:  λ₁(S_n) = 2/(n−1) exactly
→ No useful signal; λ₁ → 0 as |W| → ∞
```

**Multiplicity structure:**
```
S_4: frac_degenerate = 0.800, max_mult = 3
S_5: frac_degenerate = 0.920, max_mult = 12
Random (same N, d): frac_degenerate = 0.000

Sharp separation — but tautological:
eigenvalue multiplicities = dim(irrep)² of W.
This is the Peter-Weyl theorem, not new physics.
```

**Ramanujan property:**
```
W(A_2), W(A_3): Ramanujan (λ₁ ≥ Alon-Boppana)
W(A_4), W(A_5): NOT Ramanujan
W(E6) via character table: λ₁ = 0.833 (Ramanujan)
→ But this depends on incomplete character table
  (Σdim² = 36,833 ≠ 51,840 = |W(E6)|)
```

### Conclusion A

The Cayley graph spectrum of W is exactly its character
table (Diaconis-Shahshahani formula). No new physical
content beyond known representation theory.

---

## Plan B: XXZ Spin Chain

### Setup

```
H = −Σᵢ [SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ + Δ·SᶻᵢSᶻᵢ₊₁]

Coxeter parameter: Δ(h) = −cos(π/h)
  E6: h=12 → Δ = −0.9659
  E8: h=30 → Δ = −0.9945
  A6: h=7  → Δ = −0.9010
```

Connection to Temperley-Lieb algebra:
```
At q = exp(iπ/h): TL_n(q) ≅ quotient of C[W(A_{n-1})]
→ Same q as Coxeter representation theory
```

### Results

**Spectral gap (n=8 spins):**
```
E6:  gap = 0.509
A6:  gap = 0.482
E8:  gap = 0.520
Random Δ ∈ [−1, −0.5]: mean = 0.432 ± 0.055

Mann-Whitney p = 0.008 — appears significant.
BUT: Coxeter Δ are closer to −1 than random sample.
Gap increases monotonically with |Δ| → −1.
Effect = range selection artifact, not algebra.
```

**Entanglement entropy:**
```
n=4..12, open boundary conditions:
  S(E6) ≈ S(A6) ≈ S(random Δ)  (< 2% difference)
No area-law vs log-law distinction.
```

**Key theorem (explains all results):**
```
|Δ(h)| = |cos(π/h)| < 1 for all finite h
→ All Coxeter algebras lie in critical phase of XXZ
→ gap → 0 as n → ∞ for ALL Coxeter algebras
→ No distinction between E6, A6, E8, random |Δ| < 1
```

**Dynkin-weighted coupling:**
```
J_i from Dynkin diagram vs uniform J=1 vs random J:
All give gap = 0 at Δ = Δ(E6). No signal.
```

### Conclusion B

The Coxeter-specific values Δ(h) = −cos(π/h) are all
in the critical (gapless) phase of XXZ by theorem.
No E6-specific signal in gap, entanglement, or
level statistics.

---

## Cumulative Falsification Record

| Framework | Parts | Key test | Result |
|-----------|-------|----------|--------|
| Standard map orbits | I–VII | D_corr, τ, z_geo | ❌ All artifacts or non-specific |
| Graph geodesic fields | VIII–IX | z_geo, PCA_ratio | ❌ Code artifact + Weyl degeneracy |
| Cayley graphs W(E6) | X-A | λ₁, multiplicity | ❌ Tautology (char table) |
| XXZ spin chain | X-B | gap, entropy, r̄ | ❌ Critical phase theorem |

---

## What Remains (Mathematical Theorems)

1. **Weyl involution:** m_i + m_{r+1-i} = h always
   → ω_i = ω_{h-i} → orbit on subtorus of dim = n_unique

2. **Character table:** Spec(Cay(W,S)) encodes all irreps
   → Beautiful but known (Peter-Weyl, 1927)

3. **Critical phase:** |Δ(h)| < 1 for all finite h
   → All Coxeter algebras are XXZ-critical

4. **Memory time:** τ ≈ 237 for E6 daughters (p<0.0001)
   → Real, but not proportional to h, not unique to E6

---

## Open Directions

**A. Quantum walks on Cay(W(E6)):**
Quantum coherence survival time vs random group.
Not yet tested; may differ from classical random walk.

**B. Random Matrix Theory:**
GUE matrices with imposed W(E6) symmetry vs plain GUE.
Well-defined statistical test with known theory.

**C. Methodology paper:**
Document the complete falsification chain as a
guide for AI-assisted hypothesis testing in physics.

---

## Reproducibility

```bash
python scripts/part10/part10_step1_cayley_small_groups.py
python scripts/part10/part10_step2_generators_scaling.py
python scripts/part10/part10_step3_spectral_fingerprint.py
python scripts/part10/part10_planb_xxz_spinchain.py
```

Expected runtime: ~10 minutes total.
All results deterministic (fixed seeds).
