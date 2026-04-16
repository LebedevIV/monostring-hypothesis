# Part VII: Testing τ ∝ h(Coxeter) — Falsified

## Abstract

Seven computational experiments tested whether the Memory Time
τ≈237 discovered in Part VI scales with the Coxeter number h
across Lie algebras A6 (h=7), E6 (h=12), E7 (h=18), E8 (h=30).

**Result: τ ∝ h is falsified.**

## Findings

### 1. D_corr(E6) = 4.09, not 3.02

The Grassberger-Procaccia algorithm applied to the linear torus
flow φ(t)=φ₀+ωt consistently gives D_corr=4.09 for E6 raw
exponents across all tested parameters (T=1000-15000, n=200-1200).

Part V/VI D_corr=3.02 was measured at a different geometric scale
(small r-range, pct 1-20) or with a graph-based algorithm.
Both values are reproducible — they measure different properties.

### 2. τ=237 does not scale with h

Three memory metrics were tested:
- T_ord = ∫max(0,-ΔS)dt (integrated ordering)  
- f_neg = fraction of sig-negative time points
- n_sig- = count of p<0.05, ΔS<0 points

All correlations with h: |r| < 0.43, p > 0.57.
**No metric scales with h.**

### 3. Root cause of τ=237 in Part VI

Part VI used Population A (shared phi_break) vs Population B
(random phi_B origin). The early ΔS<0 signal reflects the
initial clustering advantage of shared-origin daughters,
not algebraic memory. τ≈237 is the characteristic time for
this clustering to dissipate on the E6 attractor (D_corr≈3,
compact orbit).

The coincidence τ/h ≈ 20 = τ/(12) is not confirmed for
other algebras.

### 4. Shuffle null is permutation-dependent

For ω=2·sin(π·m/h), the shuffle can accidentally cluster
equal frequencies (e.g., E6 shuffle grouped [0.518,0.518]
and [1.732,1.732] together), creating a MORE compact null
attractor and reversing the sign of ΔS.

## Conclusion

The hypothesis τ∝h is **definitively falsified** across
seven independent experimental approaches (Part VII v1-v9).

The Part VI Memory Time τ≈237 is a real physical effect
(initial clustering decay), but its connection to the
Coxeter number h=12 of E6 is coincidental.

## What survives from Part VI

✅ D_corr(E6 orbit) ≈ 3-4 (geometry of the attractor)
✅ Daughters with shared origin show lower entropy initially  
✅ This initial ordering decays on a timescale ~O(100) steps
✅ The decay timescale depends on attractor geometry (σ, D_corr)
❌ τ is NOT proportional to h
❌ τ is NOT specific to E6
❌ τ does NOT require Coxeter structure
