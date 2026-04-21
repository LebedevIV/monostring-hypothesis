# Part XVII: Mutual Information between Coxeter Modes

**Authors:** Igor Lebedev, Claude (Anthropic)
**Date:** 2025
**Status:** Falsified — Artifact #10 (Weyl pairing)

---

## Abstract

Mutual information I(φᵢ:φⱼ) was computed for all pairs
of modes in standard map dynamics on Coxeter algebras
E8, E6, E7, A6. Step 1 showed apparent massive signal:
TC(E8)=13.64 vs random=0.44, p=0.000. Step 2 with
structure-matched controls showed 98.9% of TC explained
by Weyl-paired modes (ωᵢ=ω_{r+1-i}). Matched controls
gave p=0.78. Artifact #10 confirmed and documented.

---

## Motivation

Direction #11 from the master research plan.
Physical interpretation: if I(Coxeter) >> I(random),
Coxeter modes "know" about each other — potential
mechanism for non-local correlations and horizon problem
solution without inflation.

---

## Step 1: Naive comparison

**Parameters:** T=100,000, warmup=10,000, κ=0.05,
N_bins=50, N_ctrl=200 (plain random per rank).

**Results:**

| Algebra | rank | TC | Random TC | p-value |
|---------|------|----|-----------|---------|
| E8 | 8 | 13.641 | 0.398±0.444 | 0.000*** |
| E6 | 6 | 10.148 | 0.202±0.301 | 0.000*** |
| E7 | 7 | 10.292 | 0.314±0.401 | 0.000*** |
| A6 | 6 | 12.979 | 0.202±0.301 | 0.000*** |

**MI matrix structure (E8):**
```
Top correlated pairs:
  φ3(ω=1.827) ↔ φ6(ω=1.827): MI=3.432  ← ω identical
  φ4(ω=1.956) ↔ φ5(ω=1.956): MI=3.432  ← ω identical
  φ1(ω=0.209) ↔ φ8(ω=0.209): MI=3.358  ← ω identical
  All non-Weyl pairs: MI ≈ 0.00–0.03
```

**Diagnostic:** r(|Δω|, MI) = −0.517 for E8.
MI not driven by frequency proximity — driven by equality.

---

## Step 2: Structure-matched controls

**Three control groups (rank=8, N=200 each):**

- **Control A** (plain random): all unique frequencies
- **Control B** (Weyl-paired): same pairing structure,
  random values — [a,b,c,d,d,c,b,a] pattern
- **Control C** (same structure as E8): identical
  degeneracy pattern, random frequency values

**Results:**

| Control | TC mean | TC std | E8 pct | p-value |
|---------|---------|--------|--------|---------|
| A (plain) | 0.436 | 0.475 | 100% | 0.000 ← unfair |
| B (Weyl-paired) | 14.035 | 0.812 | 20% | 0.800 |
| C (same struct) | 14.111 | 1.290 | 22% | 0.780 |

**TC decomposition:**

| Algebra | TC_Weyl | % | TC_nonWeyl | % |
|---------|---------|---|------------|---|
| E8 | 13.488 | 98.9% | 0.153 | 1.1% |
| E6 | 10.084 | 99.4% | 0.064 | 0.6% |
| E7 | 10.199 | 99.1% | 0.093 | 0.9% |
| A6 | 10.178 | 78.4% | 2.801 | 21.6% |

---

## Mechanism

```
Weyl involution: mᵢ + m_{r+1-i} = h  (theorem, Part IX)
→ ωᵢ = 2sin(πmᵢ/h) = 2sin(πm_{r+1-i}/h) = ω_{r+1-i}

Identical frequencies → nearly identical trajectories:
  φᵢ(t) ≈ φ_{r+1-i}(t) + const

→ MI(φᵢ, φ_{r+1-i}) ≈ H(φᵢ) ≈ log(N_bins) ≈ 3.9

E8 has 4 Weyl pairs × MI≈3.4 = 13.6 ≈ observed TC ✓
Random (plain) has 0 such pairs → TC ≈ 0.4 ✓
Random (Weyl-paired) has 4 such pairs → TC ≈ 14.0 ✓
```

---

## A6 Anomaly

A6 shows TC_nonWeyl = 2.801 (21.6%), significantly
higher than E8 (1.1%), E6 (0.6%), E7 (0.9%).

**Explanation:** A6 exponents m=[1,2,3,4,5,6] are
uniformly spaced → many mode pairs with small |Δω|
→ quasi-resonance → elevated non-Weyl MI.
This is a mathematical property of uniform spacing,
not a physical signal.

---

## Verdict

**H₀ not rejected** (p=0.78 vs structure-matched control).

**Artifact #10:** "Weyl-pairing creates spurious MI signal.
Always use structure-matched controls for MI tests."

**Lesson:** The apparent p=0.000 signal in Step 1 was
entirely due to comparing Coxeter (with Weyl pairs) to
plain random (without). This is the same Weyl degeneracy
identified in Part IX, manifesting in a new observable.

---

## Conclusion

Part XVII completes the mutual information investigation.
The monostring hypothesis is falsified across all 17
tested directions. The MI signal is Artifact #10.
