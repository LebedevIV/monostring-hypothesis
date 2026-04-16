# The Monostring Hypothesis — Part VII
# Testing τ ∝ h(Coxeter) Across Lie Algebras

**Author:** Igor Lebedev  
**AI Collaborator:** Anthropic Claude  
**Follows:** Parts I–VI ([DOI 10.5281/zenodo.18886047](https://doi.org/10.5281/zenodo.18886047))  
**Status:** ❌ Hypothesis Falsified

---

## Abstract

Part VII tests whether the Memory Time τ≈237 discovered in Part VI
scales with the Coxeter number h across Lie algebras:
A6 (h=7), E6 (h=12), E7 (h=18), E8 (h=30).

Through nine iterations of adversarial computational experiments
(v1–v9), the hypothesis **τ ∝ h is definitively falsified.**

No memory metric (integrated ordering T_ord, fraction of
significant-negative time points f_neg, or count n_sig-)
scales with h. The best observed correlation is r=−0.425 (p=0.58),
with wrong sign. The τ/h≈20 coincidence observed for E6 in Part VI
does not generalize to other algebras.

---

## 1. Motivation

Part VI found that E6 daughter strings maintain lower Shannon
entropy than null-model strings for τ≈237 steps (p<0.0001).
The observation that τ/h = 237/12 ≈ 20, where h=12 is the
Coxeter number of E6, suggested a scaling law:

**τ(algebra) ≈ 20 × h(algebra)**

If confirmed, this would constitute the first evidence that
Lie algebra identity — not just the set of frequencies —
affects fragmentation dynamics.

---

## 2. Experimental Design

### 2.1 Algebras Tested

| Algebra | Rank | h | Exponents | τ predicted |
|---------|------|---|-----------|-------------|
| A6 | 6 | 7 | [1,2,3,4,5,6] | 140 |
| E6 | 6 | 12 | [1,4,5,7,8,11] | 240 |
| E7 | 7 | 18 | [1,5,7,9,11,13,17] | 360 |
| E8 | 8 | 30 | [1,7,11,13,17,19,23,29] | 600 |

### 2.2 Frequency Convention

Following Part VI (script v10):

$$\omega_i = 2\sin\!\left(\frac{\pi m_i}{h}\right)$$

This encodes algebraic structure through ratios $m_i/h$
while keeping frequencies bounded in $[0, 2]$.

### 2.3 Dynamics

Exact Part VI convention (nonlinear standard map with noise):

$$\phi_{n+1} = \phi_n + \omega + \kappa\sin(\phi_n)
+ \mathcal{N}(0,\,\sigma_\xi)$$

with $\kappa=0.05$, $\sigma_\xi=0.006$.

### 2.4 Null Model

**Population A** (Coxeter): N daughters initialized in
σ-ball around $\phi_{\rm break}$ (shared monostring origin).

**Population B** (null): N daughters initialized in
σ-ball around a **random** $\phi_B$ (no shared origin).

This is the exact Part VI null convention.

### 2.5 Memory Metrics

Because $\Delta S(t) = S(A) - S(B)$ oscillates
quasi-periodically (due to near-resonances in
$\omega = 2\sin(\pi m/h)$), the zero-crossing time τ is
ill-defined. Three integrated metrics were used:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| $T_{\rm ord}$ | $\int_0^{T_{\rm max}} \max(0, -\Delta S)\,dt$ | Total ordering area |
| $f_{\rm neg}$ | $\#\{t: \Delta S<0, p<0.05\} / \#\{t\}$ | Fraction sig-neg |
| $n_{\rm sig-}$ | $\#\{t: \Delta S<0, p<0.05\}$ | Count sig-neg |

---

## 3. Script Genealogy

Nine versions were required to isolate the physical effect
from methodological artifacts:

| Version | Critical discovery | Outcome |
|---------|-------------------|---------|
| v1 | Wrong frequencies: raw exponents vs 2·sin(π·m/h) | Fixed |
| v2 | Linear dynamics: ΔS(shuffle)=0 by permutation invariance | Found nonlinearity needed |
| v3 | κ=3.0: full chaos, D_corr=1.3 (not physical) | Found κ=0.05 from Part VI |
| v4 | D_corr(E6,GP)=4.09 consistently, not 3.02 | Accepted: different scales |
| v5 | S_orbit=2.99 (not 15.86): Part VI uses sum, not mean | Found exact formula |
| v6 | Thermalization metric: all algebras hit T_max | Metric inadequate |
| v7 | Reproduced Part VI v10 exactly | Validated pipeline |
| v8 | τ_crossover = first oscillation (t≈20-35): meaningless | Switched to integrated metrics |
| v9 | Integrated metrics: r(T_ord, h)=-0.012 | **Hypothesis falsified** |

---

## 4. Results

### 4.1 D_corr Under Different Conditions

| Condition | D_corr(E6) | Note |
|-----------|-----------|------|
| Linear flow, GP, pct 5-45 | 4.09 | Consistent across T=1k-15k |
| Linear flow, GP, pct 1-20 | 1.3-1.5 | Small-scale regime |
| Linear flow, GP, pct 10-40 | 3.83 | Closest to 3.02 |
| Nonlinear κ=0.05, GP | ~4.0 | Similar to linear |
| Nonlinear κ=3.0, GP | 1.3-2.1 | Chaotic regime |

**Part V/VI D_corr=3.02** was measured with a specific
r-range convention (pct 10-40, linear spacing) or a
graph-based algorithm. Both results are reproducible;
they characterize different geometric scales.

### 4.2 ΔS Profiles

| Algebra | n_sig- | n_sig+ | Dominant |
|---------|--------|--------|---------|
| A6 (h=7) | 7 | 1 | Ordering (A6 more compact) |
| E6 (h=12) | 0 | 17 | **ANTI-ordering** |
| E7 (h=18) | 0 | 15 | **ANTI-ordering** |
| E8 (h=30) | 2 | 10 | ANTI-ordering |

E6 and E7 show **significantly higher** entropy than the
shuffle null throughout. This is because the specific
permutation used grouped equal frequencies (e.g., E6 shuffle
clustered [0.518, 0.518] and [1.732, 1.732]), creating a
more compact null attractor.

### 4.3 Memory Metrics vs h

| Metric | Pearson r(metric, h) | p-value | Verdict |
|--------|---------------------|---------|---------|
| T_ord | −0.012 | 0.988 | No correlation |
| f_neg | −0.425 | 0.576 | No correlation |
| n_sig- | −0.425 | 0.576 | No correlation |

**All correlations are near zero or negative.**
The hypothesis τ ∝ h predicts r ≈ +1.0.

---

## 5. Root Cause Analysis: Why τ=237 in Part VI?

### 5.1 The actual Part VI measurement

Part VI measured:
- Population A: N daughters near **shared** $\phi_{\rm break}$
- Population B: N daughters near **random** $\phi_B$

The early signal $\Delta S < 0$ (A more ordered) is caused by:
1. **Shared origin effect:** A daughters start clustered
   together; B daughters start at a random location
2. **Attractor geometry:** E6 orbit has D_corr≈3-4 (compact);
   daughters thermalize onto this attractor

τ≈237 is the time for the initial σ-ball clustering of A
to dissipate as daughters spread across the attractor.

### 5.2 Why τ/h ≈ 20 is coincidental

The spread time depends on:
- σ (initial ball size): τ is σ-independent (Part VI finding)
- Attractor dimension D_corr: more compact → different rate
- Minimum frequency $\omega_{\rm min}$: sets oscillation period

For E6: $\omega_{\rm min} = 2\sin(\pi/12) \approx 0.518$,
oscillation period $\approx 2\pi/0.518 \approx 12$ steps.
τ≈237 ≈ 20 oscillation periods.

For other algebras, $\omega_{\rm min}$ is different,
so the same relation does not hold.

### 5.3 The shuffle sensitivity problem

For $\omega = 2\sin(\pi m/h)$, many exponents produce
equal or near-equal frequencies (e.g., E6: $m=5,7$
both give $\omega \approx 1.932$). A random permutation
may cluster these, changing attractor geometry and
reversing the sign of ΔS.

This makes the shuffled-null result **permutation-dependent**:
different random seeds give qualitatively different outcomes.

---

## 6. What Survives

| Finding | Status | Evidence |
|---------|--------|---------|
| D_corr(E6 orbit) ≈ 3-4 | ✅ Confirmed | Consistent across methods |
| Shared-origin daughters show initial ordering | ✅ Real | Part VI, Part VII A6 |
| Initial ordering decays on O(100) steps | ✅ Real | τ≈237 for E6 |
| τ ∝ h(Coxeter) | ❌ **Falsified** | r<0.43, p>0.57 |
| τ specific to E6 | ❌ Falsified | A6 shows same effect |
| τ/h≈20 is a scaling law | ❌ Falsified | Coincidental for E6 |

---

## 7. Methodological Lessons

1. **Oscillating ΔS requires integrated metrics.**
   Zero-crossing τ is meaningless when ΔS changes sign
   multiple times due to frequency near-resonances.

2. **Shuffle null is permutation-sensitive.**
   For symmetric frequency sets (repeated values),
   a random permutation may accidentally create a
   more compact attractor, reversing the effect direction.

3. **D_corr depends on measurement scale.**
   The same orbit gives D_corr=1.3 to 4.1 depending on
   the r-range percentiles. Both are geometrically correct
   at their respective scales.

4. **Validate against known result before extending.**
   Seven versions were needed because the baseline (E6 τ=237)
   was not reproducible until the exact Part VI dynamics
   were recovered (v7, the 8th attempt).

---

## 8. Conclusion

The hypothesis **τ ∝ h(Coxeter)** is definitively falsified.

The Memory Time τ≈237 discovered in Part VI is a real
physical effect — initial clustering decay on a compact
attractor — but it is not specific to E6 and does not
scale with the Coxeter number. The τ/h ≈ 20 observation
was coincidental.

The Monostring Hypothesis research programme has now
falsified all major quantitative predictions:

- ❌ 6D → 4D via Lyapunov (Part I)
- ❌ Gauge Higgs mechanism (Part II)  
- ❌ d_s = 4.0 as fixed dimension (Parts III-V)
- ❌ Dark energy as geometric inevitability (Part IV)
- ❌ Daughters converge to monostring orbit (Part VI)
- ❌ **τ ∝ h(Coxeter) (Part VII)**

What survives is a collection of robust computational
findings about the geometry of quasi-periodic orbits
on high-dimensional tori, which may have independent
mathematical interest.

---

## Appendix: Script Inventory

| Script | Key test | Outcome |
|--------|---------|---------|
| part7_v1.py | GP D_corr, raw exponents | D_corr=4.09, not 3.02 |
| part7_v2.py | Linear dynamics + diagnostics | ΔS=noise |
| part7_v3.py | φ·m/h frequencies | D_corr wrong scale |
| part7_v4.py | Raw exponents, arithmetic null | A6: ΔS=0 exactly |
| part7_v5.py | Parameter scan (T, n, pct) | No config gives 3.02 |
| part7_v6.py | Nonlinear κ=0.05-3.0 | κ=3.0 chaotic |
| part7_v7.py | Thermalization 95% metric | All hit T_max |
| part7_v8.py | Exact Part VI dynamics, τ_crossover | τ≈20-35 (oscillation) |
| part7_v9.py | Integrated metrics T_ord, f_neg | r(h)≈0, **falsified** |
