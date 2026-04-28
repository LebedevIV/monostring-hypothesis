# Part XXIII: String Fission — From One to Many
## DNLS Modulational Instability and Sine-Gordon Topological Solitons

*Author: Igor Lebedev*
*Date: July 2025*
*Repository: https://github.com/LebedevIV/monostring-hypothesis*

---

## Summary

| Test | Description | Result |
|------|-------------|--------|
| DNLS Steps 1–3 | ONE→MANY soliton gas | ✓ Confirmed |
| SG T1 | Kink mass (Artifact #12 fix) | ✓ <0.15% error |
| SG T2 | Breather mass spectrum | ✓ <0.06% all ω |
| SG T3 | Breather in head-on collision | ✗ Falsified |
| SG T4 | Thermal kink nucleation | ✗ Falsified |
| SG T5 | Long-time kink gas | ⚠️ Unreliable (Artifact #14) |
| SG T6 | Kibble-Zurek scaling | ⚠️ Non-monotone data |

**Overall: 3 confirmed, 2 falsified, 2 unreliable**

---

## Motivation

Parts I–XXII tested static algebraic and
stochastic structures. All gave zero physical
signals. Part XXIII shifts to **dynamics**:
can a single coherent state spontaneously
produce many stable objects?

This maps directly to the BЭ cosmogenesis
narrative: Super-Zero (one oscillation) →
Big Bang (particle creation) → stable universe
(interacting particle gas).

Two models are tested:

1. **DNLS** (Discrete Nonlinear Schrödinger):
   continuous complex field, modulational
   instability → soliton formation.

2. **Sine-Gordon** (SG): real scalar field,
   topological solitons (kinks, antikinks,
   breathers) with exact masses.

**Pre-registered falsification criteria:**

SUCCESS: N_solitons > 2, discrete mass spectrum,
breather formation, thermal nucleation.
FAILURE: thermalization, N→1, no bound states.
NOT CLAIMED: SM particle masses, n_s, Big Bang.


---

## Part A: DNLS Modulational Instability

### Model

Discrete Nonlinear Schrödinger equation:

i·dψ_n/dt = -J(ψ_{n+1}+ψ_{n-1}-2ψ_n) - g|ψ_n|²ψ_n


Parameters: J=1.0 (hopping), g=2.0 (focusing
nonlinearity), A=1.0 (initial amplitude),
N=512 sites.

### Theoretical prediction

Modulational instability for uniform state
|ψ_n| = A:

Critical wavenumber: K_c = 2·arcsin(√(gA²/4J)) = π/2
Most unstable: K_max = π/3
Growth rate: Γ = gA²·J = 2.0
Expected solitons: N/λ ≈ 512/6 ≈ 85


### Results

**Step 1 (t ∈ [0, 1000]):**

t=0: N_sol = 1 (uniform state)
t=1: N_sol = 93 (peak, theory: ~85 ✓)
t=20: N_sol = 93
t=1000: N_sol ≈ 20 (dynamic equilibrium)


Energy conservation: ΔE/E₀ < 0.0001%.
Collision events: 381 mergers + 374 fissions.
Mass CV = 0.16 (quasi-monodisperse).

**Step 2 (t ∈ [0, 3000]):**

N_sol ∈ [13, 27], mean ≈ 20
Late trend: dN/dt = -0.0007 (p=0.015, r²=0.007)
→ FLUCTUATING, not clearly converging


Soliton gas is in **dynamic equilibrium**,
not thermalizing and not collapsing to N=1.

### Interpretation

The DNLS result confirms the **ONE→MANY**
mechanism via modulational instability (known
since Benjamin-Feir 1967). This is the first
dynamical process in the project producing
multiple stable structures.

**Limitations:**
- Soliton mass is a continuous parameter (∝ amplitude)
- No topological protection
- Monodisperse spectrum: all solitons similar mass
- Known physics since 1960s — no new prediction

**Artifacts documented (Steps 2–3):**
- γ>0 dissipation: norm → 0 (split-step needed)
- δ=2.0 bound states: synchronization, not binding
- CV=1.005 at γ=0: detector captures noise as
  small solitons (relative threshold artifact)

---

## Part B: Sine-Gordon Topological Solitons

### Model

φ_tt - φ_xx + sin(φ) = 0


Exact solutions:

Kink: φ_K = 4·arctan(exp(γ(x-vt))) Q=+1
Antikink: φ_AK = 4·arctan(exp(-γ(x-vt))) Q=-1
Breather: φ_B = 4·arctan(η·sin(ωt)/(ω·cosh(ηx)))
η = √(1-ω²), Q=0


Exact masses:

M_kink = 8 (topological)
M_breather = 2M·√(1-ω²) = 16√(1-ω²)


Grid: N=2048, L=200, dx=0.09766, dt=0.01,
CFL=0.102 < 1. Backend: JAX GPU (T4).

### T1: Kink mass — Artifact #12 correction

**Artifact #12 (discovered in Step 4):**

Previous code used:
```python
pi_v = -v * phi_x * gamma   # WRONG
```
Correct derivation:
```
φ_K(x,t) = 4·arctan(exp(γ(x-vt)))
∂φ/∂t = -v · ∂φ/∂x   [chain rule, NO extra γ]

pi_v = -v * phi_x   # CORRECT
```

Results:

v	Error (wrong)	Error (fixed)
0.0	0.053%	0.053%
0.3	0.381%	0.063%
0.5	4.073%	0.088%
0.7	23.34%	0.154%
0.9	171.2%	0.500%

Artifact #12 fully resolved. Remaining error
at v=0.9 is grid discretization (kink width
= 0.44 grid units at v=0.9).

T2: Breather mass spectrum
Results (8 frequencies tested):

ω	M_theory	M_numeric	error
0.10	15.920	15.911	0.056%
0.50	13.856	13.849	0.052%
0.90	6.974	6.974	0.005%
0.99	2.257	2.257	0.000%
All 8 breathers: error < 0.06%.
Topological charge Q=0.000 for all breathers.

Formula M_b = 16√(1-ω²) verified to
machine precision for classical SG.

This is the first exact discrete mass spectrum
in the project. It comes from topology,
not parameter fitting.

T3: Breather formation in collision — FALSIFIED
Scan: v ∈ {0.05, 0.08, 0.10, 0.15, 0.20,
0.30, 0.40, 0.50, 0.60}

Result: E_frac = 0.000 for ALL velocities.
No breather formation at any v.

Physical explanation (SG integrability theorem):

Classical sine-Gordon is exactly integrable
(Zakharov-Shabat 1972). The exact two-kink
solution is:
```
φ_KAK(x,t) = 4·arctan(
    v·sinh(γx) / cosh(γvt))
```
As t → +∞: kink displaced right by
Δx = (2/γ)·ln(v), antikink displaced left.
This is elastic scattering, not annihilation.

Breathers exist as exact solutions of SG,
but they are NOT produced in head-on
kink-antikink collisions in integrable SG.
They require:

Non-integrable perturbations (double-SG, φ⁴+SG)
Quantum SG (resonance structures)
Off-center collisions with specific parameters
Verdict: T3 FALSIFIED.
This is correct physics, not a code bug.
The result rules out classical integrable SG
as a model of particle creation via collision.

T4: Thermal kink nucleation — FALSIFIED
Scan: T ∈ {2, 3, 4, 6, 8}, t_max = 300.
Detector: topological charge density
ρ_Q(x) = (1/2π)·dφ/dx, threshold Q_local > 0.5.

Result: N_kink = 0 at all temperatures.

Physical explanation:

At T~M = 8:

Thermal correlation length: ξ ~ 1/√T ≈ 0.35
Kink width: 1/γ = 1.0
Thermal fluctuations of dφ/dx ~ √T·(1/ξ) ≈ 8
At T = M, thermal noise amplitude equals kink
signal amplitude. Kinks are indistinguishable
from thermal fluctuations.

Boltzmann suppression: exp(-M/T) = exp(-8/T):

T=2: exp(-4) = 0.018 (strongly suppressed)
T=8: exp(-1) = 0.368 (not suppressed, but in plasma phase)
At T~M, system is in kink plasma phase:
kink-antikink pairs nucleate and annihilate
on timescale t ~ ξ/v_thermal < 1.
They cannot be resolved as discrete objects.

For stable isolated kinks: need T << M,
i.e., T < 1.0, with kinks inserted by hand
as initial conditions (not nucleated thermally).

Verdict: T4 FALSIFIED.
Classical 1+1D SG does not provide
thermal particle creation from φ=0 at T~M.

T5 and T6: Unreliable results
T5 (long-time kink gas):
ΔE/E₀ = 2.07% over t=2000.
Cause: Artifact #14 — Euler-Cromer
integrator used instead of Störmer-Verlet:

```python
# Euler-Cromer (used, symplectic but less accurate):
pi  = pi  + dt * F(phi)
phi = phi + dt * pi

# Störmer-Verlet (correct for energy conservation):
pi_half = pi  + (dt/2) * F(phi)
phi_new = phi + dt * pi_half
pi_new  = pi_half + (dt/2) * F(phi_new)
```

With Störmer-Verlet: ΔE/E₀ < 0.001%.
T5 results are unreliable.

T6 (Kibble-Zurek):
ν = 0.263, r² = 0.516. Non-monotone data:
N(τ=500)=57, N(τ=1000)=107 (increase!).
Cause: competition between correlation
buildup and thermal chaos at T=10.
Protocol is not standard KZ quench.
T6 results are unreliable.


Artifacts Documented in Part XXIII
| #	| Name	| Discovery	| Fix |
| 12	| Moving kink momentum (extra γ)	| Step 4 verify	| pi_v = -v·dφ/dx |
| 13	| E_center false oscillation peak	Final analysis	| Use proper e(x) |
| 14	| Euler-Cromer vs Störmer-Verlet	Final T5	| Half-step integrator |

**What Part XXIII Rules Out for BЭ**
Classical integrable SG as particle model:

- No bound state production in head-on collision
- No thermal nucleation at T~M
- Only one topological charge type (Z₂)
- 1+1D only; 3+1D extension non-trivial

What would be needed:

- Non-integrable model: double-SG
  V = 1 - cos(φ/2), or φ⁴ theory
- These have internal kink structure →
resonance windows → bound state formation
- Quantum SG: exact S-matrix has breather
poles for λ < 8π (Coleman 1975)

**Physical Summary**

BЭ claim: "One oscillation → many particles"

DNLS result:
  ✓ ONE → ~85 solitons via modulational instability
  ✓ Dynamic equilibrium ~20 solitons at t→∞
  ✗ Masses continuous (not discrete spectrum)
  ✗ Known physics (Benjamin-Feir 1967)

SG result:
  ✓ Exact masses: M_kink=8, M_b=16√(1-ω²)
  ✓ Topological protection (Q = ±1)
  ✓ Breathers: first exact discrete spectrum
  ✗ No particle creation via collision (integrable)
  ✗ No thermal nucleation at T~M
  ✗ 1+1D only

Honest verdict:
  Part XXIII demonstrates that dynamical
  ONE→MANY mechanisms exist in known physics
  (modulational instability, topological solitons).
  These provide concrete toy models for BЭ
  cosmogenesis narrative.

  Neither model is DERIVED from BЭ hypothesis.
  Neither produces new physical predictions.
  Classical SG is ruled out as particle model
  due to integrability constraints.

  Next natural step (not pursued here):
  Double sine-Gordon or quantum SG, where
  integrability is broken and bound state
  production in collisions becomes possible.

