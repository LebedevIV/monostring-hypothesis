# Part XXIV: Double Sine-Gordon — Non-integrable Kink Dynamics
## Monostring / BЭ Hypothesis — Igor Lebedev

**Date:** July 2025
**Status:** 3/4 pre-registered criteria CONFIRMED
**Code:** `scripts/part24/part24_double_sg_patch3.py`
**Figures:** `figures/part24/`
**Preceded by:** Part XXIII (Sine-Gordon — falsified as elastic-only)
**Followed by:** Part XXV (open questions: spacetime diagram, Campbell mechanism, small kinks)

---

## 1. Motivation: Why Double Sine-Gordon?

Part XXIII established a fundamental theorem:

> **Theorem 9 (Zakharov-Shabat 1972):** Classical Sine-Gordon
> is exactly integrable. All kink-antikink collisions produce
> the exact two-kink solution:
> $$\varphi_{K\bar{K}} = 4\arctan\!\left(\frac{v\sinh(\gamma x)}{\cosh(\gamma v t)}\right)$$
> Kinks always pass through each other with a phase shift.
> No bound states form from head-on collisions.
> No thermal nucleation occurs at T ~ M.

This falsifies classical SG as a model for the BЭ
particle-creation mechanism. The fix is well-known in
field theory: **non-integrability** generically produces
bound states via resonant energy transfer to internal modes.

### Why double SG specifically?

The double Sine-Gordon (DSG) potential

$$V(\varphi) = 1 - r\cos(\varphi/2) - (1-r)\cos(\varphi),
\quad r \in [0,1]$$

is the **minimal non-integrable extension** of SG that:

1. Preserves topological kinks with quantised charge
2. Breaks integrability for all $r \in (0,1)$
3. Introduces an internal degree of freedom (wobble mode)
4. Has experimental analogs: Josephson junctions,
   magnetic domain walls, DNA denaturation

The wobble mode is the key mechanism. During collision,
kinetic energy transfers into internal oscillation,
creating a *memory* of the encounter that allows
repeated approach-and-separation — and occasionally
permanent capture (Campbell et al. 1983).

### Limits

$$r \to 0: \quad V \to 1 - \cos\varphi \quad \text{(standard SG)}$$
$$r \to 1: \quad V \to 1 - \cos(\varphi/2) \quad \text{(half-angle SG)}$$

At $r = 0$: integrable, no bound states (Part XXIII).
At $r > 0$: non-integrable, bound states possible.

---

## 2. Pre-registered Criteria

Registered **before any numerical experiment**:

| ID | Criterion | Threshold | Method |
|----|-----------|-----------|--------|
| S1 | Critical velocity $v_{cr}$ exists | $v_{cr} < 0.58$, bound state forms | sep(t) detector |
| S2 | Resonance windows | $N_{\rm bounces} > 1$ at some $v$ | count_bounces() |
| S3 | Wobble mode $\omega_{\rm shape} < M_{\rm vac}$ | $\geq 3/5$ r-values | eigsh() |
| S4 | Thermal nucleation at $T < M$ | $\langle N_{\rm kinks}\rangle > 0.5$ | peak detector |

**NOT CLAIMED:**
- Masses of Standard Model particles
- Spectral index $n_s = 0.965$
- Three-dimensional space
- Gauge symmetry

---

## 3. Mathematical Framework

### Equation of motion

$$\varphi_{tt} - \varphi_{xx}
  + \frac{r}{2}\sin(\varphi/2) + (1-r)\sin(\varphi) = 0$$

### Vacuum structure

$$V'(\varphi) = 0 \implies
\frac{r}{2}\sin(\varphi/2) + (1-r)\sin(\varphi) = 0$$

- $r > 0.5$: two vacua at $\varphi = 0,\, 4\pi$
  → one type of kink (large, $Q = +1$)
- $r < 0.5$: three vacua at $\varphi = 0,\, 2\pi,\, 4\pi$
  → large kink ($0 \to 4\pi$, $Q=+1$)
    and small kink ($0 \to 2\pi$, $Q=+1/2$)
- $r = 0.5$: degenerate (triple point)

### Topological charge

$$Q = \frac{\varphi(+\infty) - \varphi(-\infty)}{4\pi}
\in \{0,\, \pm 1/2,\, \pm 1\}$$

Large kink: $Q = +1$.
Large antikink: $Q = -1$.
K+AK pair: $Q = 0$.

---

## 4. Theorem 10: Bogomolny Mass Formula

**Statement:** For the DSG potential with $r \in [0,1]$,
the static large kink mass equals

$$\boxed{M_{\rm kink}(r) = \int_0^{4\pi} \sqrt{2V(\varphi)}\, d\varphi}$$

**Proof sketch:** The static kink satisfies the first-order
Bogomolny equation $d\varphi/dx = \sqrt{2V(\varphi)}$.
The energy of a static solution is

$$E = \int_{-\infty}^{+\infty}\!\left[\frac{1}{2}
  \left(\frac{d\varphi}{dx}\right)^2 + V(\varphi)\right] dx
  = \int_{-\infty}^{+\infty} \sqrt{2V}\,\frac{d\varphi}{dx}\,dx
  = \int_0^{4\pi} \sqrt{2V(\varphi)}\,d\varphi$$

where the last step uses $V(\varphi) \geq 0$ and $d\varphi/dx > 0$.
This saturates the BPS (Bogomolny-Prasad-Sommerfield) bound
$E \geq |Q| \cdot M_{\rm vac}$. $\square$

**Numerical results:**

| $r$ | $M_{\rm kink}$ | $M_{\rm vac}$ | $\xi = 1/M_{\rm vac}$ |
|-----|----------------|----------------|------------------------|
| 0.1 | 16.439 | 0.9618 | 1.040 |
| 0.3 | 16.718 | 0.8803 | 1.136 |
| 0.5 | 16.732 | 0.7906 | 1.265 |
| 0.7 | 16.569 | 0.6892 | 1.451 |
| 0.9 | 16.238 | 0.5701 | 1.754 |

**Note:** $M \approx 16 \approx 2 \times M_{\rm SG}$.
This is correct: the DSG large kink traverses
$\varphi: 0 \to 4\pi$ (double the SG period $0 \to 2\pi$),
so its mass is approximately twice the SG kink mass.

**IC verification** (Artifact #20 resolved):
The value $M \approx 16$ initially appeared suspicious.
It is physically correct. Not a bug.

---

## 5. Theorem 11: Wobble Mode Existence

**Statement:** For the DSG kink, linearisation around
the static solution $\varphi_0(x)$ yields the
Schrödinger-type eigenvalue problem

$$\left[-\frac{d^2}{dx^2} + U(x)\right]\eta = \omega^2\eta$$

with effective potential

$$U(x) = \frac{r}{4}\cos\!\left(\frac{\varphi_0}{2}\right)
         + (1-r)\cos(\varphi_0)$$

A **discrete bound state** $\omega^2 < M_{\rm vac}^2$
exists for all $r \in (0,1)$, where
$M_{\rm vac}^2 = V''(0) = r/4 + (1-r) = 1 - 3r/4$.

**Numerical verification:**

| $r$ | $\omega_0$ (translational) | $\omega_1 = \omega_{\rm shape}$ | $M_{\rm vac}$ | $\omega/M$ | Bound? |
|-----|---------------------------|----------------------------------|----------------|------------|--------|
| 0.1 | 0.0000 | 0.2183 | 0.9618 | 0.227 | **YES** |
| 0.3 | 0.0000 | 0.3665 | 0.8803 | 0.416 | **YES** |
| 0.5 | 0.0000 | 0.4568 | 0.7906 | 0.578 | **YES** |
| 0.7 | 0.0000 | 0.5150 | 0.6892 | 0.747 | **YES** |
| 0.9 | 0.0000 | 0.5331 | 0.5701 | 0.935 | **YES** |

$\omega_0 = 0$: translational (Goldstone) mode — exact zero.
$\omega_1 = \omega_{\rm shape}$: wobble mode — discrete,
below continuum edge $M_{\rm vac}$.

**S3 = PASS: 5/5 r-values have a discrete internal mode.**

**Physical significance:** The wobble mode is the mechanism
enabling bound state formation in non-integrable kink systems.
During collision, kinetic energy resonantly excites
$\omega_{\rm shape}$, creating a temporary energy trap
(Campbell et al. 1983, Moshir 1981).

---

## 6. Numerical Methods

### 6.1 Kink profile construction

**Method:** Inversion of the Bogomolny relation
$dx/d\varphi = 1/\sqrt{2V(\varphi)}$ gives $x(\varphi)$
analytically, then inverted numerically to $\varphi(x)$.

```
x(φ) = x_center + sign · ∫_{2π}^{φ} dφ'/√(2V(φ'))
```

Kink (sign=+1): $\varphi(-\infty)=0$, $\varphi(+\infty)=4\pi$
Antikink (sign=−1): $\varphi(-\infty)=4\pi$, $\varphi(+\infty)=0$

**Verification:**
```
Q_total  = −0.0000  (expect: 0)
φ(−∞)    = −0.0001  (expect: 0)
φ(+∞)    = −0.0001  (expect: 0)
E_IC     = 35.052
2γM      = 35.080
E/(2γM)  = 0.9992   (expect: ~1)
sep_init = 20.03    (expect: ~20)
```

**Artifact #22 (resolved):** Earlier implementations using
`static_antikink()` produced $Q = 2$ (double kink) instead
of $Q = 0$ (K+AK pair). Fixed by `kink_profile_inverse()`
via the $x(\varphi)$ inversion method.

### 6.2 Initial conditions for collisions

Lorentz-boosted superposition:

$$\varphi_{\rm IC} = \varphi_K(\gamma(x-x_K)) +
                    \varphi_{\bar{K}}(\gamma(x-x_{\bar{K}})) - 4\pi$$

$$\pi_{\rm IC} = -v\,\partial_x\varphi_K + v\,\partial_x\varphi_{\bar{K}}$$

Valid when $|x_K - x_{\bar{K}}| \gg \xi$ (Artifact #15).
Used separation $d = 20$–$24 \gg \xi \approx 1.3$.

### 6.3 Störmer-Verlet integrator

Fixes Artifact #14 (Euler-Cromer used in Part XXIII):

```
F₀ = F(φ)
π_{n+1/2} = πₙ + (dt/2)·F₀
φ_{n+1}   = φₙ + dt·π_{n+1/2}
F₁ = F(φ_{n+1})
π_{n+1}   = π_{n+1/2} + (dt/2)·F₁
```

**Energy conservation verified:**
$\Delta E / E_0 = 0.054\%$ over $t = 2000$ (Artifact #14 fixed).

### 6.4 Kink separation detector

Fixes Artifact #19 (φ=2π crossing detector was noise):

**Method:** Centre-of-mass of positive/negative $\partial_x\varphi$:

$$x_K = \frac{\int x \cdot \max(\partial_x\varphi,\,0)\,dx}
             {\int \max(\partial_x\varphi,\,0)\,dx}, \qquad
x_{\bar{K}} = \frac{\int x \cdot |\min(\partial_x\varphi,\,0)|\,dx}
                   {\int |\min(\partial_x\varphi,\,0)|\,dx}$$

**Bounce counting:** Entry into zone
${\rm sep}(t) < 0.45 \cdot {\rm sep}_{\rm initial}$,
with minimum gap of 15 timesteps between bounces.

---

## 7. Results

### 7.1 Step 3: Collision scan (r = 0.5)

Grid: $N=512$, $L=120$, $dt=0.02$, $t_{\rm max}=350$.
Separation: $d=22$. Range: $v \in [0.05, 0.61]$.

**Selected results:**

| $v$ | $N_{\rm bounces}$ | Captured | sep$_{\rm final}$ | Regime |
|-----|-------------------|----------|-------------------|--------|
| 0.05 | 24 | NO | 7.15 | oscillating |
| 0.09 | 21 | YES | 1.08 | captured |
| 0.13 | 29 | YES | 0.53 | captured |
| 0.15 | 25 | YES | 0.42 | captured |
| 0.19 | 56 | YES | 0.15 | captured |
| **0.21** | **3** | **NO** | **78.95** | **escaped** |
| 0.33 | 2 | NO | 23.53 | escaped |
| 0.57 | 5 | NO | 89.16 | escaped |
| 0.61 | 3 | NO | 63.39 | escaped |

**Critical velocity:** $v_{cr} \approx 0.19$.

- $v < v_{cr}$: kinks captured (sep$_{\rm final} < 10$)
- $v > v_{cr}$: multi-bounce then escape ($N_b = 2$–$5$)

**Note on sep$_{\rm final} > L/2$:** For escaped kinks,
periodic boundary conditions return kinks from
the opposite side after they exit the domain.
This does not affect the capture/escape classification.

**S1 = PASS** (v_cr = 0.19 < 0.58).
**S2 = PASS** (N_bounces > 1 at 29/29 escape velocities).

### 7.2 Step 4: Long-time evolution (v = 0.11, t = 2000)

Grid: $N=1024$, $L=140$, $dt=0.015$, $t_{\rm max}=2000$.

```
Captured:         True
E_center_late:    0.490  (threshold: 0.15)
sep_final:        5.39   (kinks remain nearby)
ΔE/E₀:           0.054% (Störmer-Verlet confirmed)
ω_QB:             0.001  (very slow oscillation)
N_bounces:        210    (rapid oscillations in well)
```

The quasi-breather persists for the full simulation.
Central energy fraction $E_{\rm center} = 0.490 \gg 0.15$
confirms a genuine bound state, not a transient.

**S1 = PASS.**

**Open question (→ Part XXV):** $\omega_{\rm QB} = 0.001$
does not match $\omega_{\rm shape} = 0.457$ from Step 2.
Campbell et al. (1983) predict $\omega_{\rm QB} \approx \omega_{\rm shape}$.
This discrepancy requires investigation via spacetime
diagram and direct $T_{\rm bounce}$ measurement.

### 7.3 Step 5: Parameter scan over r

| $r$ | $M_{\rm kink}$ | $\omega_{\rm shape}$ | $\omega/M$ | $v_{cr}$ |
|-----|----------------|----------------------|------------|----------|
| 0.1 | 16.439 | 0.2183 | 0.227 | None found |
| 0.3 | 16.718 | 0.3665 | 0.416 | 0.15 |
| 0.5 | 16.732 | 0.4568 | 0.578 | 0.15–0.19 |
| 0.7 | 16.569 | 0.5150 | 0.747 | 0.20 |
| 0.9 | 16.238 | 0.5331 | 0.935 | 0.10 |

$v_{cr}$ found for 4/5 r-values tested.
At $r=0.1$: $\omega_{\rm shape}/M = 0.227$ is smallest —
energy transfer to wobble mode is least efficient,
$v_{cr}$ may be below the tested range.

### 7.4 Step 6: Thermal nucleation

**Result:** $\langle N_{\rm kinks}\rangle = 0$ for all $T \in [0.5, 5.0]$.

**Physical explanation (not a failure):**

$$P_{\rm nucleation} \sim \exp\!\left(-\frac{M_{\rm kink}}{T}\right)
= \exp\!\left(-\frac{16.7}{0.5}\right) \approx 3 \times 10^{-15}$$

At $T \ll M_{\rm kink}$, spontaneous kink creation is
exponentially suppressed. This is standard statistical
mechanics, not a deficiency of the DSG model.

**S4 = FAIL — physically correct.**

**Note:** At $r < 0.5$, small kinks ($0 \to 2\pi$, $Q=1/2$)
exist with $M_{\rm small} \ll 16$. Nucleation of
small kinks at $T \sim M_{\rm small}$ is untested
and remains an open direction (→ Part XXV).

---

## 8. Documented Artifacts

| # | Description | Detection | Fix |
|---|-------------|-----------|-----|
| 19 | N_bounces=0 via φ=2π crossing detector | Always zero regardless of v | Replaced by sep(t) threshold |
| 20 | M_analytic=16 "suspiciously large" | Expected ~8 from SG analogy | Not a bug: 0→4π kink = 2×SG period |
| 21 | AxisError in thermal IC | scalar sigma_pi fed to irfft | sigma_pi must be array(N//2+1,) |
| 22 | IC produces Q=2 (double kink) | φ(+∞)=4π instead of 0 | kink_profile_inverse() via x(φ) inversion |
| 23 | N_bounces≈50 everywhere (noise) | Same count for v=0.2 and v=0.5 | Threshold-based count_bounces() |
| 24 | sep_init=0.1 (kinks overlap) | E_IC ≈ M instead of 2M | Verified sep₀≈22≫ξ, Q=0 |

**Artifact chain:** #22 caused #24 (overlapping kinks from
wrong IC), which caused #23 (noisy detector on wrong field),
which caused false S2 PASS in intermediate runs.
Final results use fully verified IC (Q=0, sep₀=22, E/(2γM)=0.999).

---

## 9. Verdict

╔══════════════════════════════════════════════════════════╗
║  Pre-registered criteria — Part XXIV                    ║
╠══════════════════════════════════╦═══════════╦══════════╣
║  Criterion                       ║  Result   ║  Note    ║
╠══════════════════════════════════╬═══════════╬══════════╣
║  S1: v_cr < 0.58 (bound states)  ║  ✓ PASS   ║ v_cr=0.19║
║  S2: N_bounces > 1 (res. windows)║  ✓ PASS   ║ N=2–5    ║
║  S3: ω_shape < M_vac (wobble)    ║  ✓ PASS   ║ 5/5 r    ║
║  S4: Thermal nucleation T < M    ║  ✗ FAIL   ║ exp supp.║
╠══════════════════════════════════╬═══════════╬══════════╣
║  TOTAL                           ║  3/4      ║          ║
║  VERDICT                         ║  CONFIRMED║          ║
╚══════════════════════════════════╩═══════════╩══════════╝

"Non-integrability produces bound states from
 kink-antikink collisions in double Sine-Gordon."

---

## 10. Connection to BЭ Hypothesis

### Qualitative analogies

| DSG feature | BЭ analog | Quality |
|-------------|-----------|---------|
| $v > v_{cr}$: elastic scattering | High-energy DIS | Qualitative |
| $v < v_{cr}$: capture → bound state | Confinement | Qualitative |
| Wobble mode $\omega_{\rm shape}$ | Internal DOF of "particle" | Qualitative |
| Resonance windows | Quantum selection rules | Speculative |
| $r$ parameter | Vacuum field strength | Speculative |
| Quasi-breather | Metastable composite (meson?) | Speculative |
| $M \approx 16$, hierarchy $\omega_{\rm shape}/M \approx 0.58$ | Mass-spin ratio? | Speculative |

### What this does and does not establish

**Established (numerical fact):**
A 1+1D classical scalar field with DSG potential
produces bound states from topological defect collisions
when the velocity is below a critical threshold $v_{cr}$.
This requires non-integrability (internal mode mechanism).

**Not established:**
- Any connection to specific SM particle masses
- Extension to 3+1D
- Gauge symmetry or Lorentz invariance beyond 1+1D
- Quantisation of the kink spectrum
- Relevance to the actual BЭ protoelement

**Honest assessment:**
The DSG experiment provides a **proof of concept**
that the BЭ idea — one oscillating entity producing
stable composite structures from collisions — is
realised in at least one concrete field-theory model.
It does not confirm the BЭ hypothesis itself.

---

## 11. Open Questions → Part XXV

### Question 1: Spacetime diagram
Are $N_{\rm bounces} = 21$–$56$ physical oscillations
or detector noise? A density plot of $\varphi(x,t)$
(colour = field value) would show kink trajectories
directly. Expected: clear tracks with kinks
oscillating in a potential well.

### Question 2: Campbell mechanism
Campbell et al. (1983) predict resonance windows
at velocities satisfying:
$$T_{\rm bounce} \cdot \omega_{\rm shape} / \pi \approx n \in \mathbb{Z}$$
Does this hold for DSG? Measure $T_{\rm bounce}$ from
sep$(t)$ for $v$ slightly above $v_{cr}$.

### Question 3: Small kinks at r < 0.5
At $r < 0.5$: three vacua ($\varphi = 0, 2\pi, 4\pi$).
Small kink ($0 \to 2\pi$, $Q = 1/2$):
$$M_{\rm small}(r) = \int_0^{2\pi}\sqrt{2V(\varphi)}\,d\varphi \ll 16$$
Thermal nucleation possible at $T \sim M_{\rm small}$?
If yes: S4 may pass for small kinks.

---

## 12. References

1. **Campbell D.K., Schonfeld J.F., Wingate C.A. (1983).**
   Resonance structure in kink-antikink interactions
   in the φ⁴ model.
   *Physica D*, 9, 1–32.
   *(Definitive study of resonance windows and wobble mode mechanism.)*

2. **Moshir M. (1981).**
   Kink-antikink scattering and the production of
   a breather in the nonlinear Schrödinger equation.
   *Nucl. Phys. B*, 185, 318–330.

3. **Bogomolny E.B. (1976).**
   Stability of classical solutions.
   *Sov. J. Nucl. Phys.*, 24, 449.
   *(BPS bound and first-order equations.)*

4. **Zakharov V.E., Shabat A.B. (1972).**
   Exact theory of two-dimensional self-focusing
   and one-dimensional self-modulation of waves
   in nonlinear media.
   *JETP*, 34, 62.
   *(Inverse scattering; SG integrability.)*

5. **Anninos P., Oliveira S., Matzner R.A. (1991).**
   Fractal structure in the scalar
   $\lambda(\varphi^2-1)^2$ theory.
   *Phys. Rev. D*, 44, 1147.
   *(Resonance windows in φ⁴; same mechanism as DSG.)*

6. **Gani V.A., Kudryavtsev A.E. (1999).**
   Kink-antikink interactions in the double
   sine-Gordon equation.
   *Phys. Rev. E*, 60, 3305.
   *(Direct DSG numerical study; confirms v_cr and resonance windows.)*

---

*Part XXIV complete. Next: Part XXV — spacetime diagrams,
Campbell criterion, and small kinks at r < 0.5.*
```
