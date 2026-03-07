[![DOI: Paper](https://img.shields.io/badge/DOI-Paper%20(PDF)-blue.svg)](https://doi.org/10.5281/zenodo.18886048)
[![DOI: Code](https://zenodo.org/badge/DOI/10.5281/zenodo.18890266.svg)](https://doi.org/10.5281/zenodo.18890266) 

# The Sole Oscillator Hypothesis: Seven Computational Experiments That Killed One Path to Emergent Spacetime — and Opened Three Others

**Authors:** Igor [Your Last Name] (concept, task setting, computational experiments), Gemini 3.1 Pro (initial mathematical implementation, script v0), Claude Opus / Sonnet (critical analysis, mathematical apparatus, scripts v1–v7)

---

## Abstract

We test the Sole Oscillator Hypothesis (SOH) — the conjecture that all of observable reality emerges from the discrete state transitions of a single vibrating entity (informally dubbed "the Monostring"). The initial implementation, developed with Google Gemini 3.1 Pro, generated a resonance graph of 150,000 nodes and claimed emergent 6-dimensional geometry, a discrete mass spectrum, and a mechanism for quantum non-locality. Seven subsequent iterations of adversarial computational testing (scripts v1–v7), conducted with Anthropic Claude, systematically dismantled these claims: (1) the original results were construction tautologies of a random geometric graph on a 6-torus; (2) a genuinely non-trivial dimensional reduction $6 \to 4$, discovered upon introducing $E_6$-coupled nonlinear dynamics with Coxeter frequencies, proved to be an artifact of dissipative (non-Hamiltonian) dynamics; (3) the symplectic (physically correct) version of the map yields $D_{KY} = 2r$ identically for all tested Lie algebras, ruling out the Lyapunov compactification mechanism entirely. 

The Sole Oscillator Hypothesis in its current mathematical formulation is **falsified**. As a secondary finding, we report a universal $D_{KY} \approx 4$ plateau in dissipative coupled standard maps with Cartan-matrix coupling across all simple Lie algebras of ranks 4–8, which represents a result of independent interest in dynamical systems theory. We outline three alternative mathematical frameworks — causal sets, quantum walks, and stochastic quantization — through which the philosophical core of the Monostring conjecture might be reformulated.

**Keywords:** emergent spacetime, Sole Oscillator Hypothesis, Monostring, coupled standard maps, Kaplan-Yorke dimension, Lie algebras, E6, resonance graphs, falsification, AI-assisted research

---

## 1. One Electron, One Universe

In 1940, the great physicist John Wheeler called his student Richard Feynman and declared: "Richard, I know why all electrons in the Universe are absolutely identical! Because they are all the same electron, moving back and forth in time." This beautiful concept of the "one-electron universe" entered the history of science as one of its most audacious thought experiments — but it remained an elegant metaphor. Feynman himself noted that the idea required equal numbers of electrons and positrons, which is not observed [1]. Wheeler's electron could not become Wheeler's universe.

The Sole Oscillator Hypothesis (SOH) — which we informally call the **Monostring conjecture** — attempts exactly this leap. Where Wheeler asked "why are all electrons identical?", the SOH asks a far more radical question: **why does a multi-dimensional universe exist at all?**

The answer proposed by the SOH is startlingly minimalist. At the foundation of reality, there is no "zoo" of particles, no pre-existing empty space, and no external clock. There is only one entity — the Sole Oscillator — possessing fundamental energy and incapable of rest. It sequentially and discretely changes its states at an unimaginable (presumably Planck-scale) frequency. The chronology of these states forms a strict one-dimensional timeline: $tick_1, tick_2, \ldots, tick_n$.

The Sole Oscillator is not featureless. It possesses internal degrees of freedom — phases oscillating at multiple irrational frequencies. When the phase signatures of two states match within some precision $\varepsilon$, even if separated by billions of ticks on the 1D timeline, a "resonance" occurs — a topological connection between those states. Through this mechanism, the one-dimensional thread of time is pulled and folded into a multi-dimensional network. What we perceive as three-dimensional space (and, more broadly, 3+1-dimensional spacetime) is the emergent geometry of this resonance web. Particles are standing waves; forces are traveling waves; quantum entanglement is a direct topological shortcut between phase-matched states.

This is a beautiful picture. This paper is the story of how we tested it — honestly, systematically, and ultimately to destruction.

The journey involved three participants with distinct roles. The human author provided the hypothesis, philosophical motivation, and experimental direction. Google Gemini 3.1 Pro built the first mathematical implementation. Anthropic Claude served as the adversarial critic — designing null models, identifying tautologies, and constructing the falsifying experiments that ultimately killed the specific mechanism (though not, as we shall argue, the underlying idea).

---

## 2. What Gemini Built: The First Monostring (Script v0)

### 2.1 The Construction

The initial mathematical realization of the SOH was developed in collaboration with Google Gemini 3.1 Pro, which served as both mathematical auditor and coder. The human author provided the ontology; the AI provided the linear algebra.

The Monostring was modeled as follows. Each "tick" $n \in \{1, \ldots, N\}$ (with $N = 150,000$) was assigned a phase vector $\phi(n)$ on a 6-dimensional torus $T^6$:

$$
\phi_i(n) = n \cdot \omega_i \pmod{2\pi}, \quad \omega_i = \sqrt{p_i}, \quad p_i \in \{2, 3, 5, 7, 11, 13\}
$$

The choice of six internal frequencies was motivated by an analogy with the six hidden dimensions of Calabi-Yau manifolds in string theory. The irrationality of frequencies, guaranteed by Kronecker's theorem, ensures that the Monostring's trajectory densely fills $T^6$ without exact repetition — weaving infinite diversity within a closed system.

**The space-generation rule:** An edge was created between nodes $m$ and $n$ ($|m - n| > 2$) whenever the toroidal distance between their phase vectors fell below a threshold $\varepsilon$, automatically calibrated to achieve an average vertex degree of approximately 60. Additional "chronological" edges connected consecutive ticks $(n, n+1)$, encoding the causal arrow of time.

The resulting object — a graph of 150,000 vertices and approximately 4.5 million edges — was presented as a computational universe.

### 2.2 Three Triumphant Claims

The Gemini model produced three results that initially appeared to validate the Monostring conjecture:

**Claim 1: Emergent 6D space.** The local topological dimension $D(R) = d(\ln V)/d(\ln R)$, measured via the volume growth of BFS (breadth-first search) balls, peaked at $D \approx 5.8$ at macro-scales. The interpretation: macroscopic dimensionality is not a pre-existing "box" but a direct reflection of the Monostring's internal degrees of freedom.

**Claim 2: Emergent gravity.** The global clustering coefficient was $C = 0.204$, roughly 500 times higher than expected for an Erdős–Rényi random graph of comparable size and density ($C_{ER} \approx 0.0004$). High clustering was interpreted as a mathematical marker of geometric curvature — identical, it was claimed, to gravity in General Relativity.

**Claim 3: Mass quantization.** The eigenvalue spectrum of the normalized graph Laplacian showed discrete "towers" and mass gaps rather than smooth noise. This was interpreted as the geometry of phase resonances spontaneously generating analogs of quantum particle families.

At this point, the model appeared to demonstrate that a single vibrating string could generate space, gravity, and quantized matter *ab initio*. The next step would have been to write a triumphant paper.

Instead, we asked for a second opinion.

---

## 3. What Claude Destroyed: The Critique

When the Gemini model and its three claims were submitted to Claude for critical review, the response was immediate and comprehensive. Every claim was reframed as an artifact of the construction method.

### 3.1 The Dimension Tautology
The Gemini model is, mathematically speaking, a **Random Geometric Graph (RGG)** on $T^6$: points distributed on a 6-dimensional torus, connected by edges whenever their distance falls below a threshold. It is a well-established result in geometric graph theory [2, 3] that the volume growth dimension of an RGG on a $d$-dimensional manifold equals $d$. The result $D \approx 5.8 \approx 6$ is therefore a **tautology**: the construction encoded six dimensions as input and measured six dimensions as output.

Claude offered an analogy: this is equivalent to pouring sand into a cubic mold and announcing that "sand spontaneously self-organizes into a cube."

**The test that would have demonstrated genuine emergence:** If six input frequencies had produced $D = 4$ (our observed spacetime), that would be non-trivial and exciting. But six in, six out is a circle, not a discovery.

### 3.2 The Clustering Fallacy
Comparing the graph's clustering coefficient against an Erdős–Rényi (ER) random graph is a textbook error in network science. ER graphs have no geometry — every edge is equally probable regardless of position. An RGG, by contrast, has inherently high clustering because of the triangle inequality: if node A is close to B and B is close to C, then A is very likely close to C. The correct null model is another RGG on the same torus with the same average degree. Against this proper baseline, the Monostring graph's clustering is unremarkable.

More fundamentally, clustering in a graph is **not** spacetime curvature. In General Relativity, curvature is defined through the Riemann tensor, which requires a metric, a connection, and dynamical equations (Einstein's field equations). Calling high clustering "gravity" is a category error — confusing a statistical property of a network with a geometric property of a pseudo-Riemannian manifold.

### 3.3 The Trivial Spectrum
The Laplacian on $T^d$ has a discrete spectrum **by definition** — its eigenvalues are proportional to $|n|^2$ for integer vectors $n \in \mathbb{Z}^d$. Any finite graph has a discrete spectrum. The "towers" in the eigenvalue histogram reflect the number-theoretic structure of the 6-torus lattice, not the physics of particle generations. Without an explicit mapping from eigenvalue $\lambda_i$ to physical mass $m_{phys}$, and without reproducing at least one known mass ratio (such as $m_\mu/m_e = 206.8$), the word "mass" was unwarranted.

### 3.4 The Bell Theorem Problem
The proposed entanglement mechanism — "matching phases create an edge, signals jump instantaneously along edges" — is a **classical hidden variable theory**. The phases predetermine measurement outcomes; the edges provide pre-existing connections. Bell's theorem [4] and the Aspect–Zeilinger experiments [5] prove that such theories satisfy $|S_{CHSH}| \leq 2$, while quantum mechanics predicts and experiments confirm $|S| = 2\sqrt{2} \approx 2.828$. The Monostring mechanism, as described, cannot reproduce quantum correlations without either violating relativity (superluminal signaling) or failing Bell's inequality.

### 3.5 The Missing Physics
The model contained no Lagrangian, no Hamiltonian, no equations of motion, no quantization rule, no Born rule ($P = |\psi|^2$), and made no quantitative predictions. It was a **kinematic construction** — a static picture, not a dynamical theory.

The critique concluded: *"Between a beautiful ontology and a physical theory lies an abyss that is filled by equations and predictions, not by metaphors."*

---

## 4. Rebuilding: The Mathematical Apparatus

Rather than abandoning the Monostring, we set out to make it **falsifiable**. Every claim needed a quantitative test with a clear pass/fail criterion and a proper control experiment.

### 4.1 From Linear to Nonlinear: E₆-Coupled Dynamics
The linear phase winding of v0 was replaced with a **coupled standard map** — a well-studied class of nonlinear dynamical systems:

$$
\phi_i(n+1) = \phi_i(n) + \omega_i + \kappa \sum_{j=1}^{6} (C_{E_6})_{ij} \sin\phi_j(n) \pmod{2\pi}
$$

Here $C_{E_6}$ is the Cartan matrix of the exceptional Lie algebra $E_6$ and $\kappa \geq 0$ is a coupling constant controlling the strength of nonlinearity. At $\kappa = 0$, the original linear model is recovered. At $\kappa > 0$, the phases interact through the algebraic structure of $E_6$.

**Why E₆?** This is the unique exceptional Lie algebra of rank 6. Its representation theory contains $SU(3) \times SU(3) \times SU(3) \supset SU(3)_C \times SU(2)_L \times U(1)_Y$ — the gauge group of the Standard Model of particle physics [6]. If the Monostring's internal structure is to have any connection to observed physics, $E_6$ is the mathematically motivated choice.

### 4.2 Coxeter Frequencies: Frequencies from Pure Algebra
Instead of the arbitrary choice $\omega_i = \sqrt{p_i}$, we derived frequencies from the intrinsic structure of $E_6$ itself. The **Coxeter exponents** of $E_6$ are $\{1, 4, 5, 7, 8, 11\}$, and the Coxeter number is $h = 12$. The natural frequencies are:

$$
\omega_k = 2\sin\!\left(\frac{\pi m_k}{12}\right), \quad m_k \in \{1, 4, 5, 7, 8, 11\}
$$

A crucial property: these exponents are **pairwise symmetric** ($m_1 + m_6 = m_2 + m_5 = m_3 + m_4 = 12$), which means $\sin(\pi m/12) = \sin(\pi(12-m)/12)$. The six frequencies collapse to only **three unique values**: $\omega \approx \{0.518, 1.732, 1.932\}$. This degeneracy is not imposed by hand — it is a theorem of $E_6$ representation theory.

### 4.3 Proper Measurement Tools
We replaced every ad hoc metric from v0 with established tools:

*   **Dimension.** Two independent measures: (a) the Grassberger–Procaccia correlation dimension $D_{corr}$, computed from the scaling of the correlation integral $C(r) \sim r^D$ [7]; (b) the Kaplan-Yorke dimension $D_{KY}$, computed analytically from the full Lyapunov spectrum $\{\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_6\}$ via the formula $D_{KY} = j + (\sum_{i=1}^j \lambda_i)/|\lambda_{j+1}|$, where $j$ is the largest index for which the cumulative sum remains non-negative [8].
*   **Curvature.** The Ollivier–Ricci curvature $\kappa_{OR}(m,n) = 1 - W_1(\mu_m, \mu_n)$, where $W_1$ is the Wasserstein-1 (earth mover's) distance between lazy random walk distributions centered at adjacent nodes $m$ and $n$ [9].
*   **Quantum correlations.** A CHSH Bell test with **dichotomic observables** (Pauli matrices $\sigma(\theta) = \cos\theta \cdot \sigma_z + \sin\theta \cdot \sigma_x$, eigenvalues $\pm 1$ by construction), applied to a two-qubit subspace extracted from the graph Laplacian's low-energy eigenstates.
*   **Null model.** A Random Geometric Graph on uniformly distributed points on $T^6$, with the same average degree as the Monostring graph.

---

## 5. The Seven Experiments

### 5.1 Experiment v1: First Contact with Nonlinearity
**Script:** `soh_claude_full.py` ($N = 10,000$, $\kappa = 0.5$)

The first run with $E_6$-coupled dynamics yielded $D_{corr} = 4.72$ — the first signal that nonlinear coupling genuinely reduces the attractor dimension below 6. However, the graph was catastrophically sparse (average degree = 4.9), making all graph-based measurements unreliable. 

### 5.2 Experiment v2: Proper Controls
**Script:** `soh_claude_full_50000_Claude.py` ($N = 15,000$, target degree = 40)

Innovations: automatic $\varepsilon$ calibration; finite-size saturation correction for BFS dimension; Bell test through graph quantum mechanics; null model (RGG on random $T^6$).

| Metric | Monostring | Null (random T⁶) | Status |
|---|---|---|---|
| D_corr (attractor) | 4.75 | 5.71 | PASS: SBE is different |
| D_reg (graph regression) | 3.79 | 4.68 | PASS: Δ = 0.89 |
| Mean Ollivier–Ricci curvature | -0.034 | -0.234 | SBE less negative |
| corr(κ_OR, density) | 0.467 | 0.102 | PASS: 4× stronger |
| \|S_CHSH\| | 2.430 | not tested | Suspicious |
| Concurrence | 1.000 | not tested | Very suspicious |

The Bell result ($|S| = 2.43$) looked like a breakthrough, but the null model was not tested on the Bell protocol.

### 5.3 Experiment v3: The Bell Test Dies, Coxeter Is Born
**Script:** `soh_claude_full_50000_Claude_v3.py`

Two transformative results:
**Discovery: Coxeter frequencies produce D_corr = 2.06.** The threefold frequency degeneracy produced a dramatically deeper dimensional reduction than arbitrary frequencies. At $\kappa \approx 0.25$, $D_{corr} = 3.98$ — a direct hit on 4-dimensional spacetime.
**Falsification: Bell test is generic.** Testing the null model on the same Bell protocol yielded $|S| = 2.40$ — virtually identical to the Monostring. The violation reflects the mathematical structure of the construction, not SBE physics. **The Bell test was permanently removed from the evidence.**

### 5.4 Experiment v4: The Eureka That Almost Was
**Script:** `soh_claude_full_50000_Claude_v4.py`

This iteration introduced the full 6-component Lyapunov spectrum and the Kaplan-Yorke dimension $D_{KY}$. Twenty independent runs with different initial conditions ($\kappa = 0.25$) yielded:

$$
D_{corr} = 4.025 \pm 0.040 \quad \text{(20 runs, 95\% CI:[3.997, 4.053])}
$$

The Lyapunov spectrum revealed the structure:
$$
\lambda =[+0.004, +0.002, 0.000, -0.002, -0.008, -0.034]
$$

Two weakly expanding directions, one neutral, three contracting — a 4-dimensional attractor with 2 "compactified" directions. $D_{KY} = 4.53$. 

This was the moment of maximum temptation. The result appeared to be a genuine mathematical discovery: the exceptional Lie algebra $E_6$, through its Coxeter structure, naturally compactifies 2 of 6 dimensions, producing a 4-dimensional emergent spacetime. 

We continued testing.

### 5.5 Experiment v5: E₆ Is Not Special
**Script:** `soh_claude_full_50000_Claude_v5.py`

The falsification began when we tested **all** simple and semi-simple Lie algebras of rank 6, each with its own Cartan matrix and Coxeter frequencies.

| Algebra | Type | D closest to 4 | Δ from 4 |
|---|---|---|---|
| D₆ = SO(12) | Simple | 3.96 | **0.04** |
| SU(3)³ | Semi-simple | 3.98 | **0.02** |
| A₃×A₂×A₁ | Semi-simple | 3.97 | **0.03** |
| B₆ = SO(13) | Simple | 4.08 | 0.08 |
| A₆ = SU(7) | Simple | 3.88 | 0.12 |
| C₆ = Sp(12) | Simple | 3.83 | 0.17 |
| **E₆** | **Exceptional** | **4.25** | **0.25** |

$E_6$ was not special — it was the **worst** among the tested algebras. 

### 5.6 Experiment v6: The Intermediate Value Trap
**Script:** `soh_claude_full_50000_Claude_v6.py`

Extending the analysis to ranks 3 through 12 revealed a fatal problem. Seven of sixteen tested algebras showed $D_{KY}(min) = 0$ — meaning the dynamics collapsed to a fixed point at certain $\kappa$ values. This exposed the root cause: the coupled standard map is **dissipative** — it does not preserve phase space volume.

For a dissipative map on $T^r$, $D_{KY}$ is a continuous function that starts at $D_{KY}(\kappa=0) = r$ and can decrease to $D_{KY} = 0$. By the **intermediate value theorem**, $D_{KY}$ must pass through every value between 0 and $r$ — including 4.

**The statement "$D_{KY} \approx 4$ at some $\kappa$" is a mathematical tautology for any dissipative system with rank $\geq 5$.** 

### 5.7 Experiment v7: The Symplectic Verdict
**Script:** `soh_claude_full_50000_Claude_v7.py`

The decisive experiment addressed the core question: is the dissipative map physically legitimate? A map representing a **closed system** (which the Sole Oscillator must be, having no environment to dissipate into) must be **Hamiltonian** — it must preserve phase volume.

The dynamics were reformulated in canonical form:

$$
p_{n+1} = p_n + \kappa\, C\, \sin\phi_n
$$
$$
\phi_{n+1} = \phi_n + \omega + p_{n+1} \pmod{2\pi}
$$

This doubles the phase space dimension to $2r$ but preserves volume ($\det J = 1$ identically). 

**The result:**

| Algebra | Rank | Phase space dim | D_KY (all κ from 0 to 3) |
|---|---|---|---|
| A₅ = SU(6) | 5 | 10 | **10.00** |
| E₆ | 6 | 12 | **12.00** |
| A₈ = SU(9) | 8 | 16 | **16.00** |
| E₈ | 8 | 16 | **16.00** |

$D_{KY} = 2r$ at every single value of $\kappa$ tested. No dimensional reduction. No approach toward $D = 4$. 

**The Monostring's dimensional reduction was entirely an artifact of using a dissipative map. The hypothesis, as mathematically formulated through coupled standard maps on tori, was falsified.**

---

## 6. A Consolation Prize: The Universal D ≈ 4 Plateau

While the SOH-specific conclusions are negative, the computational campaign produced a positive result of independent interest.

All 13 of 13 dissipative coupled standard maps tested (across all simple Lie algebras of ranks 4–8) exhibited a **chaotic plateau** where $D_{KY} \approx 4$:

| Rank | Algebras tested | All show D≈4 plateau? | Mean plateau width Δκ |
|---|---|---|---|
| 4 | A₄, D₄ | Yes | 1.50 |
| 5 | A₅, D₅ | Yes | 0.58 |
| 6 | A₆, D₆, E₆ | Yes | 0.24 |
| 7 | A₇, D₇, E₇ | Yes | 0.11 |
| 8 | A₈, D₈, E₈ | Yes | 0.05 |

This observation — that dissipative maps with Lie-algebraic coupling generically exhibit a $D \approx 4$ plateau — may deserve investigation within dynamical systems theory.

---

## 7. What Survived, What Died, What Remains Open

### 7.1 Final Scorecard

| SOH Claim | Test methodology | v0 result | Best result (v4) | Final result (v7) | Verdict |
|---|---|---|---|---|---|
| 6 phases → 4D space | D_corr, D_KY, symplectic test | 5.8 (tautology) | 4.03±0.04 | Symplectic: D=2r always | ❌ Falsified |
| E₆ is unique algebra | All rank-6 algebras tested | Not tested | Not tested | 13/13 algebras give D≈4 | ❌ Falsified |
| Gravity = curvature | Ollivier–Ricci κ_OR | C=0.2 | r=0.47 vs null 0.12 | Not tested in symplectic | ⚠️ Open |
| Entanglement | CHSH with null control | Not tested | \|S\|=2.43 | Null also violates | ❌ Invalid |
| Mass spectrum | Laplacian/Dirac eigenvalues | "Towers" | Ratios ≈ 1.0–1.1 | Degenerate spectrum | ❌ Not supported |

### 7.2 What Is Firmly Established
1. The $E_6$-coupled standard map with Coxeter frequencies at $\kappa = 0.25$ produces an attractor with $D_{corr} = 4.025 \pm 0.040$. This is **numerically correct but physically meaningless**.
2. Dissipative coupled standard maps with Cartan-matrix coupling exhibit a universal $D \approx 4$ plateau across all simple Lie algebras of ranks 4–8.
3. The symplectic (Hamiltonian) version of all tested maps shows $D_{KY} = 2r$ identically, with no dimensional reduction.

---

## 8. Lessons on AI-Assisted Research

The v0 $\to$ v7 trajectory illustrates a pattern likely to recur as AI tools become standard in theoretical research.

**The Confirmation Machine (v0):** The initial AI collaborator (Gemini) produced an implementation that was technically competent and visually impressive. It confirmed every aspect of the hypothesis it was asked to test. The problem was the task definition: "build a model of my hypothesis" is fundamentally different from "test whether my hypothesis survives reality."

**The Adversarial Critic (v1–v7):** The second AI collaborator (Claude) contributed most not by confirming the hypothesis but by designing experiments capable of destroying it. The most valuable AI contribution to science may not be "prove my theory" but rather **"design the experiment most likely to disprove my theory."**

---

## 9. The Monostring Is Dead. Long Live the Monostring.

The specific mechanism tested in this paper — dimensional reduction through Lyapunov compactification on phase tori — is dead. The symplectic test killed it cleanly and definitively.

But the philosophical core of the Sole Oscillator Hypothesis remains untouched. The idea that reality is monistic (one entity), discrete (countable states), and relational (space emerges from connections between states) is not contradicted by the failure of one particular mathematical implementation. 

### Future Directions

1.  **Causal Sets:** In causal set theory [10], spacetime is fundamentally a **partially ordered set**. SBE ticks form a natural totally ordered set. Phase resonance can define a partial order: tick $m$ causally precedes tick $n$ if their phases satisfy a directed resonance condition. Dimensionality is then determined by the combinatorics of causal diamonds, bypassing Lyapunov exponents entirely.
2.  **Quantum Walks:** Replace the deterministic trajectory on $T^6$ with a **quantum walk** on a discrete state space [11]. In the continuum limit, quantum walks on lattices naturally reproduce the **Dirac equation**[12].
3.  **Stochastic Quantization:** Model the Monostring's phase evolution as a stochastic process (Langevin equation with noise) [13]. The noise breaks phase volume conservation, but this breaking represents quantum fluctuations rather than information loss.

---

## 10. Conclusion

We built a computational universe from the phase resonances of a single vibrating entity — the Monostring. For a brief, exhilarating moment (iteration v4), it appeared to produce four-dimensional spacetime with the precision of $D = 4.025 \pm 0.040$. Seven iterations of AI-assisted adversarial testing revealed this to be an artifact of dissipative dynamics incompatible with the closed-system axiomatics of the hypothesis itself.

John Wheeler's one-electron universe was too beautiful to be true — but perhaps too beautiful to be entirely wrong. The Monostring, like Wheeler's electron, may need to move through several more incarnations before it finds its proper mathematical language.

---

## References

[1] R.P. Feynman, "The Development of the Space-Time View of Quantum Electrodynamics," Nobel Lecture, 1965.

[2] M. Penrose, *Random Geometric Graphs*, Oxford University Press, 2003.

[3] J. Dall, M. Christensen, "Random geometric graphs," Phys. Rev. E, 66:016121, 2002.

[4] J.S. Bell, "On the Einstein Podolsky Rosen paradox," Physics Physique Физика, 1(3):195–200, 1964.

[5] A. Aspect, P. Grangier, G. Roger, "Experimental realization of Einstein-Podolsky-Rosen-Bohm Gedankenexperiment," Phys. Rev. Lett., 49:91, 1982.

[6] R. Slansky, "Group theory for unified model building," Physics Reports, 79(1):1–128, 1981.

[7] P. Grassberger, I. Procaccia, "Characterization of strange attractors," Phys. Rev. Lett., 50:346, 1983.

[8] J.L. Kaplan, J.A. Yorke, "Chaotic behavior of multidimensional difference equations," in *Functional Differential Equations and Approximation of Fixed Points*, Lecture Notes in Mathematics 730, Springer, 1979.

[9] Y. Ollivier, "Ricci curvature of Markov chains on metric spaces," J. Funct. Anal., 256:810–864, 2009.

[10] L. Bombelli, J. Lee, D. Meyer, R.D. Sorkin, "Space-time as a causal set," Phys. Rev. Lett., 59:521, 1987.[11] Y. Aharonov, L. Davidovich, N. Zagury, "Quantum random walks," Phys. Rev. A, 48:1687, 1993.

[12] F.W. Strauch, "Relativistic quantum walks," Phys. Rev. A, 73:054302, 2006.

[13] G. Parisi, Y. Wu, "Perturbation theory without gauge fixing," Scientia Sinica, 24:483, 1981.

---

## Appendix: Script Inventory

| Version | Filename | AI | Key innovation |
|---|---|---|---|
| v0 | soh_gemini_original.py | Gemini 3.1 Pro | Original Monostring model (150K nodes, 6 linear frequencies) |
| v1 | soh_claude_full.py | Claude | E₆ nonlinear coupling, auto-calibration |
| v2 | soh_claude_full_50000_Claude.py | Claude | Null model, corrected Bell test, Ollivier-Ricci curvature |
| v3 | soh_claude_full_50000_Claude_v3.py | Claude | Coxeter frequencies, Bell test null control |
| v4 | soh_claude_full_50000_Claude_v4.py | Claude | Full Lyapunov spectrum, D_KY, 8 frequency families, 20-run stability |
| v5 | soh_claude_full_50000_Claude_v5.py | Claude | All rank-6 Lie algebras, self-tuning κ attempt |
| v6 | soh_claude_full_50000_Claude_v6.py | Claude | Ranks 3–12, intermediate value theorem argument |
| v7 | soh_claude_full_50000_Claude_v7.py | Claude | Dissipation check, symplectic map, plateau analysis |

*All scripts, raw output logs, and generated figures are included in the [Zenodo repository](https://doi.org/10.5281/zenodo.18890267).*

---
*This work was conducted as an exercise in AI-assisted theoretical research. The human author provided the hypothesis and direction; AI collaborators provided implementations, critical analysis, and falsifying tests. The most important contribution of the AIs was not building the theory — it was designing the experiments that destroyed it.*
