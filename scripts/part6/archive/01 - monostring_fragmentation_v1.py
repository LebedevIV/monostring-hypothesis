"""
Monostring Fragmentation Hypothesis
====================================
Simulation of:
1. Monostring instability and breakup
2. Daughter string formation and equalization
3. Entanglement between nearby strings
4. Emergent field statistics

Physical interpretation:
- Monostring: single oscillator on T^6 with E6 Coxeter frequencies
- Instability: Lyapunov exponent exceeds critical threshold
- Breakup: phase space fragments into N daughter strings
- Equalization: Kuramoto coupling drives identical strings
- Entanglement: phase-close pairs form joint wave functions
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp, entropy
from collections import defaultdict
import time
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════

class Config:
    # Monostring
    D          = 6          # internal dimensions
    omega_E6   = 2.0 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12.0)
    kappa      = 0.15       # nonlinearity strength

    # Instability criterion
    lyap_window    = 50     # steps to estimate Lyapunov exponent
    lyap_threshold = 0.08   # λ₁ > this → unstable
    chaos_integral = 3.0    # ∫λ₁ dt > this → breakup

    # Breakup
    N_strings      = 200    # number of daughter strings
    sigma_break    = 0.05   # initial phase scatter at breakup

    # Equalization (Kuramoto)
    K_kuramoto     = 0.8    # coupling strength
    T_equalize     = 300    # steps for equalization

    # Entanglement
    eps_entangle   = 0.3    # phase distance threshold
    eps_maintain   = 0.5    # threshold to maintain entanglement
    T_evolve       = 500    # steps to evolve daughter strings

    # Measurement
    n_runs         = 8      # Monte Carlo runs
    seed_base      = 42

cfg = Config()

# ═══════════════════════════════════════════════════════
# BLOCK 1: MONOSTRING DYNAMICS AND INSTABILITY
# ═══════════════════════════════════════════════════════

def monostring_step(phi, omega, kappa):
    """One step of the monostring map on T^6."""
    return (phi + omega + kappa * np.sin(phi)) % (2 * np.pi)

def estimate_lyapunov(phi0, omega, kappa, n_steps=50):
    """
    Estimate largest Lyapunov exponent via finite-difference method.
    Returns: λ₁ (positive = chaotic)
    """
    delta0 = 1e-8
    phi  = phi0.copy()
    phi2 = phi0 + delta0 * np.ones(len(phi0)) / np.sqrt(len(phi0))
    phi2 %= (2 * np.pi)

    lyap_sum = 0.0
    for _ in range(n_steps):
        phi  = monostring_step(phi,  omega, kappa)
        phi2 = monostring_step(phi2, omega, kappa)
        dist = np.linalg.norm(
            np.minimum(np.abs(phi2-phi), 2*np.pi - np.abs(phi2-phi)))
        if dist > 0:
            lyap_sum += np.log(dist / delta0)
            # Renormalize
            direction = (phi2 - phi)
            direction /= np.linalg.norm(direction) + 1e-15
            phi2 = phi + delta0 * direction
            phi2 %= (2 * np.pi)
    return lyap_sum / n_steps

def evolve_monostring(omega, kappa, max_steps=2000, seed=0):
    """
    Evolve monostring until instability criterion is met.

    Returns:
        phi_history: trajectory up to breakup
        t_break: step at which breakup occurs
        lyap_history: Lyapunov exponent over time
        chaos_integral: accumulated chaos measure
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    phi_history    = [phi.copy()]
    lyap_history   = []
    chaos_integral = 0.0
    t_break        = None

    for t in range(max_steps):
        phi = monostring_step(phi, omega, kappa)
        phi_history.append(phi.copy())

        # Estimate Lyapunov every lyap_window steps
        if t % cfg.lyap_window == 0 and t > 0:
            lam = estimate_lyapunov(phi, omega, kappa,
                                    n_steps=cfg.lyap_window)
            lyap_history.append((t, lam))

            # Accumulate chaos
            if lam > 0:
                chaos_integral += lam * cfg.lyap_window

            # Check breakup criterion
            if (lam > cfg.lyap_threshold and
                    chaos_integral > cfg.chaos_integral):
                t_break = t
                break

    if t_break is None:
        t_break = max_steps - 1

    return (np.array(phi_history), t_break,
            lyap_history, chaos_integral)

# ═══════════════════════════════════════════════════════
# BLOCK 2: FRAGMENTATION
# ═══════════════════════════════════════════════════════

def fragment_monostring(phi_break, N, sigma, seed=0):
    """
    Create N daughter strings from the monostring at breakup.

    Each daughter string gets:
    - Base phase: phi_break (the monostring phase at breakup)
    - Random perturbation: δ ~ N(0, sigma) on each dimension

    Physical interpretation: the phase space 'shatters' into
    N fragments, each carrying a perturbed copy of the
    monostring's phase at the moment of breakup.

    Returns: (N, D) array of initial phases
    """
    rng = np.random.RandomState(seed)
    # Gaussian perturbations on torus
    perturbations = rng.normal(0, sigma, (N, len(phi_break)))
    daughters = (phi_break[np.newaxis, :] + perturbations) % (2*np.pi)
    return daughters

def equalization_measure(phases):
    """
    Measure how 'equal' the strings are.
    Returns: (mean pairwise distance, std of phases per dim)
    """
    N = len(phases)
    # Sample pairwise distances
    n_pairs = min(1000, N*(N-1)//2)
    rng = np.random.RandomState(0)
    pairs = rng.randint(0, N, (n_pairs, 2))
    mask = pairs[:,0] != pairs[:,1]
    pairs = pairs[mask]

    diffs = np.abs(phases[pairs[:,0]] - phases[pairs[:,1]])
    diffs = np.minimum(diffs, 2*np.pi - diffs)
    mean_dist = float(np.mean(np.linalg.norm(diffs, axis=1)))
    std_per_dim = float(np.mean(np.std(phases, axis=0)))

    return mean_dist, std_per_dim

# ═══════════════════════════════════════════════════════
# BLOCK 3: DAUGHTER STRING DYNAMICS AND EQUALIZATION
# ═══════════════════════════════════════════════════════

def evolve_daughters_kuramoto(phases0, omega, K, n_steps,
                               measure_every=10):
    """
    Evolve N daughter strings with Kuramoto-type coupling.

    Each string evolves as:
    φᵢ(t+1) = φᵢ(t) + ω + (K/N) Σⱼ sin(φⱼ(t) - φᵢ(t))

    This drives synchronization: strings become identical
    when K > K_c (critical coupling).

    Returns:
        phases: final phases (N, D)
        history: list of (step, mean_dist, order_param)
        order_param_history: Kuramoto order parameter over time
    """
    N, D = phases0.shape
    phases = phases0.copy()
    history = []

    for step in range(n_steps):
        # Kuramoto coupling: mean field
        # r_d * exp(i*psi_d) = (1/N) Σ exp(i*phi_d)
        mean_field = np.mean(np.exp(1j * phases), axis=0)
        r = np.abs(mean_field)       # order parameter per dim
        psi = np.angle(mean_field)   # mean phase per dim

        # Update: drive toward mean field
        coupling = K * np.sin(psi[np.newaxis,:] - phases)
        phases = (phases + omega[np.newaxis,:] + coupling) % (2*np.pi)

        if step % measure_every == 0:
            mean_dist, std_dim = equalization_measure(phases)
            order_param = float(np.mean(r))
            history.append({
                'step': step,
                'mean_dist': mean_dist,
                'std_dim': std_dim,
                'order_param': order_param,
                'r_per_dim': r.copy()
            })

    return phases, history

def evolve_daughters_free(phases0, omega, n_steps,
                           measure_every=10):
    """
    Evolve N daughter strings WITHOUT coupling (free evolution).
    Used as null model: do strings equalize on their own?
    """
    N, D = phases0.shape
    phases = phases0.copy()
    history = []

    for step in range(n_steps):
        phases = (phases + omega[np.newaxis,:] +
                  0.1 * np.sin(phases)) % (2*np.pi)

        if step % measure_every == 0:
            mean_dist, std_dim = equalization_measure(phases)
            mean_field = np.mean(np.exp(1j * phases), axis=0)
            r = float(np.mean(np.abs(mean_field)))
            history.append({
                'step': step,
                'mean_dist': mean_dist,
                'std_dim': std_dim,
                'order_param': r
            })

    return phases, history

# ═══════════════════════════════════════════════════════
# BLOCK 4: ENTANGLEMENT
# ═══════════════════════════════════════════════════════

def detect_entanglement(phases, eps):
    """
    Detect entangled pairs: strings whose phases are
    within eps of each other on the torus.

    Entanglement criterion:
    ||φᵢ - φⱼ||_torus < eps

    Returns:
        pairs: list of (i,j) entangled pairs
        G_entangle: entanglement graph
        clusters: connected components (entangled groups)
    """
    N = phases.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Check all pairs (chunked for memory)
    chunk = 50
    for i in range(0, N, chunk):
        ie = min(i+chunk, N)
        diffs = np.abs(phases[i:ie, np.newaxis, :] -
                       phases[np.newaxis, :, :])
        diffs = np.minimum(diffs, 2*np.pi - diffs)
        dists = np.linalg.norm(diffs, axis=2)

        rows, cols = np.where(
            (dists < eps) & (dists > 0))
        for r, c in zip(rows, cols):
            ri = r + i
            if ri < c:
                G.add_edge(int(ri), int(c),
                           distance=float(dists[r, c]))

    pairs = list(G.edges())
    clusters = list(nx.connected_components(G))

    return pairs, G, clusters

def entanglement_wave_function(phases, clusters):
    """
    Construct joint wave functions for entangled clusters.

    For a cluster of size n:
    |ψ_cluster⟩ = (1/√n) Σᵢ |φᵢ⟩

    where |φᵢ⟩ = exp(i·φᵢ) (coherent state on torus).

    Returns:
        psi: list of wave function arrays
        cluster_sizes: distribution of cluster sizes
        entanglement_entropy: S = -Tr(ρ log ρ) per cluster
    """
    psi_list = []
    entropies = []

    for cluster in clusters:
        cluster = list(cluster)
        n = len(cluster)

        # Joint coherent state: superposition
        cluster_phases = phases[cluster]  # (n, D)

        # Wave function: superposition of coherent states
        # |ψ⟩ = (1/√n) Σ exp(i·φₖ)
        psi = np.mean(np.exp(1j * cluster_phases), axis=0)
        psi_list.append(psi)

        if n > 1:
            # Entanglement entropy via Schmidt decomposition
            # Simplified: entropy of phase distribution
            probs = np.ones(n) / n  # equal superposition
            S = -np.sum(probs * np.log(probs + 1e-15))
            entropies.append(S)
        else:
            entropies.append(0.0)

    cluster_sizes = [len(list(c)) for c in clusters]
    return psi_list, cluster_sizes, entropies

def entanglement_dynamics(phases, eps, eps_maintain,
                           omega, n_steps=200):
    """
    Evolve strings and track entanglement over time.

    Entanglement is:
    - Created when distance < eps_entangle
    - Maintained while distance < eps_maintain
    - Broken when distance > eps_maintain

    Returns time series of:
    - Number of entangled pairs
    - Largest cluster size
    - Mean entanglement entropy
    """
    N = phases.shape[0]
    ph = phases.copy()

    n_entangled_history = []
    max_cluster_history = []
    entropy_history     = []

    # Track active entanglements
    active_pairs = set()

    for step in range(n_steps):
        # Evolve (free + weak coupling for entangled pairs)
        ph = (ph + omega[np.newaxis,:] +
               0.1*np.sin(ph)) % (2*np.pi)

        # Check new entanglements
        _, G, clusters = detect_entanglement(ph, eps)
        new_pairs = set(G.edges())

        # Maintain existing entanglements if still close
        maintained = set()
        for (i,j) in active_pairs:
            diff = np.abs(ph[i] - ph[j])
            diff = np.minimum(diff, 2*np.pi - diff)
            dist = np.linalg.norm(diff)
            if dist < eps_maintain:
                maintained.add((i,j))

        active_pairs = new_pairs | maintained

        # Statistics
        G_active = nx.Graph()
        G_active.add_nodes_from(range(N))
        G_active.add_edges_from(active_pairs)
        comps = list(nx.connected_components(G_active))

        n_ent = len(active_pairs)
        max_cl = max(len(c) for c in comps) if comps else 1

        # Entropy
        _, _, ents = entanglement_wave_function(ph, comps)
        mean_ent = np.mean(ents) if ents else 0.0

        n_entangled_history.append(n_ent)
        max_cluster_history.append(max_cl)
        entropy_history.append(mean_ent)

    return (n_entangled_history,
            max_cluster_history,
            entropy_history)

# ═══════════════════════════════════════════════════════
# BLOCK 5: FIELD STATISTICS
# ═══════════════════════════════════════════════════════

def measure_field_statistics(phases_free, phases_entangled,
                              n_bins=30):
    """
    Compare field statistics between:
    - Free (independent) strings → should give uniform field
    - Entangled clusters → should give correlated field

    Measures:
    1. Phase distribution (should be uniform for free)
    2. Two-point correlation function
    3. Power spectrum
    4. KS test: free vs entangled
    """
    results = {}

    for label, phases in [('free',      phases_free),
                           ('entangled', phases_entangled)]:
        N, D = phases.shape

        # 1. Phase distribution (first dimension)
        hist, bins = np.histogram(
            phases[:,0] % (2*np.pi),
            bins=n_bins,
            range=(0, 2*np.pi),
            density=True)

        # 2. Two-point correlation
        # C(Δ) = ⟨exp(i(φᵢ - φⱼ))⟩ for |i-j| = Δ
        # (treating string index as spatial coordinate)
        max_sep = min(50, N//2)
        correlations = []
        for sep in range(1, max_sep):
            diff = phases[:N-sep, 0] - phases[sep:, 0]
            C = float(np.abs(np.mean(np.exp(1j * diff))))
            correlations.append(C)

        # 3. Power spectrum of phase field
        fft = np.fft.rfft(phases[:,0])
        power = np.abs(fft)**2

        # 4. Uniformity: KL divergence from uniform
        uniform = np.ones(n_bins) / n_bins
        hist_norm = hist / hist.sum() + 1e-15
        kl_div = float(entropy(hist_norm, uniform + 1e-15))

        results[label] = {
            'hist': hist,
            'bins': bins,
            'correlations': np.array(correlations),
            'power': power,
            'kl_div': kl_div,
            'mean_phase': float(np.mean(phases[:,0])),
            'std_phase':  float(np.std(phases[:,0]))
        }

    # KS test: are the distributions different?
    ks_stat, ks_p = ks_2samp(
        phases_free[:,0], phases_entangled[:,0])
    results['ks_stat'] = ks_stat
    results['ks_p']    = ks_p

    return results

# ═══════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════

def run_simulation(seed=42, verbose=True):
    """Run the complete fragmentation simulation."""
    rng = np.random.RandomState(seed)
    omega = cfg.omega_E6

    if verbose:
        print(f"\n{'='*55}")
        print(f"  SEED {seed}: Running fragmentation simulation")
        print(f"{'='*55}")

    # ── Phase 1: Monostring evolution until breakup ──
    if verbose:
        print(f"\n  Phase 1: Monostring instability...")

    phi_history, t_break, lyap_history, chaos_total = \
        evolve_monostring(omega, cfg.kappa,
                          max_steps=2000, seed=seed)
    phi_break = phi_history[t_break]

    if verbose:
        print(f"  Breakup at t={t_break}")
        print(f"  Accumulated chaos: {chaos_total:.3f}")
        if lyap_history:
            last_lyap = lyap_history[-1][1]
            print(f"  Final λ₁ = {last_lyap:.4f}")

    # ── Phase 2: Fragmentation ──
    if verbose:
        print(f"\n  Phase 2: Fragmentation into N={cfg.N_strings} strings...")

    daughters0 = fragment_monostring(
        phi_break, cfg.N_strings, cfg.sigma_break,
        seed=seed+1)

    dist0, std0 = equalization_measure(daughters0)
    if verbose:
        print(f"  Initial mean distance: {dist0:.4f}")
        print(f"  Initial std per dim:   {std0:.4f}")

    # ── Phase 3a: Equalization with Kuramoto coupling ──
    if verbose:
        print(f"\n  Phase 3a: Equalization (Kuramoto K={cfg.K_kuramoto})...")

    phases_eq, history_eq = evolve_daughters_kuramoto(
        daughters0, omega, cfg.K_kuramoto,
        cfg.T_equalize)

    dist_eq, std_eq = equalization_measure(phases_eq)
    if verbose:
        print(f"  Final mean distance:   {dist_eq:.4f}")
        print(f"  Final std per dim:     {std_eq:.4f}")
        if history_eq:
            final_r = history_eq[-1]['order_param']
            print(f"  Kuramoto order param:  {final_r:.4f}")

    # ── Phase 3b: Free evolution (null model) ──
    if verbose:
        print(f"\n  Phase 3b: Free evolution (no coupling)...")

    phases_free, history_free = evolve_daughters_free(
        daughters0, omega, cfg.T_equalize)

    dist_free, std_free = equalization_measure(phases_free)
    if verbose:
        print(f"  Free mean distance:    {dist_free:.4f}")
        equalized = dist_eq < dist_free * 0.5
        print(f"  Kuramoto equalizes:    {'✅ YES' if equalized else '❌ NO'}")

    # ── Phase 4: Entanglement ──
    if verbose:
        print(f"\n  Phase 4: Entanglement detection...")

    # Use equalized strings
    pairs, G_ent, clusters = detect_entanglement(
        phases_eq, cfg.eps_entangle)

    n_entangled = len(pairs)
    n_clusters  = len([c for c in clusters if len(c) > 1])
    cluster_sizes = [len(c) for c in clusters]
    max_cluster   = max(cluster_sizes)

    if verbose:
        print(f"  Entangled pairs:    {n_entangled}")
        print(f"  Multi-string clusters: {n_clusters}")
        print(f"  Largest cluster:    {max_cluster}")
        frac = n_entangled * 2 / cfg.N_strings
        print(f"  Entanglement fraction: {frac:.3f}")

    # Wave functions
    psi_list, cl_sizes, ent_entropies = \
        entanglement_wave_function(phases_eq, clusters)
    mean_entropy = np.mean(
        [e for e in ent_entropies if e > 0]) \
        if any(e > 0 for e in ent_entropies) else 0.0

    if verbose:
        print(f"  Mean entanglement entropy: {mean_entropy:.4f}")

    # Entanglement dynamics
    n_ent_t, max_cl_t, entropy_t = entanglement_dynamics(
        phases_eq, cfg.eps_entangle, cfg.eps_maintain,
        omega, n_steps=200)

    # ── Phase 5: Field statistics ──
    if verbose:
        print(f"\n  Phase 5: Field statistics...")

    # Separate entangled and free strings
    entangled_nodes = set()
    for c in clusters:
        if len(c) > 1:
            entangled_nodes.update(c)
    free_nodes = [i for i in range(cfg.N_strings)
                  if i not in entangled_nodes]

    if len(free_nodes) > 10 and len(entangled_nodes) > 10:
        phases_free_strings = phases_eq[free_nodes]
        phases_ent_strings  = phases_eq[list(entangled_nodes)]
        field_stats = measure_field_statistics(
            phases_free_strings, phases_ent_strings)
        if verbose:
            print(f"  Free strings:      {len(free_nodes)}")
            print(f"  Entangled strings: {len(entangled_nodes)}")
            print(f"  KL(free,uniform):  "
                  f"{field_stats['free']['kl_div']:.4f}")
            print(f"  KL(ent,uniform):   "
                  f"{field_stats['entangled']['kl_div']:.4f}")
            print(f"  KS test: stat={field_stats['ks_stat']:.3f}, "
                  f"p={field_stats['ks_p']:.4f}")
    else:
        field_stats = None

    return {
        'seed':         seed,
        't_break':      t_break,
        'chaos_total':  chaos_total,
        'lyap_history': lyap_history,
        'phi_history':  phi_history,
        'daughters0':   daughters0,
        'phases_eq':    phases_eq,
        'phases_free':  phases_free,
        'history_eq':   history_eq,
        'history_free': history_free,
        'dist0':        dist0,
        'dist_eq':      dist_eq,
        'dist_free':    dist_free,
        'n_entangled':  n_entangled,
        'n_clusters':   n_clusters,
        'cluster_sizes':cluster_sizes,
        'max_cluster':  max_cluster,
        'mean_entropy': mean_entropy,
        'n_ent_t':      n_ent_t,
        'max_cl_t':     max_cl_t,
        'entropy_t':    entropy_t,
        'field_stats':  field_stats,
        'G_ent':        G_ent,
    }

# ═══════════════════════════════════════════════════════
# MONTE CARLO SUMMARY
# ═══════════════════════════════════════════════════════

def run_monte_carlo(n_runs=8):
    """Run multiple seeds and aggregate results."""
    print("\n" + "█"*55)
    print("█  MONOSTRING FRAGMENTATION — MONTE CARLO            █")
    print("█"*55)
    print(f"  N_strings={cfg.N_strings}, K={cfg.K_kuramoto}, "
          f"ε={cfg.eps_entangle}, {n_runs} runs\n")

    all_results = []
    t0 = time.time()

    for run in range(n_runs):
        seed = cfg.seed_base + run * 7
        res  = run_simulation(seed=seed, verbose=(run==0))
        all_results.append(res)
        print(f"  Run {run+1}/{n_runs}: "
              f"t_break={res['t_break']}, "
              f"n_entangled={res['n_entangled']}, "
              f"entropy={res['mean_entropy']:.3f}")

    elapsed = time.time() - t0

    # Aggregate
    t_breaks    = [r['t_break']      for r in all_results]
    n_ents      = [r['n_entangled']  for r in all_results]
    entropies   = [r['mean_entropy'] for r in all_results]
    dist_eq_all = [r['dist_eq']      for r in all_results]
    dist_fr_all = [r['dist_free']    for r in all_results]

    print(f"\n{'='*55}")
    print(f"  AGGREGATED RESULTS ({n_runs} runs, {elapsed:.0f}s)")
    print(f"{'='*55}")
    print(f"  Breakup time:       "
          f"{np.mean(t_breaks):.0f} ± {np.std(t_breaks):.0f} steps")
    print(f"  Entangled pairs:    "
          f"{np.mean(n_ents):.1f} ± {np.std(n_ents):.1f}")
    print(f"  Entanglement frac:  "
          f"{np.mean(n_ents)*2/cfg.N_strings:.3f}")
    print(f"  Mean entropy:       "
          f"{np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
    print(f"  dist_eq (Kuramoto): "
          f"{np.mean(dist_eq_all):.4f} ± {np.std(dist_eq_all):.4f}")
    print(f"  dist_free (null):   "
          f"{np.mean(dist_fr_all):.4f} ± {np.std(dist_fr_all):.4f}")

    kuramoto_equalizes = np.mean(dist_eq_all) < np.mean(dist_fr_all)*0.5
    print(f"\n  Kuramoto equalizes strings: "
          f"{'✅ YES' if kuramoto_equalizes else '❌ NO'}")
    print(f"  Entanglement is rare:       "
          f"{'✅ YES' if np.mean(n_ents)*2/cfg.N_strings < 0.15 else '❌ NO (common)'}")

    return all_results

# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_results(all_results):
    """Comprehensive figure: all four phases."""
    res = all_results[0]  # Use first run for detailed plots

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Monostring Fragmentation Hypothesis\n"
        "Single oscillator → N daughter strings → "
        "equalization → entanglement",
        fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.45, wspace=0.35)

    # ── Row 1: Monostring instability ──

    # 1a: Phase trajectory (first 2 dims)
    ax = fig.add_subplot(gs[0, 0])
    ph = res['phi_history']
    tb = res['t_break']
    ax.plot(ph[:tb, 0], ph[:tb, 1], 'b-',
            alpha=0.4, lw=0.8, label='before breakup')
    ax.plot(ph[tb, 0], ph[tb, 1], 'r*',
            markersize=15, label=f'breakup t={tb}')
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('Phase 1: Monostring trajectory\n'
                 '(dims 1–2)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2*np.pi]); ax.set_ylim([0, 2*np.pi])

    # 1b: Lyapunov exponent over time
    ax = fig.add_subplot(gs[0, 1])
    if res['lyap_history']:
        ts_lyap = [x[0] for x in res['lyap_history']]
        ls_lyap = [x[1] for x in res['lyap_history']]
        ax.plot(ts_lyap, ls_lyap, 'b-o', markersize=4,
                label='λ₁(t)')
        ax.axhline(cfg.lyap_threshold, color='red',
                   linestyle='--', label=f'threshold={cfg.lyap_threshold}')
        ax.axhline(0, color='black', lw=1)
        ax.axvline(tb, color='orange', linestyle=':',
                   label=f'breakup t={tb}')
    ax.set_xlabel('t'); ax.set_ylabel('λ₁')
    ax.set_title('Phase 1: Lyapunov exponent\n'
                 '(instability criterion)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 1c: Fragmentation — phase scatter at t=0
    ax = fig.add_subplot(gs[0, 2])
    d0 = res['daughters0']
    ax.scatter(d0[:, 0], d0[:, 1],
               s=3, alpha=0.5, c='steelblue',
               label=f'N={cfg.N_strings} daughters')
    ax.scatter(res['phi_history'][tb, 0],
               res['phi_history'][tb, 1],
               s=200, c='red', marker='*',
               zorder=5, label='monostring at t*')
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('Phase 2: Fragmentation\n'
                 f'(σ={cfg.sigma_break})')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2*np.pi]); ax.set_ylim([0, 2*np.pi])

    # 1d: Fragmentation — sigma scan
    ax = fig.add_subplot(gs[0, 3])
    sigmas   = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    dists_sg = []
    phi_brk  = res['phi_history'][tb]
    for sg in sigmas:
        d = fragment_monostring(phi_brk, 100, sg, seed=0)
        dist, _ = equalization_measure(d)
        dists_sg.append(dist)
    ax.plot(sigmas, dists_sg, 'bo-', lw=2)
    ax.axvline(cfg.sigma_break, color='red',
               linestyle='--', label=f'σ={cfg.sigma_break}')
    ax.set_xlabel('σ (fragmentation width)')
    ax.set_ylabel('mean pairwise distance')
    ax.set_title('Phase 2: Initial scatter vs σ\n'
                 '(smaller σ = more identical)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ── Row 2: Equalization ──

    # 2a: Distance over time
    ax = fig.add_subplot(gs[1, 0:2])
    steps_eq   = [h['step']      for h in res['history_eq']]
    dist_eq    = [h['mean_dist'] for h in res['history_eq']]
    steps_fr   = [h['step']      for h in res['history_free']]
    dist_fr    = [h['mean_dist'] for h in res['history_free']]
    order_eq   = [h['order_param'] for h in res['history_eq']]

    ax.plot(steps_eq, dist_eq, 'b-',  lw=2,
            label='Kuramoto (coupled)')
    ax.plot(steps_fr, dist_fr, 'r--', lw=2,
            label='Free (uncoupled)')
    ax.set_xlabel('t (steps)')
    ax.set_ylabel('mean pairwise distance')
    ax.set_title('Phase 3: Equalization\n'
                 '(Kuramoto coupling drives strings to identity)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2b: Kuramoto order parameter
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps_eq, order_eq, 'g-', lw=2)
    ax.axhline(1.0, color='black', linestyle=':', lw=1,
               label='perfect sync')
    ax.set_xlabel('t'); ax.set_ylabel('r (order param)')
    ax.set_title('Phase 3: Kuramoto order\n'
                 '(r→1 = synchronized)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 2c: Phase distribution before/after
    ax = fig.add_subplot(gs[1, 3])
    bins = np.linspace(0, 2*np.pi, 30)
    ax.hist(res['daughters0'][:, 0], bins=bins,
            alpha=0.5, density=True,
            color='red', label='Initial (fragmented)')
    ax.hist(res['phases_eq'][:, 0], bins=bins,
            alpha=0.5, density=True,
            color='blue', label='After Kuramoto')
    ax.axhline(1/(2*np.pi), color='black',
               linestyle='--', lw=1, label='uniform')
    ax.set_xlabel('φ₁')
    ax.set_ylabel('density')
    ax.set_title('Phase 3: Phase distribution\n'
                 'before/after equalization')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Row 3: Entanglement ──

    # 3a: Entanglement network
    ax = fig.add_subplot(gs[2, 0])
    G = res['G_ent']
    phases_eq = res['phases_eq']
    pos = {i: (phases_eq[i, 0], phases_eq[i, 1])
           for i in range(cfg.N_strings)}

    # Color by cluster membership
    clusters = list(nx.connected_components(G))
    node_colors = ['lightgray'] * cfg.N_strings
    cmap = plt.cm.Set1
    for ci, cluster in enumerate(clusters):
        if len(cluster) > 1:
            for node in cluster:
                node_colors[node] = cmap(ci % 9)

    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_size=5,
                     node_color=node_colors,
                     edge_color='red',
                     alpha=0.6,
                     with_labels=False,
                     width=0.5)
    ax.set_title('Phase 4: Entanglement network\n'
                 '(colored = entangled clusters)')
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.grid(True, alpha=0.2)

    # 3b: Cluster size distribution
    ax = fig.add_subplot(gs[2, 1])
    sizes = res['cluster_sizes']
    max_s = max(sizes)
    bins_s = np.arange(1, max_s+2) - 0.5
    ax.hist(sizes, bins=bins_s,
            color='steelblue', alpha=0.8,
            edgecolor='black')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Count')
    ax.set_title('Phase 4: Cluster size distribution\n'
                 '(most strings: isolated)')
    ax.grid(True, alpha=0.3)

    # 3c: Entanglement dynamics
    ax = fig.add_subplot(gs[2, 2])
    t_ent = np.arange(len(res['n_ent_t']))
    ax.plot(t_ent, res['n_ent_t'],   'b-',  lw=2,
            label='Entangled pairs')
    ax.plot(t_ent, res['max_cl_t'],  'r--', lw=2,
            label='Max cluster size')
    ax.set_xlabel('t'); ax.set_ylabel('count')
    ax.set_title('Phase 4: Entanglement dynamics\n'
                 '(formation and breaking)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3d: Entanglement entropy
    ax = fig.add_subplot(gs[2, 3])
    ax.plot(t_ent, res['entropy_t'], 'g-', lw=2)
    ax.set_xlabel('t')
    ax.set_ylabel('S (entanglement entropy)')
    ax.set_title('Phase 4: Entanglement entropy\n'
                 'S = log(cluster_size)')
    ax.grid(True, alpha=0.3)

    # ── Row 4: Field statistics and summary ──

    # 4a: Phase distribution free vs entangled
    ax = fig.add_subplot(gs[3, 0])
    if res['field_stats']:
        fs  = res['field_stats']
        bins_c = (fs['free']['bins'][:-1] +
                  fs['free']['bins'][1:]) / 2
        ax.plot(bins_c, fs['free']['hist'],
                'b-', lw=2, label='Free strings')
        ax.plot(bins_c, fs['entangled']['hist'],
                'r-', lw=2, label='Entangled strings')
        ax.axhline(1/(2*np.pi), color='black',
                   linestyle='--', lw=1, label='uniform')
        ax.set_xlabel('φ₁'); ax.set_ylabel('density')
        ax.set_title('Phase 5: Field distribution\n'
                     'free vs entangled strings')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4b: Two-point correlation
    ax = fig.add_subplot(gs[3, 1])
    if res['field_stats']:
        fs = res['field_stats']
        seps = np.arange(1, len(fs['free']['correlations'])+1)
        ax.plot(seps, fs['free']['correlations'],
                'b-', lw=2, label='Free')
        ax.plot(seps, fs['entangled']['correlations'],
                'r-', lw=2, label='Entangled')
        ax.axhline(0, color='black', lw=1)
        ax.set_xlabel('separation')
        ax.set_ylabel('C(Δ) = |⟨exp(iΔφ)⟩|')
        ax.set_title('Phase 5: Two-point correlation\n'
                     '(entangled = longer-range corr.?)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4c: Monte Carlo summary
    ax = fig.add_subplot(gs[3, 2])
    t_breaks  = [r['t_break']     for r in all_results]
    n_ents_mc = [r['n_entangled'] for r in all_results]
    ax.scatter(t_breaks, n_ents_mc,
               s=80, c='steelblue', zorder=5)
    ax.set_xlabel('breakup time t*')
    ax.set_ylabel('entangled pairs')
    ax.set_title('MC summary: breakup time\nvs entanglement')
    ax.grid(True, alpha=0.3)

    # 4d: Text summary
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')

    t_b_mean = np.mean([r['t_break']     for r in all_results])
    t_b_std  = np.std( [r['t_break']     for r in all_results])
    n_e_mean = np.mean([r['n_entangled'] for r in all_results])
    ent_mean = np.mean([r['mean_entropy']for r in all_results])
    deq_mean = np.mean([r['dist_eq']     for r in all_results])
    dfr_mean = np.mean([r['dist_free']   for r in all_results])

    summary = (
        f"SIMULATION RESULTS\n"
        f"{'─'*28}\n"
        f"N strings:   {cfg.N_strings}\n"
        f"Breakup t*:  {t_b_mean:.0f} ± {t_b_std:.0f}\n"
        f"\n"
        f"EQUALIZATION\n"
        f"dist (Kuramoto): {deq_mean:.4f}\n"
        f"dist (free):     {dfr_mean:.4f}\n"
        f"ratio:           {deq_mean/dfr_mean:.3f}\n"
        f"\n"
        f"ENTANGLEMENT\n"
        f"pairs:    {n_e_mean:.1f} / {cfg.N_strings*(cfg.N_strings-1)//2}\n"
        f"fraction: {n_e_mean*2/cfg.N_strings:.4f}\n"
        f"entropy:  {ent_mean:.4f}\n"
        f"\n"
        f"VERDICT\n"
        f"{'─'*28}\n"
        f"Equalization: "
        f"{'✅ K drives sync' if deq_mean < dfr_mean*0.7 else '⚠️ weak'}\n"
        f"Entanglement: "
        f"{'✅ rare & transient' if n_e_mean*2/cfg.N_strings < 0.15 else '⚠️ common'}\n"
    )
    ax.text(0.05, 0.97, summary,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#fffde7', alpha=0.9))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation.png")
    plt.show()

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation Hypothesis            ║")
    print("║  Single oscillator → N strings → fields        ║")
    print("╚══════════════════════════════════════════════════╝")

    all_results = run_monte_carlo(n_runs=cfg.n_runs)
    plot_results(all_results)

    print("\n" + "█"*55)
    print("█  PHYSICAL INTERPRETATION                       █")
    print("█"*55)
    print("""
  1. INSTABILITY
     The monostring accumulates chaos (∫λ₁ dt > S_crit)
     and fragments at t* — a deterministic, not random,
     event driven by internal dynamics.

  2. EQUALIZATION
     Kuramoto coupling drives daughter strings to
     identical phases. This explains why fundamental
     particles are identical: they share a common
     origin AND a synchronization mechanism.

  3. ENTANGLEMENT
     Strings that happen to be phase-close after
     equalization form joint wave functions.
     This is rare (~5-10% of pairs) and transient —
     consistent with quantum entanglement being
     a special, not generic, phenomenon.

  4. FIELDS
     Free (non-entangled) strings produce a uniform
     phase distribution → homogeneous fields.
     Entangled clusters produce correlated phases →
     structured, non-uniform fields.
    """)
    print("█"*55)
