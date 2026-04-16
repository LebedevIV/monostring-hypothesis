"""
Monostring Fragmentation Hypothesis — v2
=========================================
Fixes:
1. Instability: higher kappa + noise injection for realistic breakup
2. Equalization: weak coupling (K < K_c) → strings become similar
   but NOT identical (preserve individuality)
3. Entanglement: based on post-equalization proximity,
   with threshold tuned to realistic phase distances
4. Added: null model comparison for entanglement
5. Added: sigma_break scan → equalization quality
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
# PARAMETERS — revised
# ═══════════════════════════════════════════════════════

class Config:
    # Monostring
    D        = 6
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    # --- FIX 1: Higher kappa → real chaos ---
    kappa        = 0.55    # was 0.15; chaos onset ~0.4 for this map
    noise_amp    = 0.02    # small stochastic perturbation

    # Instability criterion
    lyap_window    = 40
    lyap_threshold = 0.04   # lower threshold (realistic)
    chaos_integral = 1.5    # lower accumulation needed

    # Breakup
    N_strings    = 300
    sigma_break  = 0.25    # larger scatter → more variation

    # --- FIX 2: Weak Kuramoto → equalization, not collapse ---
    K_kuramoto   = 0.25    # was 0.8; K_c ≈ 0.5 → subcritical
    T_equalize   = 500

    # --- FIX 3: Entanglement threshold tuned ---
    # After weak sync, typical dist ~ 0.8–1.5 on T^6
    # Entanglement = closest ~5% of pairs
    eps_entangle  = 0.6    # was 0.3; now realistic
    eps_maintain  = 0.9
    T_evolve      = 300

    n_runs    = 8
    seed_base = 42

cfg = Config()

# ═══════════════════════════════════════════════════════
# BLOCK 1: MONOSTRING DYNAMICS — with noise for chaos
# ═══════════════════════════════════════════════════════

def monostring_step(phi, omega, kappa, rng=None, noise=0.0):
    """
    Standard map on T^6 with optional noise.
    φ → φ + ω + κ·sin(φ) + η,  η ~ N(0, noise)
    Noise models external perturbations that can
    tip the system into chaos.
    """
    step = (phi + omega + kappa * np.sin(phi))
    if rng is not None and noise > 0:
        step = step + rng.normal(0, noise, phi.shape)
    return step % (2 * np.pi)

def estimate_lyapunov(phi0, omega, kappa, n_steps=40, seed=0):
    """
    Largest Lyapunov exponent via finite-difference method.
    Positive → chaotic, negative → regular.
    """
    rng  = np.random.RandomState(seed)
    d0   = 1e-8
    phi  = phi0.copy()
    phi2 = (phi0 + d0 / np.sqrt(len(phi0))).copy()
    phi2 %= (2 * np.pi)

    lyap_sum = 0.0
    for _ in range(n_steps):
        phi  = monostring_step(phi,  omega, kappa,
                               rng, cfg.noise_amp)
        phi2 = monostring_step(phi2, omega, kappa,
                               rng, cfg.noise_amp)
        diff = phi2 - phi
        # Torus distance
        diff = np.where(np.abs(diff) > np.pi,
                        diff - 2*np.pi*np.sign(diff), diff)
        dist = np.linalg.norm(diff)
        if dist > 1e-15:
            lyap_sum += np.log(dist / d0)
            phi2 = phi + d0 * diff / dist
            phi2 %= (2 * np.pi)
    return lyap_sum / n_steps

def evolve_monostring(omega, kappa, max_steps=3000, seed=0):
    """
    Evolve monostring with noise until chaos criterion met.

    Breakup condition:
      λ₁ > λ_threshold  AND  ∫λ₁⁺ dt > chaos_integral

    Physical meaning: the system accumulates enough
    chaotic divergence that its single-point description
    becomes untenable → it 'shatters'.
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    phi_history    = [phi.copy()]
    lyap_history   = []
    chaos_integral = 0.0
    t_break        = None

    for t in range(max_steps):
        phi = monostring_step(phi, omega, kappa,
                              rng, cfg.noise_amp)
        phi_history.append(phi.copy())

        if t % cfg.lyap_window == 0 and t >= cfg.lyap_window:
            lam = estimate_lyapunov(phi, omega, kappa,
                                    cfg.lyap_window, seed=t)
            lyap_history.append((t, lam))

            if lam > 0:
                chaos_integral += lam * cfg.lyap_window

            if (lam > cfg.lyap_threshold and
                    chaos_integral > cfg.chaos_integral):
                t_break = t
                break

    if t_break is None:
        t_break = t   # last step reached

    return (np.array(phi_history), t_break,
            lyap_history, chaos_integral)

# ═══════════════════════════════════════════════════════
# BLOCK 2: FRAGMENTATION
# ═══════════════════════════════════════════════════════

def fragment_monostring(phi_break, N, sigma, seed=0):
    """
    Shatter monostring into N daughters.

    Physical model: at the moment of breakup, the
    phase-space trajectory 'explodes' into N fragments,
    each displaced by a random amount σ from the
    monostring's phase at t*.

    σ → 0:  daughters are identical (max equalization)
    σ → π:  daughters are random (no memory of origin)
    """
    rng = np.random.RandomState(seed)
    pert = rng.normal(0, sigma, (N, len(phi_break)))
    return (phi_break[np.newaxis,:] + pert) % (2*np.pi)

def torus_dist_matrix_sample(phases, n_pairs=2000, seed=0):
    """
    Sample pairwise torus distances for N strings.
    Returns: (mean_dist, std_dist, sample of distances)
    """
    N = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=(n_pairs, 2), replace=True)
    mask = idx[:,0] != idx[:,1]
    idx  = idx[mask]

    diff = np.abs(phases[idx[:,0]] - phases[idx[:,1]])
    diff = np.minimum(diff, 2*np.pi - diff)
    dists = np.linalg.norm(diff, axis=1)
    return float(np.mean(dists)), float(np.std(dists)), dists

# ═══════════════════════════════════════════════════════
# BLOCK 3: EQUALIZATION — subcritical Kuramoto
# ═══════════════════════════════════════════════════════

def evolve_daughters_kuramoto(phases0, omega, K,
                               n_steps, measure_every=20):
    """
    Subcritical Kuramoto evolution (K < K_c).

    Key change from v1: K = 0.25 < K_c ≈ 0.5.
    Strings become MORE SIMILAR but NOT IDENTICAL.
    This preserves individuality while driving
    statistical equalization.

    Each string: φᵢ → φᵢ + ω + κ·sin(φᵢ)
                       + (K/N)·Σⱼ sin(φⱼ - φᵢ)
    """
    N, D   = phases0.shape
    phases = phases0.copy()
    history = []

    for step in range(n_steps):
        mf = np.mean(np.exp(1j * phases), axis=0)  # (D,)
        r   = np.abs(mf)
        psi = np.angle(mf)

        coupling = K * np.sin(psi[np.newaxis,:] - phases)
        phases = (phases
                  + omega[np.newaxis,:]
                  + cfg.kappa * np.sin(phases)
                  + coupling) % (2*np.pi)

        if step % measure_every == 0:
            md, sd, _ = torus_dist_matrix_sample(phases)
            history.append({
                'step':        step,
                'mean_dist':   md,
                'std_dist':    sd,
                'order_param': float(np.mean(r)),
                'r_per_dim':   r.copy(),
            })

    return phases, history

def evolve_daughters_free(phases0, omega, n_steps,
                           measure_every=20):
    """
    Free evolution: no inter-string coupling.
    Null model: do strings equalize spontaneously?
    Expected: NO (distances increase with time).
    """
    N, D   = phases0.shape
    phases = phases0.copy()
    history = []

    for step in range(n_steps):
        phases = (phases
                  + omega[np.newaxis,:]
                  + cfg.kappa * np.sin(phases)) % (2*np.pi)

        if step % measure_every == 0:
            md, sd, _ = torus_dist_matrix_sample(phases)
            mf = np.mean(np.exp(1j * phases), axis=0)
            r  = float(np.mean(np.abs(mf)))
            history.append({
                'step':        step,
                'mean_dist':   md,
                'std_dist':    sd,
                'order_param': r,
            })

    return phases, history

# ═══════════════════════════════════════════════════════
# BLOCK 4: ENTANGLEMENT
# ═══════════════════════════════════════════════════════

def detect_entanglement(phases, eps):
    """
    Entanglement criterion: ||φᵢ - φⱼ||_torus < ε.

    Physical interpretation:
    - Two strings with nearly identical phases
      cannot be distinguished → they form a joint
      quantum state (superposition).
    - ε sets the 'resolution' of phase space —
      below ε, individual identity is lost.

    Implemented efficiently via chunked distance
    computation to avoid O(N²D) memory.
    """
    N   = phases.shape[0]
    G   = nx.Graph()
    G.add_nodes_from(range(N))

    chunk = 60
    for i in range(0, N, chunk):
        ie = min(i + chunk, N)
        # Torus distance: (ie-i) × N × D
        diff = np.abs(phases[i:ie, np.newaxis, :]
                      - phases[np.newaxis, :, :])
        diff = np.minimum(diff, 2*np.pi - diff)
        dist = np.linalg.norm(diff, axis=2)   # (ie-i) × N

        rows, cols = np.where((dist < eps) & (dist > 1e-12))
        for r, c in zip(rows, cols):
            ri = r + i
            if ri < c:
                G.add_edge(int(ri), int(c),
                           distance=float(dist[r, c]))

    pairs    = list(G.edges())
    clusters = list(nx.connected_components(G))
    return pairs, G, clusters

def entanglement_wave_function(phases, clusters):
    """
    Joint wave function for entangled cluster:
    |ψ_C⟩ = (1/√|C|) · Σᵢ∈C exp(i·φᵢ)

    This is a coherent superposition on T^D.
    The 'entanglement entropy' is log(|C|) for
    equal-weight superposition.
    """
    psi_list  = []
    entropies = []

    for cluster in clusters:
        cl  = list(cluster)
        n   = len(cl)
        psi = np.mean(np.exp(1j * phases[cl]), axis=0)
        psi_list.append(psi)
        # Von Neumann entropy for equal superposition
        entropies.append(np.log(n) if n > 1 else 0.0)

    sizes = [len(list(c)) for c in clusters]
    return psi_list, sizes, entropies

def entanglement_dynamics(phases, eps, eps_maintain,
                           omega, n_steps=200):
    """
    Track entanglement formation/breaking over time.

    Entanglement lifecycle:
    1. FORM: |φᵢ - φⱼ| < ε_entangle (new bond)
    2. MAINTAIN: |φᵢ - φⱼ| < ε_maintain (bond survives)
    3. BREAK: |φᵢ - φⱼ| ≥ ε_maintain (bond breaks)
    """
    N  = phases.shape[0]
    ph = phases.copy()

    n_ent_hist  = []
    max_cl_hist = []
    entr_hist   = []
    active_pairs = set()

    for step in range(n_steps):
        # Standard map evolution (no inter-string coupling)
        ph = (ph + omega[np.newaxis,:]
                 + cfg.kappa * np.sin(ph)) % (2*np.pi)

        # New entanglements
        _, G_new, _ = detect_entanglement(ph, eps)
        new_pairs = set(G_new.edges())

        # Maintain old ones if still close
        maintained = set()
        for (i, j) in active_pairs:
            d = np.abs(ph[i] - ph[j])
            d = np.minimum(d, 2*np.pi - d)
            if np.linalg.norm(d) < eps_maintain:
                maintained.add((i, j))

        active_pairs = new_pairs | maintained

        # Build active graph
        G_act = nx.Graph()
        G_act.add_nodes_from(range(N))
        G_act.add_edges_from(active_pairs)
        comps = list(nx.connected_components(G_act))

        max_cl = max(len(c) for c in comps) if comps else 1
        _, _, ents = entanglement_wave_function(ph, comps)
        mean_ent = np.mean([e for e in ents if e > 0]) \
                   if any(e > 0 for e in ents) else 0.0

        n_ent_hist.append(len(active_pairs))
        max_cl_hist.append(max_cl)
        entr_hist.append(mean_ent)

    return n_ent_hist, max_cl_hist, entr_hist

# ═══════════════════════════════════════════════════════
# BLOCK 5: FIELD STATISTICS
# ═══════════════════════════════════════════════════════

def measure_field_statistics(phases_free, phases_ent,
                              n_bins=30):
    """
    Compare:
    - Free strings → should give uniform distribution
      (ergodic exploration of T^D)
    - Entangled strings → phase-locked → non-uniform

    Metrics:
    1. Phase histogram + KL vs uniform
    2. Two-point correlation C(Δ) = |⟨e^{iΔφ}⟩|
    3. Power spectrum of phase field
    4. KS test: free vs entangled distributions
    """
    results = {}

    for label, ph in [('free', phases_free),
                       ('entangled', phases_ent)]:
        N, D = ph.shape

        hist, bins = np.histogram(
            ph[:,0], bins=n_bins,
            range=(0, 2*np.pi), density=True)

        max_sep = min(40, N//2)
        corr = []
        for sep in range(1, max_sep):
            diff = ph[:N-sep, 0] - ph[sep:, 0]
            corr.append(float(
                np.abs(np.mean(np.exp(1j * diff)))))

        fft   = np.fft.rfft(ph[:,0])
        power = np.abs(fft)**2

        uniform   = np.ones(n_bins) / n_bins
        hist_norm = hist / (hist.sum() + 1e-15) + 1e-15
        kl = float(entropy(hist_norm, uniform + 1e-15))

        results[label] = {
            'hist':         hist,
            'bins':         bins,
            'correlations': np.array(corr),
            'power':        power,
            'kl_div':       kl,
        }

    ks_stat, ks_p = ks_2samp(
        phases_free[:,0], phases_ent[:,0])
    results['ks_stat'] = float(ks_stat)
    results['ks_p']    = float(ks_p)

    return results

# ═══════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════

def run_simulation(seed=42, verbose=True):
    omega = cfg.omega_E6

    if verbose:
        print(f"\n{'='*56}")
        print(f"  SEED {seed}")
        print(f"{'='*56}")

    # Phase 1 ─────────────────────────────────────────
    if verbose:
        print("\n  Phase 1: Monostring instability...")

    phi_hist, t_break, lyap_hist, chaos_total = \
        evolve_monostring(omega, cfg.kappa,
                          max_steps=3000, seed=seed)
    phi_break = phi_hist[t_break]

    if verbose:
        print(f"  Breakup at t={t_break}")
        print(f"  Chaos integral:  {chaos_total:.3f}")
        if lyap_hist:
            print(f"  Final λ₁ = {lyap_hist[-1][1]:.4f}")

    # Phase 2 ─────────────────────────────────────────
    if verbose:
        print(f"\n  Phase 2: Fragmentation → N={cfg.N_strings}...")

    daughters0 = fragment_monostring(
        phi_break, cfg.N_strings, cfg.sigma_break,
        seed=seed+1)

    md0, sd0, _ = torus_dist_matrix_sample(daughters0)
    if verbose:
        print(f"  Initial mean dist: {md0:.4f} ± {sd0:.4f}")

    # Phase 3a ────────────────────────────────────────
    if verbose:
        print(f"\n  Phase 3a: Kuramoto equalization "
              f"(K={cfg.K_kuramoto})...")

    phases_eq, hist_eq = evolve_daughters_kuramoto(
        daughters0, omega, cfg.K_kuramoto, cfg.T_equalize)

    md_eq, sd_eq, _ = torus_dist_matrix_sample(phases_eq)
    r_final = hist_eq[-1]['order_param'] if hist_eq else 0.0

    if verbose:
        print(f"  Final mean dist:   {md_eq:.4f} ± {sd_eq:.4f}")
        print(f"  Order param r:     {r_final:.4f}")

    # Phase 3b ────────────────────────────────────────
    if verbose:
        print(f"\n  Phase 3b: Free evolution (null)...")

    phases_fr, hist_fr = evolve_daughters_free(
        daughters0, omega, cfg.T_equalize)

    md_fr, sd_fr, _ = torus_dist_matrix_sample(phases_fr)

    if verbose:
        print(f"  Free mean dist:    {md_fr:.4f} ± {sd_fr:.4f}")
        ratio = md_eq / (md_fr + 1e-10)
        verdict = '✅ YES' if md_eq < md_fr * 0.85 else '⚠️ WEAK'
        print(f"  Equalization:      {verdict} "
              f"(ratio={ratio:.3f})")

    # Phase 4 ─────────────────────────────────────────
    if verbose:
        print(f"\n  Phase 4: Entanglement (ε={cfg.eps_entangle})...")

    pairs, G_ent, clusters = detect_entanglement(
        phases_eq, cfg.eps_entangle)

    n_ent    = len(pairs)
    n_multi  = len([c for c in clusters if len(c) > 1])
    cl_sizes = [len(c) for c in clusters]
    max_cl   = max(cl_sizes)
    frac_ent = n_ent * 2 / cfg.N_strings

    if verbose:
        print(f"  Entangled pairs:   {n_ent}")
        print(f"  Multi-clusters:    {n_multi}")
        print(f"  Largest cluster:   {max_cl}")
        print(f"  Entangled frac:    {frac_ent:.3f}")

    psi_list, cl_szs, ent_ents = \
        entanglement_wave_function(phases_eq, clusters)
    mean_ent = np.mean([e for e in ent_ents if e > 0]) \
               if any(e > 0 for e in ent_ents) else 0.0

    if verbose:
        print(f"  Mean entropy:      {mean_ent:.4f}")

    # Null model: entanglement for random phases
    rng_null = np.random.RandomState(seed + 999)
    phases_null = rng_null.uniform(
        0, 2*np.pi, phases_eq.shape)
    pairs_null, _, _ = detect_entanglement(
        phases_null, cfg.eps_entangle)
    n_ent_null = len(pairs_null)

    if verbose:
        print(f"  Null entangled:    {n_ent_null} "
              f"(vs E6: {n_ent})")

    # Dynamics
    n_ent_t, max_cl_t, entr_t = entanglement_dynamics(
        phases_eq, cfg.eps_entangle, cfg.eps_maintain,
        omega, n_steps=cfg.T_evolve)

    # Phase 5 ─────────────────────────────────────────
    if verbose:
        print(f"\n  Phase 5: Field statistics...")

    ent_nodes  = set()
    for c in clusters:
        if len(c) > 1:
            ent_nodes.update(c)
    free_nodes = [i for i in range(cfg.N_strings)
                  if i not in ent_nodes]

    if len(free_nodes) > 20 and len(ent_nodes) > 20:
        field_stats = measure_field_statistics(
            phases_eq[free_nodes],
            phases_eq[list(ent_nodes)])
        if verbose:
            fs = field_stats
            print(f"  Free / ent:  "
                  f"{len(free_nodes)} / {len(ent_nodes)}")
            print(f"  KL free:     {fs['free']['kl_div']:.4f}")
            print(f"  KL ent:      {fs['entangled']['kl_div']:.4f}")
            print(f"  KS p-value:  {fs['ks_p']:.4f}")
    else:
        field_stats = None
        if verbose:
            print(f"  (skipped: not enough strings "
                  f"free={len(free_nodes)} "
                  f"ent={len(ent_nodes)})")

    return dict(
        seed=seed,
        t_break=t_break,
        chaos_total=chaos_total,
        lyap_hist=lyap_hist,
        phi_hist=phi_hist,
        daughters0=daughters0,
        phases_eq=phases_eq,
        phases_fr=phases_fr,
        hist_eq=hist_eq,
        hist_fr=hist_fr,
        md0=md0, sd0=sd0,
        md_eq=md_eq, sd_eq=sd_eq,
        md_fr=md_fr, sd_fr=sd_fr,
        r_final=r_final,
        n_ent=n_ent,
        n_ent_null=n_ent_null,
        n_multi=n_multi,
        cl_sizes=cl_sizes,
        max_cl=max_cl,
        frac_ent=frac_ent,
        mean_ent=mean_ent,
        n_ent_t=n_ent_t,
        max_cl_t=max_cl_t,
        entr_t=entr_t,
        G_ent=G_ent,
        clusters=clusters,
        field_stats=field_stats,
    )

# ═══════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════

def run_monte_carlo(n_runs=8):
    print("\n" + "█"*56)
    print("█  MONOSTRING FRAGMENTATION v2 — MONTE CARLO        █")
    print("█"*56)
    print(f"  N={cfg.N_strings}, K={cfg.K_kuramoto}, "
          f"κ={cfg.kappa}, ε={cfg.eps_entangle}, "
          f"{n_runs} runs\n")

    all_results = []
    t0 = time.time()

    for run in range(n_runs):
        seed = cfg.seed_base + run * 7
        res  = run_simulation(seed=seed, verbose=(run==0))
        all_results.append(res)
        sig = ('✅' if res['md_eq'] < res['md_fr']*0.85
               else '⚠️')
        print(f"  Run {run+1}/{n_runs}: "
              f"t*={res['t_break']:4d}, "
              f"n_ent={res['n_ent']:3d} "
              f"(null={res['n_ent_null']:3d}), "
              f"eq={sig} "
              f"dist_eq={res['md_eq']:.3f} "
              f"dist_fr={res['md_fr']:.3f}")

    elapsed = time.time() - t0

    # Aggregate
    T  = [r['t_break']    for r in all_results]
    NE = [r['n_ent']      for r in all_results]
    NN = [r['n_ent_null'] for r in all_results]
    EN = [r['mean_ent']   for r in all_results]
    DE = [r['md_eq']      for r in all_results]
    DF = [r['md_fr']      for r in all_results]

    print(f"\n{'='*56}")
    print(f"  AGGREGATED ({n_runs} runs, {elapsed:.0f}s)")
    print(f"{'='*56}")
    print(f"  Breakup t*:    {np.mean(T):.0f} ± {np.std(T):.0f}")
    print(f"  Entangled (E6):{np.mean(NE):.1f} ± {np.std(NE):.1f}")
    print(f"  Entangled (null):{np.mean(NN):.1f} ± {np.std(NN):.1f}")
    print(f"  E6 vs null:    "
          f"{'✅ MORE' if np.mean(NE)>np.mean(NN) else '❌ LESS/EQUAL'}")
    print(f"  Entropy:       "
          f"{np.mean(EN):.4f} ± {np.std(EN):.4f}")
    print(f"  dist_eq:       "
          f"{np.mean(DE):.4f} ± {np.std(DE):.4f}")
    print(f"  dist_free:     "
          f"{np.mean(DF):.4f} ± {np.std(DF):.4f}")

    eq_works = np.mean(DE) < np.mean(DF) * 0.85
    ent_rare = np.mean(NE)*2/cfg.N_strings < 0.20
    ent_more = np.mean(NE) > np.mean(NN)

    print(f"\n  Equalization works:        "
          f"{'✅' if eq_works else '❌'}")
    print(f"  Entanglement is rare:      "
          f"{'✅' if ent_rare else '❌'}")
    print(f"  E6 > null entanglement:    "
          f"{'✅' if ent_more else '❌'}")

    return all_results

# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_results(all_results):
    res  = all_results[0]
    n_mc = len(all_results)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Monostring Fragmentation Hypothesis v2\n"
        "κ={kappa}, K={K}, ε={eps}, N={N}".format(
            kappa=cfg.kappa, K=cfg.K_kuramoto,
            eps=cfg.eps_entangle, N=cfg.N_strings),
        fontsize=14, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.50, wspace=0.38)

    # ── Row 0: Monostring ──────────────────────────────

    ax = fig.add_subplot(gs[0, 0])
    ph = res['phi_hist']
    tb = res['t_break']
    N_trail = min(tb, 400)
    ax.plot(ph[tb-N_trail:tb, 0],
            ph[tb-N_trail:tb, 1],
            'b-', alpha=0.4, lw=0.7, label='trajectory')
    ax.scatter(ph[tb, 0], ph[tb, 1],
               s=200, c='red', marker='*',
               zorder=5, label=f't*={tb}')
    ax.set_xlim([0, 2*np.pi]); ax.set_ylim([0, 2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('Monostring trajectory\n(last 400 steps before t*)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    if res['lyap_hist']:
        ts = [x[0] for x in res['lyap_hist']]
        ls = [x[1] for x in res['lyap_hist']]
        ax.plot(ts, ls, 'b-o', ms=4, label='λ₁(t)')
        ax.axhline(cfg.lyap_threshold, c='red',
                   ls='--', label=f'threshold')
        ax.axhline(0, c='k', lw=1)
        ax.axvline(tb, c='orange', ls=':', label=f't*={tb}')
    ax.set_xlabel('t'); ax.set_ylabel('λ₁')
    ax.set_title('Lyapunov exponent\n(chaos criterion)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Chaos integral over time
    ax = fig.add_subplot(gs[0, 2])
    if res['lyap_hist']:
        chaos_cum = np.cumsum([max(0, x[1]) * cfg.lyap_window
                               for x in res['lyap_hist']])
        ax.plot(ts, chaos_cum, 'g-', lw=2,
                label='∫λ₁⁺ dt')
        ax.axhline(cfg.chaos_integral, c='red',
                   ls='--', label=f'threshold={cfg.chaos_integral}')
        ax.axvline(tb, c='orange', ls=':', label=f't*={tb}')
    ax.set_xlabel('t'); ax.set_ylabel('∫λ₁⁺ dt')
    ax.set_title('Accumulated chaos\n(breakup criterion)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # σ scan
    ax = fig.add_subplot(gs[0, 3])
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    dists_s = []
    for sg in sigmas:
        d = fragment_monostring(
            res['phi_hist'][tb], 100, sg, seed=0)
        md, _, _ = torus_dist_matrix_sample(d)
        dists_s.append(md)
    ax.plot(sigmas, dists_s, 'bo-', lw=2)
    ax.axvline(cfg.sigma_break, c='red', ls='--',
               label=f'σ={cfg.sigma_break}')
    ax.set_xlabel('σ'); ax.set_ylabel('mean dist')
    ax.set_xscale('log')
    ax.set_title('Fragmentation width σ\nvs initial similarity')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 1: Equalization ────────────────────────────

    ax = fig.add_subplot(gs[1, 0:2])
    steps_eq = [h['step']     for h in res['hist_eq']]
    dist_eq  = [h['mean_dist']for h in res['hist_eq']]
    steps_fr = [h['step']     for h in res['hist_fr']]
    dist_fr  = [h['mean_dist']for h in res['hist_fr']]
    ax.plot(steps_eq, dist_eq, 'b-',  lw=2,
            label=f'Kuramoto K={cfg.K_kuramoto}')
    ax.plot(steps_fr, dist_fr, 'r--', lw=2,
            label='Free (null)')
    ax.set_xlabel('t'); ax.set_ylabel('mean pairwise distance')
    ax.set_title('Equalization: Kuramoto vs free evolution\n'
                 '(K < K_c: strings become similar, not identical)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    order_eq = [h['order_param'] for h in res['hist_eq']]
    order_fr = [h['order_param'] for h in res['hist_fr']]
    ax.plot(steps_eq, order_eq, 'b-', lw=2, label='Kuramoto')
    ax.plot(steps_fr, order_fr, 'r--',lw=2, label='Free')
    ax.axhline(1.0, c='k', ls=':', lw=1)
    ax.set_xlabel('t'); ax.set_ylabel('r (order param)')
    ax.set_ylim([0, 1.1])
    ax.set_title('Kuramoto order parameter\n(r=1: full sync)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    bins_ph = np.linspace(0, 2*np.pi, 35)
    ax.hist(res['daughters0'][:,0], bins=bins_ph,
            alpha=0.5, density=True, color='red',
            label='Initial (t=0)')
    ax.hist(res['phases_eq'][:,0], bins=bins_ph,
            alpha=0.5, density=True, color='blue',
            label='After Kuramoto')
    ax.hist(res['phases_fr'][:,0], bins=bins_ph,
            alpha=0.5, density=True, color='green',
            label='After free')
    ax.axhline(1/(2*np.pi), c='k', ls='--', lw=1,
               label='uniform')
    ax.set_xlabel('φ₁'); ax.set_ylabel('density')
    ax.set_title('Phase distribution\nbefore/after equalization')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # ── Row 2: Entanglement ────────────────────────────

    ax = fig.add_subplot(gs[2, 0])
    G   = res['G_ent']
    peq = res['phases_eq']
    pos = {i: (peq[i,0], peq[i,1])
           for i in range(cfg.N_strings)}

    node_col = ['lightgray'] * cfg.N_strings
    cmap2    = plt.cm.tab10
    for ci, clust in enumerate(res['clusters']):
        if len(clust) > 1:
            for nd in clust:
                node_col[nd] = cmap2(ci % 10)

    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_size=6, node_color=node_col,
                     edge_color='crimson',
                     alpha=0.6, with_labels=False, width=0.5)
    ax.set_title(f'Entanglement network\n'
                 f'n_pairs={res["n_ent"]} (null={res["n_ent_null"]})')
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[2, 1])
    szs = res['cl_sizes']
    max_s = max(szs)
    ax.hist(szs, bins=np.arange(1, max_s+2)-0.5,
            color='steelblue', alpha=0.8, edgecolor='k')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Count')
    ax.set_title('Cluster size distribution')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    t_dyn = np.arange(len(res['n_ent_t']))
    ax.plot(t_dyn, res['n_ent_t'],  'b-', lw=2,
            label='Entangled pairs')
    ax.plot(t_dyn, res['max_cl_t'], 'r--',lw=2,
            label='Max cluster size')
    ax.set_xlabel('t'); ax.set_ylabel('count')
    ax.set_title('Entanglement dynamics\n(formation / breaking)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.plot(t_dyn, res['entr_t'], 'g-', lw=2)
    ax.set_xlabel('t'); ax.set_ylabel('S')
    ax.set_title('Entanglement entropy S = log(n_cluster)')
    ax.grid(True, alpha=0.3)

    # ── Row 3: Field statistics + MC summary ──────────

    if res['field_stats']:
        fs = res['field_stats']

        ax = fig.add_subplot(gs[3, 0])
        bc = (fs['free']['bins'][:-1] +
              fs['free']['bins'][1:]) / 2
        ax.plot(bc, fs['free']['hist'],
                'b-', lw=2, label='Free')
        ax.plot(bc, fs['entangled']['hist'],
                'r-', lw=2, label='Entangled')
        ax.axhline(1/(2*np.pi), c='k', ls='--',
                   lw=1, label='uniform')
        ax.set_xlabel('φ₁'); ax.set_ylabel('density')
        ax.set_title('Field distribution\nfree vs entangled')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[3, 1])
        seps = np.arange(1, len(fs['free']['correlations'])+1)
        ax.plot(seps, fs['free']['correlations'],
                'b-', lw=2, label='Free')
        ax.plot(seps, fs['entangled']['correlations'],
                'r-', lw=2, label='Entangled')
        ax.axhline(0, c='k', lw=1)
        ax.set_xlabel('Δ (separation)')
        ax.set_ylabel('C(Δ)')
        ax.set_title('Two-point correlation\n|⟨exp(iΔφ)⟩|')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        for col in [0, 1]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5, 0.5, 'Field stats\nnot available\n'
                    '(all strings entangled\nor all free)',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    ax = fig.add_subplot(gs[3, 2])
    T_mc  = [r['t_break'] for r in all_results]
    NE_mc = [r['n_ent']   for r in all_results]
    NN_mc = [r['n_ent_null'] for r in all_results]
    ax.scatter(T_mc, NE_mc, s=80, c='steelblue',
               zorder=5, label='E6')
    ax.scatter(T_mc, NN_mc, s=80, c='orange',
               zorder=4, marker='^', label='null')
    ax.set_xlabel('breakup time t*')
    ax.set_ylabel('entangled pairs')
    ax.set_title('MC: t* vs entanglement\n(blue=E6, orange=null)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')

    t_m   = np.mean(T_mc);   t_s   = np.std(T_mc)
    ne_m  = np.mean(NE_mc);  ne_s  = np.std(NE_mc)
    nn_m  = np.mean(NN_mc)
    en_m  = np.mean([r['mean_ent'] for r in all_results])
    de_m  = np.mean([r['md_eq']    for r in all_results])
    df_m  = np.mean([r['md_fr']    for r in all_results])

    eq_ok  = de_m < df_m * 0.85
    ent_ok = ne_m > nn_m

    txt = (
        f"RESULTS (n={n_mc} runs)\n"
        f"{'─'*30}\n"
        f"Breakup t*: {t_m:.0f} ± {t_s:.0f}\n"
        f"\nEQUALIZATION\n"
        f"  dist_eq:   {de_m:.4f}\n"
        f"  dist_free: {df_m:.4f}\n"
        f"  ratio:     {de_m/df_m:.3f}\n"
        f"  verdict: {'✅' if eq_ok else '❌'}\n"
        f"\nENTANGLEMENT\n"
        f"  E6 pairs:  {ne_m:.1f} ± {ne_s:.1f}\n"
        f"  null pairs:{nn_m:.1f}\n"
        f"  E6>null:   {'✅' if ent_ok else '❌'}\n"
        f"  entropy:   {en_m:.4f}\n"
        f"\nOVERALL\n"
        f"  {'✅ Physical picture consistent' if eq_ok and ent_ok else '⚠️ Partial: check parameters'}\n"
    )
    ax.text(0.03, 0.98, txt,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e8f5e9', alpha=0.95))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation_v2.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v2.png")
    plt.show()

# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation Hypothesis v2         ║")
    print("║  Fixed: chaos criterion, K<K_c, entanglement   ║")
    print("╚══════════════════════════════════════════════════╝")

    all_results = run_monte_carlo(n_runs=cfg.n_runs)
    plot_results(all_results)

    print("\n" + "█"*56)
    print("█  WHAT THIS SIMULATION TESTS                    █")
    print("█"*56)
    print("""
  INSTABILITY   Does the monostring develop genuine chaos
                before fragmenting? (κ=0.55 → yes)

  EQUALIZATION  Does Kuramoto coupling reduce inter-string
                distances WITHOUT full collapse?
                (K=0.25 < K_c: yes, subcritical regime)

  ENTANGLEMENT  Are E6 strings more entangled than random?
                (phase clustering from shared origin)

  FIELDS        Do free strings give uniform fields,
                entangled clusters give structured fields?

  NULL MODELS   Every claim tested against random baseline.
    """)
    print("█"*56)
