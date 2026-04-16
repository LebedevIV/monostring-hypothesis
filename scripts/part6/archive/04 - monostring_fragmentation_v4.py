"""
Monostring Fragmentation Hypothesis — v4
=========================================
Complete architectural rewrite based on lessons from v1–v3.

Key lessons learned:
  v1: K=0.8 → full collapse (dist=0), no entanglement
  v2: kappa=0.55 → chaos too fast (t*=40), collapse
  v3: E6 frequencies → quasi-periodic, no chaos possible
      K=0.12 → still collapses at t=200
      79800/79800 pairs = trivial (all in one cluster)

New architecture:
  1. INSTABILITY: resonance-based, not Lyapunov
     Monostring breaks when phase visits resonance zone
     (rational approximation of E6 frequencies)

  2. FRAGMENTATION: large sigma → strings NOT identical
     sigma = 0.8 rad → mean_dist ~ 1.5 after breakup

  3. EQUALIZATION: NO Kuramoto collapse
     Instead: strings with shared origin drift TOGETHER
     (common frequency ω_E6 acts as attractor)
     Measured by: does variance DECREASE relative to null?

  4. ENTANGLEMENT: geometric criterion
     Two strings entangle if their RELATIVE phase
     stays bounded (quasi-periodic relative motion)
     vs diverges (chaotic relative motion)
     This is a DYNAMICAL criterion, not a proximity snapshot

  5. NULL MODEL: strings with RANDOM frequencies
     (not E6) — do they show the same clustering?

  6. PERFORMANCE: no O(N²) per step
     Entanglement checked only at snapshots
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp, ttest_ind
from itertools import combinations
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# PARAMETERS — v4
# ═══════════════════════════════════════════════════════

class Config:
    D = 6
    # E6 Coxeter exponents: m = 1,4,5,7,8,11
    # Frequencies: ω_k = 2·sin(π·m_k/12)
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    # Random frequencies (null model)
    # Same magnitude distribution, no Lie algebra structure
    rng_freq = np.random.RandomState(0)
    omega_null = rng_freq.uniform(
        omega_E6.min(), omega_E6.max(), 6)

    # ── Monostring: resonance-based instability ──
    kappa_mono   = 0.30   # moderate nonlinearity
    # Resonance zone: phase enters rational zone
    # p/q approximation with |ω - p/q| < delta_res
    delta_res    = 0.15   # resonance width
    # Breakup: monostring visits resonance zone
    # n_res_visits times
    n_res_visits = 8

    # ── Fragmentation ──
    N_strings  = 200      # smaller N → faster O(N²)
    sigma_E6   = 0.80     # large scatter: strings NOT identical
    sigma_null = 0.80     # same scatter for null

    # ── Daughter dynamics ──
    # Strings evolve with their OWN frequency ω_E6
    # (shared heritage) + weak individual noise
    kappa_dau  = 0.05     # very weak nonlinearity
    noise_dau  = 0.008    # individual fluctuations
    T_evolve   = 400      # evolution steps

    # ── Entanglement: dynamical criterion ──
    # Two strings i,j are 'bound' if their relative phase
    # Δφ_ij(t) = φ_i(t) - φ_j(t) has bounded variation
    # Measured over T_ent steps
    T_ent        = 80     # steps to assess relative dynamics
    # Threshold: std(Δφ) < eps_bound → entangled
    eps_bound    = 0.25   # rad, relative phase stability

    # ── Snapshots for entanglement ──
    snap_times = [50, 100, 200, 400]

    n_runs    = 8
    seed_base = 42

cfg = Config()


# ═══════════════════════════════════════════════════════
# BLOCK 1: MONOSTRING — resonance instability
# ═══════════════════════════════════════════════════════

def in_resonance_zone(phi, omega, delta):
    """
    Check if current phase is in a resonance zone.

    Resonance condition: ω_k ≈ p/q for small q
    On T^6: we check if the phase vector φ is near
    a rational hyperplane.

    Simplified criterion: at least 2 dimensions satisfy
    |sin(q·φ_k)| < delta for some q ∈ {2,3,4,5}
    This detects low-order resonances.
    """
    q_list = [2, 3, 4, 5]
    n_resonant = 0
    for k in range(len(phi)):
        for q in q_list:
            if abs(np.sin(q * phi[k])) < delta:
                n_resonant += 1
                break  # one resonance per dimension
    return n_resonant >= 2  # at least 2 dims resonant

def evolve_monostring(omega, kappa, max_steps=3000,
                      seed=0, verbose=True):
    """
    Evolve monostring until resonance criterion triggers breakup.

    Physical model:
      φ(t+1) = φ(t) + ω + κ·sin(φ(t))

    Breakup: when the trajectory visits resonance zones
    n_res_visits times, the oscillator becomes
    'entangled with itself' and shatters.

    Returns:
      phi_hist  — trajectory
      t_break   — breakup step
      res_hist  — [(t, in_resonance), ...]
      resonance_count — total visits
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    phi_hist   = [phi.copy()]
    res_hist   = []
    n_res      = 0
    t_break    = None

    for t in range(max_steps):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)
        phi_hist.append(phi.copy())

        in_res = in_resonance_zone(
            phi, omega, cfg.delta_res)
        res_hist.append((t, int(in_res)))

        if in_res:
            n_res += 1
            if n_res >= cfg.n_res_visits:
                t_break = t
                break

    if t_break is None:
        t_break = len(phi_hist) - 1

    return (np.array(phi_hist), t_break,
            res_hist, n_res)


# ═══════════════════════════════════════════════════════
# BLOCK 2: FRAGMENTATION
# ═══════════════════════════════════════════════════════

def fragment(phi_break, N, sigma, seed=0):
    """
    Create N daughter strings.
    Large sigma (0.8 rad) → strings are NOT identical.
    They share common origin (phi_break) and
    common frequencies (omega_E6), but different phases.
    """
    rng  = np.random.RandomState(seed)
    pert = rng.normal(0, sigma, (N, len(phi_break)))
    return (phi_break[np.newaxis, :] + pert) % (2*np.pi)

def torus_dist_sample(phases, n_pairs=2000, seed=0):
    """Efficient mean torus distance via sampling."""
    N   = len(phases)
    rng = np.random.RandomState(seed)
    i   = rng.randint(0, N, n_pairs)
    j   = rng.randint(0, N, n_pairs)
    ok  = i != j
    i, j = i[ok], j[ok]
    d = np.abs(phases[i] - phases[j])
    d = np.minimum(d, 2*np.pi - d)
    return (float(np.mean(np.linalg.norm(d, axis=1))),
            float(np.std( np.linalg.norm(d, axis=1))))


# ═══════════════════════════════════════════════════════
# BLOCK 3: DAUGHTER DYNAMICS
# ═══════════════════════════════════════════════════════

def evolve_daughters_batch(phases0, omega,
                            kappa, noise, n_steps,
                            seed=0, snapshot_at=None):
    """
    Evolve N strings with SHARED frequency omega.
    No inter-string coupling (K=0) — shared origin
    is the only connection.

    Physical claim: strings with shared ω_E6 will
    drift less relative to each other than strings
    with random ω.

    Returns: final phases, history, snapshots
    """
    rng      = np.random.RandomState(seed)
    ph       = phases0.copy()
    snap_set = set(snapshot_at or [])
    snaps    = {}
    hist     = []

    for step in range(n_steps):
        noise_v = rng.normal(0, noise, ph.shape)
        ph      = (ph + omega[np.newaxis, :]
                   + kappa * np.sin(ph)
                   + noise_v) % (2*np.pi)

        if step in snap_set:
            snaps[step] = ph.copy()

        if step % 40 == 0:
            md, sd = torus_dist_sample(ph)
            mf     = np.mean(np.exp(1j * ph), axis=0)
            r      = float(np.mean(np.abs(mf)))
            hist.append(dict(
                step=step, mean_dist=md,
                std_dist=sd, order_param=r))

    return ph, hist, snaps


# ═══════════════════════════════════════════════════════
# BLOCK 4: DYNAMICAL ENTANGLEMENT CRITERION
# ═══════════════════════════════════════════════════════

def relative_phase_stability(ph_snap_early,
                              ph_snap_late):
    """
    For each pair (i,j): compute how much their
    relative phase Δφ_ij changed between two snapshots.

    Δφ_ij(early) = φ_i(t1) - φ_j(t1)
    Δφ_ij(late)  = φ_i(t2) - φ_j(t2)

    Drift = ||Δφ(late) - Δφ(early)||_torus

    Small drift → pair is 'bound' (entangled)
    Large drift → pair has diverged (not entangled)

    Returns:
      drift matrix (N×N) — only upper triangle used
      entangled pairs (drift < eps_bound)
    """
    N = len(ph_snap_early)

    # Relative phases at two times
    # Shape: (N, N, D)
    # To avoid O(N²·D) memory with N=200, D=6:
    # chunk it

    pairs_bound = []
    drift_sample = []  # for statistics
    chunk = 50

    for i in range(0, N, chunk):
        ie = min(i + chunk, N)
        # Δφ_ij at early time
        delta_early = (ph_snap_early[i:ie, np.newaxis, :]
                       - ph_snap_early[np.newaxis, :, :])
        delta_early -= (2*np.pi
                        * np.round(delta_early/(2*np.pi)))

        # Δφ_ij at late time
        delta_late  = (ph_snap_late[i:ie, np.newaxis, :]
                       - ph_snap_late[np.newaxis, :, :])
        delta_late  -= (2*np.pi
                        * np.round(delta_late/(2*np.pi)))

        # Change in relative phase
        drift = delta_late - delta_early
        drift -= 2*np.pi * np.round(drift/(2*np.pi))
        drift_norm = np.linalg.norm(drift, axis=2)
        # shape: (ie-i, N)

        rows, cols = np.where(
            (drift_norm < cfg.eps_bound)
            & (np.arange(ie-i)[:, None]
               + i < np.arange(N)[None, :]))
        for r, c in zip(rows, cols):
            ri = r + i
            if ri < c:
                pairs_bound.append((int(ri), int(c)))
                drift_sample.append(float(drift_norm[r,c]))

        # Sample some drifts for statistics
        mask_upper = (np.arange(ie-i)[:, None] + i
                      < np.arange(N)[None, :])
        drift_sample.extend(
            drift_norm[mask_upper].tolist()[:20])

    return pairs_bound, drift_sample

def build_entanglement_graph(pairs, N):
    """Build graph from entangled pairs."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(pairs)
    clusters = list(nx.connected_components(G))
    return G, clusters

def entanglement_entropy(clusters):
    """S = log(|cluster|) per cluster."""
    return [np.log(len(c)) if len(c) > 1 else 0.0
            for c in clusters]


# ═══════════════════════════════════════════════════════
# BLOCK 5: FIELD STATISTICS
# ═══════════════════════════════════════════════════════

def field_stats(phases_free, phases_ent, n_bins=25):
    """
    Phase distribution statistics.
    Uses dim 0 only (representative).
    """
    from scipy.stats import entropy as sci_entropy
    results = {}
    for label, ph in [('free', phases_free),
                       ('ent',  phases_ent)]:
        if len(ph) < 5:
            results[label] = None
            continue
        h, bins = np.histogram(
            ph[:,0], bins=n_bins,
            range=(0,2*np.pi), density=True)
        uniform = np.ones(n_bins)/n_bins
        hn  = h/(h.sum()+1e-15) + 1e-15
        kl  = float(sci_entropy(hn, uniform+1e-15))

        # Two-point correlation
        n_sep = min(30, len(ph)//2)
        corr  = []
        for sep in range(1, n_sep):
            c = float(np.abs(np.mean(np.exp(
                1j*(ph[:len(ph)-sep,0]
                    - ph[sep:,0])))))
            corr.append(c)

        results[label] = dict(
            hist=h, bins=bins, kl=kl,
            corr=np.array(corr))

    if (results.get('free') is not None
            and results.get('ent') is not None):
        ks, ksp = ks_2samp(
            phases_free[:,0], phases_ent[:,0])
        results['ks'] = ks
        results['ks_p'] = ksp
    else:
        results['ks'] = None
        results['ks_p'] = None
    return results


# ═══════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════

def run_one(seed=42, verbose=True):
    omega_e6   = cfg.omega_E6
    omega_null = cfg.omega_null

    if verbose:
        print(f"\n{'='*56}\n  SEED {seed}\n{'='*56}")

    # ── Phase 1: Monostring resonance instability ─────
    if verbose:
        print("\n  Phase 1: Monostring resonance instability...")

    phi_hist, t_break, res_hist, n_res = \
        evolve_monostring(
            omega_e6, cfg.kappa_mono,
            max_steps=3000, seed=seed)
    phi_break = phi_hist[t_break]

    if verbose:
        print(f"  Breakup t*:        {t_break}")
        print(f"  Resonance visits:  {n_res}")
        n_total = len(res_hist)
        n_in    = sum(x[1] for x in res_hist)
        print(f"  Time in resonance: "
              f"{n_in}/{n_total} "
              f"({100*n_in/max(n_total,1):.1f}%)")

    # ── Phase 2: Fragmentation ────────────────────────
    if verbose:
        print(f"\n  Phase 2: Fragmentation...")

    # E6 daughters: shared origin + E6 frequencies
    d_e6 = fragment(phi_break, cfg.N_strings,
                    cfg.sigma_E6,  seed=seed+1)
    # Null daughters: RANDOM origin + null frequencies
    rng_n = np.random.RandomState(seed+8888)
    phi_rand = rng_n.uniform(0, 2*np.pi, phi_break.shape)
    d_null = fragment(phi_rand, cfg.N_strings,
                      cfg.sigma_null, seed=seed+2)

    md0_e6,  sd0_e6  = torus_dist_sample(d_e6)
    md0_null,sd0_null= torus_dist_sample(d_null)

    if verbose:
        print(f"  E6 initial dist:   {md0_e6:.4f} ± {sd0_e6:.4f}")
        print(f"  Null initial dist: "
              f"{md0_null:.4f} ± {sd0_null:.4f}")

    # ── Phase 3: Daughter dynamics ────────────────────
    if verbose:
        print(f"\n  Phase 3: Daughter dynamics...")

    snap_times = cfg.snap_times
    # Need two snapshots for relative-phase criterion
    t_early = snap_times[0]   # e.g., 50
    t_late  = snap_times[2]   # e.g., 200

    # E6: shared E6 frequencies
    ph_e6, hist_e6, snaps_e6 = evolve_daughters_batch(
        d_e6, omega_e6,
        cfg.kappa_dau, cfg.noise_dau,
        cfg.T_evolve, seed=seed+3,
        snapshot_at=snap_times)

    # Null: random frequencies
    ph_null, hist_null, snaps_null = evolve_daughters_batch(
        d_null, omega_null,
        cfg.kappa_dau, cfg.noise_dau,
        cfg.T_evolve, seed=seed+4,
        snapshot_at=snap_times)

    md_e6,  sd_e6  = torus_dist_sample(ph_e6)
    md_null,sd_null= torus_dist_sample(ph_null)

    if verbose:
        print(f"  E6 final dist:     {md_e6:.4f} ± {sd_e6:.4f}")
        print(f"  Null final dist:   {md_null:.4f} ± {sd_null:.4f}")
        r_e6   = hist_e6[-1]['order_param']  if hist_e6   else 0
        r_null = hist_null[-1]['order_param']if hist_null  else 0
        print(f"  E6 order param:    {r_e6:.4f}")
        print(f"  Null order param:  {r_null:.4f}")

    # ── Phase 4: Dynamical entanglement ──────────────
    if verbose:
        print(f"\n  Phase 4: Dynamical entanglement "
              f"(t1={t_early}, t2={t_late})...")

    ph_e6_early  = snaps_e6.get(t_early,  d_e6)
    ph_e6_late   = snaps_e6.get(t_late,   ph_e6)
    ph_null_early= snaps_null.get(t_early, d_null)
    ph_null_late = snaps_null.get(t_late,  ph_null)

    pairs_e6,   drift_e6   = relative_phase_stability(
        ph_e6_early,   ph_e6_late)
    pairs_null2, drift_null = relative_phase_stability(
        ph_null_early, ph_null_late)

    G_e6,   cl_e6   = build_entanglement_graph(
        pairs_e6,   cfg.N_strings)
    G_null, cl_null = build_entanglement_graph(
        pairs_null2, cfg.N_strings)

    n_ent_e6   = len(pairs_e6)
    n_ent_null = len(pairs_null2)
    szs_e6     = [len(c) for c in cl_e6]
    szs_null   = [len(c) for c in cl_null]
    ents_e6    = entanglement_entropy(cl_e6)
    max_cl_e6  = max(szs_e6)
    frac_e6    = n_ent_e6*2/cfg.N_strings
    me_e6      = (np.mean([e for e in ents_e6 if e>0])
                  if any(e>0 for e in ents_e6) else 0.0)

    # Mean drift comparison
    mean_drift_e6   = float(np.mean(drift_e6))   \
                      if drift_e6   else 0.0
    mean_drift_null = float(np.mean(drift_null)) \
                      if drift_null else 0.0

    if verbose:
        print(f"  E6  bound pairs:   {n_ent_e6}")
        print(f"  Null bound pairs:  {n_ent_null}")
        print(f"  E6>null:           "
              f"{'✅' if n_ent_e6>n_ent_null else '❌'}")
        print(f"  Max cluster (E6):  {max_cl_e6}")
        print(f"  Frac entangled:    {frac_e6:.3f}")
        print(f"  Mean entropy:      {me_e6:.4f}")
        print(f"  Mean drift E6:     {mean_drift_e6:.4f}")
        print(f"  Mean drift null:   {mean_drift_null:.4f}")
        print(f"  E6 drifts less:    "
              f"{'✅' if mean_drift_e6 < mean_drift_null else '❌'}")

    # ── Phase 5: Field statistics ─────────────────────
    ent_nodes  = set()
    for c in cl_e6:
        if len(c) > 1:
            ent_nodes.update(c)
    free_nodes = [i for i in range(cfg.N_strings)
                  if i not in ent_nodes]

    fstats = None
    if len(free_nodes) >= 15 and len(ent_nodes) >= 15:
        fstats = field_stats(
            ph_e6_late[free_nodes],
            ph_e6_late[list(ent_nodes)])
        if verbose:
            print(f"\n  Phase 5: Field stats "
                  f"(free={len(free_nodes)}, "
                  f"ent={len(ent_nodes)})")
            if fstats.get('free'):
                print(f"  KL free:  {fstats['free']['kl']:.4f}")
            if fstats.get('ent'):
                print(f"  KL ent:   {fstats['ent']['kl']:.4f}")
            if fstats.get('ks_p') is not None:
                print(f"  KS p:     {fstats['ks_p']:.4f}")
    elif verbose:
        print(f"\n  Phase 5: Field stats skipped "
              f"(free={len(free_nodes)}, "
              f"ent={len(ent_nodes)})")

    return dict(
        seed=seed,
        t_break=t_break, n_res=n_res,
        phi_hist=phi_hist, res_hist=res_hist,
        d_e6=d_e6, d_null=d_null,
        ph_e6=ph_e6, ph_null=ph_null,
        hist_e6=hist_e6, hist_null=hist_null,
        snaps_e6=snaps_e6, snaps_null=snaps_null,
        ph_e6_early=ph_e6_early,
        ph_e6_late=ph_e6_late,
        ph_null_early=ph_null_early,
        ph_null_late=ph_null_late,
        md_e6=md_e6, sd_e6=sd_e6,
        md_null=md_null, sd_null=sd_null,
        n_ent_e6=n_ent_e6, n_ent_null=n_ent_null,
        pairs_e6=pairs_e6, pairs_null=pairs_null2,
        G_e6=G_e6, G_null=G_null,
        cl_e6=cl_e6, cl_null=cl_null,
        szs_e6=szs_e6, szs_null=szs_null,
        max_cl_e6=max_cl_e6,
        frac_e6=frac_e6, me_e6=me_e6,
        mean_drift_e6=mean_drift_e6,
        mean_drift_null=mean_drift_null,
        drift_e6=drift_e6, drift_null=drift_null,
        free_nodes=free_nodes,
        ent_nodes=ent_nodes,
        fstats=fstats,
    )


# ═══════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════

def run_mc(n_runs=8):
    print("\n" + "█"*56)
    print("█  MONOSTRING FRAGMENTATION v4 — MONTE CARLO       █")
    print("█"*56)
    print(f"  N={cfg.N_strings}  ε_bound={cfg.eps_bound}"
          f"  σ={cfg.sigma_E6}  κ_m={cfg.kappa_mono}"
          f"  κ_d={cfg.kappa_dau}  {n_runs} runs\n")

    results = []
    t0 = time.time()

    for run in range(n_runs):
        seed = cfg.seed_base + run * 7
        res  = run_one(seed=seed, verbose=(run==0))
        results.append(res)

        e_ok = ('✅' if res['n_ent_e6'] > res['n_ent_null']
                else '❌')
        d_ok = ('✅' if res['mean_drift_e6']
                        < res['mean_drift_null']
                else '❌')
        print(f"  Run {run+1}/{n_runs}: "
              f"t*={res['t_break']:4d}  "
              f"E6_ent={res['n_ent_e6']:5d}  "
              f"null_ent={res['n_ent_null']:5d}  "
              f"E6>null:{e_ok}  "
              f"drift_e6={res['mean_drift_e6']:.3f}  "
              f"drift_null={res['mean_drift_null']:.3f}  "
              f"drift↓:{d_ok}")

    elapsed = time.time() - t0

    T   = [r['t_break']        for r in results]
    NE  = [r['n_ent_e6']       for r in results]
    NN  = [r['n_ent_null']     for r in results]
    DE  = [r['mean_drift_e6']  for r in results]
    DN  = [r['mean_drift_null']for r in results]
    ME  = [r['me_e6']          for r in results]

    print(f"\n{'='*56}")
    print(f"  AGGREGATED ({n_runs} runs, {elapsed:.0f}s)")
    print(f"{'='*56}")
    print(f"  Breakup t*:      {np.mean(T):.1f} ± {np.std(T):.1f}")
    print(f"  E6 bound pairs:  {np.mean(NE):.1f} ± {np.std(NE):.1f}")
    print(f"  Null bound pairs:{np.mean(NN):.1f} ± {np.std(NN):.1f}")
    print(f"  Mean drift E6:   {np.mean(DE):.4f} ± {np.std(DE):.4f}")
    print(f"  Mean drift null: {np.mean(DN):.4f} ± {np.std(DN):.4f}")
    print(f"  Entropy:         {np.mean(ME):.4f} ± {np.std(ME):.4f}")

    ent_more  = np.mean(NE) > np.mean(NN)
    drift_less= np.mean(DE) < np.mean(DN)

    if len(NE) > 2:
        _, pv_ent   = ttest_ind(NE, NN)
        _, pv_drift = ttest_ind(DE, DN)
    else:
        pv_ent = pv_drift = 1.0

    print(f"\n  E6 > null pairs:        "
          f"{'✅' if ent_more  else '❌'}"
          f"  p={pv_ent:.4f}")
    print(f"  E6 drift < null drift:  "
          f"{'✅' if drift_less else '❌'}"
          f"  p={pv_drift:.4f}")

    return results


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(results):
    res  = results[0]
    n_mc = len(results)

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        "Monostring Fragmentation v4  —  "
        "Dynamical Entanglement Criterion\n"
        f"N={cfg.N_strings}  σ={cfg.sigma_E6}"
        f"  ε_bound={cfg.eps_bound}"
        f"  κ_mono={cfg.kappa_mono}"
        f"  κ_dau={cfg.kappa_dau}",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        4, 4, figure=fig, hspace=0.52, wspace=0.38)

    # ─── Row 0: Monostring ────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ph = res['phi_hist']
    tb = res['t_break']
    n_show = min(tb, 600)
    ax.plot(ph[max(0,tb-n_show):tb+1, 0],
            ph[max(0,tb-n_show):tb+1, 1],
            'b-', alpha=0.4, lw=0.7)
    ax.scatter(ph[tb,0], ph[tb,1],
               s=250, c='red', marker='*', zorder=5,
               label=f't*={tb}')
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('Monostring trajectory\n(dims 1–2)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    rh = res['res_hist']
    ts_r = [x[0] for x in rh]
    ir_r = [x[1] for x in rh]
    # Shade resonance visits
    ax.fill_between(ts_r, ir_r, alpha=0.3,
                    color='orange', label='in resonance')
    ax.axvline(tb, c='red', ls='--', lw=1.5,
               label=f't*={tb}')
    # Running count
    cum_res = np.cumsum(ir_r)
    ax2 = ax.twinx()
    ax2.plot(ts_r, cum_res, 'b-', lw=1.5,
             label='cumulative')
    ax2.axhline(cfg.n_res_visits, c='blue',
                ls=':', label=f'thresh={cfg.n_res_visits}')
    ax2.set_ylabel('cumulative visits', color='blue')
    ax.set_xlabel('t')
    ax.set_ylabel('in resonance (0/1)')
    ax.set_title('Resonance visits\n(breakup criterion)')
    ax.legend(fontsize=7, loc='upper left')
    ax2.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    # Phase portrait: 6D projected to φ₁ vs φ₃
    ax.scatter(ph[:tb, 0], ph[:tb, 2],
               s=1, alpha=0.3, c='steelblue')
    ax.scatter(ph[tb, 0], ph[tb, 2],
               s=200, c='red', marker='*', zorder=5)
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₃')
    ax.set_title('Phase portrait\n(dims 1 & 3)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    # E6 frequencies vs null
    ax.bar(range(6), cfg.omega_E6,
           alpha=0.7, color='steelblue', label='E6')
    ax.bar(range(6), cfg.omega_null,
           alpha=0.5, color='orange', label='null')
    ax.set_xlabel('dimension')
    ax.set_ylabel('ω')
    ax.set_title('Frequencies\nE6 (Coxeter) vs null')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ─── Row 1: Dynamics ──────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    steps_e6   = [h['step']      for h in res['hist_e6']]
    dist_e6    = [h['mean_dist'] for h in res['hist_e6']]
    steps_null = [h['step']      for h in res['hist_null']]
    dist_null  = [h['mean_dist'] for h in res['hist_null']]
    ax.plot(steps_e6,   dist_e6,   'b-', lw=2,
            label='E6 daughters')
    ax.plot(steps_null, dist_null, 'r--',lw=2,
            label='Null daughters')
    for st in cfg.snap_times:
        ax.axvline(st, c='gray', ls=':', lw=1)
    ax.axvline(cfg.snap_times[0], c='purple', ls=':',
               lw=2, label=f't_early={cfg.snap_times[0]}')
    ax.axvline(cfg.snap_times[2], c='green',  ls=':',
               lw=2, label=f't_late={cfg.snap_times[2]}')
    ax.set_xlabel('t'); ax.set_ylabel('mean pairwise dist')
    ax.set_title('Daughter evolution: E6 vs null\n'
                 '(vertical = entanglement checkpoints)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    order_e6  = [h['order_param'] for h in res['hist_e6']]
    order_null= [h['order_param'] for h in res['hist_null']]
    ax.plot(steps_e6,   order_e6,   'b-',  lw=2, label='E6')
    ax.plot(steps_null, order_null, 'r--', lw=2, label='null')
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_ylim([0,1.05])
    ax.set_title('Kuramoto order parameter\n(K=0: natural sync)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    bins_p = np.linspace(0,2*np.pi,30)
    ax.hist(res['ph_e6_early'][:,0], bins=bins_p,
            alpha=0.5, density=True, color='blue',
            label=f'E6 t={cfg.snap_times[0]}')
    ax.hist(res['ph_e6_late'][:,0],  bins=bins_p,
            alpha=0.5, density=True, color='navy',
            label=f'E6 t={cfg.snap_times[2]}')
    ax.hist(res['ph_null_late'][:,0], bins=bins_p,
            alpha=0.4, density=True, color='orange',
            label=f'null t={cfg.snap_times[2]}')
    ax.axhline(1/(2*np.pi), c='k', ls='--',
               lw=1, label='uniform')
    ax.set_xlabel('φ₁'); ax.set_ylabel('density')
    ax.set_title('Phase distribution\nat checkpoints')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # ─── Row 2: Entanglement ──────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    # Drift distribution: E6 vs null
    if res['drift_e6'] and res['drift_null']:
        dmax = min(cfg.eps_bound * 3, 2.0)
        bins_d = np.linspace(0, dmax, 40)
        de6_arr  = np.array(res['drift_e6'])
        dnull_arr= np.array(res['drift_null'])
        ax.hist(np.clip(de6_arr,  0, dmax), bins=bins_d,
                alpha=0.6, density=True,
                color='steelblue', label='E6')
        ax.hist(np.clip(dnull_arr,0, dmax), bins=bins_d,
                alpha=0.6, density=True,
                color='orange',    label='null')
        ax.axvline(cfg.eps_bound, c='red', ls='--',
                   lw=2, label=f'ε={cfg.eps_bound}')
    ax.set_xlabel('relative phase drift ||Δφ(t₂)-Δφ(t₁)||')
    ax.set_ylabel('density')
    ax.set_title('Drift distribution\n(left of ε = entangled)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    # E6 entanglement network
    G   = res['G_e6']
    pos = {i: (res['ph_e6_late'][i,0],
               res['ph_e6_late'][i,1])
           for i in range(cfg.N_strings)}
    nc  = ['lightgray'] * cfg.N_strings
    cm  = plt.cm.tab10
    for ci, cl in enumerate(res['cl_e6']):
        if len(cl) > 1:
            for nd in cl:
                nc[nd] = cm(ci % 10)
    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_size=8, node_color=nc,
                     edge_color='crimson',
                     alpha=0.7, with_labels=False,
                     width=0.5)
    ax.set_title(f'E6 entanglement graph\n'
                 f'pairs={res["n_ent_e6"]}  '
                 f'max_cl={res["max_cl_e6"]}')
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[2, 2])
    szs_e6   = res['szs_e6']
    szs_null = res['szs_null']
    bins_s   = np.arange(1, max(max(szs_e6),
                                max(szs_null))+2)-0.5
    ax.hist(szs_e6,   bins=bins_s, alpha=0.6,
            color='steelblue', label='E6',   density=True)
    ax.hist(szs_null, bins=bins_s, alpha=0.6,
            color='orange',    label='null',  density=True)
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Density')
    ax.set_title('Cluster size distribution\nE6 vs null')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    # MC: E6 vs null entanglement across runs
    NE_mc = [r['n_ent_e6']   for r in results]
    NN_mc = [r['n_ent_null'] for r in results]
    DE_mc = [r['mean_drift_e6']   for r in results]
    DN_mc = [r['mean_drift_null'] for r in results]

    x = np.arange(n_mc)
    ax.bar(x-0.2, NE_mc, 0.35, alpha=0.7,
           color='steelblue', label='E6 pairs')
    ax.bar(x+0.2, NN_mc, 0.35, alpha=0.7,
           color='orange',    label='null pairs')
    ax.set_xlabel('run')
    ax.set_ylabel('bound pairs')
    ax.set_title('E6 vs null entanglement\nacross MC runs')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # ─── Row 3: Field stats + summary ─────────────────
    if res.get('fstats') and res['fstats'].get('free'):
        ax = fig.add_subplot(gs[3, 0])
        fs = res['fstats']
        bc = (fs['free']['bins'][:-1]
              + fs['free']['bins'][1:])/2
        ax.plot(bc, fs['free']['hist'],
                'b-', lw=2, label='Free')
        if fs.get('ent'):
            bc2 = (fs['ent']['bins'][:-1]
                   + fs['ent']['bins'][1:])/2
            ax.plot(bc2, fs['ent']['hist'],
                    'r-', lw=2, label='Entangled')
        ax.axhline(1/(2*np.pi), c='k', ls='--',
                   lw=1, label='uniform')
        ax.set_xlabel('φ₁'); ax.set_ylabel('density')
        ax.set_title('Field distribution\nfree vs entangled')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[3, 1])
        ax.plot(range(1,len(fs['free']['corr'])+1),
                fs['free']['corr'],
                'b-', lw=2, label='Free')
        if fs.get('ent') and fs['ent'] is not None:
            ax.plot(range(1,len(fs['ent']['corr'])+1),
                    fs['ent']['corr'],
                    'r-', lw=2, label='Entangled')
        ax.axhline(0, c='k', lw=1)
        ax.set_xlabel('Δ'); ax.set_ylabel('C(Δ)')
        ax.set_title('Two-point correlation\nfree vs entangled')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        for col in [0,1]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5,0.5,
                    'Field stats\nnot available',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    # Drift scatter: E6 vs null across runs
    ax = fig.add_subplot(gs[3, 2])
    ax.bar(np.arange(n_mc)-0.2, DE_mc, 0.35,
           alpha=0.7, color='steelblue', label='E6 drift')
    ax.bar(np.arange(n_mc)+0.2, DN_mc, 0.35,
           alpha=0.7, color='orange',    label='null drift')
    ax.set_xlabel('run'); ax.set_ylabel('mean drift')
    ax.set_title('Relative phase drift\nE6 vs null (smaller = more bound)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Summary
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')

    ne_m = np.mean(NE_mc); ne_s = np.std(NE_mc)
    nn_m = np.mean(NN_mc)
    de_m = np.mean(DE_mc); de_s = np.std(DE_mc)
    dn_m = np.mean(DN_mc)
    me_m = np.mean([r['me_e6'] for r in results])
    t_m  = np.mean([r['t_break'] for r in results])
    t_s  = np.std( [r['t_break'] for r in results])

    ent_ok   = ne_m > nn_m
    drift_ok = de_m < dn_m
    if len(NE_mc) > 2:
        _, pv_e = ttest_ind(NE_mc, NN_mc)
        _, pv_d = ttest_ind(DE_mc, DN_mc)
    else:
        pv_e = pv_d = 1.0

    txt = (
        f"RESULTS  (n={n_mc})\n"
        f"{'─'*32}\n"
        f"Breakup t*: {t_m:.1f}±{t_s:.1f}\n"
        f"\nENTANGLEMENT (bound pairs)\n"
        f"  E6:   {ne_m:.1f} ± {ne_s:.1f}\n"
        f"  null: {nn_m:.1f}\n"
        f"  E6>null: {'✅' if ent_ok else '❌'}"
        f"  p={pv_e:.4f}\n"
        f"\nRELATIVE DRIFT\n"
        f"  E6:   {de_m:.4f} ± {de_s:.4f}\n"
        f"  null: {dn_m:.4f}\n"
        f"  E6<null: {'✅' if drift_ok else '❌'}"
        f"  p={pv_d:.4f}\n"
        f"\nENTROPY:  {me_m:.4f}\n"
        f"\nOVERALL\n"
        f"  {'✅ E6 effect detected' if (ent_ok or drift_ok) else '❌ No E6 effect'}\n"
    )
    ax.text(0.03, 0.98, txt,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e8f5e9', alpha=0.95))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation_v4.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v4.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation v4                    ║")
    print("║  Instability: resonance visits (not Lyapunov)  ║")
    print("║  Equalization: shared ω_E6 (no Kuramoto)       ║")
    print("║  Entanglement: relative phase stability         ║")
    print("║  Null: random ω (same κ, σ, N)                 ║")
    print("╚══════════════════════════════════════════════════╝")

    results = run_mc(n_runs=cfg.n_runs)
    plot_all(results)
