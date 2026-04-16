"""
Monostring Fragmentation Hypothesis — v3
=========================================
Architecture fix:
  - Monostring uses high κ (chaotic) → breaks at realistic t*
  - Daughter strings use LOW κ (quasi-periodic) + weak Kuramoto
    → equalize but remain distinguishable
  - Entanglement detected BEFORE full equalization (at t_ent < T_eq)
  - Null model: random initial phases, same daughter dynamics

Key physical distinction:
  Monostring κ = 0.55  →  chaotic, self-destructs
  Daughters   κ = 0.08  →  quasi-periodic, long-lived
  This asymmetry is the core of the fragmentation hypothesis.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp, entropy, ttest_ind
import time
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════

class Config:
    # Internal dimensions (E6 rank)
    D = 6
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12.0)

    # ── Monostring (chaotic) ──
    kappa_mono    = 0.55   # strong nonlinearity → chaos
    noise_mono    = 0.01   # small environmental noise
    # Instability criterion
    lyap_window   = 30
    lyap_thresh   = 0.05   # λ₁ > this
    chaos_thresh  = 0.8    # ∫λ₁⁺dt > this → breakup

    # ── Fragmentation ──
    N_strings   = 400
    sigma_break = 0.20     # scatter at breakup

    # ── Daughter strings (quasi-periodic) ──
    kappa_daughter = 0.08  # WEAK nonlinearity → no chaos
    noise_daughter = 0.003
    K_kuramoto     = 0.12  # subcritical coupling
    T_equalize     = 600   # equalization steps

    # ── Entanglement ──
    # Measured at t_ent = T_equalize // 3 (partial equalization)
    # At this point strings are similar but not identical
    eps_entangle   = 0.45
    eps_maintain   = 0.70
    T_dyn          = 250   # dynamics steps after equalization

    # ── Null model ──
    # Random phases with SAME daughter dynamics (κ=0.08, K=0.12)
    # Tests whether E6 origin matters

    n_runs    = 8
    seed_base = 42

cfg = Config()


# ═══════════════════════════════════════════════════════
# BLOCK 1: MONOSTRING — chaotic, self-destructs
# ═══════════════════════════════════════════════════════

def mono_step(phi, omega, kappa, rng, noise):
    """Standard map on T^6 with additive noise."""
    return (phi + omega + kappa * np.sin(phi)
            + rng.normal(0, noise, phi.shape)) % (2*np.pi)

def lyapunov_estimate(phi, omega, kappa, n=30, seed=0):
    """
    Largest Lyapunov exponent (finite-difference).
    Positive → chaos, negative → regular motion.
    """
    rng = np.random.RandomState(seed)
    d0  = 1e-8
    p   = phi.copy()
    q   = (phi + d0 / np.sqrt(len(phi))) % (2*np.pi)
    lam = 0.0
    for _ in range(n):
        p = mono_step(p, omega, kappa, rng, 0.0)
        q = mono_step(q, omega, kappa, rng, 0.0)
        diff = q - p
        # Wrap to [-π, π]
        diff -= 2*np.pi * np.round(diff / (2*np.pi))
        dist = np.linalg.norm(diff)
        if dist > 1e-15:
            lam += np.log(dist / d0)
            q = p + d0 * diff / dist
            q %= (2*np.pi)
    return lam / n

def evolve_monostring(omega, kappa, max_steps=2000, seed=0):
    """
    Evolve monostring until chaos criterion triggers breakup.

    Breakup condition (AND):
      λ₁(t) > lyap_thresh        — currently chaotic
      ∫λ₁⁺ dt > chaos_thresh     — enough chaos accumulated

    Returns:
      phi_hist  — trajectory array
      t_break   — breakup step
      lyap_hist — [(t, λ₁), ...]
      chaos_acc — final ∫λ₁⁺ dt
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    phi_hist  = [phi.copy()]
    lyap_hist = []
    chaos_acc = 0.0
    t_break   = None

    for t in range(max_steps):
        phi = mono_step(phi, omega, kappa, rng, cfg.noise_mono)
        phi_hist.append(phi.copy())

        if t % cfg.lyap_window == 0 and t >= cfg.lyap_window:
            lam = lyapunov_estimate(
                phi, omega, kappa,
                n=cfg.lyap_window, seed=t)
            lyap_hist.append((t, lam))

            if lam > 0:
                chaos_acc += lam * cfg.lyap_window

            if (lam > cfg.lyap_thresh
                    and chaos_acc > cfg.chaos_thresh):
                t_break = t
                break

    if t_break is None:
        t_break = len(phi_hist) - 1

    return np.array(phi_hist), t_break, lyap_hist, chaos_acc


# ═══════════════════════════════════════════════════════
# BLOCK 2: FRAGMENTATION
# ═══════════════════════════════════════════════════════

def fragment(phi_break, N, sigma, seed=0):
    """
    Create N daughter strings near phi_break.
    Each daughter: φᵢ = φ_break + δᵢ,  δᵢ ~ N(0, σ)
    σ encodes how 'violent' the breakup was.
    """
    rng = np.random.RandomState(seed)
    pert = rng.normal(0, sigma, (N, len(phi_break)))
    return (phi_break[np.newaxis, :] + pert) % (2*np.pi)

def torus_dist_sample(phases, n_pairs=3000, seed=0):
    """Sample mean torus distance between strings."""
    N   = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, (n_pairs, 2), replace=True)
    ok  = idx[:, 0] != idx[:, 1]
    idx = idx[ok]
    d   = np.abs(phases[idx[:,0]] - phases[idx[:,1]])
    d   = np.minimum(d, 2*np.pi - d)
    dists = np.linalg.norm(d, axis=1)
    return float(np.mean(dists)), float(np.std(dists))


# ═══════════════════════════════════════════════════════
# BLOCK 3: DAUGHTER DYNAMICS — quasi-periodic + weak sync
# ═══════════════════════════════════════════════════════

def daughter_step(phases, omega, kappa, K, rng, noise):
    """
    One step for N daughter strings.

    φᵢ → φᵢ + ω + κ·sin(φᵢ)           (individual map)
            + (K/N)·Σⱼ sin(φⱼ - φᵢ)   (Kuramoto coupling)
            + η                          (small noise)

    With κ=0.08: quasi-periodic (λ₁ < 0)
    With K=0.12 < K_c: partial sync, not collapse
    """
    N, D = phases.shape
    mf   = np.mean(np.exp(1j * phases), axis=0)  # mean field
    psi  = np.angle(mf)                           # mean phase
    r    = np.abs(mf)                             # order param

    coupling = K * np.sin(psi[np.newaxis, :] - phases)
    noise_v  = rng.normal(0, noise, phases.shape)

    new = (phases + omega[np.newaxis, :]
           + kappa * np.sin(phases)
           + coupling + noise_v) % (2*np.pi)
    return new, float(np.mean(r))

def evolve_daughters(phases0, omega, kappa, K,
                     n_steps, noise, seed=0,
                     measure_every=25,
                     snapshot_at=None):
    """
    Evolve daughter strings.
    snapshot_at: list of steps to save phase snapshot.
    Returns phases, history, snapshots dict.
    """
    rng      = np.random.RandomState(seed)
    phases   = phases0.copy()
    history  = []
    snaps    = {}
    snap_set = set(snapshot_at or [])

    for step in range(n_steps):
        phases, r = daughter_step(
            phases, omega, kappa, K, rng, noise)

        if step in snap_set:
            snaps[step] = phases.copy()

        if step % measure_every == 0:
            md, sd = torus_dist_sample(phases)
            history.append(dict(
                step=step, mean_dist=md, std_dist=sd,
                order_param=r))

    return phases, history, snaps


# ═══════════════════════════════════════════════════════
# BLOCK 4: ENTANGLEMENT
# ═══════════════════════════════════════════════════════

def detect_entanglement(phases, eps):
    """
    Find entangled pairs: ||φᵢ - φⱼ||_torus < ε, i≠j.
    Uses chunked O(N²) scan.
    Returns: pairs list, graph G, connected components.
    """
    N  = phases.shape[0]
    G  = nx.Graph()
    G.add_nodes_from(range(N))
    chunk = 80

    for i in range(0, N, chunk):
        ie   = min(i + chunk, N)
        diff = np.abs(phases[i:ie, np.newaxis, :]
                      - phases[np.newaxis, :, :])
        diff = np.minimum(diff, 2*np.pi - diff)
        dist = np.linalg.norm(diff, axis=2)
        rows, cols = np.where(
            (dist < eps) & (dist > 1e-10))
        for r, c in zip(rows, cols):
            ri = r + i
            if ri < c:
                G.add_edge(int(ri), int(c),
                           w=float(dist[r, c]))

    pairs    = list(G.edges())
    clusters = list(nx.connected_components(G))
    return pairs, G, clusters

def entanglement_stats(phases, clusters):
    """
    Compute entanglement entropy for each cluster.
    S_cluster = log(|cluster|) for equal superposition.
    """
    entropies = []
    for cl in clusters:
        n = len(cl)
        entropies.append(np.log(n) if n > 1 else 0.0)
    sizes = [len(list(c)) for c in clusters]
    return sizes, entropies

def track_entanglement(phases, omega, eps, eps_m,
                       n_steps, seed=0):
    """
    Evolve phases and track entanglement over time.
    Records: n_pairs, max_cluster, mean_entropy per step.
    """
    rng    = np.random.RandomState(seed)
    ph     = phases.copy()
    active = set()
    hist   = []

    for step in range(n_steps):
        ph, _ = daughter_step(
            ph, omega, cfg.kappa_daughter,
            cfg.K_kuramoto, rng, cfg.noise_daughter)

        _, G_new, _ = detect_entanglement(ph, eps)
        new_pairs   = set(G_new.edges())

        # Maintain bonds still within eps_m
        kept = set()
        for (a, b) in active:
            d = np.abs(ph[a] - ph[b])
            d = np.minimum(d, 2*np.pi - d)
            if np.linalg.norm(d) < eps_m:
                kept.add((a, b))
        active = new_pairs | kept

        G_act = nx.Graph()
        G_act.add_nodes_from(range(len(ph)))
        G_act.add_edges_from(active)
        comps  = list(nx.connected_components(G_act))
        szs, ents = entanglement_stats(ph, comps)
        max_cl = max(szs)
        me     = (np.mean([e for e in ents if e > 0])
                  if any(e > 0 for e in ents) else 0.0)

        hist.append(dict(
            step=step,
            n_pairs=len(active),
            max_cluster=max_cl,
            mean_entropy=me))

    return hist


# ═══════════════════════════════════════════════════════
# BLOCK 5: FIELD STATISTICS
# ═══════════════════════════════════════════════════════

def field_statistics(phases_free, phases_ent, n_bins=30):
    """
    Compare phase distributions.
    free     → should be near-uniform (ergodic)
    entangled → phase-locked, non-uniform
    Returns dict with histograms, correlations, KL, KS.
    """
    out = {}
    for label, ph in [('free', phases_free),
                       ('ent',  phases_ent)]:
        if len(ph) < 5:
            out[label] = None
            continue

        h, bins = np.histogram(
            ph[:, 0], bins=n_bins,
            range=(0, 2*np.pi), density=True)

        # Two-point correlation
        n_sep = min(40, len(ph)//2)
        corr  = []
        for sep in range(1, n_sep):
            c = np.abs(np.mean(
                np.exp(1j*(ph[:len(ph)-sep, 0]
                           - ph[sep:, 0]))))
            corr.append(c)

        uniform   = np.ones(n_bins) / n_bins
        hn        = h / (h.sum() + 1e-15) + 1e-15
        kl        = float(entropy(hn, uniform + 1e-15))

        out[label] = dict(
            hist=h, bins=bins,
            corr=np.array(corr), kl=kl)

    if (out.get('free') is not None
            and out.get('ent') is not None):
        ks, ksp = ks_2samp(
            phases_free[:, 0], phases_ent[:, 0])
        out['ks_stat'] = float(ks)
        out['ks_p']    = float(ksp)
    else:
        out['ks_stat'] = None
        out['ks_p']    = None

    return out


# ═══════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════

def run_one(seed=42, verbose=True):
    omega = cfg.omega_E6

    if verbose:
        print(f"\n{'='*56}\n  SEED {seed}\n{'='*56}")

    # ── Phase 1: Monostring chaos & breakup ───────────
    if verbose:
        print("\n  Phase 1: Monostring instability...")

    phi_hist, t_break, lyap_hist, chaos_acc = \
        evolve_monostring(omega, cfg.kappa_mono,
                          max_steps=2000, seed=seed)
    phi_break = phi_hist[t_break]

    if verbose:
        last_lam = lyap_hist[-1][1] if lyap_hist else 0.0
        print(f"  Breakup t*:    {t_break}")
        print(f"  Chaos ∫λ:      {chaos_acc:.3f}")
        print(f"  Final λ₁:      {last_lam:.4f}")

    # ── Phase 2: Fragmentation ────────────────────────
    if verbose:
        print(f"\n  Phase 2: Fragment → N={cfg.N_strings}...")

    daughters0 = fragment(phi_break, cfg.N_strings,
                          cfg.sigma_break, seed=seed+1)
    md0, sd0   = torus_dist_sample(daughters0)

    # Null: random initial phases (same N, same dynamics)
    rng_null     = np.random.RandomState(seed + 9999)
    daughters0_null = rng_null.uniform(
        0, 2*np.pi, daughters0.shape)

    if verbose:
        print(f"  Initial dist:  {md0:.4f} ± {sd0:.4f}")

    # ── Phase 3: Daughter evolution ───────────────────
    # Snapshot at t_ent = T_equalize//3 for entanglement
    t_ent = cfg.T_equalize // 3
    snaps_wanted = [t_ent, cfg.T_equalize - 1]

    if verbose:
        print(f"\n  Phase 3: Daughter evolution "
              f"(κ={cfg.kappa_daughter}, K={cfg.K_kuramoto})...")

    # E6 daughters
    phases_final, hist_eq, snaps_eq = evolve_daughters(
        daughters0, omega,
        cfg.kappa_daughter, cfg.K_kuramoto,
        cfg.T_equalize, cfg.noise_daughter,
        seed=seed+2,
        snapshot_at=snaps_wanted)

    # Null daughters (random origin)
    phases_null_final, hist_null, snaps_null = evolve_daughters(
        daughters0_null, omega,
        cfg.kappa_daughter, cfg.K_kuramoto,
        cfg.T_equalize, cfg.noise_daughter,
        seed=seed+3,
        snapshot_at=snaps_wanted)

    # Free E6 daughters (no Kuramoto, null for equalization)
    phases_free_final, hist_free, snaps_free = evolve_daughters(
        daughters0, omega,
        cfg.kappa_daughter, 0.0,
        cfg.T_equalize, cfg.noise_daughter,
        seed=seed+4,
        snapshot_at=snaps_wanted)

    md_eq,   sd_eq   = torus_dist_sample(phases_final)
    md_null, sd_null = torus_dist_sample(phases_null_final)
    md_free, sd_free = torus_dist_sample(phases_free_final)
    r_eq   = hist_eq[-1]['order_param']   if hist_eq   else 0.0
    r_null = hist_null[-1]['order_param'] if hist_null  else 0.0

    if verbose:
        print(f"  dist E6+Kura:  {md_eq:.4f} ± {sd_eq:.4f}")
        print(f"  dist null+K:   {md_null:.4f} ± {sd_null:.4f}")
        print(f"  dist E6 free:  {md_free:.4f} ± {sd_free:.4f}")
        print(f"  Order E6:      {r_eq:.4f}")
        eq_ok = md_eq < md_free * 0.90
        print(f"  Equalization:  {'✅' if eq_ok else '⚠️'}")

    # ── Phase 4: Entanglement ─────────────────────────
    if verbose:
        print(f"\n  Phase 4: Entanglement at t={t_ent}...")

    # Use partial-equalization snapshot
    ph_ent_e6   = snaps_eq.get(t_ent, phases_final)
    ph_ent_null = snaps_null.get(t_ent, phases_null_final)

    pairs_e6,   G_e6,   clusters_e6   = detect_entanglement(
        ph_ent_e6,   cfg.eps_entangle)
    pairs_null2, G_null, clusters_null = detect_entanglement(
        ph_ent_null, cfg.eps_entangle)

    n_ent_e6   = len(pairs_e6)
    n_ent_null = len(pairs_null2)
    szs_e6, ents_e6 = entanglement_stats(ph_ent_e6, clusters_e6)
    max_cl  = max(szs_e6)
    frac    = n_ent_e6 * 2 / cfg.N_strings
    me_e6   = (np.mean([e for e in ents_e6 if e > 0])
               if any(e > 0 for e in ents_e6) else 0.0)

    if verbose:
        print(f"  E6  pairs:     {n_ent_e6}")
        print(f"  Null pairs:    {n_ent_null}")
        print(f"  E6>null:       {'✅' if n_ent_e6 > n_ent_null else '❌'}")
        print(f"  Max cluster:   {max_cl}")
        print(f"  Ent fraction:  {frac:.3f}")
        print(f"  Mean entropy:  {me_e6:.4f}")

    # Entanglement dynamics
    dyn_hist = track_entanglement(
        ph_ent_e6, omega,
        cfg.eps_entangle, cfg.eps_maintain,
        cfg.T_dyn, seed=seed+5)

    # ── Phase 5: Field statistics ─────────────────────
    if verbose:
        print(f"\n  Phase 5: Field statistics...")

    ent_nodes  = set()
    for c in clusters_e6:
        if len(c) > 1:
            ent_nodes.update(c)
    free_nodes = [i for i in range(cfg.N_strings)
                  if i not in ent_nodes]

    fstats = None
    if len(free_nodes) >= 20 and len(ent_nodes) >= 20:
        fstats = field_statistics(
            ph_ent_e6[free_nodes],
            ph_ent_e6[list(ent_nodes)])
        if verbose and fstats:
            print(f"  Free / ent:    "
                  f"{len(free_nodes)} / {len(ent_nodes)}")
            kl_f = fstats['free']['kl'] if fstats['free'] else None
            kl_e = fstats['ent']['kl']  if fstats['ent']  else None
            print(f"  KL free:       "
                  f"{kl_f:.4f}" if kl_f else "  KL free: N/A")
            print(f"  KL ent:        "
                  f"{kl_e:.4f}" if kl_e else "  KL ent: N/A")
            if fstats['ks_p'] is not None:
                print(f"  KS p-value:    {fstats['ks_p']:.4f}")
    elif verbose:
        print(f"  Skipped (free={len(free_nodes)}, "
              f"ent={len(ent_nodes)})")

    return dict(
        seed=seed,
        t_break=t_break, chaos_acc=chaos_acc,
        lyap_hist=lyap_hist,
        phi_hist=phi_hist,
        daughters0=daughters0,
        ph_ent_e6=ph_ent_e6,
        ph_ent_null=ph_ent_null,
        phases_final=phases_final,
        phases_free_final=phases_free_final,
        hist_eq=hist_eq, hist_free=hist_free,
        hist_null=hist_null,
        md0=md0, sd0=sd0,
        md_eq=md_eq, sd_eq=sd_eq,
        md_free=md_free, sd_free=sd_free,
        r_eq=r_eq,
        n_ent_e6=n_ent_e6, n_ent_null=n_ent_null,
        szs_e6=szs_e6, max_cl=max_cl,
        frac=frac, me_e6=me_e6,
        G_e6=G_e6, clusters_e6=clusters_e6,
        dyn_hist=dyn_hist,
        fstats=fstats,
        free_nodes=free_nodes,
        ent_nodes=ent_nodes,
    )


# ═══════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════

def run_mc(n_runs=8):
    print("\n" + "█"*56)
    print("█  MONOSTRING FRAGMENTATION v3 — MONTE CARLO       █")
    print("█"*56)
    print(f"  N={cfg.N_strings}  κ_mono={cfg.kappa_mono}"
          f"  κ_dau={cfg.kappa_daughter}"
          f"  K={cfg.K_kuramoto}  ε={cfg.eps_entangle}"
          f"  {n_runs} runs\n")

    results = []
    t0 = time.time()

    for run in range(n_runs):
        seed = cfg.seed_base + run * 7
        res  = run_one(seed=seed, verbose=(run == 0))
        results.append(res)
        e_ok = '✅' if res['n_ent_e6'] > res['n_ent_null'] else '❌'
        q_ok = '✅' if res['md_eq'] < res['md_free'] * 0.90 else '⚠️'
        print(f"  Run {run+1}/{n_runs}:  "
              f"t*={res['t_break']:4d}  "
              f"e6_ent={res['n_ent_e6']:4d}  "
              f"null_ent={res['n_ent_null']:4d}  "
              f"E6>null:{e_ok}  "
              f"eq:{q_ok}  "
              f"dist_eq={res['md_eq']:.3f}  "
              f"dist_free={res['md_free']:.3f}")

    elapsed = time.time() - t0

    # Aggregates
    T   = [r['t_break']     for r in results]
    NE  = [r['n_ent_e6']    for r in results]
    NN  = [r['n_ent_null']  for r in results]
    ME  = [r['me_e6']       for r in results]
    DEQ = [r['md_eq']       for r in results]
    DFR = [r['md_free']     for r in results]

    print(f"\n{'='*56}")
    print(f"  AGGREGATED ({n_runs} runs, {elapsed:.0f}s)")
    print(f"{'='*56}")
    print(f"  Breakup t*:      {np.mean(T):.1f} ± {np.std(T):.1f}")
    print(f"  E6 entangled:    {np.mean(NE):.1f} ± {np.std(NE):.1f}")
    print(f"  Null entangled:  {np.mean(NN):.1f} ± {np.std(NN):.1f}")
    print(f"  Entropy:         {np.mean(ME):.4f} ± {np.std(ME):.4f}")
    print(f"  dist_eq:         {np.mean(DEQ):.4f} ± {np.std(DEQ):.4f}")
    print(f"  dist_free:       {np.mean(DFR):.4f} ± {np.std(DFR):.4f}")

    eq_works  = np.mean(DEQ) < np.mean(DFR) * 0.90
    ent_rare  = np.mean(NE)*2 / cfg.N_strings < 0.25
    ent_more  = np.mean(NE) > np.mean(NN)

    # t-test: E6 vs null entanglement
    if len(NE) > 2 and np.std(NE) > 0:
        _, pval = ttest_ind(NE, NN)
    else:
        pval = 1.0

    print(f"\n  Equalization works:     {'✅' if eq_works  else '❌'}")
    print(f"  Entanglement is rare:   {'✅' if ent_rare   else '⚠️ common'}")
    print(f"  E6 > null (count):      {'✅' if ent_more   else '❌'}")
    print(f"  E6 vs null (t-test p):  {pval:.4f} "
          f"{'✅ p<0.05' if pval < 0.05 else '❌ not sig'}")

    return results


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(results):
    res  = results[0]
    n_mc = len(results)

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        "Monostring Fragmentation v3\n"
        f"κ_mono={cfg.kappa_mono}  κ_dau={cfg.kappa_daughter}"
        f"  K={cfg.K_kuramoto}  ε={cfg.eps_entangle}"
        f"  N={cfg.N_strings}",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        4, 4, figure=fig, hspace=0.52, wspace=0.38)

    # ─── Row 0: Monostring ────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ph = res['phi_hist']
    tb = res['t_break']
    n_show = min(tb, 500)
    ax.plot(ph[max(0,tb-n_show):tb, 0],
            ph[max(0,tb-n_show):tb, 1],
            'b-', alpha=0.4, lw=0.7)
    ax.scatter(ph[tb, 0], ph[tb, 1],
               s=250, c='red', marker='*', zorder=5,
               label=f't*={tb}')
    ax.set_xlim([0, 2*np.pi]); ax.set_ylim([0, 2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('Monostring trajectory\n(dims 1–2, to breakup)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    if res['lyap_hist']:
        ts = [x[0] for x in res['lyap_hist']]
        ls = [x[1] for x in res['lyap_hist']]
        ax.plot(ts, ls, 'b-o', ms=4)
        ax.axhline(cfg.lyap_thresh, c='red', ls='--',
                   label=f'λ_thresh={cfg.lyap_thresh}')
        ax.axhline(0, c='k', lw=0.8)
        ax.axvline(tb, c='orange', ls=':', lw=1.5,
                   label=f't*={tb}')
    ax.set_xlabel('t'); ax.set_ylabel('λ₁')
    ax.set_title('Lyapunov exponent\n(chaos criterion)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if res['lyap_hist']:
        chaos_cum = np.cumsum(
            [max(0, x[1]) * cfg.lyap_window
             for x in res['lyap_hist']])
        ax.plot(ts, chaos_cum, 'g-', lw=2,
                label='∫λ₁⁺ dt')
        ax.axhline(cfg.chaos_thresh, c='red', ls='--',
                   label=f'thresh={cfg.chaos_thresh}')
        ax.axvline(tb, c='orange', ls=':', lw=1.5,
                   label=f't*={tb}')
    ax.set_xlabel('t'); ax.set_ylabel('∫λ₁⁺ dt')
    ax.set_title('Accumulated chaos\n(breakup criterion)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    dists  = []
    for sg in sigmas:
        d = fragment(ph[tb], 150, sg, seed=0)
        m, _ = torus_dist_sample(d)
        dists.append(m)
    ax.plot(sigmas, dists, 'bo-', lw=2)
    ax.axvline(cfg.sigma_break, c='red', ls='--',
               label=f'σ={cfg.sigma_break}')
    ax.set_xscale('log')
    ax.set_xlabel('σ (breakup scatter)')
    ax.set_ylabel('initial mean dist')
    ax.set_title('Fragmentation width σ\nvs initial scatter')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ─── Row 1: Equalization ──────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    steps_eq   = [h['step']      for h in res['hist_eq']]
    dist_eq    = [h['mean_dist'] for h in res['hist_eq']]
    steps_fr   = [h['step']      for h in res['hist_free']]
    dist_fr    = [h['mean_dist'] for h in res['hist_free']]
    steps_null = [h['step']      for h in res['hist_null']]
    dist_null  = [h['mean_dist'] for h in res['hist_null']]

    ax.plot(steps_eq,   dist_eq,   'b-',  lw=2,
            label=f'E6 + Kuramoto K={cfg.K_kuramoto}')
    ax.plot(steps_fr,   dist_fr,   'r--', lw=2,
            label='E6 free (K=0)')
    ax.plot(steps_null, dist_null, 'g:',  lw=2,
            label='Null + Kuramoto')
    ax.axvline(cfg.T_equalize//3, c='purple', ls=':',
               lw=1.5, label=f't_ent={cfg.T_equalize//3}')
    ax.set_xlabel('t'); ax.set_ylabel('mean pairwise dist')
    ax.set_title('Equalization: E6 vs null vs free\n'
                 '(vertical line = entanglement measurement point)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    order_eq   = [h['order_param'] for h in res['hist_eq']]
    order_fr   = [h['order_param'] for h in res['hist_free']]
    order_null = [h['order_param'] for h in res['hist_null']]
    ax.plot(steps_eq,   order_eq,   'b-',  lw=2, label='E6+K')
    ax.plot(steps_fr,   order_fr,   'r--', lw=2, label='E6 free')
    ax.plot(steps_null, order_null, 'g:',  lw=2, label='Null+K')
    ax.axhline(1.0, c='k', ls=':', lw=1)
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_ylim([0, 1.05])
    ax.set_title('Order parameter r\n(r=1: full sync)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    bins_p = np.linspace(0, 2*np.pi, 35)
    ax.hist(res['daughters0'][:, 0], bins=bins_p,
            alpha=0.45, density=True, color='red',
            label='Initial')
    ax.hist(res['ph_ent_e6'][:, 0], bins=bins_p,
            alpha=0.45, density=True, color='blue',
            label=f'E6 at t={cfg.T_equalize//3}')
    ax.hist(res['ph_ent_null'][:, 0], bins=bins_p,
            alpha=0.45, density=True, color='green',
            label='Null at same t')
    ax.axhline(1/(2*np.pi), c='k', ls='--', lw=1,
               label='uniform')
    ax.set_xlabel('φ₁'); ax.set_ylabel('density')
    ax.set_title('Phase distribution\nat entanglement checkpoint')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # ─── Row 2: Entanglement ──────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    G   = res['G_e6']
    pos = {i: (res['ph_ent_e6'][i, 0],
               res['ph_ent_e6'][i, 1])
           for i in range(cfg.N_strings)}
    nc = ['lightgray'] * cfg.N_strings
    cm = plt.cm.tab10
    for ci, cl in enumerate(res['clusters_e6']):
        if len(cl) > 1:
            for nd in cl:
                nc[nd] = cm(ci % 10)
    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_size=5, node_color=nc,
                     edge_color='crimson',
                     alpha=0.6, with_labels=False,
                     width=0.6)
    ax.set_title(f'Entanglement network (E6)\n'
                 f'pairs={res["n_ent_e6"]}  '
                 f'null={res["n_ent_null"]}')
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[2, 1])
    szs = res['szs_e6']
    ax.hist(szs, bins=np.arange(1, max(szs)+2)-0.5,
            color='steelblue', alpha=0.8, edgecolor='k')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Count')
    ax.set_title('Cluster sizes (E6)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    t_d    = [h['step']         for h in res['dyn_hist']]
    np_d   = [h['n_pairs']      for h in res['dyn_hist']]
    mc_d   = [h['max_cluster']  for h in res['dyn_hist']]
    ax.plot(t_d, np_d, 'b-', lw=2, label='Entangled pairs')
    ax.plot(t_d, mc_d, 'r--',lw=2, label='Max cluster')
    ax.set_xlabel('t'); ax.set_ylabel('count')
    ax.set_title('Entanglement dynamics\n(formation / breaking)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    me_d = [h['mean_entropy'] for h in res['dyn_hist']]
    ax.plot(t_d, me_d, 'g-', lw=2)
    ax.set_xlabel('t')
    ax.set_ylabel('S = log(cluster size)')
    ax.set_title('Entanglement entropy')
    ax.grid(True, alpha=0.3)

    # ─── Row 3: Field stats + MC ──────────────────────
    if res['fstats'] and res['fstats'].get('free'):
        ax = fig.add_subplot(gs[3, 0])
        fs = res['fstats']
        bc = (fs['free']['bins'][:-1]
              + fs['free']['bins'][1:]) / 2
        ax.plot(bc, fs['free']['hist'],
                'b-', lw=2, label='Free strings')
        if fs.get('ent'):
            bc2 = (fs['ent']['bins'][:-1]
                   + fs['ent']['bins'][1:]) / 2
            ax.plot(bc2, fs['ent']['hist'],
                    'r-', lw=2, label='Entangled')
        ax.axhline(1/(2*np.pi), c='k', ls='--', lw=1,
                   label='uniform')
        ax.set_xlabel('φ₁'); ax.set_ylabel('density')
        ax.set_title('Field distribution\nfree vs entangled')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[3, 1])
        ax.plot(range(1, len(fs['free']['corr'])+1),
                fs['free']['corr'],
                'b-', lw=2, label='Free')
        if fs.get('ent') and fs['ent'] is not None:
            ax.plot(range(1, len(fs['ent']['corr'])+1),
                    fs['ent']['corr'],
                    'r-', lw=2, label='Entangled')
        ax.axhline(0, c='k', lw=1)
        ax.set_xlabel('Δ'); ax.set_ylabel('C(Δ)')
        ax.set_title('Two-point correlation')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        for col in [0, 1]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5, 0.5,
                    'Field stats\nnot available',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    # MC scatter: t* vs entanglement
    ax = fig.add_subplot(gs[3, 2])
    T_mc  = [r['t_break']    for r in results]
    NE_mc = [r['n_ent_e6']   for r in results]
    NN_mc = [r['n_ent_null'] for r in results]
    ax.scatter(T_mc, NE_mc, s=90, c='steelblue',
               zorder=5, label='E6')
    ax.scatter(T_mc, NN_mc, s=90, c='orange',
               zorder=4, marker='^', label='null')
    ax.set_xlabel('breakup t*')
    ax.set_ylabel('entangled pairs')
    ax.set_title('MC: t* vs entanglement\nblue=E6, orange=null')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Summary text
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')

    ne_m  = np.mean(NE_mc); ne_s = np.std(NE_mc)
    nn_m  = np.mean(NN_mc)
    de_m  = np.mean([r['md_eq']   for r in results])
    df_m  = np.mean([r['md_free'] for r in results])
    me_m  = np.mean([r['me_e6']   for r in results])
    t_m   = np.mean(T_mc);  t_s  = np.std(T_mc)

    eq_ok  = de_m < df_m * 0.90
    ent_ok = ne_m > nn_m

    if len(NE_mc) > 2 and np.std(NE_mc) > 0:
        _, pv = ttest_ind(NE_mc, NN_mc)
    else:
        pv = 1.0

    txt = (
        f"RESULTS  (n={n_mc} runs)\n"
        f"{'─'*32}\n"
        f"Breakup t*:  {t_m:.1f} ± {t_s:.1f}\n"
        f"\nEQUALIZATION\n"
        f"  dist_eq:   {de_m:.4f}\n"
        f"  dist_free: {df_m:.4f}\n"
        f"  ratio:     {de_m/df_m:.3f}\n"
        f"  verdict:   {'✅ works' if eq_ok else '❌ fails'}\n"
        f"\nENTANGLEMENT\n"
        f"  E6:        {ne_m:.1f} ± {ne_s:.1f}\n"
        f"  null:      {nn_m:.1f}\n"
        f"  E6>null:   {'✅' if ent_ok else '❌'}\n"
        f"  t-test p:  {pv:.4f} "
        f"{'✅' if pv < 0.05 else '❌'}\n"
        f"  entropy:   {me_m:.4f}\n"
        f"\nOVERALL\n"
        f"  {'✅ Consistent' if eq_ok and ent_ok else '⚠️  Partial'}\n"
    )
    ax.text(0.03, 0.98, txt,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e8f5e9', alpha=0.95))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation_v3.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v3.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation Hypothesis v3         ║")
    print("║  Monostring: κ=0.55 (chaotic)                  ║")
    print("║  Daughters:  κ=0.08 (quasi-periodic)           ║")
    print("║  Entanglement measured at partial equalization  ║")
    print("╚══════════════════════════════════════════════════╝")

    results = run_mc(n_runs=cfg.n_runs)
    plot_all(results)

    print("\n" + "█"*56)
    print("█  KEY PHYSICAL CLAIMS TESTED                    █")
    print("█"*56)
    print("""
  ① Instability:   Monostring accumulates chaos → breaks
                   at finite t* (not immediately, not never)

  ② Equalization:  Daughter strings become similar via
                   weak Kuramoto coupling (K < K_c)
                   WITHOUT full collapse (individuality preserved)

  ③ Entanglement:  E6-origin daughters show MORE entanglement
                   than random-origin daughters (same dynamics)
                   This would be the first non-trivial E6 effect
                   in this simulation series.

  ④ Fields:        Free daughters → uniform phase distribution
                   Entangled clusters → structured distribution
    """)
    print("█"*56)
