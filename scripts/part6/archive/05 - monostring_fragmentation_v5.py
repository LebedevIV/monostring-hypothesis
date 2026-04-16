"""
Monostring Fragmentation v5
============================
Key insight from v4:
  E6 Coxeter frequencies are MAXIMALLY IRRATIONAL
  → relative phases drift MORE, not less
  → naive "proximity = entanglement" always fails for E6

Correct physical picture (consistent with Part V results):
  D_corr(E6) ≈ 3.02 means the E6 orbit is quasi-3D
  → 3 dimensions are "bound" (slow drift)
  → 3 dimensions are "free" (fast drift)

This is the ANISOTROPIC ENTANGLEMENT hypothesis:
  Entanglement is not uniform across dimensions.
  E6 daughters show 3D binding + 3D freedom.
  Null daughters show isotropic drift (no structure).

What we measure:
  1. Per-dimension drift: which dims are bound?
  2. Drift anisotropy: max/min ratio across dims
  3. Cluster structure: are E6 clusters more regular?
  4. Dimensional reduction: do bound dims = 3?

New null model: SHUFFLED E6 frequencies
  Same values, random assignment to dimensions
  → destroys Coxeter structure, keeps magnitude
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind, ks_2samp
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════

class Config:
    D = 6
    # E6 Coxeter frequencies (maximally irrational)
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    # Null 1: shuffled E6 (same values, random order)
    rng0 = np.random.RandomState(0)
    omega_shuffled = rng0.permutation(omega_E6.copy())

    # Null 2: uniform random in same range
    omega_random = rng0.uniform(
        omega_E6.min(), omega_E6.max(), 6)

    # ── Monostring ──
    kappa_mono   = 0.30
    delta_res    = 0.15   # resonance width
    n_res_visits = 15     # more visits → later breakup

    # ── Fragmentation ──
    N_strings  = 300
    sigma      = 0.80     # large scatter

    # ── Daughter dynamics (no coupling) ──
    kappa_dau  = 0.05
    noise_dau  = 0.006
    T_evolve   = 500

    # ── Snapshots for entanglement ──
    t_early = 100
    t_late  = 400

    # ── Anisotropic entanglement ──
    # Per-dimension drift threshold
    eps_dim    = 0.20     # bound if per-dim drift < eps_dim

    n_runs    = 8
    seed_base = 42

cfg = Config()


# ═══════════════════════════════════════════════════════
# BLOCK 1: MONOSTRING
# ═══════════════════════════════════════════════════════

def in_resonance(phi, delta):
    """At least 2 dims near low-order rational."""
    n = 0
    for q in [2, 3, 4, 5]:
        hits = np.abs(np.sin(q * phi)) < delta
        n += int(np.any(hits))
    return n >= 2

def evolve_monostring(omega, kappa, max_steps=3000,
                      seed=0):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    phi_hist  = [phi.copy()]
    res_hist  = []
    n_res     = 0
    t_break   = None

    for t in range(max_steps):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
        phi_hist.append(phi.copy())
        ir = in_resonance(phi, cfg.delta_res)
        res_hist.append(int(ir))
        if ir:
            n_res += 1
            if n_res >= cfg.n_res_visits:
                t_break = t
                break

    if t_break is None:
        t_break = len(phi_hist) - 1

    return np.array(phi_hist), t_break, res_hist, n_res


# ═══════════════════════════════════════════════════════
# BLOCK 2 & 3: FRAGMENTATION + DYNAMICS
# ═══════════════════════════════════════════════════════

def fragment(phi_break, N, sigma, seed=0):
    rng  = np.random.RandomState(seed)
    pert = rng.normal(0, sigma, (N, len(phi_break)))
    return (phi_break[np.newaxis,:] + pert) % (2*np.pi)

def evolve(phases0, omega, kappa, noise, n_steps,
           seed=0, snaps=()):
    """
    Evolve N strings with shared omega.
    Returns final phases, history, snapshot dict.
    """
    rng    = np.random.RandomState(seed)
    ph     = phases0.copy()
    snaps  = set(snaps)
    saved  = {}
    hist   = []

    for step in range(n_steps):
        ph = (ph + omega[np.newaxis,:]
               + kappa*np.sin(ph)
               + rng.normal(0, noise, ph.shape)) % (2*np.pi)

        if step in snaps:
            saved[step] = ph.copy()

        if step % 50 == 0:
            mf = np.mean(np.exp(1j*ph), axis=0)
            r  = float(np.mean(np.abs(mf)))
            # Per-dimension order parameter
            r_per_dim = np.abs(mf)
            hist.append(dict(
                step=step, order_param=r,
                r_per_dim=r_per_dim.copy()))

    return ph, hist, saved


# ═══════════════════════════════════════════════════════
# BLOCK 4: ANISOTROPIC ENTANGLEMENT
# ═══════════════════════════════════════════════════════

def per_dim_drift(ph_early, ph_late):
    """
    For each pair (i,j), compute per-dimension drift:
    drift_k(i,j) = |Δφ_k(t2) - Δφ_k(t1)|_torus

    Returns:
      drift_matrix: (N, N, D) — per-dim drift for all pairs
      (computed as upper-triangle only, chunked)

    Physical meaning:
      Small drift in dim k → dims i,j are "bound" in dim k
      E6 hypothesis: exactly 3 dims should be bound
    """
    N, D = ph_early.shape

    # Per-dimension relative phase change
    # We want: for each pair (i,j) and dim k:
    # |( φᵢₖ(t2)-φⱼₖ(t2) ) - ( φᵢₖ(t1)-φⱼₖ(t1) )|

    # Result arrays
    drift_per_dim = []   # list of (D,) arrays, one per pair
    pairs_list    = []

    chunk = 60
    for i in range(0, N, chunk):
        ie = min(i+chunk, N)

        # Relative phase at early time: (ie-i, N, D)
        dE = (ph_early[i:ie, np.newaxis, :]
              - ph_early[np.newaxis, :, :])
        dE -= 2*np.pi * np.round(dE/(2*np.pi))

        # Relative phase at late time: (ie-i, N, D)
        dL = (ph_late[i:ie, np.newaxis, :]
              - ph_late[np.newaxis, :, :])
        dL -= 2*np.pi * np.round(dL/(2*np.pi))

        # Per-dim drift: (ie-i, N, D)
        dr = np.abs(dL - dE)
        dr = np.minimum(dr, 2*np.pi - dr)

        # Only upper triangle
        for r in range(ie - i):
            ri = r + i
            for c in range(ri+1, N):
                drift_per_dim.append(dr[r, c, :])
                pairs_list.append((ri, c))

    if not drift_per_dim:
        return [], [], np.array([])

    drift_arr = np.array(drift_per_dim)  # (n_pairs, D)
    return pairs_list, drift_arr

def anisotropic_entanglement(pairs_list, drift_arr,
                              eps_dim, N):
    """
    Classify pairs by anisotropic binding:

    A pair (i,j) is 'dim-k bound' if drift_arr[:,k] < eps_dim
    Count how many dims each pair is bound in.

    Key question: does E6 show ~3 bound dims per pair
    while null shows isotropic (0, 1, 2, 3, 4, 5, 6)?

    Returns:
      n_bound_dims: array (n_pairs,) — bound dim count per pair
      bound_mask:   (n_pairs, D) boolean
      dim_binding_rate: (D,) — fraction of pairs bound per dim
    """
    if len(drift_arr) == 0:
        return np.array([]), np.array([]), np.zeros(6)

    bound_mask       = drift_arr < eps_dim  # (n_pairs, D)
    n_bound_dims     = bound_mask.sum(axis=1)  # (n_pairs,)
    dim_binding_rate = bound_mask.mean(axis=0)  # (D,)

    return n_bound_dims, bound_mask, dim_binding_rate

def sample_pairs_drift(pairs_list, drift_arr, n_sample=5000):
    """Sample a subset for statistics (speed)."""
    n = len(pairs_list)
    if n <= n_sample:
        return pairs_list, drift_arr
    idx = np.random.choice(n, n_sample, replace=False)
    return ([pairs_list[i] for i in idx],
            drift_arr[idx])

def torus_dist_mean(phases, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    N   = len(phases)
    i   = rng.randint(0, N, n)
    j   = rng.randint(0, N, n)
    ok  = i != j
    d   = np.abs(phases[i[ok]] - phases[j[ok]])
    d   = np.minimum(d, 2*np.pi - d)
    return float(np.mean(np.linalg.norm(d, axis=1)))


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def run_one(seed=42, verbose=True):
    if verbose:
        print(f"\n{'='*56}\n  SEED {seed}\n{'='*56}")

    # ── Monostring ────────────────────────────────────
    if verbose:
        print("\n  Phase 1: Monostring...")
    phi_hist, t_break, res_hist, n_res = \
        evolve_monostring(cfg.omega_E6, cfg.kappa_mono,
                          max_steps=3000, seed=seed)
    phi_break = phi_hist[t_break]
    if verbose:
        print(f"  t* = {t_break},  res_visits = {n_res}")

    # ── Fragmentation: 3 populations ─────────────────
    if verbose:
        print("\n  Phase 2: Fragmentation...")

    # Population A: E6 frequencies + shared origin
    d_e6  = fragment(phi_break, cfg.N_strings,
                     cfg.sigma, seed=seed+1)
    # Population B: shuffled E6 + shared origin
    d_shuf = fragment(phi_break, cfg.N_strings,
                      cfg.sigma, seed=seed+2)
    # Population C: E6 frequencies + RANDOM origin
    rng_c  = np.random.RandomState(seed+9000)
    phi_rand = rng_c.uniform(0, 2*np.pi, phi_break.shape)
    d_rand = fragment(phi_rand, cfg.N_strings,
                      cfg.sigma, seed=seed+3)

    if verbose:
        print(f"  E6  init dist: {torus_dist_mean(d_e6):.4f}")
        print(f"  Shuf init dist: {torus_dist_mean(d_shuf):.4f}")
        print(f"  Rand init dist: {torus_dist_mean(d_rand):.4f}")

    # ── Daughter dynamics ─────────────────────────────
    if verbose:
        print("\n  Phase 3: Daughter dynamics...")

    snaps = (cfg.t_early, cfg.t_late)

    # A: E6 omega, shared origin
    ph_e6, hist_e6, sv_e6 = evolve(
        d_e6, cfg.omega_E6, cfg.kappa_dau,
        cfg.noise_dau, cfg.T_evolve,
        seed=seed+10, snaps=snaps)

    # B: shuffled omega, shared origin
    ph_shuf, hist_shuf, sv_shuf = evolve(
        d_shuf, cfg.omega_shuffled, cfg.kappa_dau,
        cfg.noise_dau, cfg.T_evolve,
        seed=seed+11, snaps=snaps)

    # C: E6 omega, random origin
    ph_rand, hist_rand, sv_rand = evolve(
        d_rand, cfg.omega_E6, cfg.kappa_dau,
        cfg.noise_dau, cfg.T_evolve,
        seed=seed+12, snaps=snaps)

    if verbose:
        for lbl, ph in [('E6',ph_e6),
                         ('shuf',ph_shuf),
                         ('rand',ph_rand)]:
            print(f"  {lbl} final dist: "
                  f"{torus_dist_mean(ph):.4f}")

    # ── Anisotropic entanglement ──────────────────────
    if verbose:
        print("\n  Phase 4: Anisotropic entanglement...")

    results_ent = {}
    for lbl, sv in [('e6', sv_e6),
                     ('shuf', sv_shuf),
                     ('rand', sv_rand)]:
        ph_early = sv.get(cfg.t_early)
        ph_late  = sv.get(cfg.t_late)
        if ph_early is None or ph_late is None:
            if verbose:
                print(f"  {lbl}: snapshots missing!")
            results_ent[lbl] = None
            continue

        pairs, drift_arr = per_dim_drift(ph_early, ph_late)

        if len(pairs) == 0:
            results_ent[lbl] = None
            continue

        # Sample for speed
        pairs_s, drift_s = sample_pairs_drift(
            pairs, drift_arr, n_sample=8000)

        n_bd, bm, dbr = anisotropic_entanglement(
            pairs_s, drift_s, cfg.eps_dim,
            cfg.N_strings)

        # Mean drift per dimension
        mean_drift_per_dim = drift_s.mean(axis=0)
        std_drift_per_dim  = drift_s.std(axis=0)

        # Anisotropy: how unequal is binding across dims?
        anisotropy = float(dbr.max() / (dbr.min() + 1e-8))

        # Distribution of bound-dim counts
        bd_hist = np.bincount(n_bd, minlength=7)

        results_ent[lbl] = dict(
            pairs=pairs_s,
            drift_arr=drift_s,
            n_bound_dims=n_bd,
            bound_mask=bm,
            dim_binding_rate=dbr,
            mean_drift_dim=mean_drift_per_dim,
            std_drift_dim=std_drift_per_dim,
            anisotropy=anisotropy,
            bd_hist=bd_hist,
        )

        if verbose:
            print(f"\n  [{lbl.upper()}]")
            print(f"  Dim binding rate: "
                  f"{np.round(dbr, 3)}")
            print(f"  Mean drift/dim:   "
                  f"{np.round(mean_drift_per_dim, 4)}")
            print(f"  Anisotropy:       {anisotropy:.3f}")
            print(f"  Bound-dim dist:   "
                  f"{bd_hist} (0..6 bound dims)")
            # Key: how many pairs have exactly 3 bound dims?
            frac3 = bd_hist[3] / max(bd_hist.sum(), 1)
            print(f"  Frac with 3 bound dims: {frac3:.4f}")

    return dict(
        seed=seed,
        t_break=t_break, n_res=n_res,
        phi_hist=phi_hist, res_hist=res_hist,
        d_e6=d_e6, ph_e6=ph_e6,
        hist_e6=hist_e6, hist_shuf=hist_shuf,
        hist_rand=hist_rand,
        sv_e6=sv_e6, sv_shuf=sv_shuf,
        sv_rand=sv_rand,
        ent=results_ent,
    )


# ═══════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════

def run_mc(n_runs=8):
    print("\n" + "█"*56)
    print("█  MONOSTRING FRAGMENTATION v5 — MONTE CARLO       █")
    print("█"*56)
    print(f"  N={cfg.N_strings}  ε_dim={cfg.eps_dim}"
          f"  σ={cfg.sigma}  κ_d={cfg.kappa_dau}"
          f"  t_early={cfg.t_early}  t_late={cfg.t_late}\n")

    results = []
    t0 = time.time()

    for run in range(n_runs):
        seed = cfg.seed_base + run*7
        res  = run_one(seed=seed, verbose=(run==0))
        results.append(res)

        # Extract key metrics
        def get_aniso(lbl):
            e = res['ent'].get(lbl)
            return e['anisotropy'] if e else 0.0
        def get_frac3(lbl):
            e = res['ent'].get(lbl)
            if e is None: return 0.0
            h = e['bd_hist']
            return h[3]/max(h.sum(),1)
        def get_dbr(lbl):
            e = res['ent'].get(lbl)
            return e['dim_binding_rate'] if e else np.zeros(6)

        ae6  = get_aniso('e6')
        ashuf= get_aniso('shuf')
        arand= get_aniso('rand')
        f3e6 = get_frac3('e6')
        f3s  = get_frac3('shuf')
        f3r  = get_frac3('rand')

        aniso_ok = ae6 > max(ashuf, arand)
        frac3_ok = f3e6 > max(f3s, f3r)

        print(f"  Run {run+1}/{n_runs}: "
              f"t*={res['t_break']:4d}  "
              f"aniso e6={ae6:.2f} sh={ashuf:.2f} rn={arand:.2f}  "
              f"{'✅' if aniso_ok else '❌'}  "
              f"frac3: e6={f3e6:.3f} sh={f3s:.3f} rn={f3r:.3f}  "
              f"{'✅' if frac3_ok else '❌'}")

    elapsed = time.time() - t0

    # Aggregate
    def agg(lbl, key):
        vals = []
        for r in results:
            e = r['ent'].get(lbl)
            if e:
                vals.append(e[key])
        return vals

    aniso_e6   = agg('e6',   'anisotropy')
    aniso_shuf = agg('shuf', 'anisotropy')
    aniso_rand = agg('rand', 'anisotropy')

    frac3_e6   = [r['ent']['e6']['bd_hist'][3]
                  /max(r['ent']['e6']['bd_hist'].sum(),1)
                  for r in results
                  if r['ent'].get('e6')]
    frac3_shuf = [r['ent']['shuf']['bd_hist'][3]
                  /max(r['ent']['shuf']['bd_hist'].sum(),1)
                  for r in results
                  if r['ent'].get('shuf')]
    frac3_rand = [r['ent']['rand']['bd_hist'][3]
                  /max(r['ent']['rand']['bd_hist'].sum(),1)
                  for r in results
                  if r['ent'].get('rand')]

    print(f"\n{'='*56}")
    print(f"  AGGREGATED ({n_runs} runs, {elapsed:.0f}s)")
    print(f"{'='*56}")
    print(f"  Breakup t*: "
          f"{np.mean([r['t_break'] for r in results]):.1f} "
          f"± {np.std([r['t_break'] for r in results]):.1f}")
    print(f"\n  ANISOTROPY (max/min dim binding rate):")
    print(f"    E6:      "
          f"{np.mean(aniso_e6):.3f} ± {np.std(aniso_e6):.3f}")
    print(f"    Shuffled:{np.mean(aniso_shuf):.3f} "
          f"± {np.std(aniso_shuf):.3f}")
    print(f"    Random:  "
          f"{np.mean(aniso_rand):.3f} ± {np.std(aniso_rand):.3f}")

    print(f"\n  FRAC PAIRS WITH EXACTLY 3 BOUND DIMS:")
    print(f"    E6:      "
          f"{np.mean(frac3_e6):.4f} ± {np.std(frac3_e6):.4f}")
    print(f"    Shuffled:{np.mean(frac3_shuf):.4f} "
          f"± {np.std(frac3_shuf):.4f}")
    print(f"    Random:  "
          f"{np.mean(frac3_rand):.4f} ± {np.std(frac3_rand):.4f}")

    # Statistical tests
    if len(aniso_e6) > 2 and len(aniso_shuf) > 2:
        _, pv_aniso = ttest_ind(aniso_e6, aniso_shuf)
        _, pv_f3    = ttest_ind(frac3_e6, frac3_shuf)
    else:
        pv_aniso = pv_f3 = 1.0

    aniso_ok = np.mean(aniso_e6) > np.mean(aniso_shuf)
    frac3_ok = np.mean(frac3_e6) > np.mean(frac3_shuf)

    print(f"\n  E6 more anisotropic than shuffled: "
          f"{'✅' if aniso_ok else '❌'}  "
          f"p={pv_aniso:.4f}")
    print(f"  E6 more 3-dim-bound than shuffled: "
          f"{'✅' if frac3_ok else '❌'}  "
          f"p={pv_f3:.4f}")

    # Show mean dim binding rates
    print(f"\n  MEAN DIM BINDING RATES (E6):")
    dbr_all = np.array([r['ent']['e6']['dim_binding_rate']
                        for r in results
                        if r['ent'].get('e6')])
    if len(dbr_all) > 0:
        print(f"    {np.round(dbr_all.mean(axis=0), 4)}")
        print(f"    (sorted: "
              f"{np.round(np.sort(dbr_all.mean(axis=0))[::-1], 4)})")

    return results


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(results):
    res  = results[0]
    n_mc = len(results)

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        "Monostring Fragmentation v5  —  "
        "Anisotropic Entanglement\n"
        f"N={cfg.N_strings}  σ={cfg.sigma}"
        f"  ε_dim={cfg.eps_dim}"
        f"  t_early={cfg.t_early}"
        f"  t_late={cfg.t_late}",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        4, 4, figure=fig, hspace=0.52, wspace=0.40)

    # ─── Row 0: Monostring ────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ph = res['phi_hist']
    tb = res['t_break']
    ax.plot(ph[:tb+1, 0], ph[:tb+1, 1],
            'b-', alpha=0.5, lw=0.8)
    ax.scatter(ph[tb,0], ph[tb,1],
               s=250, c='red', marker='*', zorder=5,
               label=f't*={tb}')
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('Monostring (dims 1–2)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    rh = res['res_hist']
    ax.fill_between(range(len(rh)), rh,
                    alpha=0.4, color='orange',
                    label='resonance')
    cum = np.cumsum(rh)
    ax2 = ax.twinx()
    ax2.plot(cum, 'b-', lw=1.5)
    ax2.axhline(cfg.n_res_visits, c='blue', ls=':',
                label=f'thresh={cfg.n_res_visits}')
    ax2.set_ylabel('cumulative', color='blue')
    ax.axvline(tb, c='red', ls='--', lw=1.5,
               label=f't*={tb}')
    ax.set_xlabel('t')
    ax.set_ylabel('in resonance')
    ax.set_title('Resonance visits')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(6), cfg.omega_E6,
           alpha=0.7, color='steelblue', label='E6')
    ax.bar(range(6), cfg.omega_shuffled,
           alpha=0.5, color='orange', label='shuffled')
    ax.bar(range(6), cfg.omega_random,
           alpha=0.4, color='green', label='random')
    ax.set_xlabel('dim k'); ax.set_ylabel('ω_k')
    ax.set_title('Frequencies: E6 / shuffled / random')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[0, 3])
    # Order param over time for all 3 populations
    for lbl, hist, c, ls in [
            ('E6',   res['hist_e6'],   'blue',   '-'),
            ('shuf', res['hist_shuf'], 'orange', '--'),
            ('rand', res['hist_rand'], 'green',  ':')]:
        if hist:
            steps = [h['step'] for h in hist]
            rvals = [h['order_param'] for h in hist]
            ax.plot(steps, rvals, c=c, ls=ls,
                    lw=2, label=lbl)
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_ylim([0, 1.05])
    ax.set_title('Order param r(t)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ─── Row 1: Per-dim binding rates ─────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    dims = np.arange(6)
    width = 0.25

    for idx, (lbl, c) in enumerate([
            ('e6',   'steelblue'),
            ('shuf', 'orange'),
            ('rand', 'green')]):
        e = res['ent'].get(lbl)
        if e is not None:
            dbr = e['dim_binding_rate']
            ax.bar(dims + (idx-1)*width, dbr, width,
                   alpha=0.8, color=c, label=lbl,
                   edgecolor='k', linewidth=0.5)

    ax.axhline(1/6, c='red', ls='--', lw=1,
               label='uniform (1/6)')
    ax.set_xlabel('Dimension k')
    ax.set_ylabel('Binding rate (frac pairs bound)')
    ax.set_title('Per-dimension binding rate\n'
                 'E6 hypothesis: 3 dims high, 3 dims low')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(dims)

    ax = fig.add_subplot(gs[1, 2])
    # Mean drift per dim
    for lbl, c in [('e6','steelblue'),
                    ('shuf','orange'),
                    ('rand','green')]:
        e = res['ent'].get(lbl)
        if e is not None:
            md = e['mean_drift_dim']
            ax.plot(dims, md, 'o-', color=c,
                    lw=2, ms=6, label=lbl)
    ax.axhline(cfg.eps_dim, c='red', ls='--',
               lw=1.5, label=f'ε={cfg.eps_dim}')
    ax.set_xlabel('dim k')
    ax.set_ylabel('mean drift')
    ax.set_title('Mean per-dim drift\n'
                 '(below ε = bound)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)

    ax = fig.add_subplot(gs[1, 3])
    # Distribution of bound-dim counts (0..6)
    bins_bd = np.arange(8) - 0.5
    for lbl, c in [('e6','steelblue'),
                    ('shuf','orange'),
                    ('rand','green')]:
        e = res['ent'].get(lbl)
        if e is not None:
            bh = e['bd_hist']
            bh_norm = bh / bh.sum()
            ax.plot(range(7), bh_norm, 'o-',
                    color=c, lw=2, ms=5, label=lbl)
    ax.axvline(3, c='red', ls='--', lw=1.5,
               label='3 (hypothesis)')
    ax.set_xlabel('n bound dims per pair')
    ax.set_ylabel('fraction of pairs')
    ax.set_title('Bound-dim count distribution\n'
                 '(E6 hypothesis: peak at 3)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ─── Row 2: MC aggregates ─────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    # Anisotropy across runs
    for idx, (lbl, c) in enumerate([
            ('e6','steelblue'),
            ('shuf','orange'),
            ('rand','green')]):
        vals = [r['ent'][lbl]['anisotropy']
                for r in results
                if r['ent'].get(lbl)]
        x = np.arange(len(vals)) + idx*0.25
        ax.bar(x, vals, 0.22, alpha=0.8, color=c,
               label=lbl)
    ax.set_xlabel('run'); ax.set_ylabel('anisotropy')
    ax.set_title('Anisotropy (max/min binding rate)\nper run')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[2, 1])
    # Frac-3 across runs
    for idx, (lbl, c) in enumerate([
            ('e6','steelblue'),
            ('shuf','orange'),
            ('rand','green')]):
        vals = [r['ent'][lbl]['bd_hist'][3]
                /max(r['ent'][lbl]['bd_hist'].sum(),1)
                for r in results
                if r['ent'].get(lbl)]
        x = np.arange(len(vals)) + idx*0.25
        ax.bar(x, vals, 0.22, alpha=0.8, color=c,
               label=lbl)
    ax.set_xlabel('run')
    ax.set_ylabel('frac pairs with 3 bound dims')
    ax.set_title('3-dim binding fraction\n(E6 hypothesis)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[2, 2])
    # Mean per-dim binding across MC
    dbr_e6   = np.array([r['ent']['e6']['dim_binding_rate']
                          for r in results
                          if r['ent'].get('e6')])
    dbr_shuf = np.array([r['ent']['shuf']['dim_binding_rate']
                          for r in results
                          if r['ent'].get('shuf')])
    if len(dbr_e6) > 0:
        mu_e6 = dbr_e6.mean(axis=0)
        se_e6 = dbr_e6.std(axis=0)/np.sqrt(len(dbr_e6))
        ax.errorbar(dims, mu_e6, yerr=2*se_e6,
                    fmt='o-', color='steelblue',
                    lw=2, ms=6, capsize=4, label='E6')
    if len(dbr_shuf) > 0:
        mu_sh = dbr_shuf.mean(axis=0)
        se_sh = dbr_shuf.std(axis=0)/np.sqrt(len(dbr_shuf))
        ax.errorbar(dims+0.1, mu_sh, yerr=2*se_sh,
                    fmt='s--', color='orange',
                    lw=2, ms=6, capsize=4, label='shuffled')
    ax.axhline(cfg.eps_dim, c='red', ls=':', lw=1)
    ax.set_xlabel('dim k'); ax.set_ylabel('binding rate')
    ax.set_title('Per-dim binding (MC mean ± 2SE)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)

    ax = fig.add_subplot(gs[2, 3])
    # Drift distribution: E6 vs shuffled
    e_e6   = res['ent'].get('e6')
    e_shuf = res['ent'].get('shuf')
    if e_e6 and e_shuf:
        # Total drift (norm over dims)
        d_e6_tot   = np.linalg.norm(
            e_e6['drift_arr'],   axis=1)
        d_shuf_tot = np.linalg.norm(
            e_shuf['drift_arr'], axis=1)
        bins_dr = np.linspace(0, min(
            d_e6_tot.max(), d_shuf_tot.max(), 3.0), 40)
        ax.hist(d_e6_tot,   bins=bins_dr, alpha=0.6,
                density=True, color='steelblue',
                label='E6')
        ax.hist(d_shuf_tot, bins=bins_dr, alpha=0.6,
                density=True, color='orange',
                label='shuffled')
        ax.set_xlabel('total drift ||Δφ||')
        ax.set_ylabel('density')
        ax.set_title('Total drift distribution\nE6 vs shuffled')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ─── Row 3: Summary ───────────────────────────────
    ax = fig.add_subplot(gs[3, 0:2])
    # Sorted binding rates: E6 vs shuffled (MC mean)
    if len(dbr_e6) > 0:
        sorted_e6   = np.sort(dbr_e6.mean(axis=0))[::-1]
        sorted_shuf = np.sort(dbr_shuf.mean(axis=0))[::-1] \
                      if len(dbr_shuf) > 0 \
                      else np.zeros(6)
        x = np.arange(6)
        ax.plot(x, sorted_e6,   'o-', color='steelblue',
                lw=2.5, ms=8, label='E6 (sorted)')
        ax.plot(x, sorted_shuf, 's--', color='orange',
                lw=2.5, ms=8, label='shuffled (sorted)')
        ax.axhline(cfg.eps_dim, c='red', ls='--',
                   lw=1.5, label=f'ε={cfg.eps_dim}')
        ax.axvline(2.5, c='purple', ls=':',
                   lw=2, label='3|3 split (hypothesis)')
    ax.set_xlabel('dimension rank (sorted)')
    ax.set_ylabel('binding rate')
    ax.set_title('KEY RESULT: Sorted binding rates\n'
                 'E6 hypothesis: gap at rank 3 (3 bound, 3 free)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    # Bound-dim count distributions: MC mean
    bd_e6_mc = np.array([
        r['ent']['e6']['bd_hist']
        /max(r['ent']['e6']['bd_hist'].sum(),1)
        for r in results if r['ent'].get('e6')])
    bd_shuf_mc = np.array([
        r['ent']['shuf']['bd_hist']
        /max(r['ent']['shuf']['bd_hist'].sum(),1)
        for r in results if r['ent'].get('shuf')])
    bd_rand_mc = np.array([
        r['ent']['rand']['bd_hist']
        /max(r['ent']['rand']['bd_hist'].sum(),1)
        for r in results if r['ent'].get('rand')])

    x = np.arange(7)
    for arr, c, lbl in [(bd_e6_mc,   'steelblue', 'E6'),
                         (bd_shuf_mc, 'orange',    'shuffled'),
                         (bd_rand_mc, 'green',     'random')]:
        if len(arr) > 0:
            mu = arr.mean(axis=0)
            se = arr.std(axis=0)/np.sqrt(len(arr))
            ax.errorbar(x, mu, yerr=2*se,
                        fmt='o-', color=c, lw=2,
                        ms=5, capsize=3, label=lbl)
    ax.axvline(3, c='red', ls='--', lw=1.5)
    ax.set_xlabel('bound dims per pair')
    ax.set_ylabel('fraction')
    ax.set_title('Bound-dim distribution (MC)\n3D hypothesis')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')

    # Summary stats
    aniso_e6_m   = np.mean([r['ent']['e6']['anisotropy']
                             for r in results
                             if r['ent'].get('e6')])
    aniso_shuf_m = np.mean([r['ent']['shuf']['anisotropy']
                             for r in results
                             if r['ent'].get('shuf')])
    frac3_e6_m   = np.mean([
        r['ent']['e6']['bd_hist'][3]
        /max(r['ent']['e6']['bd_hist'].sum(),1)
        for r in results if r['ent'].get('e6')])
    frac3_shuf_m = np.mean([
        r['ent']['shuf']['bd_hist'][3]
        /max(r['ent']['shuf']['bd_hist'].sum(),1)
        for r in results if r['ent'].get('shuf')])

    aniso_ok2 = aniso_e6_m > aniso_shuf_m
    frac3_ok2 = frac3_e6_m > frac3_shuf_m

    all_aniso = ([r['ent']['e6']['anisotropy']
                  for r in results if r['ent'].get('e6')],
                 [r['ent']['shuf']['anisotropy']
                  for r in results if r['ent'].get('shuf')])
    all_f3    = ([r['ent']['e6']['bd_hist'][3]
                  /max(r['ent']['e6']['bd_hist'].sum(),1)
                  for r in results if r['ent'].get('e6')],
                 [r['ent']['shuf']['bd_hist'][3]
                  /max(r['ent']['shuf']['bd_hist'].sum(),1)
                  for r in results if r['ent'].get('shuf')])

    pv_a = ttest_ind(*all_aniso)[1] if all(
        len(x)>2 for x in all_aniso) else 1.0
    pv_f = ttest_ind(*all_f3)[1] if all(
        len(x)>2 for x in all_f3) else 1.0

    t_m = np.mean([r['t_break'] for r in results])
    t_s = np.std( [r['t_break'] for r in results])

    txt = (
        f"RESULTS  (n={n_mc})\n"
        f"{'─'*32}\n"
        f"Breakup t*: {t_m:.1f} ± {t_s:.1f}\n"
        f"\nANISOTROPY\n"
        f"  E6:      {aniso_e6_m:.3f}\n"
        f"  shuffled:{aniso_shuf_m:.3f}\n"
        f"  E6>shuf: {'✅' if aniso_ok2 else '❌'}"
        f"  p={pv_a:.4f}\n"
        f"\n3-DIM BINDING FRAC\n"
        f"  E6:      {frac3_e6_m:.4f}\n"
        f"  shuffled:{frac3_shuf_m:.4f}\n"
        f"  E6>shuf: {'✅' if frac3_ok2 else '❌'}"
        f"  p={pv_f:.4f}\n"
        f"\nOVERALL\n"
        f"  {'✅ Anisotropic 3D binding detected' if (aniso_ok2 and frac3_ok2) else '⚠️ Partial or no E6 effect'}\n"
        f"\nCONSISTENCY with Part V:\n"
        f"  D_corr(E6)≈3 ↔ 3 bound dims\n"
        f"  {'✅ Consistent' if frac3_ok2 else '❌ Inconsistent'}\n"
    )
    ax.text(0.03, 0.98, txt,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e8f5e9',
                      alpha=0.95))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation_v5.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v5.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation v5                    ║")
    print("║  Core claim: E6 daughters show ANISOTROPIC     ║")
    print("║  binding — 3 dims bound, 3 dims free           ║")
    print("║  Consistent with D_corr(E6) ≈ 3.02 (Part V)   ║")
    print("╚══════════════════════════════════════════════════╝")

    results = run_mc(n_runs=cfg.n_runs)
    plot_all(results)

    print("\n" + "█"*56)
    print("█  PHYSICAL INTERPRETATION                       █")
    print("█"*56)
    print("""
  The E6 Coxeter frequencies ω = 2sin(πm/12)
  are maximally irrational — they DON'T synchronize.

  But their specific pattern creates ANISOTROPIC binding:
  3 internal dimensions drift slowly relative to each other
  3 internal dimensions drift fast (free)

  This is the same effect as D_corr(E6) ≈ 3.02:
  the orbit occupies a quasi-3D submanifold of T^6.

  Daughter strings with E6 heritage therefore show:
  - 3 'bound' relative dimensions (proto-space)
  - 3 'free' relative dimensions (internal dof)

  If confirmed: this is the first NEW prediction
  of the fragmentation hypothesis, going BEYOND
  what was shown in Parts I–V.
    """)
    print("█"*56)
