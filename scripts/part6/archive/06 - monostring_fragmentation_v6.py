"""
Monostring Fragmentation v6
============================
Correct metric: proximity to monostring orbit

Key insight from v5:
  Relative phase Δφ_ij = φ_i - φ_j cancels ω exactly.
  → E6 vs shuffled gives identical relative drift.
  → Per-dim analysis shows no E6 signature.

Correct approach:
  The monostring traces a quasi-3D orbit on T^6.
  D_corr(E6) ≈ 3.02 (Part V) means the orbit
  is confined to a 3D submanifold of T^6.

  Daughter strings (with same ω_E6) should ALSO
  stay near this quasi-3D submanifold.

  Null (shuffled ω) traces a DIFFERENT orbit —
  different submanifold, different dimension.

Measurements:
  1. Correlation dimension of daughter cloud
     Does D_corr(daughters_E6) ≈ 3?
     Does D_corr(daughters_shuf) ≠ 3?

  2. Distance to reference orbit
     Generate reference E6 orbit (long run).
     Measure: how close are daughters to this orbit?
     E6 daughters should stay closer than shuffled.

  3. Dimensional reduction via PCA
     Project daughter cloud onto principal axes.
     E6: 3 large eigenvalues, 3 small (quasi-3D)?
     Shuffled: more uniform spectrum?

  4. Breakup: resonance accumulation (fixed from v4)
     Keep n_res_visits=15 but add diagnostic plots.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind
from scipy.spatial.distance import cdist
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════

class Config:
    D = 6
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    # Null 1: shuffled E6 (same magnitudes, no Coxeter)
    rng0 = np.random.RandomState(1)
    omega_shuf = rng0.permutation(omega_E6.copy())

    # Null 2: uniform random same range
    omega_rand = rng0.uniform(
        omega_E6.min(), omega_E6.max(), 6)

    # ── Monostring ──
    kappa_mono   = 0.30
    delta_res    = 0.15
    n_res_visits = 15

    # ── Reference orbit ──
    # Long run of monostring to define the "E6 orbit"
    T_ref        = 5000   # length of reference orbit

    # ── Fragmentation ──
    N_strings    = 200
    sigma        = 0.60   # scatter

    # ── Daughter dynamics ──
    kappa_dau    = 0.05
    noise_dau    = 0.006
    T_evolve     = 600
    snap_times   = [50, 150, 300, 500]

    # ── Correlation dimension ──
    # Estimated from daughter cloud at T_evolve
    n_dcorr_pairs = 3000  # pairs to sample
    r_lo          = 0.05  # distance range for power law
    r_hi          = 0.80

    # ── Distance to orbit ──
    n_ref_sample  = 500   # reference orbit points to use
    n_dau_sample  = 100   # daughter strings to sample

    n_runs    = 8
    seed_base = 42

cfg = Config()


# ═══════════════════════════════════════════════════════
# BLOCK 0: REFERENCE ORBIT
# ═══════════════════════════════════════════════════════

def generate_reference_orbit(omega, kappa, T, seed=0):
    """
    Generate the monostring's natural orbit on T^6.
    This is the attractor that daughters should follow
    if they share the same frequencies.

    Returns: (T, D) array of orbit points
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    orbit = []

    # Warm up (discard transient)
    for _ in range(200):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)

    for _ in range(T):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
        orbit.append(phi.copy())

    return np.array(orbit)

def torus_dist_to_orbit(phases, orbit_pts, n_orb=500):
    """
    For each daughter string, find its minimum torus
    distance to the reference orbit.

    Uses n_orb randomly sampled orbit points.
    Returns: (N_daughters,) array of min distances.
    """
    idx  = np.random.choice(len(orbit_pts), n_orb,
                             replace=False)
    orb  = orbit_pts[idx]  # (n_orb, D)

    min_dists = []
    for ph in phases:
        d = np.abs(ph[np.newaxis,:] - orb)
        d = np.minimum(d, 2*np.pi - d)
        dist = np.linalg.norm(d, axis=1)  # (n_orb,)
        min_dists.append(float(dist.min()))
    return np.array(min_dists)


# ═══════════════════════════════════════════════════════
# BLOCK 1: MONOSTRING INSTABILITY
# ═══════════════════════════════════════════════════════

def in_resonance(phi, delta):
    n = sum(int(np.any(np.abs(np.sin(q*phi)) < delta))
            for q in [2,3,4,5])
    return n >= 2

def evolve_monostring(omega, kappa, max_steps=3000,
                      seed=0):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    phi_hist = [phi.copy()]
    res_hist = []
    n_res = 0
    t_break = None

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
# BLOCK 2: FRAGMENTATION
# ═══════════════════════════════════════════════════════

def fragment(phi_break, N, sigma, seed=0):
    rng  = np.random.RandomState(seed)
    pert = rng.normal(0, sigma, (N, len(phi_break)))
    return (phi_break[np.newaxis,:] + pert) % (2*np.pi)


# ═══════════════════════════════════════════════════════
# BLOCK 3: DAUGHTER DYNAMICS
# ═══════════════════════════════════════════════════════

def evolve_daughters(phases0, omega, kappa, noise,
                     n_steps, seed=0, snaps=()):
    rng   = np.random.RandomState(seed)
    ph    = phases0.copy()
    saved = {}
    hist  = []

    for step in range(n_steps):
        ph = (ph + omega[np.newaxis,:]
               + kappa*np.sin(ph)
               + rng.normal(0, noise, ph.shape)) % (2*np.pi)

        if step in snaps:
            saved[step] = ph.copy()

        if step % 50 == 0:
            mf = np.mean(np.exp(1j*ph), axis=0)
            r_per_dim = np.abs(mf)
            hist.append(dict(
                step=step,
                order_param=float(np.mean(r_per_dim)),
                r_per_dim=r_per_dim.copy()))

    return ph, hist, saved


# ═══════════════════════════════════════════════════════
# BLOCK 4: CORRELATION DIMENSION
# ═══════════════════════════════════════════════════════

def correlation_dimension(phases, n_pairs=3000,
                           r_lo=0.05, r_hi=0.80,
                           n_bins=15, seed=0):
    """
    Estimate correlation dimension D_corr of the
    daughter string cloud.

    C(r) = (2/N²) Σᵢ<ⱼ θ(r - ||φᵢ-φⱼ||)
    D_corr = d log C(r) / d log r

    Uses torus metric.
    Returns: D_corr estimate, (log_r, log_C) arrays.
    """
    rng = np.random.RandomState(seed)
    N   = len(phases)
    n   = min(n_pairs, N*(N-1)//2)

    # Sample pairs
    i = rng.randint(0, N, n)
    j = rng.randint(0, N, n)
    ok = i != j
    i, j = i[ok], j[ok]

    d = np.abs(phases[i] - phases[j])
    d = np.minimum(d, 2*np.pi - d)
    dists = np.linalg.norm(d, axis=1)

    # Correlation integral
    r_vals = np.logspace(np.log10(r_lo),
                          np.log10(r_hi), n_bins)
    C_vals = np.array([float(np.mean(dists < r))
                       for r in r_vals])

    # Remove zeros
    ok = C_vals > 0
    if ok.sum() < 3:
        return float('nan'), r_vals, C_vals

    log_r = np.log10(r_vals[ok])
    log_C = np.log10(C_vals[ok])

    # Linear fit in log-log space
    coeffs = np.polyfit(log_r, log_C, 1)
    D_corr = float(coeffs[0])

    return D_corr, r_vals, C_vals


# ═══════════════════════════════════════════════════════
# BLOCK 5: PCA ANALYSIS
# ═══════════════════════════════════════════════════════

def pca_analysis(phases):
    """
    Principal component analysis of the daughter cloud.

    Torus: project via (cos φ, sin φ) per dimension,
    giving 2D embedding per dim → 12D total.
    Then PCA on the 12D representation.

    Alternatively: linear PCA on phases directly
    (valid if spread < π in each dim).

    Returns: eigenvalues (12,), explained variance ratio.
    """
    N, D = phases.shape

    # 2D embedding per dimension
    X = np.zeros((N, 2*D))
    for k in range(D):
        X[:, 2*k]   = np.cos(phases[:, k])
        X[:, 2*k+1] = np.sin(phases[:, k])

    # Center
    X -= X.mean(axis=0)

    # SVD
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    eigvals = s**2 / (N-1)
    total   = eigvals.sum()
    ratio   = eigvals / total if total > 0 else eigvals

    return eigvals, ratio


# ═══════════════════════════════════════════════════════
# BLOCK 6: DISTANCE TO REFERENCE ORBIT
# ═══════════════════════════════════════════════════════

# Already defined above as torus_dist_to_orbit


# ═══════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════

def run_one(seed=42, verbose=True):
    if verbose:
        print(f"\n{'='*56}\n  SEED {seed}\n{'='*56}")

    # ── Reference orbits ──────────────────────────────
    if verbose:
        print("\n  Generating reference orbits...")

    orbit_e6   = generate_reference_orbit(
        cfg.omega_E6,   cfg.kappa_mono,
        cfg.T_ref, seed=seed+5000)
    orbit_shuf = generate_reference_orbit(
        cfg.omega_shuf, cfg.kappa_mono,
        cfg.T_ref, seed=seed+5001)
    orbit_rand = generate_reference_orbit(
        cfg.omega_rand, cfg.kappa_mono,
        cfg.T_ref, seed=seed+5002)

    # D_corr of reference orbits (ground truth)
    dc_orb_e6,   _, _ = correlation_dimension(
        orbit_e6[::5],   seed=seed)
    dc_orb_shuf, _, _ = correlation_dimension(
        orbit_shuf[::5], seed=seed)
    dc_orb_rand, _, _ = correlation_dimension(
        orbit_rand[::5], seed=seed)

    if verbose:
        print(f"  D_corr(orbit E6):   {dc_orb_e6:.3f}")
        print(f"  D_corr(orbit shuf): {dc_orb_shuf:.3f}")
        print(f"  D_corr(orbit rand): {dc_orb_rand:.3f}")

    # ── Monostring ────────────────────────────────────
    if verbose:
        print("\n  Phase 1: Monostring...")
    phi_hist, t_break, res_hist, n_res = \
        evolve_monostring(cfg.omega_E6, cfg.kappa_mono,
                          max_steps=3000, seed=seed)
    phi_break = phi_hist[t_break]
    if verbose:
        print(f"  t* = {t_break},  res_visits = {n_res}")

    # ── Fragmentation ─────────────────────────────────
    if verbose:
        print("\n  Phase 2: Fragmentation...")

    d_e6  = fragment(phi_break, cfg.N_strings,
                     cfg.sigma, seed=seed+1)

    # Null A: shuffled omega, SAME origin
    d_shuf = fragment(phi_break, cfg.N_strings,
                      cfg.sigma, seed=seed+2)

    # Null B: E6 omega, RANDOM origin
    rng_b = np.random.RandomState(seed+8888)
    phi_rnd = rng_b.uniform(0, 2*np.pi, phi_break.shape)
    d_rand = fragment(phi_rnd, cfg.N_strings,
                      cfg.sigma, seed=seed+3)

    # ── Daughter dynamics ─────────────────────────────
    if verbose:
        print("\n  Phase 3: Daughter dynamics...")

    snaps = set(cfg.snap_times)

    ph_e6, hist_e6, sv_e6 = evolve_daughters(
        d_e6,  cfg.omega_E6,   cfg.kappa_dau,
        cfg.noise_dau, cfg.T_evolve,
        seed=seed+10, snaps=snaps)

    ph_shuf, hist_shuf, sv_shuf = evolve_daughters(
        d_shuf, cfg.omega_shuf, cfg.kappa_dau,
        cfg.noise_dau, cfg.T_evolve,
        seed=seed+11, snaps=snaps)

    ph_rand, hist_rand, sv_rand = evolve_daughters(
        d_rand, cfg.omega_E6,   cfg.kappa_dau,
        cfg.noise_dau, cfg.T_evolve,
        seed=seed+12, snaps=snaps)

    # ── Correlation dimension of daughter clouds ──────
    if verbose:
        print("\n  Phase 4a: Correlation dimensions...")

    dc_e6,   rv_e6,   cv_e6   = correlation_dimension(
        ph_e6,   seed=seed+20)
    dc_shuf, rv_shuf, cv_shuf = correlation_dimension(
        ph_shuf, seed=seed+21)
    dc_rand, rv_rand, cv_rand = correlation_dimension(
        ph_rand, seed=seed+22)

    if verbose:
        print(f"  D_corr(daughters E6):   {dc_e6:.3f}")
        print(f"  D_corr(daughters shuf): {dc_shuf:.3f}")
        print(f"  D_corr(daughters rand): {dc_rand:.3f}")
        print(f"  Reference orbit E6:     {dc_orb_e6:.3f}")
        print(f"  Hypothesis: |dc_e6 - dc_orb_e6| < others")
        dist_e6_to_orb  = abs(dc_e6  - dc_orb_e6)
        dist_shuf_to_orb= abs(dc_shuf - dc_orb_e6)
        print(f"  |dc_e6  - dc_orb|: {dist_e6_to_orb:.3f}")
        print(f"  |dc_shuf - dc_orb|:{dist_shuf_to_orb:.3f}")
        if dist_e6_to_orb < dist_shuf_to_orb:
            print("  E6 daughters closer to orbit dim: ✅")
        else:
            print("  E6 daughters closer to orbit dim: ❌")

    # ── PCA analysis ──────────────────────────────────
    if verbose:
        print("\n  Phase 4b: PCA analysis...")

    eig_e6,   rat_e6   = pca_analysis(ph_e6)
    eig_shuf, rat_shuf = pca_analysis(ph_shuf)
    eig_rand, rat_rand = pca_analysis(ph_rand)

    # How many PCs needed for 80% variance?
    def pcs_for_var(rat, threshold=0.80):
        cumvar = np.cumsum(rat)
        idx    = np.searchsorted(cumvar, threshold)
        return int(idx) + 1

    n80_e6   = pcs_for_var(rat_e6)
    n80_shuf = pcs_for_var(rat_shuf)
    n80_rand = pcs_for_var(rat_rand)

    # Effective dimension via participation ratio
    def eff_dim(rat):
        return float(1.0 / np.sum(rat**2))

    eff_e6   = eff_dim(rat_e6)
    eff_shuf = eff_dim(rat_shuf)
    eff_rand = eff_dim(rat_rand)

    if verbose:
        print(f"  PCs for 80% var: "
              f"E6={n80_e6}, shuf={n80_shuf}, rand={n80_rand}")
        print(f"  Eff dim (PR):    "
              f"E6={eff_e6:.2f}, shuf={eff_shuf:.2f}, "
              f"rand={eff_rand:.2f}")
        print(f"  Top-6 eigenval E6:   "
              f"{np.round(rat_e6[:6]*100,1)}%")
        print(f"  Top-6 eigenval shuf: "
              f"{np.round(rat_shuf[:6]*100,1)}%")

    # ── Distance to reference orbit ───────────────────
    if verbose:
        print("\n  Phase 4c: Distance to E6 reference orbit...")

    # Sample daughters for speed
    idx_s = np.random.choice(
        cfg.N_strings, cfg.n_dau_sample, replace=False)

    dist_e6_orb   = torus_dist_to_orbit(
        ph_e6[idx_s],   orbit_e6,
        n_orb=cfg.n_ref_sample)
    dist_shuf_orb = torus_dist_to_orbit(
        ph_shuf[idx_s], orbit_e6,
        n_orb=cfg.n_ref_sample)
    dist_rand_orb = torus_dist_to_orbit(
        ph_rand[idx_s], orbit_e6,
        n_orb=cfg.n_ref_sample)

    mean_dist_e6   = float(dist_e6_orb.mean())
    mean_dist_shuf = float(dist_shuf_orb.mean())
    mean_dist_rand = float(dist_rand_orb.mean())

    if verbose:
        print(f"  Mean dist to E6 orbit:")
        print(f"    E6 daughters:   {mean_dist_e6:.4f}")
        print(f"    Shuf daughters: {mean_dist_shuf:.4f}")
        print(f"    Rand daughters: {mean_dist_rand:.4f}")
        closer = mean_dist_e6 < mean_dist_shuf
        print(f"  E6 closer to orbit: "
              f"{'✅' if closer else '❌'}")

    return dict(
        seed=seed,
        t_break=t_break, n_res=n_res,
        phi_hist=phi_hist, res_hist=res_hist,
        # Orbits
        orbit_e6=orbit_e6,
        dc_orb_e6=dc_orb_e6,
        dc_orb_shuf=dc_orb_shuf,
        dc_orb_rand=dc_orb_rand,
        # Daughters
        ph_e6=ph_e6, ph_shuf=ph_shuf, ph_rand=ph_rand,
        hist_e6=hist_e6, hist_shuf=hist_shuf,
        hist_rand=hist_rand,
        sv_e6=sv_e6, sv_shuf=sv_shuf, sv_rand=sv_rand,
        # D_corr
        dc_e6=dc_e6, dc_shuf=dc_shuf, dc_rand=dc_rand,
        rv_e6=rv_e6, cv_e6=cv_e6,
        rv_shuf=rv_shuf, cv_shuf=cv_shuf,
        rv_rand=rv_rand, cv_rand=cv_rand,
        # PCA
        rat_e6=rat_e6, rat_shuf=rat_shuf,
        rat_rand=rat_rand,
        n80_e6=n80_e6, n80_shuf=n80_shuf,
        n80_rand=n80_rand,
        eff_e6=eff_e6, eff_shuf=eff_shuf,
        eff_rand=eff_rand,
        # Orbit distance
        dist_e6_orb=dist_e6_orb,
        dist_shuf_orb=dist_shuf_orb,
        dist_rand_orb=dist_rand_orb,
        mean_dist_e6=mean_dist_e6,
        mean_dist_shuf=mean_dist_shuf,
        mean_dist_rand=mean_dist_rand,
    )


# ═══════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════

def run_mc(n_runs=8):
    print("\n" + "█"*56)
    print("█  MONOSTRING FRAGMENTATION v6 — MONTE CARLO       █")
    print("█"*56)
    print(f"  N={cfg.N_strings}  σ={cfg.sigma}"
          f"  κ_d={cfg.kappa_dau}"
          f"  T={cfg.T_evolve}  {n_runs} runs\n")

    results = []
    t0 = time.time()

    for run in range(n_runs):
        seed = cfg.seed_base + run*7
        res  = run_one(seed=seed, verbose=(run==0))
        results.append(res)

        dc_ok = abs(res['dc_e6']  - res['dc_orb_e6']) < \
                abs(res['dc_shuf']- res['dc_orb_e6'])
        dist_ok = res['mean_dist_e6'] < res['mean_dist_shuf']
        eff_ok  = res['eff_e6'] < res['eff_shuf']

        print(f"  Run {run+1}/{n_runs}: "
              f"t*={res['t_break']:3d}  "
              f"dc: e6={res['dc_e6']:.2f} "
              f"sh={res['dc_shuf']:.2f} "
              f"orb={res['dc_orb_e6']:.2f} "
              f"{'✅' if dc_ok else '❌'}  "
              f"dist: e6={res['mean_dist_e6']:.3f} "
              f"sh={res['mean_dist_shuf']:.3f} "
              f"{'✅' if dist_ok else '❌'}  "
              f"eff: e6={res['eff_e6']:.1f} "
              f"sh={res['eff_shuf']:.1f} "
              f"{'✅' if eff_ok else '❌'}")

    elapsed = time.time() - t0

    DC_e6   = [r['dc_e6']    for r in results]
    DC_shuf = [r['dc_shuf']  for r in results]
    DC_rand = [r['dc_rand']  for r in results]
    DC_orb  = [r['dc_orb_e6']for r in results]
    MD_e6   = [r['mean_dist_e6']   for r in results]
    MD_shuf = [r['mean_dist_shuf'] for r in results]
    EF_e6   = [r['eff_e6']   for r in results]
    EF_shuf = [r['eff_shuf'] for r in results]

    print(f"\n{'='*56}")
    print(f"  AGGREGATED ({n_runs} runs, {elapsed:.0f}s)")
    print(f"{'='*56}")
    print(f"  t*: {np.mean([r['t_break'] for r in results]):.1f}"
          f" ± {np.std([r['t_break'] for r in results]):.1f}")
    print(f"\n  CORRELATION DIMENSION (daughter clouds)")
    print(f"    E6:      {np.mean(DC_e6):.3f} ± {np.std(DC_e6):.3f}")
    print(f"    Shuf:    {np.mean(DC_shuf):.3f} ± {np.std(DC_shuf):.3f}")
    print(f"    Rand:    {np.mean(DC_rand):.3f} ± {np.std(DC_rand):.3f}")
    print(f"    Orbit:   {np.mean(DC_orb):.3f} ± {np.std(DC_orb):.3f}")
    print(f"\n  DISTANCE TO E6 REFERENCE ORBIT")
    print(f"    E6:      {np.mean(MD_e6):.4f} ± {np.std(MD_e6):.4f}")
    print(f"    Shuf:    {np.mean(MD_shuf):.4f} ± {np.std(MD_shuf):.4f}")
    print(f"\n  EFFECTIVE DIMENSION (PCA)")
    print(f"    E6:      {np.mean(EF_e6):.3f} ± {np.std(EF_e6):.3f}")
    print(f"    Shuf:    {np.mean(EF_shuf):.3f} ± {np.std(EF_shuf):.3f}")

    # Statistical tests
    _, pv_dc   = ttest_ind(DC_e6, DC_shuf)
    _, pv_dist = ttest_ind(MD_e6, MD_shuf)
    _, pv_eff  = ttest_ind(EF_e6, EF_shuf)

    dc_orb_mean = np.mean(DC_orb)
    dc_ok   = (abs(np.mean(DC_e6)  - dc_orb_mean) <
               abs(np.mean(DC_shuf)- dc_orb_mean))
    dist_ok = np.mean(MD_e6) < np.mean(MD_shuf)
    eff_ok  = np.mean(EF_e6) < np.mean(EF_shuf)

    print(f"\n  E6 D_corr closer to orbit:  "
          f"{'✅' if dc_ok   else '❌'}  p={pv_dc:.4f}")
    print(f"  E6 closer to orbit (dist):  "
          f"{'✅' if dist_ok else '❌'}  p={pv_dist:.4f}")
    print(f"  E6 lower eff dim than shuf: "
          f"{'✅' if eff_ok  else '❌'}  p={pv_eff:.4f}")

    n80_e6   = np.mean([r['n80_e6']   for r in results])
    n80_shuf = np.mean([r['n80_shuf'] for r in results])
    print(f"\n  PCs for 80% variance:")
    print(f"    E6:   {n80_e6:.1f}  "
          f"Shuf: {n80_shuf:.1f}")

    return results


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(results):
    res  = results[0]
    n_mc = len(results)

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        "Monostring Fragmentation v6  —  "
        "D_corr + PCA + Orbit Distance\n"
        f"N={cfg.N_strings}  σ={cfg.sigma}"
        f"  κ_d={cfg.kappa_dau}  T={cfg.T_evolve}",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        4, 4, figure=fig, hspace=0.52, wspace=0.40)

    # ─── Row 0: Reference orbits ──────────────────────
    ax = fig.add_subplot(gs[0, 0])
    orb = res['orbit_e6']
    ax.scatter(orb[::10, 0], orb[::10, 1],
               s=1, alpha=0.3, c='steelblue',
               label=f'E6 orbit\nD_corr={res["dc_orb_e6"]:.2f}')
    ax.scatter(res['ph_e6'][:, 0],
               res['ph_e6'][:, 1],
               s=8, alpha=0.6, c='red',
               label=f'daughters\nD_corr={res["dc_e6"]:.2f}')
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title('E6 orbit + daughters\n(dims 1–2)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    orb_s = res['orbit_e6']  # using E6 orbit as reference
    # Compare orbits with different frequencies
    ax.scatter(orb_s[::10, 0], orb_s[::10, 2],
               s=1, alpha=0.3, c='steelblue', label='E6')
    ax.scatter(res['ph_shuf'][:,0], res['ph_shuf'][:,2],
               s=8, alpha=0.5, c='orange', label='shuf daughters')
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₃')
    ax.set_title('Shuf daughters vs E6 orbit\n(dims 1&3)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    # Monostring + resonances
    ph = res['phi_hist']
    tb = res['t_break']
    ax.plot(ph[:tb+1,0], ph[:tb+1,1],
            'b-', alpha=0.5, lw=0.8)
    ax.scatter(ph[tb,0], ph[tb,1], s=250,
               c='red', marker='*', zorder=5,
               label=f't*={tb}')
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xlabel('φ₁'); ax.set_ylabel('φ₂')
    ax.set_title(f'Monostring breakup\nt*={tb}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    # D_corr of orbits
    labels  = ['E6\norbit', 'Shuf\norbit', 'Rand\norbit']
    dc_orbs = [res['dc_orb_e6'],
               res['dc_orb_shuf'],
               res['dc_orb_rand']]
    colors  = ['steelblue', 'orange', 'green']
    ax.bar(labels, dc_orbs, color=colors, alpha=0.8,
           edgecolor='k')
    ax.axhline(3.0, c='red', ls='--', lw=1.5,
               label='D=3 (Part V)')
    ax.set_ylabel('D_corr')
    ax.set_title('D_corr of reference orbits\n(E6→≈3, others→?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # ─── Row 1: D_corr of daughter clouds ─────────────
    ax = fig.add_subplot(gs[1, 0:2])
    # Log-log correlation integral
    for lbl, rv, cv, c in [
            ('E6',   res['rv_e6'],   res['cv_e6'],   'steelblue'),
            ('Shuf', res['rv_shuf'], res['cv_shuf'], 'orange'),
            ('Rand', res['rv_rand'], res['cv_rand'], 'green')]:
        ok = cv > 0
        if ok.sum() > 2:
            ax.plot(np.log10(rv[ok]), np.log10(cv[ok]),
                    'o-', color=c, lw=2, ms=5, label=lbl)
    ax.set_xlabel('log₁₀ r')
    ax.set_ylabel('log₁₀ C(r)')
    ax.set_title('Correlation integral C(r)\n'
                 'slope = D_corr')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    # MC D_corr comparison
    DC_e6   = [r['dc_e6']   for r in results]
    DC_shuf = [r['dc_shuf'] for r in results]
    DC_rand = [r['dc_rand'] for r in results]
    DC_orb  = [r['dc_orb_e6'] for r in results]
    x = np.arange(n_mc)
    ax.plot(x, DC_e6,   'o-', color='steelblue',
            lw=2, ms=6, label='E6 daughters')
    ax.plot(x, DC_shuf, 's--', color='orange',
            lw=2, ms=6, label='Shuf daughters')
    ax.plot(x, DC_rand, '^:', color='green',
            lw=2, ms=6, label='Rand daughters')
    ax.plot(x, DC_orb,  'D-', color='black',
            lw=1.5, ms=5, label='E6 orbit (ref)')
    ax.axhline(3.0, c='red', ls='--', lw=1, label='D=3')
    ax.set_xlabel('run')
    ax.set_ylabel('D_corr')
    ax.set_title('D_corr per run\n(E6 hypothesis: ~orbit)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    # Bar comparison: mean D_corr
    lbls = ['E6\ndaught.', 'Shuf\ndaught.',
            'Rand\ndaught.', 'E6\norbit']
    means = [np.mean(DC_e6), np.mean(DC_shuf),
             np.mean(DC_rand), np.mean(DC_orb)]
    errs  = [np.std(DC_e6),  np.std(DC_shuf),
             np.std(DC_rand), np.std(DC_orb)]
    clrs  = ['steelblue','orange','green','black']
    ax.bar(lbls, means, yerr=errs, color=clrs,
           alpha=0.8, capsize=6, edgecolor='k')
    ax.axhline(3.0, c='red', ls='--', lw=1.5,
               label='D=3')
    ax.set_ylabel('D_corr')
    ax.set_title('Mean D_corr ± std\n(MC aggregate)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # ─── Row 2: PCA + orbit distance ──────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    # PCA eigenvalue spectra
    dims_pca = np.arange(1, 13)
    for lbl, rat, c, ls in [
            ('E6',   res['rat_e6'],   'steelblue', '-'),
            ('Shuf', res['rat_shuf'], 'orange',    '--'),
            ('Rand', res['rat_rand'], 'green',     ':')]:
        n = min(12, len(rat))
        ax.plot(dims_pca[:n], rat[:n]*100,
                color=c, ls=ls, lw=2, marker='o',
                ms=5, label=lbl)
    ax.axhline(100/12, c='gray', ls=':',
               lw=1, label='uniform (1/12)')
    ax.set_xlabel('PC rank')
    ax.set_ylabel('% variance')
    ax.set_title('PCA eigenvalue spectrum\n'
                 '(E6: steeper drop → lower eff dim?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(dims_pca)

    ax = fig.add_subplot(gs[2, 2])
    # Effective dimension MC
    EF_e6   = [r['eff_e6']   for r in results]
    EF_shuf = [r['eff_shuf'] for r in results]
    EF_rand = [r['eff_rand'] for r in results]
    x = np.arange(n_mc)
    ax.plot(x, EF_e6,   'o-', color='steelblue',
            lw=2, ms=6, label='E6')
    ax.plot(x, EF_shuf, 's--', color='orange',
            lw=2, ms=6, label='Shuf')
    ax.plot(x, EF_rand, '^:', color='green',
            lw=2, ms=6, label='Rand')
    ax.set_xlabel('run'); ax.set_ylabel('eff dim (PR)')
    ax.set_title('Effective dimension (PCA)\nper run')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    # Distance to orbit: distributions
    ax.hist(res['dist_e6_orb'],   bins=25, alpha=0.6,
            density=True, color='steelblue',
            label=f'E6 μ={res["mean_dist_e6"]:.3f}')
    ax.hist(res['dist_shuf_orb'], bins=25, alpha=0.6,
            density=True, color='orange',
            label=f'Shuf μ={res["mean_dist_shuf"]:.3f}')
    ax.hist(res['dist_rand_orb'], bins=25, alpha=0.6,
            density=True, color='green',
            label=f'Rand μ={res["mean_dist_rand"]:.3f}')
    ax.set_xlabel('min dist to E6 orbit')
    ax.set_ylabel('density')
    ax.set_title('Distance to E6 reference orbit\n'
                 '(E6 daughters should be closer)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ─── Row 3: MC summary ────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    # Orbit distance MC
    MD_e6   = [r['mean_dist_e6']   for r in results]
    MD_shuf = [r['mean_dist_shuf'] for r in results]
    MD_rand = [r['mean_dist_rand'] for r in results]
    x = np.arange(n_mc)
    ax.plot(x, MD_e6,   'o-', color='steelblue',
            lw=2, ms=6, label='E6')
    ax.plot(x, MD_shuf, 's--', color='orange',
            lw=2, ms=6, label='Shuf')
    ax.plot(x, MD_rand, '^:', color='green',
            lw=2, ms=6, label='Rand')
    ax.set_xlabel('run')
    ax.set_ylabel('mean dist to E6 orbit')
    ax.set_title('Orbit proximity per run')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    # Snapshot evolution of D_corr
    snap_dc_e6   = []
    snap_dc_shuf = []
    for st in cfg.snap_times:
        sv_e6  = results[0]['sv_e6'].get(st)
        sv_shuf= results[0]['sv_shuf'].get(st)
        if sv_e6 is not None:
            dc, _, _ = correlation_dimension(sv_e6, seed=0)
            snap_dc_e6.append(dc)
        if sv_shuf is not None:
            dc, _, _ = correlation_dimension(sv_shuf, seed=0)
            snap_dc_shuf.append(dc)

    if snap_dc_e6:
        ax.plot(cfg.snap_times[:len(snap_dc_e6)],
                snap_dc_e6, 'o-', color='steelblue',
                lw=2, ms=6, label='E6')
    if snap_dc_shuf:
        ax.plot(cfg.snap_times[:len(snap_dc_shuf)],
                snap_dc_shuf, 's--', color='orange',
                lw=2, ms=6, label='Shuf')
    ax.axhline(res['dc_orb_e6'], c='black', ls=':',
               lw=1.5, label=f'orbit={res["dc_orb_e6"]:.2f}')
    ax.axhline(3.0, c='red', ls='--', lw=1, label='D=3')
    ax.set_xlabel('t')
    ax.set_ylabel('D_corr')
    ax.set_title('D_corr evolution over time\n'
                 '(converges to orbit dim?)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    # Summary bar: all metrics
    categories = ['D_corr\n(daughter)', 'Dist to\norbit',
                  'Eff dim\n(PCA)']
    e6_vals   = [np.mean(DC_e6),   np.mean(MD_e6),
                 np.mean(EF_e6)]
    shuf_vals = [np.mean(DC_shuf), np.mean(MD_shuf),
                 np.mean(EF_shuf)]
    x = np.arange(3)
    ax.bar(x-0.2, e6_vals,   0.35, alpha=0.8,
           color='steelblue', label='E6', edgecolor='k')
    ax.bar(x+0.2, shuf_vals, 0.35, alpha=0.8,
           color='orange',    label='Shuf', edgecolor='k')
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_title('Summary: E6 vs Shuf\n(all 3 metrics)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Text summary
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')

    dc_orb_m = np.mean(DC_orb)
    dc_ok    = (abs(np.mean(DC_e6) - dc_orb_m) <
                abs(np.mean(DC_shuf) - dc_orb_m))
    dist_ok  = np.mean(MD_e6) < np.mean(MD_shuf)
    eff_ok   = np.mean(EF_e6) < np.mean(EF_shuf)

    _, pv_dc   = ttest_ind(DC_e6,  DC_shuf)
    _, pv_dist = ttest_ind(MD_e6,  MD_shuf)
    _, pv_eff  = ttest_ind(EF_e6,  EF_shuf)

    txt = (
        f"RESULTS  (n={n_mc})\n"
        f"{'─'*32}\n"
        f"ORBIT D_corr:  {dc_orb_m:.3f}\n"
        f"\nDAUGHTER D_corr\n"
        f"  E6:   {np.mean(DC_e6):.3f}±{np.std(DC_e6):.3f}\n"
        f"  Shuf: {np.mean(DC_shuf):.3f}±{np.std(DC_shuf):.3f}\n"
        f"  E6≈orbit: {'✅' if dc_ok else '❌'} p={pv_dc:.4f}\n"
        f"\nORBIT DISTANCE\n"
        f"  E6:   {np.mean(MD_e6):.4f}\n"
        f"  Shuf: {np.mean(MD_shuf):.4f}\n"
        f"  E6<Shuf: {'✅' if dist_ok else '❌'} p={pv_dist:.4f}\n"
        f"\nEFF DIM (PCA)\n"
        f"  E6:   {np.mean(EF_e6):.2f}\n"
        f"  Shuf: {np.mean(EF_shuf):.2f}\n"
        f"  E6<Shuf: {'✅' if eff_ok else '❌'} p={pv_eff:.4f}\n"
        f"\nOVERALL\n"
        f"  {'✅ E6 effect confirmed' if sum([dc_ok,dist_ok,eff_ok])>=2 else '❌ No consistent E6 effect'}\n"
        f"  (2/3 metrics needed)\n"
    )
    ax.text(0.03, 0.98, txt,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e8f5e9', alpha=0.95))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation_v6.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v6.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation v6                    ║")
    print("║  Three measurements:                            ║")
    print("║  1. D_corr of daughter cloud                   ║")
    print("║  2. Distance to E6 reference orbit             ║")
    print("║  3. PCA effective dimension                     ║")
    print("║  Null: shuffled E6 frequencies                 ║")
    print("╚══════════════════════════════════════════════════╝")

    results = run_mc(n_runs=cfg.n_runs)
    plot_all(results)
