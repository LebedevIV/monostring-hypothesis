"""
Monostring Fragmentation v7
============================
Integrates the full experimental roadmap.

Sprint 1 (CRITICAL FIRST):
  Exp 0.1 — D_corr calibration on known manifolds
  Exp 0.2 — D_corr convergence with N
  Exp 0.3 — WHY v6 gave 4.327 instead of 3.02

Sprint 2:
  Exp 1.1 — Poincaré recurrence as breakup mechanism
  Exp 1.2 — Entropy arrow of time

Sprint 3:
  Exp 2.1 — Correlation graph (relational space)
  Exp 2.2 — Spectral dimension of the graph

All unexpected findings are tracked and reported.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind, linregress
from scipy.linalg import eigh
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# GLOBAL CONFIG
# ═══════════════════════════════════════════════════════

class Config:
    D = 6
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    rng0 = np.random.RandomState(1)
    omega_shuf = rng0.permutation(omega_E6.copy())
    omega_rand = rng0.uniform(
        omega_E6.min(), omega_E6.max(), 6)

    kappa_mono  = 0.30
    kappa_dau   = 0.05
    noise_dau   = 0.006
    delta_res   = 0.15
    n_res_visits = 15

    N_strings = 300
    sigma     = 0.50
    T_evolve  = 1000

    n_runs    = 8
    seed_base = 42

cfg = Config()


# ═══════════════════════════════════════════════════════
# SPRINT 1 — МЕТРОЛОГИЧЕСКАЯ БАЗА
# ═══════════════════════════════════════════════════════

def correlation_dimension_core(points, n_pairs=5000,
                                r_lo=None, r_hi=None,
                                n_bins=20, seed=0):
    """
    Grassberger-Procaccia correlation dimension.

    C(r) = fraction of pairs with distance < r
    D_corr = d log C(r) / d log r

    Works on any point cloud in R^d or T^d.
    r_lo, r_hi: fitting range (auto if None)
    """
    rng = np.random.RandomState(seed)
    N   = len(points)
    n   = min(n_pairs, N*(N-1)//2)

    i = rng.randint(0, N, n*2)
    j = rng.randint(0, N, n*2)
    ok = i != j
    i, j = i[ok][:n], j[ok][:n]

    # Torus metric
    d = np.abs(points[i] - points[j])
    d = np.minimum(d, 2*np.pi - d)
    dists = np.linalg.norm(d, axis=1)

    if r_lo is None:
        r_lo = np.percentile(dists, 5)
    if r_hi is None:
        r_hi = np.percentile(dists, 60)

    r_vals = np.logspace(np.log10(r_lo + 1e-10),
                          np.log10(r_hi), n_bins)
    C_vals = np.array([float(np.mean(dists < r))
                       for r in r_vals])

    ok = C_vals > 0.01
    if ok.sum() < 4:
        return float('nan'), r_vals, C_vals, (r_lo, r_hi)

    log_r = np.log10(r_vals[ok])
    log_C = np.log10(C_vals[ok])
    slope, intercept, r_val, p_val, _ = linregress(
        log_r, log_C)

    return float(slope), r_vals, C_vals, (r_lo, r_hi)


# ──────────────────────────────────────────────────────
# Exp 0.1: Калибровка на известных многообразиях
# ──────────────────────────────────────────────────────

def exp_01_calibration(N=2000, n_runs=5):
    """
    Measure D_corr on manifolds with known dimension.
    Tests: T^d for d = 1..6, uniform in [0,2π]^d

    Key question: at what N and r-range does D_corr
    reliably recover the true dimension?

    Expected: D_corr(T^d) ≈ d ± 0.15
    """
    print("\n" + "═"*56)
    print("  EXP 0.1: D_corr Calibration on T^d")
    print("═"*56)

    results = {}
    for d in range(1, 7):
        dc_list = []
        for run in range(n_runs):
            rng    = np.random.RandomState(run * 13 + d)
            # Uniform sample on T^d, embedded in T^6
            pts    = np.zeros((N, 6))
            pts[:, :d] = rng.uniform(0, 2*np.pi, (N, d))

            # Adjust r-range for dimension
            r_lo = 0.05 * (6/d)**0.5
            r_hi = 0.60 * (6/d)**0.5
            r_lo = min(r_lo, 0.8)
            r_hi = min(r_hi, 2.5)

            dc, _, _, _ = correlation_dimension_core(
                pts, n_pairs=5000,
                r_lo=r_lo, r_hi=r_hi, seed=run)
            dc_list.append(dc)

        mu = np.nanmean(dc_list)
        sd = np.nanstd(dc_list)
        ok = abs(mu - d) < 0.3
        print(f"  T^{d}: D_corr = {mu:.3f} ± {sd:.3f}"
              f"  (true={d})  {'✅' if ok else '❌'}")
        results[d] = dict(mean=mu, std=sd, true=d)

    return results


# ──────────────────────────────────────────────────────
# Exp 0.2: Сходимость D_corr(E6 orbit) по N
# ──────────────────────────────────────────────────────

def generate_orbit(omega, kappa, T, seed=0, warmup=500):
    """Generate monostring orbit (single trajectory)."""
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
    orbit = []
    for _ in range(T):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
        orbit.append(phi.copy())
    return np.array(orbit)

def exp_02_convergence(N_list=None, kappa=0.30,
                        n_runs=5):
    """
    D_corr(E6 orbit) as function of N points.

    v6 gave 4.327, Part V gave 3.02.
    Hypothesis: v6 used wrong r-range (too large).
    Here we find the correct r-range by calibration.

    Expected: D_corr → 3.0 at N > 1000
    """
    print("\n" + "═"*56)
    print("  EXP 0.2: D_corr(E6 orbit) vs N")
    print("═"*56)

    if N_list is None:
        N_list = [200, 500, 1000, 2000, 5000]

    results = {}
    for N in N_list:
        dc_list = []
        for run in range(n_runs):
            orbit = generate_orbit(
                cfg.omega_E6, kappa, N,
                seed=run*17, warmup=500)

            # Use calibrated r-range
            # For 6D torus with quasi-3D orbit:
            # distances cluster around 2-3 rad
            # Use 5th-40th percentile
            rng2   = np.random.RandomState(run)
            idx    = rng2.choice(N, min(500, N),
                                  replace=False)
            sample = orbit[idx]
            d_tmp  = np.abs(
                sample[:, None, :] - sample[None, :, :])
            d_tmp  = np.minimum(d_tmp, 2*np.pi-d_tmp)
            d_tmp  = np.linalg.norm(d_tmp, axis=2)
            d_flat = d_tmp[np.triu_indices(
                len(sample), k=1)]
            r_lo = float(np.percentile(d_flat, 5))
            r_hi = float(np.percentile(d_flat, 40))

            dc, _, _, _ = correlation_dimension_core(
                orbit, n_pairs=5000,
                r_lo=r_lo, r_hi=r_hi, seed=run)
            dc_list.append(dc)

        mu = np.nanmean(dc_list)
        sd = np.nanstd(dc_list)
        print(f"  N={N:5d}: D_corr = {mu:.3f} ± {sd:.3f}")
        results[N] = dict(mean=mu, std=sd)

    return results


# ──────────────────────────────────────────────────────
# Exp 0.3: Диагностика расхождения v6 (4.327 vs 3.02)
# ──────────────────────────────────────────────────────

def exp_03_diagnose_v6():
    """
    WHY did v6 give D_corr(orbit E6) = 4.327
    when Part V gives 3.02?

    Systematically vary:
    1. r_lo, r_hi (fitting range)
    2. N (orbit length)
    3. warmup steps
    4. kappa

    Find which parameter causes the discrepancy.
    """
    print("\n" + "═"*56)
    print("  EXP 0.3: Diagnose v6 discrepancy")
    print("═"*56)

    orbit_ref = generate_orbit(
        cfg.omega_E6, cfg.kappa_mono,
        T=5000, seed=0, warmup=500)

    print("\n  A) Varying r-range (N=5000 fixed):")
    r_ranges = [
        (0.05, 0.40),  # tight, small r
        (0.10, 0.80),  # v6-like (likely cause)
        (0.05, 0.40),  # Part V calibrated
        (0.02, 0.25),  # very tight
        (0.20, 1.20),  # very loose
    ]
    labels = ['tight','v6-like','calibrated',
              'very tight','very loose']
    for (r_lo, r_hi), lbl in zip(r_ranges, labels):
        dc, _, _, _ = correlation_dimension_core(
            orbit_ref, n_pairs=5000,
            r_lo=r_lo, r_hi=r_hi, seed=0)
        print(f"    [{lbl:12s}] r=[{r_lo:.2f},{r_hi:.2f}]:"
              f" D_corr={dc:.3f}")

    print("\n  B) Varying orbit length (r calibrated):")
    for T in [500, 1000, 2000, 5000, 10000]:
        orbit = generate_orbit(
            cfg.omega_E6, cfg.kappa_mono,
            T=T, seed=0)
        dc_list = []
        for s in range(5):
            dc, _, _, _ = correlation_dimension_core(
                orbit, n_pairs=3000,
                r_lo=0.05, r_hi=0.40, seed=s)
            dc_list.append(dc)
        print(f"    T={T:6d}: D_corr = "
              f"{np.nanmean(dc_list):.3f} "
              f"± {np.nanstd(dc_list):.3f}")

    print("\n  C) Varying kappa (T=5000, r calibrated):")
    for kappa in [0.10, 0.20, 0.30, 0.40, 0.50]:
        orbit = generate_orbit(
            cfg.omega_E6, kappa, T=5000, seed=0)
        dc, _, _, _ = correlation_dimension_core(
            orbit, n_pairs=5000,
            r_lo=0.05, r_hi=0.40, seed=0)
        print(f"    κ={kappa:.2f}: D_corr = {dc:.3f}")


# ═══════════════════════════════════════════════════════
# SPRINT 2 — МЕХАНИКА РАЗРЫВА
# ═══════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────
# Exp 1.1: Пуанкаре — время рекуррентности
# ──────────────────────────────────────────────────────

def poincare_recurrence(omega, kappa, phi0,
                         epsilon=0.3,
                         T_max=5000, seed=0):
    """
    Find first return to epsilon-neighbourhood of phi0.

    For quasi-periodic systems on T^D:
    T_rec ~ exp(D * complexity(omega))

    If T_rec > T_max → "breakup" (system forgets itself).

    Physical interpretation:
    The monostring ceases to be a coherent oscillator
    when it can no longer return to its initial state.
    This is an intrinsic, not external, criterion.
    """
    phi = phi0.copy()
    for t in range(1, T_max+1):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
        d = np.abs(phi - phi0)
        d = np.minimum(d, 2*np.pi - d)
        if np.linalg.norm(d) < epsilon:
            return t
    return None  # No return within T_max


def exp_11_poincare(n_runs=20, T_max=8000,
                     epsilon=0.35):
    """
    Compare Poincaré recurrence times for:
    - E6 frequencies (maximally irrational)
    - Shuffled E6 (same magnitudes, no Coxeter)
    - Random uniform

    Hypothesis: T_rec(E6) >> T_rec(others)
    → E6 monostring "forgets itself" first
    → breaks up first (without external trigger)

    This would explain WHY E6 specifically
    is the fragmentation candidate.
    """
    print("\n" + "═"*56)
    print("  EXP 1.1: Poincaré Recurrence Times")
    print("═"*56)
    print(f"  ε={epsilon}, T_max={T_max}, "
          f"{n_runs} runs\n")

    results = {}
    for omega, lbl in [
            (cfg.omega_E6,   'E6      '),
            (cfg.omega_shuf, 'Shuffled'),
            (cfg.omega_rand, 'Random  ')]:

        T_recs = []
        n_none = 0
        for run in range(n_runs):
            rng  = np.random.RandomState(run * 31)
            phi0 = rng.uniform(0, 2*np.pi, 6)
            T_r  = poincare_recurrence(
                omega, cfg.kappa_mono, phi0,
                epsilon=epsilon, T_max=T_max,
                seed=run)
            if T_r is None:
                n_none += 1
                T_recs.append(T_max)  # censored
            else:
                T_recs.append(T_r)

        T_recs = np.array(T_recs)
        mu  = np.mean(T_recs)
        med = np.median(T_recs)
        print(f"  {lbl}: mean={mu:.0f}  "
              f"median={med:.0f}  "
              f"no-return={n_none}/{n_runs}")
        results[lbl.strip()] = dict(
            T_recs=T_recs, mean=mu,
            median=med, n_none=n_none)

    # Test: E6 has higher T_rec?
    _, pv = ttest_ind(
        results['E6']['T_recs'],
        results['Shuffled']['T_recs'])
    e6_higher = (results['E6']['mean']
                 > results['Shuffled']['mean'])
    print(f"\n  E6 T_rec > Shuffled: "
          f"{'✅' if e6_higher else '❌'}  "
          f"p={pv:.4f}")
    print("  Physical: higher T_rec → E6 more likely")
    print("  to 'forget itself' → intrinsic breakup")
    return results


# ──────────────────────────────────────────────────────
# Exp 1.2: Стрела времени через энтропию
# ──────────────────────────────────────────────────────

def shannon_entropy_cloud(phases, bins=15):
    """
    Shannon entropy of daughter string cloud.
    H = sum over dims of H_d(histogram of phi_d)

    Before breakup: H ≈ low (concentrated near phi_break)
    After breakup:  H grows (strings spread out)

    Hypothesis: H is monotonically increasing after t_break.
    This defines the arrow of time.
    """
    H = 0.0
    for d in range(phases.shape[1]):
        hist, _ = np.histogram(
            phases[:, d], bins=bins,
            range=(0, 2*np.pi))
        p = hist / (hist.sum() + 1e-15)
        H += -np.sum(p * np.log(p + 1e-15))
    return H

def exp_12_arrow_of_time(seed=42):
    """
    Measure entropy S(t) of daughter cloud.

    Three phases:
    1. Pre-breakup:  monostring trajectory (1 string)
                    S ≈ low (concentrated)
    2. At breakup:   N strings appear near phi_break
                    S jumps (fragmentation)
    3. Post-breakup: strings evolve freely
                    S grows (arrow of time)

    Key test: is S(t) monotonically increasing?
    If yes: breakup defines t=0 for the universe.
    """
    print("\n" + "═"*56)
    print("  EXP 1.2: Arrow of Time via Entropy")
    print("═"*56)

    rng = np.random.RandomState(seed)
    omega = cfg.omega_E6

    # Generate orbit (pre-breakup)
    orbit = generate_orbit(omega, cfg.kappa_mono,
                           T=500, seed=seed)

    # Compute S along orbit (as if 1 string = delta function)
    # Use sliding window of nearby orbit points
    S_pre = []
    window = 30
    for t in range(window, len(orbit)):
        pts = orbit[t-window:t]
        # Add small noise to get a cloud
        pts_noisy = (pts + rng.normal(
            0, 0.05, pts.shape)) % (2*np.pi)
        S_pre.append(shannon_entropy_cloud(pts_noisy))

    # At breakup: fragment
    phi_break = orbit[-1]
    daughters0 = (
        phi_break[np.newaxis,:] +
        rng.normal(0, cfg.sigma,
                   (cfg.N_strings, 6))) % (2*np.pi)
    S_break = shannon_entropy_cloud(daughters0)

    # Post-breakup: evolve and measure S(t)
    ph = daughters0.copy()
    S_post = [S_break]
    t_steps = []

    for step in range(cfg.T_evolve):
        ph = (ph + omega[np.newaxis,:]
               + cfg.kappa_dau * np.sin(ph)
               + rng.normal(0, cfg.noise_dau,
                            ph.shape)) % (2*np.pi)

        if step % 20 == 0:
            S_post.append(shannon_entropy_cloud(ph))
            t_steps.append(step)

    S_post = np.array(S_post)
    S_pre_mean = np.mean(S_pre)
    S_max = -6 * np.log(1/15)  # maximum entropy

    # Is S increasing?
    dS = np.diff(S_post)
    frac_increasing = float(np.mean(dS > 0))

    print(f"  S(pre-breakup):  {S_pre_mean:.3f}")
    print(f"  S(at breakup):   {S_break:.3f}")
    print(f"  S(final):        {S_post[-1]:.3f}")
    print(f"  S(max possible): {S_max:.3f}")
    print(f"  Fraction increasing: {frac_increasing:.3f}")
    print(f"  Arrow of time:   "
          f"{'✅ YES' if frac_increasing > 0.6 else '❌ NO'}")

    # Compare E6 vs shuffled entropy evolution
    ph_shuf = daughters0.copy()
    S_shuf = [S_break]
    for step in range(cfg.T_evolve):
        ph_shuf = (ph_shuf + cfg.omega_shuf[np.newaxis,:]
                   + cfg.kappa_dau * np.sin(ph_shuf)
                   + rng.normal(0, cfg.noise_dau,
                                ph_shuf.shape)) % (2*np.pi)
        if step % 20 == 0:
            S_shuf.append(shannon_entropy_cloud(ph_shuf))

    print(f"\n  E6 final entropy:   {S_post[-1]:.3f}")
    print(f"  Shuf final entropy: {np.array(S_shuf)[-1]:.3f}")
    print(f"  E6 lower entropy:   "
          f"{'✅ more ordered' if S_post[-1] < np.array(S_shuf)[-1] else '❌ no difference'}")

    return dict(
        S_pre=S_pre, S_pre_mean=S_pre_mean,
        S_post=S_post, S_shuf=np.array(S_shuf),
        S_break=S_break,
        t_steps=t_steps,
        frac_increasing=frac_increasing)


# ═══════════════════════════════════════════════════════
# SPRINT 3 — РЕЛЯЦИОННОЕ ПРОСТРАНСТВО
# ═══════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────
# Exp 2.1: Граф корреляций
# ──────────────────────────────────────────────────────

def build_correlation_graph(phases_history,
                             threshold=0.5,
                             n_sample=150):
    """
    Build graph G where edge(i,j) if corr(i,j) > threshold.

    corr(i,j) = mean over dims of
                |Pearson(phi_i(t), phi_j(t))|

    Physical meaning: two strings are "connected"
    if their oscillations are correlated over time.
    This is a RELATIONAL notion of proximity —
    not where they are, but how they move together.

    Key question: does this graph have D_corr ≈ 3?
    """
    T, N, D = phases_history.shape

    # Sample N_sample strings for speed
    idx = np.random.choice(N, min(n_sample, N),
                            replace=False)
    ph  = phases_history[:, idx, :]  # (T, n_sample, D)
    n   = len(idx)

    # Correlation matrix: (n, n)
    C = np.zeros((n, n))
    for d in range(D):
        series = ph[:, :, d]  # (T, n)
        # Pearson correlation
        mean = series.mean(axis=0, keepdims=True)
        std  = series.std(axis=0, keepdims=True) + 1e-15
        normed = (series - mean) / std
        C_d = np.dot(normed.T, normed) / T
        C += np.abs(C_d)
    C /= D

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(
        (C > threshold) & (np.eye(n) == 0))
    for r, c in zip(rows, cols):
        if r < c:
            G.add_edge(int(r), int(c),
                       weight=float(C[r,c]))

    return G, C


def spectral_dim_graph(G, t_range=None):
    """
    Spectral dimension of graph via heat kernel.
    d_s = -2 d log K(t) / d log t
    where K(t) = Tr(exp(-tL)) / N, L = normalized Laplacian.

    This is the same method as Part IV.
    For a d-dim manifold: K(t) ~ t^(-d/2).
    """
    N = G.number_of_nodes()
    if N < 4:
        return float('nan'), [], []

    # Laplacian
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigvals = np.sort(np.linalg.eigvalsh(L))

    # Heat kernel trace
    if t_range is None:
        t_range = np.logspace(-1, 2, 30)

    K_vals = []
    for t in t_range:
        K = float(np.mean(np.exp(-t * eigvals)))
        K_vals.append(K)
    K_vals = np.array(K_vals)

    # Spectral dimension from slope
    ok = K_vals > 0.01
    if ok.sum() < 4:
        return float('nan'), t_range, K_vals

    log_t = np.log(t_range[ok])
    log_K = np.log(K_vals[ok])
    slope, _, _, _, _ = linregress(log_t, log_K)
    d_s = float(-2 * slope)

    return d_s, t_range, K_vals


def exp_21_correlation_graph(seed=42, n_runs=6):
    """
    Build correlation graphs for E6 and null daughters.
    Compare spectral dimensions.

    Key hypotheses:
    H1: d_s(graph E6) ≈ 3  (quasi-3D relational space)
    H2: d_s(graph E6) < d_s(graph shuf)  (E6 more structured)
    H3: graph E6 is more connected (fewer components)
    """
    print("\n" + "═"*56)
    print("  EXP 2.1: Correlation Graph Analysis")
    print("═"*56)

    results = {'e6': [], 'shuf': [], 'rand': []}

    for run in range(n_runs):
        rng = np.random.RandomState(seed + run*7)

        # Generate daughters
        phi_break = rng.uniform(0, 2*np.pi, 6)
        daughters0 = (
            phi_break[np.newaxis,:] +
            rng.normal(0, cfg.sigma,
                       (cfg.N_strings, 6))) % (2*np.pi)
        phi_rand_origin = rng.uniform(0, 2*np.pi, 6)
        daughters0_shuf = (
            phi_rand_origin[np.newaxis,:] +
            rng.normal(0, cfg.sigma,
                       (cfg.N_strings, 6))) % (2*np.pi)

        T_hist = 200  # shorter for memory
        measure_every = 4

        # Evolve and record history
        def evolve_history(ph0, omega):
            ph   = ph0.copy()
            hist = []
            for step in range(T_hist):
                ph = (ph + omega[np.newaxis,:]
                       + cfg.kappa_dau * np.sin(ph)
                       + rng.normal(
                           0, cfg.noise_dau,
                           ph.shape)) % (2*np.pi)
                if step % measure_every == 0:
                    hist.append(ph.copy())
            return np.array(hist)  # (T/me, N, D)

        hist_e6   = evolve_history(
            daughters0,      cfg.omega_E6)
        hist_shuf = evolve_history(
            daughters0_shuf, cfg.omega_shuf)

        for lbl, hist, thresh in [
                ('e6',   hist_e6,   0.50),
                ('shuf', hist_shuf, 0.50)]:
            G, C = build_correlation_graph(
                hist, threshold=thresh, n_sample=120)

            d_s, _, _ = spectral_dim_graph(G)

            n_comp = nx.number_connected_components(G)
            n_edge = G.number_of_edges()
            n_node = G.number_of_nodes()
            density = (2*n_edge / max(n_node*(n_node-1), 1))

            results[lbl].append(dict(
                d_s=d_s, n_comp=n_comp,
                n_edge=n_edge, density=density,
                n_node=n_node))

            if run == 0:
                print(f"  [{lbl.upper()}] run={run}: "
                      f"d_s={d_s:.3f}  "
                      f"n_comp={n_comp}  "
                      f"edges={n_edge}  "
                      f"density={density:.3f}")

    # Aggregate
    print(f"\n  AGGREGATED ({n_runs} runs):")
    for lbl in ['e6', 'shuf']:
        ds_vals = [r['d_s'] for r in results[lbl]
                   if not np.isnan(r['d_s'])]
        nc_vals = [r['n_comp'] for r in results[lbl]]
        de_vals = [r['density'] for r in results[lbl]]
        print(f"  [{lbl.upper()}]: "
              f"d_s={np.nanmean(ds_vals):.3f}±"
              f"{np.nanstd(ds_vals):.3f}  "
              f"n_comp={np.mean(nc_vals):.1f}  "
              f"density={np.mean(de_vals):.4f}")

    # Statistical comparison
    ds_e6   = [r['d_s'] for r in results['e6']
               if not np.isnan(r['d_s'])]
    ds_shuf = [r['d_s'] for r in results['shuf']
               if not np.isnan(r['d_s'])]
    if len(ds_e6) > 2 and len(ds_shuf) > 2:
        _, pv = ttest_ind(ds_e6, ds_shuf)
        print(f"\n  d_s E6 vs shuf: p={pv:.4f}")
        print(f"  E6 closer to 3: "
              f"{'✅' if abs(np.mean(ds_e6)-3) < abs(np.mean(ds_shuf)-3) else '❌'}")

    return results


# ═══════════════════════════════════════════════════════
# UNEXPECTED FINDINGS TRACKER
# ═══════════════════════════════════════════════════════

class FindingsTracker:
    """
    Track all unexpected results across experiments.
    These may be emergent discoveries.
    """
    def __init__(self):
        self.findings = []

    def add(self, experiment, finding,
            value, expected, significance):
        self.findings.append(dict(
            exp=experiment,
            finding=finding,
            value=value,
            expected=expected,
            significance=significance))

    def report(self):
        print("\n" + "█"*56)
        print("█  UNEXPECTED FINDINGS (Emergent Discoveries) █")
        print("█"*56)
        for f in self.findings:
            print(f"\n  [{f['exp']}]")
            print(f"  Finding:    {f['finding']}")
            print(f"  Value:      {f['value']}")
            print(f"  Expected:   {f['expected']}")
            print(f"  Importance: {f['significance']}")


tracker = FindingsTracker()


# ═══════════════════════════════════════════════════════
# MAIN — RUN ALL SPRINTS
# ═══════════════════════════════════════════════════════

def run_all():
    t0 = time.time()

    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation v7                    ║")
    print("║  Integrated Experimental Roadmap               ║")
    print("╚══════════════════════════════════════════════════╝")

    all_results = {}

    # ── Sprint 1: Metrology ───────────────────────────
    print("\n" + "▓"*56)
    print("▓  SPRINT 1: METROLOGICAL BASE")
    print("▓"*56)

    r01 = exp_01_calibration(N=2000, n_runs=5)
    all_results['exp01'] = r01

    # Check for unexpected D_corr values
    for d, res in r01.items():
        if abs(res['mean'] - d) > 0.5:
            tracker.add('Exp 0.1',
                f'D_corr(T^{d}) = {res["mean"]:.2f}',
                res['mean'], d,
                '⭐⭐ Wrong r-range invalidates all D_corr')

    r02 = exp_02_convergence(
        N_list=[200, 500, 1000, 2000, 5000],
        n_runs=4)
    all_results['exp02'] = r02

    # Check convergence
    vals = [r02[N]['mean'] for N in sorted(r02.keys())]
    if not np.isnan(vals[-1]):
        if abs(vals[-1] - 3.0) < 0.3:
            tracker.add('Exp 0.2',
                f'D_corr(E6 orbit) → {vals[-1]:.2f}',
                vals[-1], 3.0,
                '✅✅ Confirms Part V result')
        else:
            tracker.add('Exp 0.2',
                f'D_corr(E6 orbit) = {vals[-1]:.2f} ≠ 3',
                vals[-1], 3.0,
                '⭐⭐⭐ Part V result not reproducible here')

    exp_03_diagnose_v6()

    # ── Sprint 2: Breakup Mechanics ───────────────────
    print("\n" + "▓"*56)
    print("▓  SPRINT 2: BREAKUP MECHANICS")
    print("▓"*56)

    r11 = exp_11_poincare(n_runs=15, T_max=6000)
    all_results['exp11'] = r11

    e6_mean   = r11['E6']['mean']
    shuf_mean = r11['Shuffled']['mean']
    if e6_mean > shuf_mean * 1.5:
        tracker.add('Exp 1.1',
            f'T_rec(E6)={e6_mean:.0f} >> T_rec(Shuf)={shuf_mean:.0f}',
            e6_mean/shuf_mean, 1.0,
            '⭐⭐⭐ Intrinsic breakup mechanism confirmed')
    elif e6_mean > shuf_mean:
        tracker.add('Exp 1.1',
            f'T_rec(E6)={e6_mean:.0f} > T_rec(Shuf)={shuf_mean:.0f}',
            e6_mean/shuf_mean, 1.0,
            '⭐ Weak signal: E6 slightly longer recurrence')
    else:
        tracker.add('Exp 1.1',
            f'T_rec(E6)={e6_mean:.0f} ≤ T_rec(Shuf)={shuf_mean:.0f}',
            e6_mean/shuf_mean, '>1',
            '❌ Poincaré mechanism does not favor E6')

    r12 = exp_12_arrow_of_time(seed=42)
    all_results['exp12'] = r12

    if r12['frac_increasing'] > 0.65:
        tracker.add('Exp 1.2',
            f'Entropy monotone: {r12["frac_increasing"]:.2f}',
            r12['frac_increasing'], '>0.65',
            '⭐⭐ Arrow of time confirmed')
    if r12['S_post'][-1] < np.array(r12['S_shuf'])[-1]:
        tracker.add('Exp 1.2',
            'E6 entropy < Shuf entropy (more ordered)',
            r12['S_post'][-1], 'S_shuf',
            '⭐⭐ E6 maintains lower entropy than null')

    # ── Sprint 3: Relational Space ────────────────────
    print("\n" + "▓"*56)
    print("▓  SPRINT 3: RELATIONAL SPACE (GRAPH)")
    print("▓"*56)

    r21 = exp_21_correlation_graph(seed=42, n_runs=6)
    all_results['exp21'] = r21

    ds_e6_mean = np.nanmean(
        [r['d_s'] for r in r21['e6']
         if not np.isnan(r['d_s'])])
    if abs(ds_e6_mean - 3.0) < 0.5:
        tracker.add('Exp 2.1',
            f'd_s(graph E6) = {ds_e6_mean:.2f} ≈ 3',
            ds_e6_mean, 3.0,
            '⭐⭐⭐ Relational space has 3D structure!')
    elif not np.isnan(ds_e6_mean):
        tracker.add('Exp 2.1',
            f'd_s(graph E6) = {ds_e6_mean:.2f} ≠ 3',
            ds_e6_mean, 3.0,
            '❌ Relational space ≠ 3D')

    # ── Final Report ──────────────────────────────────
    elapsed = time.time() - t0
    tracker.report()

    print("\n" + "═"*56)
    print(f"  Total runtime: {elapsed:.0f}s")
    print("═"*56)

    return all_results


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(all_results):
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Monostring Fragmentation v7 — Integrated Roadmap\n"
        "Sprint 1: Metrology | Sprint 2: Breakup | "
        "Sprint 3: Relational Space",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        3, 4, figure=fig, hspace=0.50, wspace=0.38)

    # ── Row 0: D_corr Calibration ─────────────────────
    ax = fig.add_subplot(gs[0, 0])
    r01 = all_results.get('exp01', {})
    if r01:
        true_d  = list(r01.keys())
        meas_d  = [r01[d]['mean'] for d in true_d]
        err_d   = [r01[d]['std']  for d in true_d]
        ax.errorbar(true_d, meas_d, yerr=err_d,
                    fmt='o-', color='steelblue',
                    lw=2, ms=8, capsize=5,
                    label='measured')
        ax.plot([1,6],[1,6],'k--',lw=1.5,label='ideal')
        ax.fill_between(true_d,
                        [d-0.3 for d in true_d],
                        [d+0.3 for d in true_d],
                        alpha=0.15, color='green',
                        label='±0.3 band')
    ax.set_xlabel('True dimension d')
    ax.set_ylabel('Measured D_corr')
    ax.set_title('Exp 0.1: D_corr Calibration\non T^d')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    r02 = all_results.get('exp02', {})
    if r02:
        Ns   = sorted(r02.keys())
        mus  = [r02[N]['mean'] for N in Ns]
        sds  = [r02[N]['std']  for N in Ns]
        ax.errorbar(Ns, mus, yerr=sds,
                    fmt='o-', color='steelblue',
                    lw=2, ms=8, capsize=5)
        ax.axhline(3.0, c='red', ls='--',
                   lw=1.5, label='Part V: 3.02')
        ax.axhline(4.327, c='orange', ls=':',
                   lw=1.5, label='v6 result: 4.33')
        ax.set_xscale('log')
    ax.set_xlabel('N (orbit points)')
    ax.set_ylabel('D_corr')
    ax.set_title('Exp 0.2: D_corr convergence\nE6 orbit vs N')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Row 1: Breakup ────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    r11 = all_results.get('exp11', {})
    if r11:
        for lbl, c in [('E6','steelblue'),
                        ('Shuffled','orange'),
                        ('Random','green')]:
            key = lbl.strip()
            if key in r11:
                T_recs = r11[key]['T_recs']
                ax.hist(T_recs, bins=15, alpha=0.6,
                        density=True, color=c,
                        label=lbl)
    ax.set_xlabel('Poincaré recurrence time')
    ax.set_ylabel('density')
    ax.set_title('Exp 1.1: Recurrence times\n(higher = breaks sooner)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1:3])
    r12 = all_results.get('exp12', {})
    if r12:
        pre_x = np.arange(len(r12['S_pre']))
        post_x = np.arange(len(r12['S_post']))
        # Pre-breakup
        ax.plot(pre_x - len(pre_x),
                r12['S_pre'],
                'gray', lw=1.5, alpha=0.7,
                label='Pre-breakup (orbit window)')
        # Post-breakup: E6
        ax.plot(post_x, r12['S_post'],
                'b-', lw=2, label='E6 daughters')
        # Post-breakup: shuffled
        ax.plot(post_x,
                r12['S_shuf'][:len(post_x)],
                'r--', lw=2, label='Shuf daughters')
        ax.axvline(0, c='red', ls=':', lw=2,
                   label='t* (breakup)')
        ax.axhline(r12['S_break'], c='purple',
                   ls=':', lw=1,
                   label=f'S(breakup)={r12["S_break"]:.2f}')
    ax.set_xlabel('t (0 = breakup)')
    ax.set_ylabel('Shannon entropy S(t)')
    ax.set_title('Exp 1.2: Arrow of Time\n'
                 'S grows after breakup → irreversibility')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    if r12:
        dS = np.diff(r12['S_post'])
        ax.hist(dS, bins=20, color='steelblue',
                alpha=0.8, edgecolor='k',
                density=True)
        ax.axvline(0, c='red', ls='--', lw=2)
        frac = float(np.mean(dS > 0))
        ax.set_xlabel('ΔS (entropy step)')
        ax.set_ylabel('density')
        ax.set_title(f'Entropy increments\n'
                     f'frac>0: {frac:.2f} '
                     f'{"✅" if frac>0.6 else "❌"}')
        ax.grid(True, alpha=0.3)

    # ── Row 2: Relational Graph ───────────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    r21 = all_results.get('exp21', {})
    if r21:
        for lbl, c in [('e6','steelblue'),
                        ('shuf','orange')]:
            ds_vals = [r['d_s'] for r in r21[lbl]
                       if not np.isnan(r['d_s'])]
            if ds_vals:
                x = np.arange(len(ds_vals))
                ax.bar(x + (0 if lbl=='e6' else 0.35),
                       ds_vals, 0.32, alpha=0.8,
                       color=c, label=lbl.upper(),
                       edgecolor='k')
        ax.axhline(3.0, c='red', ls='--', lw=1.5,
                   label='d_s=3 (hypothesis)')
        ax.set_xlabel('run')
        ax.set_ylabel('d_s (spectral dim of graph)')
        ax.set_title('Exp 2.1: Spectral dimension\n'
                     'of correlation graph E6 vs Shuf')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[2, 2])
    if r21:
        for lbl, c in [('e6','steelblue'),
                        ('shuf','orange')]:
            nc_vals = [r['n_comp']
                       for r in r21[lbl]]
            x = np.arange(len(nc_vals))
            ax.plot(x, nc_vals, 'o-', color=c,
                    lw=2, ms=6, label=lbl.upper())
    ax.set_xlabel('run'); ax.set_ylabel('n_components')
    ax.set_title('Graph connectivity\n(1=fully connected)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')

    # Summary of findings
    lines = ["UNEXPECTED FINDINGS\n" + "─"*30]
    for f in tracker.findings[:6]:
        lines.append(
            f"\n[{f['exp']}] {f['significance'][:2]}\n"
            f"{f['finding'][:40]}\n"
            f"val={str(f['value'])[:8]} "
            f"exp={str(f['expected'])[:8]}")
    txt = "\n".join(lines)
    ax.text(0.02, 0.98, txt,
            fontsize=7.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#fff9c4', alpha=0.95))
    ax.set_title('Emergent Findings')

    plt.savefig('monostring_fragmentation_v7.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v7.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = run_all()
    plot_all(all_results)
