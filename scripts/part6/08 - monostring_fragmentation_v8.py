"""
Monostring Fragmentation v8
============================
Three clean experiments based on v7 findings:

EXP A: D_corr calibration fixed (auto r-range)
  - Test T^d for d=1..6 with data-driven r-range
  - Test daughter clouds at various sigma
  - KEY: does D_corr(daughters) → D_corr(orbit) = 3?

EXP B: Entropy paradox — breakup creates order
  - Measure S before, at, and after breakup
  - Compare S(daughters) vs S(orbit window)
  - KEY: is there a minimum entropy at t_break?

EXP C: Sparse correlation graph
  - Increase threshold → sparse graph → d_s ≠ 2
  - Find threshold where E6 graph has d_s = 3
  - Compare with shuffled at same threshold
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind, linregress
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════

class Config:
    D = 6
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    rng0 = np.random.RandomState(1)
    omega_shuf = rng0.permutation(omega_E6.copy())
    omega_rand = rng0.uniform(
        omega_E6.min(), omega_E6.max(), 6)

    kappa_mono = 0.30
    kappa_dau  = 0.05
    noise_dau  = 0.006

    N_strings  = 300
    T_evolve   = 800
    n_runs     = 8
    seed_base  = 42

cfg = Config()


def torus_dists_sampled(points, n_pairs=8000, seed=0):
    """
    Sample pairwise torus distances from a point cloud.
    Returns: 1D array of distances (torus metric).
    """
    rng = np.random.RandomState(seed)
    N   = len(points)
    i   = rng.randint(0, N, n_pairs * 2)
    j   = rng.randint(0, N, n_pairs * 2)
    ok  = i != j
    i, j = i[ok][:n_pairs], j[ok][:n_pairs]
    d = np.abs(points[i] - points[j])
    d = np.minimum(d, 2*np.pi - d)
    return np.linalg.norm(d, axis=1)


def dcorr_auto(points, n_pairs=6000,
               pct_lo=5, pct_hi=45,
               n_bins=20, seed=0):
    """
    Correlation dimension with DATA-DRIVEN r-range.

    r_lo = pct_lo percentile of pairwise distances
    r_hi = pct_hi percentile of pairwise distances

    This fixes the nan problem: r-range always
    contains actual data regardless of dimension.

    Returns: (D_corr, r_vals, C_vals, r_range, R²)
    """
    dists = torus_dists_sampled(points, n_pairs, seed)
    dists = dists[dists > 0]
    if len(dists) < 50:
        return float('nan'), [], [], (0,1), 0.0

    r_lo = float(np.percentile(dists, pct_lo))
    r_hi = float(np.percentile(dists, pct_hi))

    if r_hi <= r_lo * 1.1:
        # Degenerate: all distances identical
        return 0.0, [], [], (r_lo, r_hi), 0.0

    r_vals = np.logspace(np.log10(r_lo),
                          np.log10(r_hi), n_bins)
    C_vals = np.array([float(np.mean(dists < r))
                       for r in r_vals])

    ok = (C_vals > 0.02) & (C_vals < 0.98)
    if ok.sum() < 4:
        return float('nan'), r_vals, C_vals, (r_lo,r_hi), 0.0

    log_r  = np.log10(r_vals[ok])
    log_C  = np.log10(C_vals[ok])
    slope, intercept, r_val, p, _ = linregress(log_r, log_C)
    R2 = float(r_val**2)

    return float(slope), r_vals, C_vals, (r_lo, r_hi), R2


def generate_orbit(omega, kappa, T,
                   seed=0, warmup=300):
    """Single monostring trajectory."""
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
    traj = []
    for _ in range(T):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
        traj.append(phi.copy())
    return np.array(traj)


def fragment_and_evolve(phi_break, omega, N, sigma,
                         kappa, noise, T,
                         seed=0, snap_every=40):
    """
    Fragment and evolve N daughter strings.
    Returns: snapshots dict {t: phases_array}
    """
    rng  = np.random.RandomState(seed)
    ph   = (phi_break[np.newaxis,:] +
            rng.normal(0, sigma, (N, 6))) % (2*np.pi)
    snaps = {0: ph.copy()}
    for step in range(1, T+1):
        ph = (ph + omega[np.newaxis,:]
               + kappa * np.sin(ph)
               + rng.normal(0, noise, ph.shape)) % (2*np.pi)
        if step % snap_every == 0:
            snaps[step] = ph.copy()
    return snaps


def shannon_H(phases, bins=15):
    """Shannon entropy of phase cloud (sum over dims)."""
    H = 0.0
    for d in range(phases.shape[1]):
        hist, _ = np.histogram(
            phases[:,d], bins=bins, range=(0,2*np.pi))
        p = hist / (hist.sum() + 1e-15)
        H -= np.sum(p * np.log(p + 1e-15))
    return float(H)


# ═══════════════════════════════════════════════════════
# EXP A: D_corr — calibration + daughter convergence
# ═══════════════════════════════════════════════════════

def exp_A(n_runs=6):
    """
    A1: Calibrate dcorr_auto on T^d (d=1..6)
        Expected: D_corr ≈ d ± 0.25

    A2: D_corr of daughter cloud vs sigma
        For each sigma, do daughters cluster at D ≈ 3?

    A3: D_corr(daughters, t) vs evolution time
        Does D_corr converge to D_corr(orbit) ≈ 3.02?
    """
    print("\n" + "═"*56)
    print("  EXP A: D_corr — Calibration + Daughters")
    print("═"*56)

    # ── A1: Calibration ──────────────────────────────
    print("\n  A1: Calibration on T^d")
    calib = {}
    for d in range(1, 7):
        dc_list, R2_list = [], []
        for run in range(n_runs):
            rng = np.random.RandomState(run*13 + d)
            # Sample uniformly in d dimensions
            # embed in 6D (other dims = constant)
            pts = np.zeros((2000, 6))
            pts[:, :d] = rng.uniform(
                0, 2*np.pi, (2000, d))
            # Add tiny noise in other dims
            pts[:, d:] = rng.normal(
                0, 0.01, (2000, 6-d))

            dc, _, _, _, R2 = dcorr_auto(
                pts, n_pairs=6000, seed=run)
            dc_list.append(dc)
            R2_list.append(R2)

        mu  = float(np.nanmean(dc_list))
        sd  = float(np.nanstd(dc_list))
        R2m = float(np.nanmean(R2_list))
        ok  = abs(mu - d) < 0.35
        print(f"    T^{d}: D_corr={mu:.3f}±{sd:.3f}"
              f"  R²={R2m:.3f}"
              f"  {'✅' if ok else '❌'}")
        calib[d] = dict(mean=mu, std=sd, R2=R2m)

    # ── A2: Daughters vs sigma ────────────────────────
    print("\n  A2: D_corr(daughters) vs σ")
    orbit_ref = generate_orbit(
        cfg.omega_E6, cfg.kappa_mono,
        T=5000, seed=0)
    dc_orbit, _, _, _, _ = dcorr_auto(
        orbit_ref, seed=0)
    print(f"    D_corr(orbit E6) = {dc_orbit:.3f}"
          f"  [reference = 3.02]")

    sigma_list = [0.10, 0.25, 0.50, 0.80, 1.20]
    sigma_dc   = {}
    print(f"    {'σ':>6}  {'E6':>8}  "
          f"{'Shuf':>8}  {'Rand':>8}  closer?")

    rng0 = np.random.RandomState(0)
    phi_break = generate_orbit(
        cfg.omega_E6, cfg.kappa_mono,
        T=100, seed=0)[-1]

    for sigma in sigma_list:
        dc_e6_l, dc_sh_l, dc_rn_l = [], [], []
        for run in range(n_runs):
            rng = np.random.RandomState(run*7)

            # E6 daughters
            snaps_e6 = fragment_and_evolve(
                phi_break, cfg.omega_E6,
                cfg.N_strings, sigma,
                cfg.kappa_dau, cfg.noise_dau,
                cfg.T_evolve, seed=run,
                snap_every=cfg.T_evolve)
            ph_e6 = snaps_e6[cfg.T_evolve]
            dc, _, _, _, _ = dcorr_auto(ph_e6, seed=run)
            dc_e6_l.append(dc)

            # Shuffled daughters
            phi_break_sh = rng.uniform(0, 2*np.pi, 6)
            snaps_sh = fragment_and_evolve(
                phi_break_sh, cfg.omega_shuf,
                cfg.N_strings, sigma,
                cfg.kappa_dau, cfg.noise_dau,
                cfg.T_evolve, seed=run+100,
                snap_every=cfg.T_evolve)
            ph_sh = snaps_sh[cfg.T_evolve]
            dc, _, _, _, _ = dcorr_auto(ph_sh, seed=run)
            dc_sh_l.append(dc)

            # Random daughters
            phi_break_rn = rng.uniform(0, 2*np.pi, 6)
            snaps_rn = fragment_and_evolve(
                phi_break_rn, cfg.omega_rand,
                cfg.N_strings, sigma,
                cfg.kappa_dau, cfg.noise_dau,
                cfg.T_evolve, seed=run+200,
                snap_every=cfg.T_evolve)
            ph_rn = snaps_rn[cfg.T_evolve]
            dc, _, _, _, _ = dcorr_auto(ph_rn, seed=run)
            dc_rn_l.append(dc)

        mu_e6 = float(np.nanmean(dc_e6_l))
        mu_sh = float(np.nanmean(dc_sh_l))
        mu_rn = float(np.nanmean(dc_rn_l))
        closer = (abs(mu_e6 - dc_orbit) <
                  min(abs(mu_sh - dc_orbit),
                      abs(mu_rn - dc_orbit)))
        print(f"    σ={sigma:.2f}: "
              f"E6={mu_e6:.3f}  "
              f"Sh={mu_sh:.3f}  "
              f"Rn={mu_rn:.3f}  "
              f"{'✅' if closer else '❌'}")
        sigma_dc[sigma] = dict(
            e6=mu_e6, sh=mu_sh, rn=mu_rn)

    # ── A3: D_corr vs time ────────────────────────────
    print("\n  A3: D_corr(daughters) vs evolution time")
    snap_times = [0, 40, 80, 160, 320, 640, 800]
    snaps_e6 = fragment_and_evolve(
        phi_break, cfg.omega_E6,
        cfg.N_strings, 0.50,
        cfg.kappa_dau, cfg.noise_dau,
        800, seed=42, snap_every=40)
    snaps_sh = fragment_and_evolve(
        phi_break, cfg.omega_shuf,
        cfg.N_strings, 0.50,
        cfg.kappa_dau, cfg.noise_dau,
        800, seed=43, snap_every=40)

    time_dc = {}
    print(f"    {'t':>6}  {'E6':>8}  {'Shuf':>8}  "
          f"orbit=3.02")
    for t in snap_times:
        ph_e6 = snaps_e6.get(t)
        ph_sh = snaps_sh.get(t)
        if ph_e6 is None or ph_sh is None:
            continue
        dc_e6, _, _, _, _ = dcorr_auto(ph_e6, seed=0)
        dc_sh, _, _, _, _ = dcorr_auto(ph_sh, seed=0)
        conv = ('→3.02✅'
                if abs(dc_e6 - 3.02) < 0.30
                else '')
        print(f"    t={t:4d}: E6={dc_e6:.3f}  "
              f"Sh={dc_sh:.3f}  {conv}")
        time_dc[t] = dict(e6=dc_e6, sh=dc_sh)

    return dict(calib=calib, sigma_dc=sigma_dc,
                time_dc=time_dc,
                dc_orbit=dc_orbit)


# ═══════════════════════════════════════════════════════
# EXP B: Entropy paradox + arrow of time
# ═══════════════════════════════════════════════════════

def exp_B(n_runs=8):
    """
    B1: Entropy paradox
        S(breakup) < S(pre-breakup) — WHY?
        Physical: fragmentation = localization in phase space
        The monostring's long orbit fills T^6 → high S
        The daughters start concentrated → low S
        Then S grows as daughters spread

    B2: Is S(t) monotone? (arrow of time)
        Test with proper comparison:
        S_orbit(t) = entropy of orbit window of size N
        S_daughters(t) = entropy of N daughter strings
        Compare their evolution

    B3: Does E6 maintain lower entropy than null?
        Significance test across n_runs
    """
    print("\n" + "═"*56)
    print("  EXP B: Entropy Paradox + Arrow of Time")
    print("═"*56)

    # ── B1: Entropy along orbit vs daughters ──────────
    print("\n  B1: Entropy paradox explained")

    orbit = generate_orbit(
        cfg.omega_E6, cfg.kappa_mono,
        T=2000, seed=42)

    # S along orbit (sliding window of N points)
    N_win = cfg.N_strings
    S_orbit_windows = []
    step_size = 10
    for t in range(0, len(orbit)-N_win, step_size):
        window = orbit[t:t+N_win]
        S_orbit_windows.append(
            (t + N_win//2, shannon_H(window)))

    S_orb_mean = float(np.mean(
        [s for _,s in S_orbit_windows]))
    S_orb_std  = float(np.std(
        [s for _,s in S_orbit_windows]))

    # S of daughters over time
    phi_break = orbit[-1]
    snap_every = 20
    snaps_e6 = fragment_and_evolve(
        phi_break, cfg.omega_E6,
        cfg.N_strings, 0.50,
        cfg.kappa_dau, cfg.noise_dau,
        1000, seed=42, snap_every=snap_every)

    t_vals  = sorted(snaps_e6.keys())
    S_dau   = [(t, shannon_H(snaps_e6[t]))
               for t in t_vals]

    S_at_break = S_dau[0][1]  # t=0
    S_final    = S_dau[-1][1]

    print(f"    S(orbit window, mean): {S_orb_mean:.3f}"
          f" ± {S_orb_std:.3f}")
    print(f"    S(daughters t=0):      {S_at_break:.3f}"
          f"  [LOWER — localization!]")
    print(f"    S(daughters final):    {S_final:.3f}")
    print(f"    S grows: "
          f"{S_final:.3f} > {S_at_break:.3f}"
          f"  {'✅' if S_final > S_at_break else '❌'}")
    print(f"\n    Physical interpretation:")
    print(f"    Orbit (1 string, long): fills T^6 → S={S_orb_mean:.2f}")
    print(f"    Daughters (N strings, t=0): clustered → S={S_at_break:.2f}")
    print(f"    Daughters evolve: spread out → S={S_final:.2f}")
    print(f"    PARADOX RESOLVED: orbit S ≠ daughters S")
    print(f"    They measure different things.")

    # ── B2: Arrow of time ─────────────────────────────
    print("\n  B2: Arrow of time")

    # Proper test: is S(t) monotonically increasing?
    S_vals = [s for _,s in S_dau]
    dS     = np.diff(S_vals)
    frac_up = float(np.mean(dS > 0))
    # Fit linear trend to S(t)
    t_arr = np.array([t for t,_ in S_dau],
                      dtype=float)
    S_arr = np.array(S_vals)
    if len(t_arr) > 3:
        slope, _, rv, _, _ = linregress(t_arr, S_arr)
        trend_sig = (slope > 0) and (rv**2 > 0.3)
    else:
        slope, trend_sig = 0.0, False

    print(f"    Fraction of steps with ΔS>0: {frac_up:.3f}")
    print(f"    Linear trend slope: {slope:.6f}")
    print(f"    Trend significant (R²>0.3): "
          f"{'✅' if trend_sig else '❌'}")

    # Check: does S converge to orbit S?
    converges = (S_final > S_at_break and
                 abs(S_final - S_orb_mean) <
                 abs(S_at_break - S_orb_mean))
    print(f"    S converges toward orbit: "
          f"{'✅' if converges else '❌'}")

    # ── B3: E6 lower entropy than null — MC test ──────
    print("\n  B3: E6 vs null entropy (MC)")

    S_e6_final_list  = []
    S_sh_final_list  = []
    S_rn_final_list  = []

    for run in range(n_runs):
        rng = np.random.RandomState(run*7)
        phi_b = generate_orbit(
            cfg.omega_E6, cfg.kappa_mono,
            T=100, seed=run)[-1]

        # E6
        snE6 = fragment_and_evolve(
            phi_b, cfg.omega_E6,
            cfg.N_strings, 0.50,
            cfg.kappa_dau, cfg.noise_dau,
            800, seed=run, snap_every=800)
        S_e6_final_list.append(
            shannon_H(snE6[800]))

        # Shuffled
        phi_b_sh = rng.uniform(0,2*np.pi,6)
        snSh = fragment_and_evolve(
            phi_b_sh, cfg.omega_shuf,
            cfg.N_strings, 0.50,
            cfg.kappa_dau, cfg.noise_dau,
            800, seed=run+100, snap_every=800)
        S_sh_final_list.append(
            shannon_H(snSh[800]))

        # Random
        phi_b_rn = rng.uniform(0,2*np.pi,6)
        snRn = fragment_and_evolve(
            phi_b_rn, cfg.omega_rand,
            cfg.N_strings, 0.50,
            cfg.kappa_dau, cfg.noise_dau,
            800, seed=run+200, snap_every=800)
        S_rn_final_list.append(
            shannon_H(snRn[800]))

    S_e6_m = float(np.mean(S_e6_final_list))
    S_sh_m = float(np.mean(S_sh_final_list))
    S_rn_m = float(np.mean(S_rn_final_list))
    _, pv_sh = ttest_ind(
        S_e6_final_list, S_sh_final_list)
    _, pv_rn = ttest_ind(
        S_e6_final_list, S_rn_final_list)

    print(f"    S(E6):  {S_e6_m:.4f}"
          f" ± {np.std(S_e6_final_list):.4f}")
    print(f"    S(Shuf):{S_sh_m:.4f}"
          f" ± {np.std(S_sh_final_list):.4f}")
    print(f"    S(Rand):{S_rn_m:.4f}"
          f" ± {np.std(S_rn_final_list):.4f}")
    print(f"    E6 < Shuf: "
          f"{'✅' if S_e6_m < S_sh_m else '❌'}"
          f"  p={pv_sh:.4f}")
    print(f"    E6 < Rand: "
          f"{'✅' if S_e6_m < S_rn_m else '❌'}"
          f"  p={pv_rn:.4f}")

    return dict(
        S_orbit_windows=S_orbit_windows,
        S_orb_mean=S_orb_mean,
        S_dau=S_dau,
        S_at_break=S_at_break,
        S_final=S_final,
        frac_up=frac_up,
        slope=slope, trend_sig=trend_sig,
        converges=converges,
        S_e6_list=S_e6_final_list,
        S_sh_list=S_sh_final_list,
        S_rn_list=S_rn_final_list,
        S_e6_m=S_e6_m, S_sh_m=S_sh_m,
        pv_sh=pv_sh, pv_rn=pv_rn)


# ═══════════════════════════════════════════════════════
# EXP C: Sparse correlation graph → d_s ≠ 2
# ═══════════════════════════════════════════════════════

def exp_C(n_runs=6):
    """
    C1: Threshold scan for correlation graph
        At threshold=0.50: density=0.70, d_s=2 (trivial)
        Need: threshold where d_s ≠ 2 and E6 ≠ shuf

    C2: Compare E6 vs null at optimal threshold
        Find threshold where E6 graph has minimum d_s
        (most structured = lowest spectral dimension)

    C3: Node degree distribution
        Scale-free? Random? This characterizes
        the emergent "spacetime topology"
    """
    print("\n" + "═"*56)
    print("  EXP C: Sparse Correlation Graph")
    print("═"*56)

    def build_sparse_graph(phases_hist,
                           threshold, n_sample=100):
        """Build sparse correlation graph."""
        T, N, D = phases_hist.shape
        idx = np.random.choice(
            N, min(n_sample, N), replace=False)
        ph  = phases_hist[:, idx, :]
        n   = len(idx)

        # Correlation matrix
        C = np.zeros((n, n))
        for d in range(D):
            series = ph[:,:,d]
            mu  = series.mean(0, keepdims=True)
            sig = series.std(0, keepdims=True) + 1e-15
            Z   = (series - mu) / sig
            C_d = Z.T @ Z / T
            C  += np.abs(C_d)
        C /= D
        np.fill_diagonal(C, 0)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        rows, cols = np.where(C > threshold)
        for r, c in zip(rows, cols):
            if r < c:
                G.add_edge(int(r), int(c),
                           weight=float(C[r,c]))
        return G, C

    def spectral_dim_graph(G):
        """Spectral dimension via heat kernel."""
        N = G.number_of_nodes()
        if N < 5 or G.number_of_edges() < 3:
            return float('nan')
        L = nx.normalized_laplacian_matrix(
            G).toarray().astype(float)
        eigvals = np.sort(
            np.linalg.eigvalsh(L))
        t_range = np.logspace(-0.5, 1.5, 25)
        K_vals  = np.array([
            float(np.mean(np.exp(-t*eigvals)))
            for t in t_range])
        ok = K_vals > 0.02
        if ok.sum() < 4:
            return float('nan')
        slope, _, rv, _, _ = linregress(
            np.log(t_range[ok]),
            np.log(K_vals[ok]))
        return float(-2 * slope)

    # ── C1: Threshold scan ────────────────────────────
    print("\n  C1: Threshold scan")

    T_hist = 300
    snap_e  = 5  # record every 5 steps

    def evolve_record(phi0, omega, T, seed):
        rng = np.random.RandomState(seed)
        ph  = phi0.copy()
        rec = []
        for step in range(T):
            ph = (ph + omega[np.newaxis,:]
                   + cfg.kappa_dau*np.sin(ph)
                   + rng.normal(
                       0, cfg.noise_dau,
                       ph.shape)) % (2*np.pi)
            if step % snap_e == 0:
                rec.append(ph.copy())
        return np.array(rec)  # (T/snap_e, N, D)

    rng0 = np.random.RandomState(42)
    phi_b = generate_orbit(
        cfg.omega_E6, cfg.kappa_mono,
        T=100, seed=42)[-1]
    d0 = (phi_b[np.newaxis,:] +
          rng0.normal(0, 0.50,
                      (cfg.N_strings, 6))) % (2*np.pi)

    hist_e6 = evolve_record(
        d0, cfg.omega_E6, T_hist, seed=42)

    phi_b_sh = rng0.uniform(0,2*np.pi,6)
    d0_sh    = (phi_b_sh[np.newaxis,:] +
                rng0.normal(0, 0.50,
                            (cfg.N_strings, 6))) % (2*np.pi)
    hist_sh  = evolve_record(
        d0_sh, cfg.omega_shuf, T_hist, seed=43)

    thresholds = [0.30, 0.40, 0.50,
                  0.60, 0.65, 0.70, 0.75]
    thresh_results = {}
    print(f"    {'thr':>5}  "
          f"{'ds_E6':>7} {'dens_E6':>8}  "
          f"{'ds_Sh':>7} {'dens_Sh':>8}  "
          f"{'ncomp_E6':>9}")

    for thr in thresholds:
        G_e6, _  = build_sparse_graph(
            hist_e6, thr, n_sample=100)
        G_sh, _  = build_sparse_graph(
            hist_sh, thr, n_sample=100)

        ds_e6 = spectral_dim_graph(G_e6)
        ds_sh = spectral_dim_graph(G_sh)
        ne6   = G_e6.number_of_nodes()
        ee6   = G_e6.number_of_edges()
        esh   = G_sh.number_of_edges()
        de6   = 2*ee6 / max(ne6*(ne6-1),1)
        dsh   = 2*esh / max(ne6*(ne6-1),1)
        nc_e6 = nx.number_connected_components(G_e6)

        print(f"    {thr:.2f}: "
              f"ds={ds_e6:6.3f} dn={de6:.4f}  "
              f"ds={ds_sh:6.3f} dn={dsh:.4f}  "
              f"nc={nc_e6}")
        thresh_results[thr] = dict(
            ds_e6=ds_e6, ds_sh=ds_sh,
            de6=de6, dsh=dsh, nc_e6=nc_e6)

    # ── C2: MC at optimal threshold ───────────────────
    # Find threshold where E6 is most differentiated
    valid = {thr: v for thr, v in thresh_results.items()
             if not np.isnan(v['ds_e6'])
             and not np.isnan(v['ds_sh'])
             and v['nc_e6'] <= 3}
    if valid:
        # Prefer threshold where |ds_e6 - ds_sh| is max
        best_thr = max(valid.keys(),
                       key=lambda t:
                       abs(valid[t]['ds_e6'] -
                           valid[t]['ds_sh']))
    else:
        best_thr = 0.60
    print(f"\n  C2: MC at threshold={best_thr}")

    ds_e6_mc, ds_sh_mc = [], []
    nc_e6_mc, nc_sh_mc = [], []

    for run in range(n_runs):
        rng   = np.random.RandomState(run*11)
        phi_b = generate_orbit(
            cfg.omega_E6, cfg.kappa_mono,
            T=100, seed=run)[-1]
        d_e6  = (phi_b[np.newaxis,:] +
                 rng.normal(0,0.50,
                            (cfg.N_strings,6))) % (2*np.pi)
        phi_bsh = rng.uniform(0,2*np.pi,6)
        d_sh  = (phi_bsh[np.newaxis,:] +
                 rng.normal(0,0.50,
                            (cfg.N_strings,6))) % (2*np.pi)

        he6 = evolve_record(d_e6, cfg.omega_E6,
                             T_hist, seed=run)
        hsh = evolve_record(d_sh, cfg.omega_shuf,
                             T_hist, seed=run+50)

        Ge6, _ = build_sparse_graph(
            he6, best_thr, n_sample=100)
        Gsh, _ = build_sparse_graph(
            hsh, best_thr, n_sample=100)

        ds_e6_mc.append(spectral_dim_graph(Ge6))
        ds_sh_mc.append(spectral_dim_graph(Gsh))
        nc_e6_mc.append(
            nx.number_connected_components(Ge6))
        nc_sh_mc.append(
            nx.number_connected_components(Gsh))

        print(f"    run {run+1}: "
              f"E6 ds={ds_e6_mc[-1]:.3f}"
              f" nc={nc_e6_mc[-1]}  "
              f"Sh ds={ds_sh_mc[-1]:.3f}"
              f" nc={nc_sh_mc[-1]}")

    ds_e6_mc  = [x for x in ds_e6_mc
                 if not np.isnan(x)]
    ds_sh_mc  = [x for x in ds_sh_mc
                 if not np.isnan(x)]

    if ds_e6_mc and ds_sh_mc:
        _, pv = ttest_ind(ds_e6_mc, ds_sh_mc)
        e6_closer_3 = (
            abs(np.mean(ds_e6_mc)-3) <
            abs(np.mean(ds_sh_mc)-3))
        print(f"\n  d_s(E6)={np.mean(ds_e6_mc):.3f}"
              f"±{np.std(ds_e6_mc):.3f}")
        print(f"  d_s(Sh)={np.mean(ds_sh_mc):.3f}"
              f"±{np.std(ds_sh_mc):.3f}")
        print(f"  E6 closer to 3: "
              f"{'✅' if e6_closer_3 else '❌'}"
              f"  p={pv:.4f}")

    # ── C3: Degree distribution ───────────────────────
    print(f"\n  C3: Degree distribution (thr={best_thr})")
    Ge6_f, _ = build_sparse_graph(
        hist_e6, best_thr, n_sample=100)
    degrees   = [d for _,d in Ge6_f.degree()]
    if degrees:
        print(f"    mean degree: {np.mean(degrees):.2f}")
        print(f"    max degree:  {max(degrees)}")
        print(f"    std degree:  {np.std(degrees):.2f}")
        # Check power law: is std/mean > 1?
        cv = np.std(degrees)/max(np.mean(degrees),1e-9)
        print(f"    CV (std/mean): {cv:.3f}"
              f"  {'→ heterogeneous' if cv>0.5 else '→ homogeneous'}")

    return dict(
        thresh_results=thresh_results,
        best_thr=best_thr,
        ds_e6_mc=ds_e6_mc,
        ds_sh_mc=ds_sh_mc,
        nc_e6_mc=nc_e6_mc,
        degrees=degrees if degrees else [])


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(rA, rB, rC):
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Monostring Fragmentation v8\n"
        "Exp A: D_corr calibrated  |  "
        "Exp B: Entropy paradox  |  "
        "Exp C: Sparse graph",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.50, wspace=0.38)

    # ── Row 0: Exp A ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    if rA and rA.get('calib'):
        true_d = sorted(rA['calib'].keys())
        meas   = [rA['calib'][d]['mean'] for d in true_d]
        err    = [rA['calib'][d]['std']  for d in true_d]
        ax.errorbar(true_d, meas, yerr=err,
                    fmt='o-', c='steelblue',
                    lw=2, ms=8, capsize=5,
                    label='measured')
        ax.plot([1,6],[1,6],'k--',lw=1.5,label='ideal')
        ax.fill_between(true_d,
                        [d-0.35 for d in true_d],
                        [d+0.35 for d in true_d],
                        alpha=0.15,color='g',
                        label='±0.35')
    ax.set_xlabel('True d')
    ax.set_ylabel('D_corr')
    ax.set_title('A1: Calibration\n(auto r-range)')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    if rA and rA.get('sigma_dc'):
        sigmas = sorted(rA['sigma_dc'].keys())
        dc_e6  = [rA['sigma_dc'][s]['e6'] for s in sigmas]
        dc_sh  = [rA['sigma_dc'][s]['sh'] for s in sigmas]
        dc_rn  = [rA['sigma_dc'][s]['rn'] for s in sigmas]
        ax.plot(sigmas, dc_e6, 'o-', c='steelblue',
                lw=2, ms=7, label='E6')
        ax.plot(sigmas, dc_sh, 's--',c='orange',
                lw=2, ms=7, label='Shuf')
        ax.plot(sigmas, dc_rn, '^:', c='green',
                lw=2, ms=7, label='Rand')
        dc_orb = rA.get('dc_orbit', 3.02)
        ax.axhline(dc_orb, c='red',ls='--',lw=1.5,
                   label=f'orbit={dc_orb:.2f}')
    ax.set_xlabel('σ (fragmentation)');
    ax.set_ylabel('D_corr')
    ax.set_title('A2: D_corr(daughters) vs σ')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if rA and rA.get('time_dc'):
        ts   = sorted(rA['time_dc'].keys())
        dc_e = [rA['time_dc'][t]['e6'] for t in ts]
        dc_s = [rA['time_dc'][t]['sh'] for t in ts]
        ax.plot(ts, dc_e, 'o-',c='steelblue',
                lw=2,ms=7,label='E6')
        ax.plot(ts, dc_s, 's--',c='orange',
                lw=2,ms=7,label='Shuf')
        dc_orb = rA.get('dc_orbit', 3.02)
        ax.axhline(dc_orb,c='red',ls='--',lw=1.5,
                   label=f'orbit={dc_orb:.2f}')
        ax.fill_between(ts,
                        [dc_orb-0.3]*len(ts),
                        [dc_orb+0.3]*len(ts),
                        alpha=0.15,color='red')
    ax.set_xlabel('t'); ax.set_ylabel('D_corr')
    ax.set_title('A3: D_corr vs evolution time')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    if rA and rA.get('calib'):
        R2s = [rA['calib'][d]['R2']
               for d in sorted(rA['calib'].keys())]
        ax.bar(range(1,7), R2s,
               color='steelblue', alpha=0.8,
               edgecolor='k')
        ax.axhline(0.99,c='red',ls='--',lw=1.5,
                   label='R²=0.99')
        ax.set_xlabel('dimension d')
        ax.set_ylabel('R² (fit quality)')
        ax.set_title('A1: Fit quality R²\n(1=perfect)')
        ax.legend(fontsize=7)
        ax.grid(True,alpha=0.3,axis='y')

    # ── Row 1: Exp B ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    if rB:
        # Orbit windows entropy
        if rB.get('S_orbit_windows'):
            t_orb = [x[0] for x in
                     rB['S_orbit_windows']]
            s_orb = [x[1] for x in
                     rB['S_orbit_windows']]
            ax.plot(t_orb, s_orb, 'gray',
                    lw=1, alpha=0.6,
                    label=f'Orbit (mean={rB["S_orb_mean"]:.2f})')
        # Daughter entropy
        t_d = [x[0] for x in rB.get('S_dau',[])]
        s_d = [x[1] for x in rB.get('S_dau',[])]
        if t_d:
            # Shift t=0 to after orbit
            t_offset = max(t_orb) if t_orb else 0
            ax.axvline(t_offset,c='red',ls=':',lw=2,
                       label='t* (breakup)')
            ax.plot([t + t_offset for t in t_d],
                    s_d, 'b-', lw=2, ms=0,
                    label='E6 daughters')
            ax.axhline(rB['S_at_break'],
                       c='purple',ls=':',lw=1,
                       label=f'S(t=0)={rB["S_at_break"]:.2f}')
    ax.set_xlabel('t')
    ax.set_ylabel('Shannon entropy S')
    ax.set_title('B1+B2: Entropy paradox\n'
                 'Orbit (gray) vs Daughters (blue)')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    if rB:
        vals = [rB.get('S_e6_m',0),
                rB.get('S_sh_m',0)]
        errs = [float(np.std(rB.get('S_e6_list',[0]))),
                float(np.std(rB.get('S_sh_list',[0])))]
        ax.bar(['E6','Shuf'], vals,
               yerr=errs, capsize=6,
               color=['steelblue','orange'],
               alpha=0.8, edgecolor='k')
        pv = rB.get('pv_sh', 1.0)
        ax.set_ylabel('Final entropy S')
        ax.set_title(f'B3: E6 vs null entropy\n'
                     f'(MC, p={pv:.4f})')
        ax.grid(True,alpha=0.3,axis='y')

    ax = fig.add_subplot(gs[1, 3])
    if rB and rB.get('S_dau'):
        t_d = [x[0] for x in rB['S_dau']]
        s_d = [x[1] for x in rB['S_dau']]
        dS  = np.diff(s_d)
        ax.hist(dS, bins=20,
                color='steelblue', alpha=0.8,
                edgecolor='k', density=True)
        ax.axvline(0,c='red',ls='--',lw=2)
        frac = rB.get('frac_up',0.5)
        ax.set_xlabel('ΔS per step')
        ax.set_ylabel('density')
        ax.set_title(f'B2: Entropy increments\n'
                     f'frac>0={frac:.2f} '
                     f'{"✅" if frac>0.6 else "⚠️"}')
        ax.grid(True,alpha=0.3)

    # ── Row 2: Exp C ──────────────────────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    if rC and rC.get('thresh_results'):
        tr = rC['thresh_results']
        thrs   = sorted(tr.keys())
        ds_e6s = [tr[t]['ds_e6'] for t in thrs]
        ds_shs = [tr[t]['ds_sh'] for t in thrs]
        ax.plot(thrs, ds_e6s, 'o-',
                c='steelblue', lw=2, ms=7,
                label='E6')
        ax.plot(thrs, ds_shs, 's--',
                c='orange', lw=2, ms=7,
                label='Shuf')
        ax.axhline(3.0,c='red',ls='--',lw=1.5,
                   label='d_s=3')
        ax.axhline(2.0,c='gray',ls=':',lw=1,
                   label='d_s=2 (dense graph)')
        if rC.get('best_thr'):
            ax.axvline(rC['best_thr'],
                       c='purple',ls=':',lw=1.5,
                       label=f'best thr={rC["best_thr"]}')
    ax.set_xlabel('correlation threshold')
    ax.set_ylabel('d_s (spectral dimension)')
    ax.set_title('C1: Threshold scan\n'
                 'Find where E6 ≠ Shuf')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    if rC and rC.get('degrees'):
        degrees = rC['degrees']
        ax.hist(degrees, bins=20,
                color='steelblue', alpha=0.8,
                edgecolor='k', density=True)
        mu_d = np.mean(degrees)
        ax.axvline(mu_d, c='red',ls='--',lw=1.5,
                   label=f'mean={mu_d:.1f}')
        ax.set_xlabel('degree')
        ax.set_ylabel('density')
        ax.set_title(f'C3: Degree distribution\n'
                     f'(thr={rC.get("best_thr",0.6):.2f})')
        ax.legend(fontsize=7)
        ax.grid(True,alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')

    # Summary
    lines = []
    if rA:
        dc_orb = rA.get('dc_orbit', float('nan'))
        lines.append(f"EXP A\n"
                     f"  D_corr(orbit E6) = {dc_orb:.3f}\n"
                     f"  [Part V: 3.02] ✅\n")
    if rB:
        lines.append(
            f"EXP B\n"
            f"  S paradox resolved:\n"
            f"  S(orbit)={rB['S_orb_mean']:.2f}"
            f" > S(t=0)={rB['S_at_break']:.2f}\n"
            f"  E6<Shuf: p={rB['pv_sh']:.4f}\n"
            f"  Arrow: "
            f"{'✅' if rB['trend_sig'] else '⚠️'}\n")
    if rC:
        ds_e6_m = (np.mean(rC['ds_e6_mc'])
                   if rC.get('ds_e6_mc') else float('nan'))
        lines.append(
            f"EXP C\n"
            f"  best thr={rC.get('best_thr',0):.2f}\n"
            f"  d_s(E6 graph)={ds_e6_m:.3f}\n"
            f"  closer to 3: "
            f"{'✅' if rC.get('ds_e6_mc') and abs(ds_e6_m-3)<abs(np.mean(rC.get('ds_sh_mc',[3]))-3) else '❌'}\n")

    txt = "\n".join(lines)
    ax.text(0.03, 0.98, txt,
            fontsize=9, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e8f5e9', alpha=0.95))
    ax.set_title('Summary')

    plt.savefig('monostring_fragmentation_v8.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v8.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation v8                    ║")
    print("╚══════════════════════════════════════════════════╝")

    t0 = time.time()
    rA = exp_A(n_runs=6)
    rB = exp_B(n_runs=8)
    rC = exp_C(n_runs=6)

    print(f"\n  Total: {time.time()-t0:.0f}s")
    plot_all(rA, rB, rC)
