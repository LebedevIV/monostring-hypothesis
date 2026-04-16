"""
Part VII: τ ∝ h(Coxeter) Test
==============================
Tests whether the memory time τ scales with the Coxeter number h
across different Lie algebras: A6, E6, E7, E8

Null hypotheses:
H0_1: τ is the same for all algebras (τ depends only on dimension, not algebra)
H0_2: τ ∝ rank (not h)
H0_3: τ ∝ sqrt(dim) (not h)

If τ ∝ h is confirmed → first evidence that E6 algebraic structure
matters for fragmentation dynamics, not just the set of frequencies.

Author: Igor Lebedev + Claude
Date: Part VII of the Monostring Hypothesis series
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 1. LIE ALGEBRA DEFINITIONS
#    Coxeter frequencies from Coxeter exponents m_i
#    ω_i = 2π × m_i / h  (normalized to unit circle)
#    This ensures algebraic structure is encoded in frequency ratios
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    # name: (rank, coxeter_h, exponents, predicted_tau)
    'A6': {
        'rank': 6,
        'h': 7,
        'exponents': [1, 2, 3, 4, 5, 6],      # m_i for A_n: 1..n
        'dim': 48,                              # dim = n(n+2) = 6*8
        'predicted_tau': 7 * 20,               # = 140
        'color': '#9C27B0',
    },
    'E6': {
        'rank': 6,
        'h': 12,
        'exponents': [1, 4, 5, 7, 8, 11],      # standard E6 exponents
        'dim': 78,
        'predicted_tau': 12 * 20,              # = 240 ≈ 237 (observed)
        'color': '#2196F3',
    },
    'E7': {
        'rank': 7,
        'h': 18,
        'exponents': [1, 5, 7, 9, 11, 13, 17], # standard E7 exponents
        'dim': 133,
        'predicted_tau': 18 * 20,              # = 360
        'color': '#FF9800',
    },
    'E8': {
        'rank': 8,
        'h': 30,
        'exponents': [1, 7, 11, 13, 17, 19, 23, 29],  # standard E8 exponents
        'dim': 248,
        'predicted_tau': 30 * 20,             # = 600
        'color': '#4CAF50',
    },
}


def get_coxeter_frequencies(algebra_name):
    """
    Compute normalized Coxeter frequencies:
    ω_i = 2π × m_i / h

    These encode the algebraic structure through rational ratios m_i/h,
    but individual values are irrational (multiples of π).
    """
    alg = ALGEBRAS[algebra_name]
    h = alg['h']
    exponents = np.array(alg['exponents'], dtype=float)
    # Normalize: frequencies are 2π × m_i / h
    # We use the raw ratios m_i/h multiplied by a base frequency
    # to stay on T^rank
    omega = 2 * np.pi * exponents / h
    return omega


def get_shuffled_frequencies(algebra_name, rng):
    """Shuffle frequencies - destroys algebraic structure, preserves magnitudes."""
    omega = get_coxeter_frequencies(algebra_name)
    shuffled = omega.copy()
    rng.shuffle(shuffled)
    return shuffled


# ══════════════════════════════════════════════════════════════════
# 2. CORE DYNAMICS: Phase evolution on T^rank
# ══════════════════════════════════════════════════════════════════

def evolve_monostring(omega, T_steps, phi0=None, rng=None):
    """
    Evolve a single string on T^rank for T_steps.
    phi(t) = phi0 + omega * t  (mod 2π)
    Returns trajectory: shape (T_steps, rank)
    """
    rank = len(omega)
    if phi0 is None:
        phi0 = rng.uniform(0, 2*np.pi, rank)

    t_arr = np.arange(T_steps, dtype=float)
    # Broadcasting: (T, 1) * (1, rank) + (1, rank)
    traj = (phi0[np.newaxis, :] +
            omega[np.newaxis, :] * t_arr[:, np.newaxis]) % (2 * np.pi)
    return traj


def shannon_entropy_cloud(positions, n_bins=20):
    """
    Compute Shannon entropy of N-string cloud.
    positions: shape (N_strings, rank)
    Bins each dimension independently, computes H = -sum(p log p)
    """
    rank = positions.shape[1]
    H_total = 0.0

    for d in range(rank):
        counts, _ = np.histogram(positions[:, d],
                                 bins=n_bins,
                                 range=(0, 2*np.pi))
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # avoid log(0)
        H_total += -np.sum(probs * np.log(probs))

    return H_total / rank  # normalize by rank for cross-algebra comparison


def fragmentation_experiment(algebra_name,
                             N_daughters=50,
                             T_max=1500,
                             sigma=0.5,
                             n_runs=30,
                             n_bins=20,
                             rng=None):
    """
    Full fragmentation experiment for one algebra.

    Returns:
    - t_arr: time array
    - dS_mean: mean ΔS(t) = S(Coxeter) - S(Shuffled) over runs
    - dS_std: std of ΔS(t) over runs
    - p_values: Mann-Whitney p-value at each time point
    - tau_estimate: crossover time (where ΔS first crosses 0 from below)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega_cox = get_coxeter_frequencies(algebra_name)
    rank = len(omega_cox)

    # Time points to measure (log-spaced for better coverage)
    t_points = np.unique(np.concatenate([
        np.arange(10, 100, 10),
        np.arange(100, 400, 25),
        np.arange(400, 800, 50),
        np.arange(800, T_max+1, 100),
    ])).astype(int)
    t_points = t_points[t_points <= T_max]

    # Storage
    dS_runs = np.zeros((n_runs, len(t_points)))

    for run in range(n_runs):
        # Choose breakup point: random point on monostring orbit
        phi_break = rng.uniform(0, 2*np.pi, rank)

        # Birth condition: N daughters in σ-ball around phi_break
        init_cox = phi_break[np.newaxis, :] + \
                   rng.normal(0, sigma, (N_daughters, rank))
        init_cox = init_cox % (2 * np.pi)

        # Shuffled daughters: same initial positions, different frequencies
        omega_shuf = get_shuffled_frequencies(algebra_name, rng)
        init_shuf = init_cox.copy()  # SAME initial conditions

        for ti, t in enumerate(t_points):
            # Evolve each daughter independently
            # pos_i(t) = init_i + omega * t  (mod 2π)
            pos_cox = (init_cox +
                      omega_cox[np.newaxis, :] * t) % (2 * np.pi)
            pos_shuf = (init_shuf +
                       omega_shuf[np.newaxis, :] * t) % (2 * np.pi)

            S_cox  = shannon_entropy_cloud(pos_cox,  n_bins=n_bins)
            S_shuf = shannon_entropy_cloud(pos_shuf, n_bins=n_bins)

            dS_runs[run, ti] = S_cox - S_shuf

    # Statistics across runs
    dS_mean = dS_runs.mean(axis=0)
    dS_std  = dS_runs.std(axis=0)
    dS_sem  = dS_std / np.sqrt(n_runs)

    # Mann-Whitney p-values: test if S_cox < S_shuf at each time point
    # We need per-run values, so recompute split
    S_cox_runs  = np.zeros((n_runs, len(t_points)))
    S_shuf_runs = np.zeros((n_runs, len(t_points)))

    for run in range(n_runs):
        phi_break = rng.uniform(0, 2*np.pi, rank)
        init_cox = phi_break[np.newaxis, :] + \
                   rng.normal(0, sigma, (N_daughters, rank))
        init_cox = init_cox % (2 * np.pi)
        omega_shuf = get_shuffled_frequencies(algebra_name, rng)
        init_shuf = init_cox.copy()

        for ti, t in enumerate(t_points):
            pos_cox  = (init_cox +
                       omega_cox[np.newaxis,:]*t) % (2*np.pi)
            pos_shuf = (init_shuf +
                       omega_shuf[np.newaxis,:]*t) % (2*np.pi)
            S_cox_runs[run,ti]  = shannon_entropy_cloud(pos_cox,  n_bins)
            S_shuf_runs[run,ti] = shannon_entropy_cloud(pos_shuf, n_bins)

    dS_runs2 = S_cox_runs - S_shuf_runs
    dS_mean2 = dS_runs2.mean(axis=0)
    dS_std2  = dS_runs2.std(axis=0)

    p_values = np.zeros(len(t_points))
    for ti in range(len(t_points)):
        _, p = stats.mannwhitneyu(
            S_cox_runs[:, ti], S_shuf_runs[:, ti],
            alternative='less')  # H1: S_cox < S_shuf
        p_values[ti] = p

    # Estimate τ: first zero crossing from negative → positive
    # in the significant region (t > 20)
    tau_estimate = estimate_tau(t_points, dS_mean2, p_values)

    return t_points, dS_mean2, dS_std2, p_values, tau_estimate


def estimate_tau(t_points, dS_mean, p_values,
                 significance=0.05, min_t=20):
    """
    Estimate τ as: the last time point where dS < 0 AND p < significance,
    before the signal becomes noisy/positive.

    More robust: find the center of the "significant negative" region.
    """
    mask_sig_neg = (dS_mean < 0) & (p_values < significance) & \
                   (t_points >= min_t)

    if mask_sig_neg.sum() == 0:
        return np.nan

    # Last significant negative point
    last_sig = t_points[mask_sig_neg][-1]

    # Find first zero crossing after last significant point
    # Look at running mean to smooth noise
    window = 3
    dS_smooth = np.convolve(dS_mean,
                            np.ones(window)/window,
                            mode='same')

    after_last = t_points >= last_sig
    crossings = np.where(np.diff(np.sign(dS_smooth[after_last])))[0]

    if len(crossings) > 0:
        idx = np.where(after_last)[0][crossings[0]]
        tau = (t_points[idx] + t_points[min(idx+1, len(t_points)-1)]) / 2
    else:
        tau = float(last_sig)

    return tau


# ══════════════════════════════════════════════════════════════════
# 3. CORRELATION DIMENSION (from Part V/VI validated method)
# ══════════════════════════════════════════════════════════════════

def correlation_dimension_torus(omega, T=5000,
                                 n_sample=500, n_trials=5,
                                 rng=None):
    """
    Compute D_corr of the orbit on T^rank using
    data-driven percentile auto r-range (validated in Part VI).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rank = len(omega)
    results = []

    for _ in range(n_trials):
        phi0 = rng.uniform(0, 2*np.pi, rank)
        t_arr = np.arange(T, dtype=float)
        traj = (phi0[np.newaxis,:] +
                omega[np.newaxis,:] * t_arr[:,np.newaxis]) % (2*np.pi)

        # Subsample
        idx = rng.choice(T, n_sample, replace=False)
        pts = traj[idx]

        # Pairwise torus distances
        dists = _torus_distances(pts)

        # Auto r-range: 5th–45th percentile of distances
        r_min = np.percentile(dists, 5)
        r_max = np.percentile(dists, 45)

        if r_max <= r_min:
            continue

        r_arr = np.linspace(r_min, r_max, 25)
        C_r   = np.array([np.mean(dists < r) for r in r_arr])

        # log-log fit
        mask = C_r > 0.01
        if mask.sum() < 5:
            continue

        slope, _, r_val, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))

        if r_val**2 > 0.95:  # only good fits
            results.append(slope)

    if len(results) == 0:
        return np.nan, np.nan

    return np.mean(results), np.std(results)


def _torus_distances(pts):
    """Pairwise torus (T^n) distances: min(|Δφ|, 2π-|Δφ|) summed."""
    n = len(pts)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            diff = np.abs(pts[i] - pts[j])
            diff = np.minimum(diff, 2*np.pi - diff)
            dists.append(np.sqrt(np.sum(diff**2)))
    return np.array(dists)


# ══════════════════════════════════════════════════════════════════
# 4. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_all_experiments(seed=2025):
    """Run the full τ ∝ h test across all algebras."""
    rng = np.random.default_rng(seed)

    print("="*70)
    print("PART VII: τ ∝ h(Coxeter) TEST")
    print("="*70)
    print(f"{'Algebra':<8} {'rank':<6} {'h':<5} "
          f"{'h×20':<8} {'τ_obs':<10} {'D_corr':<12} {'Status'}")
    print("-"*70)

    results = {}

    for alg_name, alg_info in ALGEBRAS.items():
        print(f"\nProcessing {alg_name} "
              f"(rank={alg_info['rank']}, h={alg_info['h']})...")

        # --- Correlation dimension of orbit ---
        omega = get_coxeter_frequencies(alg_name)
        D_corr, D_std = correlation_dimension_torus(
            omega, T=5000, n_sample=400, n_trials=8, rng=rng)

        # --- Fragmentation experiment ---
        t_pts, dS_mean, dS_std, p_vals, tau_obs = \
            fragmentation_experiment(
                alg_name,
                N_daughters=60,
                T_max=1200 if alg_info['h'] <= 12 else 2000,
                sigma=0.5,
                n_runs=25,
                n_bins=18,
                rng=rng
            )

        tau_pred = alg_info['predicted_tau']
        h = alg_info['h']

        # Status
        if not np.isnan(tau_obs):
            ratio = tau_obs / h
            within_factor2 = 10 < ratio < 40  # expect ~20
            status = "✓ CONFIRMS" if within_factor2 else "✗ REJECTS"
        else:
            ratio = np.nan
            status = "? NO SIGNAL"

        results[alg_name] = {
            'h': h,
            'rank': alg_info['rank'],
            'dim': alg_info['dim'],
            'tau_obs': tau_obs,
            'tau_pred': tau_pred,
            'tau_ratio': ratio,
            'D_corr': D_corr,
            'D_std': D_std,
            't_pts': t_pts,
            'dS_mean': dS_mean,
            'dS_std': dS_std,
            'p_vals': p_vals,
            'color': alg_info['color'],
            'status': status,
        }

        print(f"{alg_name:<8} {alg_info['rank']:<6} {h:<5} "
              f"{tau_pred:<8} {tau_obs:<10.1f} "
              f"{D_corr:.3f}±{D_std:.3f}  {status}")
        print(f"         τ/h = {ratio:.1f} "
              f"(expected ~20, range 10-40)")

    return results


# ══════════════════════════════════════════════════════════════════
# 5. STATISTICAL TESTS OF τ ∝ h
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results):
    """
    Test H0: τ is independent of h.
    Test H1: τ ∝ h (linear through origin).
    Compare with alternative scalings: τ ∝ rank, τ ∝ dim.
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: τ ∝ h vs alternatives")
    print("="*70)

    alg_names = list(results.keys())
    h_vals    = np.array([results[a]['h']    for a in alg_names])
    rank_vals = np.array([results[a]['rank'] for a in alg_names])
    dim_vals  = np.array([results[a]['dim']  for a in alg_names])
    tau_vals  = np.array([results[a]['tau_obs'] for a in alg_names])
    D_vals    = np.array([results[a]['D_corr']  for a in alg_names])

    # Remove NaN
    valid = ~np.isnan(tau_vals)
    h_v, rank_v, dim_v = h_vals[valid], rank_vals[valid], dim_vals[valid]
    tau_v = tau_vals[valid]
    alg_v = [alg_names[i] for i in range(len(alg_names)) if valid[i]]

    print(f"\nValid data points: {valid.sum()}/{len(alg_names)}")

    if valid.sum() < 3:
        print("WARNING: Too few data points for regression!")
        return None

    # Model 1: τ ∝ h (forced through origin)
    slope_h = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    resid_h = tau_v - slope_h * h_v
    ss_res_h = np.sum(resid_h**2)
    ss_tot = np.sum((tau_v - tau_v.mean())**2)
    r2_h0 = 1 - ss_res_h / ss_tot  # R² for forced-origin

    # Model 2: τ = a*h + b (free intercept)
    slope2, intercept2, r2_h, p_h, se_h = stats.linregress(h_v, tau_v)

    # Model 3: τ ∝ rank
    slope_r2, intercept_r2, r2_rank, p_rank, _ = \
        stats.linregress(rank_v, tau_v)

    # Model 4: τ ∝ dim
    slope_d2, intercept_d2, r2_dim, p_dim, _ = \
        stats.linregress(dim_v, tau_v)

    print(f"\nModel 1: τ ∝ h (through origin)")
    print(f"  slope = {slope_h:.2f}  "
          f"[expected: 20.0 from Part VI E6]")
    print(f"  R² = {r2_h0:.4f}")

    print(f"\nModel 2: τ = a×h + b (free intercept)")
    print(f"  a={slope2:.2f}, b={intercept2:.1f}")
    print(f"  R²={r2_h**2:.4f}, p={p_h:.4f}")

    print(f"\nModel 3: τ = a×rank + b")
    print(f"  R²={r2_rank**2:.4f}, p={p_rank:.4f}")

    print(f"\nModel 4: τ = a×dim + b")
    print(f"  R²={r2_dim**2:.4f}, p={p_dim:.4f}")

    print(f"\nBest model by R²:")
    models = {
        'τ ∝ h (free)':   r2_h**2,
        'τ ∝ rank':       r2_rank**2,
        'τ ∝ dim':        r2_dim**2,
    }
    best = max(models, key=models.get)
    for m, r2 in sorted(models.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if m == best else ""
        print(f"  {m:<20}: R²={r2:.4f}{marker}")

    # D_corr analysis
    print(f"\nD_corr across algebras:")
    for i, a in enumerate(alg_v):
        print(f"  {a}: D_corr = {D_vals[valid][i]:.3f} "
              f"± {results[a]['D_std']:.3f}")

    return {
        'slope_h0':    slope_h,
        'r2_h0':       r2_h0,
        'r2_h_free':   r2_h**2,
        'r2_rank':     r2_rank**2,
        'r2_dim':      r2_dim**2,
        'p_h':         p_h,
        'best_model':  best,
        'h_vals':      h_v,
        'tau_vals':    tau_v,
        'alg_names':   alg_v,
        'slope2':      slope2,
        'intercept2':  intercept2,
    }


# ══════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def make_figure(results, stats_out):
    """Create the Part VII summary figure."""

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII\n"
        "Testing τ ∝ h(Coxeter) Across Lie Algebras: "
        "A6, E6, E7, E8",
        fontsize=16, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.50, wspace=0.42)

    # ── Row 0: ΔS(t) profiles for each algebra ────────────────
    for i, (alg_name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        t   = res['t_pts']
        dS  = res['dS_mean']
        dSs = res['dS_std']
        pv  = res['p_vals']
        clr = res['color']

        ax.fill_between(t, dS - dSs, dS + dSs,
                        alpha=0.25, color=clr)
        ax.plot(t, dS, '-', color=clr, lw=2.5,
                label=f'{alg_name}', zorder=4)
        ax.axhline(0, c='k', lw=1.5, zorder=3)

        # Significant points
        for ti_idx, (ti, di, pi) in \
                enumerate(zip(t, dS, pv)):
            if pi < 0.01 and di < 0:
                ax.scatter(ti, di, s=80,
                           c='#2E7D32', zorder=6,
                           edgecolors='k', lw=1)
            elif pi < 0.01:
                ax.scatter(ti, di, s=80,
                           c='#C62828', zorder=6,
                           edgecolors='k', lw=1)

        # Mark τ
        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.axvline(tau, c='red', ls='--', lw=2,
                       label=f'τ={tau:.0f}')
            ax.axvline(res['tau_pred'],
                       c='orange', ls=':', lw=2,
                       label=f'pred={res["tau_pred"]:.0f}')

        h = res['h']
        rank = res['rank']
        ax.set_title(
            f'{alg_name}: rank={rank}, h={h}\n'
            f'τ_obs={tau:.0f} | τ_pred={res["tau_pred"]}',
            fontsize=10, fontweight='bold')
        ax.set_xlabel('t (steps)', fontsize=9)
        ax.set_ylabel('ΔS = S(Cox)−S(Shuf)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Row 1, col 0-1: τ vs h scatter ──────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    h_all   = [ALGEBRAS[a]['h'] for a in results]
    tau_all = [results[a]['tau_obs'] for a in results]
    clrs    = [results[a]['color'] for a in results]
    names   = list(results.keys())

    # Prediction line: τ = 20 × h
    h_line = np.linspace(5, 35, 100)
    ax.plot(h_line, 20 * h_line, '--',
            c='gray', lw=2, alpha=0.7,
            label='τ = 20×h (prediction from E6)',
            zorder=2)
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.08, color='gray',
                    label='×2 band (10h–40h)')

    for h_v, tau_v, c, n in zip(h_all, tau_all, clrs, names):
        if not np.isnan(tau_v):
            ax.scatter(h_v, tau_v, s=300, c=c,
                       zorder=5, edgecolors='k',
                       linewidth=2)
            ax.annotate(f'{n}\n(τ={tau_v:.0f})',
                        xy=(h_v, tau_v),
                        xytext=(h_v + 0.5, tau_v + 15),
                        fontsize=10, fontweight='bold',
                        color=c)

    # Fit line if stats available
    if stats_out and stats_out['r2_h_free'] > 0.5:
        h_fit = stats_out['h_vals']
        tau_fit = (stats_out['slope2'] * h_line +
                   stats_out['intercept2'])
        ax.plot(h_line, tau_fit, '-',
                c='red', lw=2.5, alpha=0.8,
                label=f"fit: τ={stats_out['slope2']:.1f}h"
                      f"+{stats_out['intercept2']:.0f}"
                      f" (R²={stats_out['r2_h_free']:.3f})")

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Memory Time τ (steps)', fontsize=12)
    ax.set_title('KEY RESULT: τ vs h(Coxeter)\n'
                 'Tests if τ ∝ h across algebras',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # ── Row 1, col 2-3: D_corr vs rank ──────────────────────
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    rank_all  = [ALGEBRAS[a]['rank']         for a in results]
    Dcorr_all = [results[a]['D_corr']        for a in results]
    Dstd_all  = [results[a]['D_std']         for a in results]

    for rank_v, Dc, Ds, c, n in zip(
            rank_all, Dcorr_all, Dstd_all, clrs, names):
        if not np.isnan(Dc):
            ax.errorbar(rank_v, Dc, yerr=Ds,
                        fmt='o', color=c, ms=14,
                        capsize=8, elinewidth=2,
                        markeredgecolor='k',
                        markeredgewidth=1.5,
                        label=f'{n} (rank={rank_v})',
                        zorder=5)

    # Reference: D_corr should be ~3 for irrational Coxeter freqs
    ax.axhline(3.02, c='blue', ls='--', lw=2,
               alpha=0.7, label='D_corr=3.02 (Part V, E6)')
    ax.axhline(3.0, c='blue', ls=':', lw=1,
               alpha=0.4)

    ax.set_xlabel('Lie algebra rank', fontsize=12)
    ax.set_ylabel('D_corr (correlation dimension)', fontsize=12)
    ax.set_title('D_corr vs Rank\n'
                 'Does D_corr ≈ 3 generalize?',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 8])

    # ── Row 2: Summary scorecard + statistics ─────────────────
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    # Left: Results table
    ax.text(0.02, 0.95,
            "RESULTS SUMMARY",
            fontsize=13, fontweight='bold',
            transform=ax.transAxes, color='#1a1a2e')

    header = f"{'Algebra':<8} {'h':<5} {'rank':<6} " \
             f"{'τ_pred':<9} {'τ_obs':<10} " \
             f"{'τ/h':<7} {'D_corr':<14} {'Verdict'}"
    ax.text(0.02, 0.82, header,
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            color='#1a1a2e', fontweight='bold')
    ax.text(0.02, 0.77, "-"*85,
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            color='gray')

    y_pos = 0.70
    for alg_name, res in results.items():
        tau_obs = res['tau_obs']
        tau_str = f"{tau_obs:.0f}" if not np.isnan(tau_obs) else "N/A"
        ratio_str = f"{res['tau_obs']/res['h']:.1f}" \
                    if not np.isnan(tau_obs) else "N/A"
        d_str = f"{res['D_corr']:.3f}+-{res['D_std']:.3f}" \
                if not np.isnan(res['D_corr']) else "N/A"

        row = (f"{alg_name:<8} {res['h']:<5} {res['rank']:<6} "
               f"{res['tau_pred']:<9} {tau_str:<10} "
               f"{ratio_str:<7} {d_str:<14} {res['status']}")

        color = '#2E7D32' if 'CONFIRMS' in res['status'] else \
                '#C62828' if 'REJECTS' in res['status'] else '#777'

        ax.text(0.02, y_pos, row,
                transform=ax.transAxes,
                fontsize=9, fontfamily='monospace',
                color=color)
        y_pos -= 0.07

    # Right: Statistical conclusion
    if stats_out:
        concl_x = 0.62
        ax.text(concl_x, 0.95,
                "STATISTICAL CONCLUSION",
                fontsize=13, fontweight='bold',
                transform=ax.transAxes, color='#1a1a2e')

        lines = [
            f"Best model: {stats_out['best_model']}",
            f"  R²(τ~h) = {stats_out['r2_h_free']:.4f}",
            f"  R²(τ~rank) = {stats_out['r2_rank']:.4f}",
            f"  R²(τ~dim) = {stats_out['r2_dim']:.4f}",
            "",
            f"τ ∝ h (forced origin):",
            f"  slope = {stats_out['slope_h0']:.2f}",
            f"  [E6 gives slope=237/12=19.75~20]",
            f"  R² = {stats_out['r2_h0']:.4f}",
            "",
            "VERDICT:",
        ]

        if stats_out['r2_h_free'] > 0.85:
            verdict = "STRONG EVIDENCE: τ ∝ h confirmed!"
            v_color = '#2E7D32'
        elif stats_out['r2_h_free'] > 0.6:
            verdict = "MODERATE EVIDENCE: τ ∝ h plausible"
            v_color = '#FF9800'
        else:
            verdict = "WEAK/NO EVIDENCE: τ ∝ h rejected"
            v_color = '#C62828'

        for i, line in enumerate(lines):
            ax.text(concl_x, 0.82 - i*0.065, line,
                    transform=ax.transAxes,
                    fontsize=10, fontfamily='monospace',
                    color='#1a1a2e')

        ax.text(concl_x, 0.82 - len(lines)*0.065,
                verdict,
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color=v_color)

    # Bottom interpretation
    ax.text(0.50, 0.02,
            "If τ ∝ h is confirmed: Memory Time is set by the "
            "algebraic structure of the frequency set (Coxeter number),\n"
            "not just by the rank or dimension. "
            "This would be the FIRST evidence that Lie algebra identity "
            "affects fragmentation dynamics.",
            transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_tau_vs_coxeter.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_tau_vs_coxeter.png")


# ══════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("\nPart VII: τ ∝ h(Coxeter) Test")
    print("Algebras: A6 (h=7), E6 (h=12), E7 (h=18), E8 (h=30)")
    print("Prediction from Part VI: τ ≈ 20 × h\n")

    # Run experiments
    results = run_all_experiments(seed=2025)

    # Statistical analysis
    stats_out = statistical_analysis(results)

    # Figure
    make_figure(results, stats_out)

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if stats_out:
        r2 = stats_out['r2_h_free']
        slope = stats_out['slope_h0']

        print(f"\nτ ∝ h hypothesis:")
        print(f"  R² = {r2:.4f}")
        print(f"  slope(forced) = {slope:.2f} "
              f"(expected ~20)")

        if r2 > 0.85 and 10 < slope < 35:
            print("\n[CONFIRMED] τ ∝ h with slope ≈ 20")
            print("  → Coxeter number governs memory time")
            print("  → Lie algebra structure affects dynamics")
            print("  → First algebraic fingerprint in fragmentation")
        elif r2 > 0.5:
            print("\n[PARTIAL] τ correlates with h but not cleanly")
            print("  → Possible, but not definitive")
            print("  → Check: is τ ∝ h or τ ∝ some other invariant?")
        else:
            print("\n[REJECTED] τ does NOT scale with h")
            print("  → Memory time is not set by Coxeter number")
            print("  → Coincidence for E6")

    print("\nOpen questions for Part VIII:")
    print("  1. If τ ∝ h: WHY? Poincaré recurrence time?")
    print("  2. Does D_corr ≈ 3 hold for E7 (rank 7), E8 (rank 8)?")
    print("  3. What sets D_corr for higher-rank algebras?")
    print("  4. Is τ the Poincaré recurrence time of the orbit?")
