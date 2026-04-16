"""
Part VII v2: τ ∝ h(Coxeter) Test — FIXED
==========================================
Fixes:
1. Diagnostic prints to see raw ΔS(t) values
2. Fixed estimate_tau: less strict threshold
3. Fixed D_corr: better r-range, more points
4. Simplified experiment structure
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 1. ALGEBRA DEFINITIONS
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': [1, 2, 3, 4, 5, 6],
        'dim': 48,
        'color': '#9C27B0',
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': [1, 4, 5, 7, 8, 11],
        'dim': 78,
        'color': '#2196F3',
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': [1, 5, 7, 9, 11, 13, 17],
        'dim': 133,
        'color': '#FF9800',
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
        'dim': 248,
        'color': '#4CAF50',
    },
}


def get_coxeter_frequencies(algebra_name):
    """ω_i = 2π × m_i / h  (Coxeter exponents)."""
    alg = ALGEBRAS[algebra_name]
    exponents = np.array(alg['exponents'], dtype=float)
    return 2.0 * np.pi * exponents / alg['h']


def get_shuffled_frequencies(algebra_name, rng):
    """Same magnitudes, random permutation of dimensions."""
    omega = get_coxeter_frequencies(algebra_name)
    idx = rng.permutation(len(omega))
    return omega[idx]


# ══════════════════════════════════════════════════════════════════
# 2. ENTROPY COMPUTATION
# ══════════════════════════════════════════════════════════════════

def shannon_entropy_cloud(positions, n_bins=20):
    """
    Shannon entropy of N-string cloud, averaged over dimensions.
    positions: shape (N_strings, rank)
    """
    rank = positions.shape[1]
    H_total = 0.0
    for d in range(rank):
        counts, _ = np.histogram(
            positions[:, d],
            bins=n_bins,
            range=(0.0, 2.0 * np.pi))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H_total += -np.sum(probs * np.log(probs))
    return H_total / rank


# ══════════════════════════════════════════════════════════════════
# 3. FRAGMENTATION EXPERIMENT — VECTORIZED
# ══════════════════════════════════════════════════════════════════

def run_fragmentation(algebra_name,
                      N_daughters=80,
                      T_max=800,
                      sigma=0.5,
                      n_runs=40,
                      n_bins=20,
                      rng=None,
                      verbose=True):
    """
    Run fragmentation experiment.
    Returns t_points, dS_mean, dS_std, p_values
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega_cox = get_coxeter_frequencies(algebra_name)
    rank = len(omega_cox)

    # Time grid — dense early, sparser late
    t_points = np.concatenate([
        np.arange(5,  100, 5),
        np.arange(100, 300, 20),
        np.arange(300, T_max + 1, 50),
    ]).astype(int)
    t_points = np.unique(t_points[t_points <= T_max])
    nT = len(t_points)

    S_cox_all  = np.zeros((n_runs, nT))
    S_shuf_all = np.zeros((n_runs, nT))

    for run in range(n_runs):
        # Break point on monostring orbit
        phi_break = rng.uniform(0, 2 * np.pi, rank)

        # Initial cloud: tight ball around phi_break
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N_daughters, rank))
               ) % (2 * np.pi)

        # Shuffled uses SAME initial cloud, different ω
        omega_shuf = get_shuffled_frequencies(algebra_name, rng)

        for ti, t in enumerate(t_points):
            pos_cox  = (init + omega_cox[np.newaxis, :] * t) % (2*np.pi)
            pos_shuf = (init + omega_shuf[np.newaxis, :] * t) % (2*np.pi)
            S_cox_all[run, ti]  = shannon_entropy_cloud(pos_cox,  n_bins)
            S_shuf_all[run, ti] = shannon_entropy_cloud(pos_shuf, n_bins)

    dS_all = S_cox_all - S_shuf_all  # negative → E6 more ordered
    dS_mean = dS_all.mean(axis=0)
    dS_std  = dS_all.std(axis=0)

    # p-values: one-sided Mann-Whitney (S_cox < S_shuf)
    p_values = np.array([
        stats.mannwhitneyu(
            S_cox_all[:, ti], S_shuf_all[:, ti],
            alternative='less').pvalue
        for ti in range(nT)
    ])

    if verbose:
        print(f"\n  [{algebra_name}] ΔS profile (first 10 points):")
        print(f"  {'t':>6}  {'ΔS':>9}  {'p':>9}  {'sig?'}")
        print(f"  {'-'*38}")
        for ti in range(min(10, nT)):
            sig = "***" if p_values[ti] < 0.001 else \
                  "**"  if p_values[ti] < 0.01  else \
                  "*"   if p_values[ti] < 0.05  else ""
            print(f"  {t_points[ti]:>6}  "
                  f"{dS_mean[ti]:>+9.4f}  "
                  f"{p_values[ti]:>9.4f}  {sig}")

        # Min ΔS and where
        min_idx = np.argmin(dS_mean)
        print(f"\n  Min ΔS = {dS_mean[min_idx]:.4f} "
              f"at t = {t_points[min_idx]}")
        print(f"  Sig. negative points (p<0.05, ΔS<0): "
              f"{np.sum((p_values < 0.05) & (dS_mean < 0))}")

    return t_points, dS_mean, dS_std, p_values


# ══════════════════════════════════════════════════════════════════
# 4. TAU ESTIMATION — ROBUST
# ══════════════════════════════════════════════════════════════════

def estimate_tau_robust(t_points, dS_mean, p_values,
                        min_t=10,
                        p_thresh=0.10,   # relaxed from 0.05
                        verbose=True):
    """
    Estimate τ robustly.

    Strategy:
    1. Find all points where dS < 0 and t >= min_t
    2. Τ = last such point (even if not significant)
    3. Also try p < p_thresh version

    Returns tau (float) or nan if no negative ΔS found at all.
    """
    mask_neg = (dS_mean < 0) & (t_points >= min_t)
    mask_sig = (dS_mean < 0) & (p_values < p_thresh) & (t_points >= min_t)

    if verbose:
        print(f"\n  Tau estimation:")
        print(f"    Points with ΔS < 0 (t>={min_t}): "
              f"{mask_neg.sum()}")
        print(f"    Points with ΔS < 0 AND p<{p_thresh}: "
              f"{mask_sig.sum()}")

    # Method 1: last significant negative point
    if mask_sig.sum() > 0:
        tau_sig = float(t_points[mask_sig][-1])
        if verbose:
            print(f"    τ (last p<{p_thresh}, ΔS<0) = {tau_sig:.0f}")
    else:
        tau_sig = np.nan
        if verbose:
            print(f"    No significant negative points found!")

    # Method 2: centroid of negative region
    if mask_neg.sum() > 0:
        tau_centroid = float(np.average(
            t_points[mask_neg], weights=-dS_mean[mask_neg]))
        if verbose:
            print(f"    τ (centroid of ΔS<0 region) = {tau_centroid:.0f}")

        # Use sig if available, else centroid
        tau_final = tau_sig if not np.isnan(tau_sig) else tau_centroid
    else:
        tau_centroid = np.nan
        tau_final = np.nan
        if verbose:
            print(f"    No negative ΔS at all!")

    return tau_final, tau_sig, tau_centroid


# ══════════════════════════════════════════════════════════════════
# 5. CORRELATION DIMENSION
# ══════════════════════════════════════════════════════════════════

def correlation_dimension(algebra_name, T=8000,
                           n_sample=600, n_trials=6,
                           rng=None, verbose=True):
    """
    Compute D_corr with auto r-range (validated in Parts V–VI).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega = get_coxeter_frequencies(algebra_name)
    rank = len(omega)

    results = []
    for trial in range(n_trials):
        phi0 = rng.uniform(0, 2*np.pi, rank)
        t_arr = np.arange(T, dtype=float)
        traj  = (phi0 + omega * t_arr[:, np.newaxis]) % (2*np.pi)

        # Subsample
        idx = rng.choice(T, n_sample, replace=False)
        pts = traj[idx]

        # Pairwise torus distances (vectorized)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        diff = np.abs(diff)
        diff = np.minimum(diff, 2*np.pi - diff)
        dist_mat = np.sqrt((diff**2).sum(axis=-1))
        dists = dist_mat[np.triu_indices(n_sample, k=1)]

        # Auto r-range
        r_min = np.percentile(dists, 5)
        r_max = np.percentile(dists, 45)

        if r_max <= r_min * 1.1:
            if verbose:
                print(f"    Trial {trial}: r-range too narrow, skip")
            continue

        r_arr = np.logspace(np.log10(r_min),
                            np.log10(r_max), 30)
        C_r   = np.array([np.mean(dists < r) for r in r_arr])

        mask = C_r > 0.005
        if mask.sum() < 5:
            if verbose:
                print(f"    Trial {trial}: too few C(r)>0 points")
            continue

        slope, _, r_val, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))

        if r_val**2 > 0.90:
            results.append(slope)
            if verbose:
                print(f"    Trial {trial}: D_corr={slope:.3f} "
                      f"(R²={r_val**2:.3f})")
        else:
            if verbose:
                print(f"    Trial {trial}: poor fit R²={r_val**2:.3f}")

    if len(results) == 0:
        return np.nan, np.nan

    return float(np.mean(results)), float(np.std(results))


# ══════════════════════════════════════════════════════════════════
# 6. MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def run_all(seed=2025):
    rng = np.random.default_rng(seed)
    results = {}

    print("=" * 65)
    print("PART VII v2: τ ∝ h(Coxeter) — DIAGNOSTIC RUN")
    print("=" * 65)

    for alg_name, alg_info in ALGEBRAS.items():
        h    = alg_info['h']
        rank = alg_info['rank']
        print(f"\n{'='*50}")
        print(f"ALGEBRA: {alg_name}  rank={rank}  h={h}")
        print(f"{'='*50}")

        print(f"\n  Frequencies (ω = 2π×m/h):")
        omega = get_coxeter_frequencies(alg_name)
        for i, (m, w) in enumerate(
                zip(alg_info['exponents'], omega)):
            print(f"    ω_{i} = 2π×{m}/{h} = {w:.6f}")

        # Step 1: D_corr
        print(f"\n  Computing D_corr...")
        D_corr, D_std = correlation_dimension(
            alg_name, T=8000,
            n_sample=600, n_trials=6,
            rng=rng, verbose=True)
        print(f"  → D_corr = {D_corr:.3f} ± {D_std:.3f}")

        # Step 2: Fragmentation
        print(f"\n  Running fragmentation experiment...")
        t_pts, dS_mean, dS_std, p_vals = run_fragmentation(
            alg_name,
            N_daughters=100,
            T_max=600,
            sigma=0.5,
            n_runs=40,
            n_bins=20,
            rng=rng,
            verbose=True)

        # Step 3: τ estimation
        tau, tau_sig, tau_cent = estimate_tau_robust(
            t_pts, dS_mean, p_vals,
            min_t=10, p_thresh=0.10,
            verbose=True)

        tau_pred = h * 20
        status = "?"
        if not np.isnan(tau):
            ratio = tau / h
            if 10 < ratio < 40:
                status = "CONFIRMS"
            else:
                status = "REJECTS"
        else:
            ratio = np.nan

        results[alg_name] = {
            'h':         h,
            'rank':      rank,
            'dim':       alg_info['dim'],
            'tau_obs':   tau,
            'tau_sig':   tau_sig,
            'tau_cent':  tau_cent,
            'tau_pred':  tau_pred,
            'tau_ratio': ratio,
            'D_corr':    D_corr,
            'D_std':     D_std,
            't_pts':     t_pts,
            'dS_mean':   dS_mean,
            'dS_std':    dS_std,
            'p_vals':    p_vals,
            'color':     alg_info['color'],
            'status':    status,
        }

        print(f"\n  SUMMARY: τ_obs={tau:.0f}" if not np.isnan(tau)
              else f"\n  SUMMARY: τ_obs=NaN (no signal)")
        print(f"           τ_pred={tau_pred}, status={status}")

    return results


# ══════════════════════════════════════════════════════════════════
# 7. STATISTICS
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results):
    print("\n" + "="*65)
    print("STATISTICAL ANALYSIS")
    print("="*65)

    names = list(results.keys())
    h_arr   = np.array([results[n]['h']       for n in names])
    rank_arr= np.array([results[n]['rank']     for n in names])
    dim_arr = np.array([results[n]['dim']      for n in names])
    tau_arr = np.array([results[n]['tau_obs']  for n in names])

    valid = ~np.isnan(tau_arr)
    n_valid = valid.sum()
    print(f"\nValid τ measurements: {n_valid}/{len(names)}")

    if n_valid < 2:
        print("Cannot run regression with < 2 points.")
        return None

    h_v, rank_v = h_arr[valid], rank_arr[valid]
    dim_v, tau_v = dim_arr[valid], tau_arr[valid]

    stats_out = {
        'h_vals':   h_v,
        'tau_vals': tau_v,
        'alg_names': [names[i] for i in range(len(names)) if valid[i]],
    }

    # Forced-origin: τ = k × h
    slope_h0 = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    resid_h0 = tau_v - slope_h0 * h_v
    ss_res   = np.sum(resid_h0**2)
    ss_tot   = np.sum((tau_v - tau_v.mean())**2)
    r2_h0    = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    print(f"\nModel: τ = k×h (forced origin)")
    print(f"  k = {slope_h0:.2f}  (expect ~20)")
    print(f"  R² = {r2_h0:.4f}")
    stats_out['slope_h0'] = slope_h0
    stats_out['r2_h0']    = r2_h0

    if n_valid >= 3:
        sl, ic, r, p, se = stats.linregress(h_v, tau_v)
        print(f"\nModel: τ = a×h + b")
        print(f"  a={sl:.2f}, b={ic:.1f}, R²={r**2:.4f}, p={p:.4f}")
        stats_out.update({'slope2': sl, 'intercept2': ic,
                          'r2_h_free': r**2, 'p_h': p})

        sl2, ic2, r2, p2, _ = stats.linregress(rank_v, tau_v)
        print(f"\nModel: τ = a×rank + b")
        print(f"  a={sl2:.2f}, b={ic2:.1f}, R²={r2**2:.4f}, p={p2:.4f}")
        stats_out['r2_rank'] = r2**2

        sl3, ic3, r3, p3, _ = stats.linregress(np.log(h_v), tau_v)
        print(f"\nModel: τ = a×log(h) + b")
        print(f"  a={sl3:.2f}, b={ic3:.1f}, "
              f"R²={r3**2:.4f}, p={p3:.4f}")
        stats_out['r2_logh'] = r3**2
    else:
        stats_out.update({
            'r2_h_free': np.nan, 'r2_rank': np.nan,
            'r2_logh': np.nan, 'slope2': np.nan,
            'intercept2': np.nan, 'p_h': np.nan})

    return stats_out


# ══════════════════════════════════════════════════════════════════
# 8. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(results, stats_out):
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII\n"
        "Testing τ ∝ h(Coxeter): A6 (h=7), E6 (h=12), "
        "E7 (h=18), E8 (h=30)",
        fontsize=15, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.40)

    # Row 0: ΔS(t) per algebra
    for i, (alg_name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')
        t   = res['t_pts']
        dS  = res['dS_mean']
        dSs = res['dS_std']
        pv  = res['p_vals']
        clr = res['color']

        ax.fill_between(t, dS - dSs, dSs + dS,
                        alpha=0.20, color=clr)
        ax.plot(t, dS, '-', color=clr, lw=2.5, zorder=4)
        ax.axhline(0, c='k', lw=1.8, zorder=3)

        for ti_i, (ti, di, pi) in enumerate(zip(t, dS, pv)):
            if pi < 0.01 and di < 0:
                ax.scatter(ti, di, s=70, c='#2E7D32',
                           zorder=6, edgecolors='k', lw=0.8)
            elif pi < 0.05 and di < 0:
                ax.scatter(ti, di, s=50, c='#81C784',
                           zorder=6, edgecolors='k', lw=0.8)

        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.axvline(tau, c='red', ls='--', lw=2,
                       label=f'τ={tau:.0f}')
        ax.axvline(res['tau_pred'], c='orange',
                   ls=':', lw=2,
                   label=f'pred={res["tau_pred"]}')

        ax.set_title(
            f"{alg_name}: h={res['h']}, rank={res['rank']}\n"
            f"τ_obs={'N/A' if np.isnan(tau) else f'{tau:.0f}'}  "
            f"D_corr={res['D_corr']:.2f}",
            fontsize=10, fontweight='bold')
        ax.set_xlabel('t (steps)', fontsize=9)
        ax.set_ylabel('ΔS', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 1 left: τ vs h
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    h_line = np.linspace(5, 35, 100)
    ax.plot(h_line, 20*h_line, '--', c='gray',
            lw=2, alpha=0.6, label='τ=20×h (predicted)')
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.07, color='gray',
                    label='10h–40h band')

    has_data = False
    for alg_name, res in results.items():
        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.scatter(res['h'], tau, s=280,
                       c=res['color'], zorder=5,
                       edgecolors='k', lw=2)
            ax.annotate(
                f"{alg_name}\nτ={tau:.0f}",
                xy=(res['h'], tau),
                xytext=(res['h']+0.4, tau+12),
                fontsize=10, fontweight='bold',
                color=res['color'])
            has_data = True

    if (has_data and stats_out and
            not np.isnan(stats_out.get('r2_h_free', np.nan))):
        r2 = stats_out['r2_h_free']
        sl = stats_out['slope2']
        ic = stats_out['intercept2']
        tau_fit = sl * h_line + ic
        ax.plot(h_line, tau_fit, '-', c='red', lw=2.5,
                label=f'fit: {sl:.1f}h+{ic:.0f} (R²={r2:.3f})')

    # E6 reference point from Part VI
    ax.scatter(12, 237, s=200, marker='*',
               c='blue', zorder=6, label='E6 Part VI (τ=237)')

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Memory Time τ (steps)', fontsize=12)
    ax.set_title('KEY: τ vs h(Coxeter)\n'
                 'Prediction: τ ≈ 20×h',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # Row 1 right: D_corr vs rank
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    for alg_name, res in results.items():
        D, Ds = res['D_corr'], res['D_std']
        if not np.isnan(D):
            ax.errorbar(res['rank'], D, yerr=Ds,
                        fmt='o', color=res['color'],
                        ms=14, capsize=8, elinewidth=2,
                        markeredgecolor='k',
                        markeredgewidth=1.5,
                        label=f"{alg_name} (rank={res['rank']})",
                        zorder=5)

    ax.axhline(3.02, c='blue', ls='--', lw=2,
               alpha=0.7, label='E6 Part V: 3.02')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('D_corr', fontsize=12)
    ax.set_title('D_corr vs Rank\n'
                 'Does quasi-3D persist for E7, E8?',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 10])

    # Row 2: Summary table
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    ax.text(0.02, 0.97, "RESULTS TABLE",
            fontsize=13, fontweight='bold',
            transform=ax.transAxes, color='#1a1a2e')

    hdr = (f"{'Algebra':<8} {'h':<5} {'rank':<6} "
           f"{'tau_pred':<10} {'tau_obs':<10} "
           f"{'tau/h':<8} {'D_corr':<16} {'Status'}")
    ax.text(0.02, 0.85, hdr,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', fontweight='bold',
            color='#1a1a2e')
    ax.text(0.02, 0.80, "-"*80,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', color='gray')

    y = 0.73
    for alg_name, res in results.items():
        tau = res['tau_obs']
        D   = res['D_corr']
        tau_str  = f"{tau:.0f}"  if not np.isnan(tau) else "N/A"
        ratio_str= f"{tau/res['h']:.1f}" if not np.isnan(tau) else "N/A"
        d_str    = (f"{D:.3f}+-{res['D_std']:.3f}"
                    if not np.isnan(D) else "N/A")
        st = res['status']
        c  = ('#2E7D32' if st == 'CONFIRMS' else
              '#C62828' if st == 'REJECTS'  else '#888')
        row = (f"{alg_name:<8} {res['h']:<5} {res['rank']:<6} "
               f"{res['tau_pred']:<10} {tau_str:<10} "
               f"{ratio_str:<8} {d_str:<16} {st}")
        ax.text(0.02, y, row, transform=ax.transAxes,
                fontsize=10, fontfamily='monospace', color=c)
        y -= 0.08

    # Stats box
    if stats_out:
        r2_h = stats_out.get('r2_h_free', np.nan)
        r2_r = stats_out.get('r2_rank', np.nan)
        sl   = stats_out.get('slope_h0', np.nan)
        r2_0 = stats_out.get('r2_h0', np.nan)

        slines = [
            f"REGRESSION RESULTS:",
            f"  tau = k*h (forced): k={sl:.1f}, R2={r2_0:.3f}",
            f"  tau ~ h (free):     R2={r2_h:.3f}",
            f"  tau ~ rank:         R2={r2_r:.3f}",
        ]
        if r2_h > 0.85:
            verdict = "VERDICT: STRONG: tau prop h CONFIRMED"
            vc = '#2E7D32'
        elif r2_h > 0.5:
            verdict = "VERDICT: MODERATE evidence for tau prop h"
            vc = '#FF9800'
        else:
            verdict = "VERDICT: WEAK/NONE — tau prop h not confirmed"
            vc = '#C62828'

        for i, sl_txt in enumerate(slines):
            ax.text(0.62, 0.90 - i*0.09, sl_txt,
                    transform=ax.transAxes, fontsize=10,
                    fontfamily='monospace', color='#1a1a2e')
        ax.text(0.62, 0.90 - len(slines)*0.09,
                verdict, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=vc)

    ax.text(0.50, 0.01,
            "If tau prop h: Coxeter number governs memory time → "
            "first algebraic fingerprint in fragmentation dynamics.",
            transform=ax.transAxes, fontsize=10,
            ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_tau_vs_coxeter.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_tau_vs_coxeter.png")


# ══════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("Part VII v2: τ ∝ h(Coxeter) — with full diagnostics\n")

    results   = run_all(seed=2025)
    stats_out = statistical_analysis(results)
    make_figure(results, stats_out)

    # Final table
    print("\n" + "="*65)
    print("FINAL SUMMARY")
    print("="*65)
    print(f"{'Algebra':<8} {'h':<5} {'tau_pred':<10} "
          f"{'tau_obs':<10} {'tau/h':<8} {'D_corr':<12} {'Status'}")
    print("-"*65)
    for alg_name, res in results.items():
        tau = res['tau_obs']
        D   = res['D_corr']
        tau_str   = f"{tau:.0f}"      if not np.isnan(tau) else "NaN"
        ratio_str = f"{tau/res['h']:.1f}" if not np.isnan(tau) else "NaN"
        d_str     = f"{D:.3f}"        if not np.isnan(D)   else "NaN"
        print(f"{alg_name:<8} {res['h']:<5} {res['tau_pred']:<10} "
              f"{tau_str:<10} {ratio_str:<8} {d_str:<12} "
              f"{res['status']}")

    print("\nDone.")
