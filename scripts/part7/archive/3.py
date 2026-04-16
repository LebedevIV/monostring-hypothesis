"""
Part VII v3: τ ∝ h(Coxeter) — PHYSICALLY CORRECT
==================================================
Root cause fixes:

FIX 1: D_corr failure
  ω_i = 2π×m_i/h are RATIONAL multiples of 2π
  → orbit is PERIODIC with period h steps
  → Grassberger-Procaccia sees a closed curve, not fractal
  Solution: add irrational base frequency ω_base = sqrt(2) or φ
  ω_i = 2π × m_i × φ / h  (φ=golden ratio → irrational)
  This preserves RATIOS m_i/h (algebraic structure)
  while making individual frequencies irrational (quasi-periodic)

FIX 2: ΔS ≈ 0 signal failure
  Shuffling permutes dimensions → same set of ω values
  → for linear dynamics, entropy is permutation-invariant
  Solution: null model = RANDOM frequencies from same range
  (not shuffle). This is physically correct: we test whether
  E6-specific ratios matter vs. arbitrary irrational frequencies.

FIX 3: Validate against Part VI
  E6 must reproduce τ ≈ 237 before testing other algebras.
  If E6 fails → experiment is broken regardless.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 1. ALGEBRA DEFINITIONS — IRRATIONAL COXETER FREQUENCIES
#    ω_i = 2π × m_i × α / h
#    where α = golden ratio φ = (1+√5)/2 makes them irrational
#    Ratios ω_i/ω_j = m_i/m_j preserved → algebraic structure intact
# ══════════════════════════════════════════════════════════════════

PHI = (1.0 + np.sqrt(5.0)) / 2.0   # golden ratio — irrational

ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': [1, 2, 3, 4, 5, 6],
        'color': '#9C27B0',
        'tau_pred': 140,
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': [1, 4, 5, 7, 8, 11],
        'color': '#2196F3',
        'tau_pred': 240,
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': [1, 5, 7, 9, 11, 13, 17],
        'color': '#FF9800',
        'tau_pred': 360,
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
        'color': '#4CAF50',
        'tau_pred': 600,
    },
}


def get_frequencies(algebra_name, mode='coxeter'):
    """
    mode='coxeter' : ω_i = 2π × m_i × φ / h  (irrational, preserves ratios)
    mode='random'  : ω_i uniform in [ω_min, ω_max] of Coxeter set
    mode='shuffle' : random permutation of Coxeter ω (same magnitudes)
    """
    alg = ALGEBRAS[algebra_name]
    exponents = np.array(alg['exponents'], dtype=float)
    h = alg['h']
    # Irrational base: φ makes ω_i = 2π×m_i×φ/h irrational
    omega_cox = 2.0 * np.pi * exponents * PHI / h
    return omega_cox


def get_null_frequencies(omega_cox, mode, rng):
    """
    Generate null-model frequencies for comparison.

    mode='random': uniform in [ω_min, ω_max] — tests if specific
                   E6 ratios matter vs. arbitrary irrationals
    mode='shuffle': permutation — tests if ORDER of assignment matters
    mode='arithmetic': linearly spaced — tests role of irrationality
    """
    n = len(omega_cox)
    if mode == 'random':
        return rng.uniform(omega_cox.min(), omega_cox.max(), n)
    elif mode == 'shuffle':
        idx = rng.permutation(n)
        return omega_cox[idx]
    elif mode == 'arithmetic':
        return np.linspace(omega_cox.min(), omega_cox.max(), n)
    else:
        raise ValueError(f"Unknown null mode: {mode}")


# ══════════════════════════════════════════════════════════════════
# 2. VALIDATE: Check orbit is quasi-periodic
# ══════════════════════════════════════════════════════════════════

def check_orbit_quasiperiodic(algebra_name, T=500):
    """
    Quick sanity check: does orbit return near start?
    Quasi-periodic → T_rec >> T_max
    Periodic → returns exactly after h steps
    """
    omega = get_frequencies(algebra_name)
    phi0  = np.zeros(len(omega))  # start at origin

    phi_T = (phi0 + omega * T) % (2 * np.pi)
    dist  = np.sqrt(np.sum((phi_T - phi0)**2))

    # For periodic: dist should be ~0 at t = h
    phi_h = (phi0 + omega * ALGEBRAS[algebra_name]['h']) % (2*np.pi)
    dist_h = np.sqrt(np.sum((phi_h - phi0)**2))

    print(f"    Orbit check:")
    print(f"      dist(t=0, t={T}) = {dist:.4f}  "
          f"[quasi-periodic → large]")
    print(f"      dist(t=0, t=h={ALGEBRAS[algebra_name]['h']}) "
          f"= {dist_h:.4f}  [periodic → ~0]")
    return dist > 0.1  # True = quasi-periodic (good)


# ══════════════════════════════════════════════════════════════════
# 3. CORRELATION DIMENSION — with orbit validation
# ══════════════════════════════════════════════════════════════════

def correlation_dimension(algebra_name, T=10000,
                           n_sample=800, n_trials=8,
                           rng=None, verbose=True):
    """
    D_corr with auto r-range (Parts V–VI validated).
    Now uses irrational frequencies → quasi-periodic orbit.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega = get_frequencies(algebra_name)
    rank  = len(omega)

    slopes = []
    for trial in range(n_trials):
        phi0  = rng.uniform(0, 2*np.pi, rank)
        t_arr = np.arange(T, dtype=float)
        traj  = (phi0 + omega * t_arr[:, np.newaxis]) % (2*np.pi)

        # Subsample
        idx = rng.choice(T, n_sample, replace=False)
        pts = traj[idx]

        # Vectorized pairwise torus distances
        diff = np.abs(pts[:, np.newaxis, :] - pts[np.newaxis, :, :])
        diff = np.minimum(diff, 2*np.pi - diff)
        dist_mat = np.sqrt((diff**2).sum(axis=-1))
        dists = dist_mat[np.triu_indices(n_sample, k=1)]

        r_min = np.percentile(dists, 5)
        r_max = np.percentile(dists, 45)

        if r_max <= r_min * 1.05:
            continue

        r_arr = np.logspace(np.log10(r_min),
                            np.log10(r_max), 35)
        C_r   = np.array([np.mean(dists < r) for r in r_arr])

        mask = C_r > 0.005
        if mask.sum() < 6:
            continue

        slope, _, r_val, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))

        if verbose:
            print(f"    Trial {trial}: slope={slope:.3f}, "
                  f"R²={r_val**2:.3f}")

        if r_val**2 > 0.85:
            slopes.append(slope)

    if len(slopes) == 0:
        return np.nan, np.nan
    return float(np.mean(slopes)), float(np.std(slopes))


# ══════════════════════════════════════════════════════════════════
# 4. SHANNON ENTROPY
# ══════════════════════════════════════════════════════════════════

def shannon_entropy(positions, n_bins=25):
    """
    Mean Shannon entropy across dimensions.
    positions: (N, rank)
    """
    rank = positions.shape[1]
    H = 0.0
    for d in range(rank):
        counts, _ = np.histogram(
            positions[:, d], bins=n_bins,
            range=(0.0, 2.0 * np.pi))
        p = counts / counts.sum()
        p = p[p > 0]
        H += -np.sum(p * np.log(p))
    return H / rank


# ══════════════════════════════════════════════════════════════════
# 5. FRAGMENTATION EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_fragmentation(algebra_name,
                      N_daughters=100,
                      T_max=1000,
                      sigma=0.5,
                      n_runs=50,
                      n_bins=25,
                      null_mode='random',
                      rng=None,
                      verbose=True):
    """
    Fragmentation experiment with RANDOM null model.

    Null model: random frequencies in same range as Coxeter.
    This tests: do specific E6 ratios matter vs. arbitrary irrationals?
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega_cox = get_frequencies(algebra_name)
    rank      = len(omega_cox)

    # Dense early time grid, sparser later
    t_early = np.arange(5, 80, 5)
    t_mid   = np.arange(80, 300, 15)
    t_late  = np.arange(300, T_max + 1, 40)
    t_points = np.unique(
        np.concatenate([t_early, t_mid, t_late])
    ).astype(int)
    t_points = t_points[t_points <= T_max]
    nT = len(t_points)

    S_cox_all  = np.zeros((n_runs, nT))
    S_null_all = np.zeros((n_runs, nT))

    for run in range(n_runs):
        phi_break = rng.uniform(0, 2*np.pi, rank)

        # Initial cloud: tight σ-ball
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N_daughters, rank))
               ) % (2 * np.pi)

        # Null frequencies
        omega_null = get_null_frequencies(
            omega_cox, null_mode, rng)

        for ti, t in enumerate(t_points):
            pos_cox  = (init
                        + omega_cox[np.newaxis, :] * t
                       ) % (2 * np.pi)
            pos_null = (init
                        + omega_null[np.newaxis, :] * t
                       ) % (2 * np.pi)

            S_cox_all[run, ti]  = shannon_entropy(pos_cox,  n_bins)
            S_null_all[run, ti] = shannon_entropy(pos_null, n_bins)

    dS_all  = S_cox_all - S_null_all
    dS_mean = dS_all.mean(axis=0)
    dS_std  = dS_all.std(axis=0)
    dS_sem  = dS_std / np.sqrt(n_runs)

    # Mann-Whitney p-values (one-sided: S_cox < S_null)
    p_values = np.array([
        stats.mannwhitneyu(
            S_cox_all[:, ti], S_null_all[:, ti],
            alternative='less').pvalue
        for ti in range(nT)
    ])

    if verbose:
        print(f"\n  [{algebra_name}] ΔS = S(Cox) - S({null_mode})")
        print(f"  {'t':>5}  {'ΔS':>9}  {'±sem':>8}  "
              f"{'p':>9}  sig")
        print(f"  {'-'*45}")
        # Print t=5,10,20,30,40,50,70,100,150,200,300,500
        show_t = [5, 10, 20, 30, 40, 50, 70,
                  100, 150, 200, 300, 500]
        for t_show in show_t:
            idx = np.searchsorted(t_points, t_show)
            if idx >= nT:
                continue
            t_  = t_points[idx]
            dS_ = dS_mean[idx]
            sem_= dS_sem[idx]
            p_  = p_values[idx]
            sig = ("***" if p_ < 0.001 else
                   "**"  if p_ < 0.01  else
                   "*"   if p_ < 0.05  else
                   "."   if p_ < 0.10  else "")
            print(f"  {t_:>5}  {dS_:>+9.4f}  "
                  f"±{sem_:>6.4f}  {p_:>9.4f}  {sig}")

        n_sig_neg = np.sum((p_values < 0.05) & (dS_mean < 0))
        print(f"\n  Significant negative (p<0.05, ΔS<0): "
              f"{n_sig_neg}/{nT}")
        min_idx = np.argmin(dS_mean)
        print(f"  Min ΔS={dS_mean[min_idx]:.4f} "
              f"at t={t_points[min_idx]}")
        max_neg_p_idx = (
            np.where((dS_mean < 0) & (p_values < 0.05))[0]
        )
        if len(max_neg_p_idx) > 0:
            print(f"  Significant negative region: "
                  f"t={t_points[max_neg_p_idx[0]]}–"
                  f"{t_points[max_neg_p_idx[-1]]}")

    return t_points, dS_mean, dS_std, p_values, S_cox_all, S_null_all


def estimate_tau(t_points, dS_mean, p_values,
                 min_t=10, p_thresh=0.05,
                 verbose=True):
    """
    τ = last time point in the significant-negative region.
    Falls back to centroid if no significant points.
    """
    mask_sig_neg = ((dS_mean < 0) &
                    (p_values < p_thresh) &
                    (t_points >= min_t))
    mask_any_neg = (dS_mean < 0) & (t_points >= min_t)

    if verbose:
        print(f"\n  τ estimation (p_thresh={p_thresh}):")
        print(f"    Sig. negative points: {mask_sig_neg.sum()}")
        print(f"    Any negative points:  {mask_any_neg.sum()}")

    if mask_sig_neg.sum() > 0:
        tau = float(t_points[mask_sig_neg][-1])
        method = "last sig-neg"
    elif mask_any_neg.sum() > 0:
        # Weighted centroid
        tau = float(np.average(
            t_points[mask_any_neg],
            weights=-dS_mean[mask_any_neg]))
        method = "centroid (no sig)"
    else:
        tau = np.nan
        method = "NaN (no negative ΔS)"

    if verbose:
        print(f"    τ = {tau:.0f}  [{method}]")

    return tau, method


# ══════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_all(seed=2025):
    rng = np.random.default_rng(seed)
    results = {}

    print("=" * 65)
    print("PART VII v3: τ ∝ h — PHYSICALLY CORRECT EXPERIMENT")
    print(f"Frequencies: ω_i = 2π × m_i × φ / h  (φ={PHI:.6f})")
    print("Null model: RANDOM frequencies in [ω_min, ω_max]")
    print("=" * 65)

    # Validation reference point: E6 should give τ ≈ 237
    print("\n=== VALIDATION: E6 reference (must give τ≈237) ===")
    t_v, dS_v, _, pv_v, _, _ = run_fragmentation(
        'E6', N_daughters=100, T_max=600,
        sigma=0.5, n_runs=50, n_bins=25,
        null_mode='random', rng=rng, verbose=True)
    tau_E6_val, meth = estimate_tau(t_v, dS_v, pv_v,
                                    verbose=True)
    print(f"\n  E6 validation τ = {tau_E6_val:.0f} "
          f"(expected: ~237, Part VI)")
    if not np.isnan(tau_E6_val) and 150 < tau_E6_val < 350:
        print("  [OK] E6 validation PASSED")
    else:
        print("  [WARN] E6 validation outside expected range!")
    print()

    # Full experiment
    for alg_name in ['A6', 'E6', 'E7', 'E8']:
        alg_info = ALGEBRAS[alg_name]
        h    = alg_info['h']
        rank = alg_info['rank']

        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}  rank={rank}  h={h}")
        print(f"  Coxeter exponents: {alg_info['exponents']}")
        print(f"  τ predicted: {alg_info['tau_pred']} "
              f"(= {h} × 20)")

        # Orbit sanity check
        is_qp = check_orbit_quasiperiodic(alg_name)
        print(f"    → {'Quasi-periodic ✓' if is_qp else 'PERIODIC! Fix needed'}")

        # D_corr
        print(f"\n  D_corr (irrational frequencies):")
        D, Ds = correlation_dimension(
            alg_name, T=10000, n_sample=800,
            n_trials=8, rng=rng, verbose=True)
        print(f"  → D_corr = {D:.3f} ± {Ds:.3f}")

        # Fragmentation
        print(f"\n  Fragmentation (null=random):")
        T_max = min(200 * h, 1500)   # scale T_max with h
        t_pts, dS_m, dS_s, pv, S_cx, S_nl = run_fragmentation(
            alg_name,
            N_daughters=100,
            T_max=T_max,
            sigma=0.5,
            n_runs=50,
            n_bins=25,
            null_mode='random',
            rng=rng,
            verbose=True)

        tau, method = estimate_tau(
            t_pts, dS_m, pv,
            min_t=10, p_thresh=0.05,
            verbose=True)

        tau_pred = alg_info['tau_pred']
        if not np.isnan(tau):
            ratio = tau / h
            if 10 < ratio < 40:
                status = "CONFIRMS τ∝h"
            else:
                status = "REJECTS τ∝h"
        else:
            ratio = np.nan
            status = "NO SIGNAL"

        results[alg_name] = {
            'h':         h,
            'rank':      rank,
            'tau_pred':  tau_pred,
            'tau_obs':   tau,
            'tau_ratio': ratio,
            'tau_method':method,
            'D_corr':    D,
            'D_std':     Ds,
            't_pts':     t_pts,
            'dS_mean':   dS_m,
            'dS_std':    dS_s,
            'p_vals':    pv,
            'color':     alg_info['color'],
            'status':    status,
        }

        print(f"\n  RESULT: τ_obs={tau:.0f}, "
              f"τ_pred={tau_pred}, "
              f"τ/h={ratio:.1f}, "
              f"status={status}"
              if not np.isnan(tau)
              else f"\n  RESULT: NO SIGNAL, status={status}")

    return results


# ══════════════════════════════════════════════════════════════════
# 7. STATISTICS
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results):
    print("\n" + "="*65)
    print("STATISTICAL ANALYSIS: τ ∝ h vs alternatives")
    print("="*65)

    names   = list(results.keys())
    h_arr   = np.array([results[n]['h']      for n in names])
    rank_arr= np.array([results[n]['rank']   for n in names])
    tau_arr = np.array([results[n]['tau_obs']for n in names])

    valid = ~np.isnan(tau_arr)
    n_v = valid.sum()
    print(f"Valid data points: {n_v}/{len(names)}")

    if n_v < 2:
        print("Cannot run regression.")
        return None

    h_v   = h_arr[valid]
    r_v   = rank_arr[valid]
    tau_v = tau_arr[valid]

    # Model 1: τ = k×h (forced origin)
    k0 = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    res0 = tau_v - k0 * h_v
    ss_res = np.sum(res0**2)
    ss_tot = np.sum((tau_v - tau_v.mean())**2)
    r2_0 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    print(f"\nModel τ=k×h (origin): k={k0:.2f}, R²={r2_0:.4f}")
    print(f"  [k=20 predicted; τ=20h]")

    out = {'k0': k0, 'r2_h0': r2_0,
           'h_vals': h_v, 'tau_vals': tau_v,
           'alg_names': [names[i] for i in range(len(names))
                         if valid[i]]}

    if n_v >= 3:
        sl, ic, r, p, _ = stats.linregress(h_v, tau_v)
        print(f"\nModel τ=a×h+b: "
              f"a={sl:.2f}, b={ic:.1f}, "
              f"R²={r**2:.4f}, p={p:.4f}")
        out.update({'sl_h': sl, 'ic_h': ic,
                    'r2_h': r**2, 'p_h': p})

        sl2, ic2, r2, p2, _ = stats.linregress(r_v, tau_v)
        print(f"Model τ=a×rank+b: "
              f"R²={r2**2:.4f}, p={p2:.4f}")
        out['r2_rank'] = r2**2

        # Pearson for 4 points
        rho, p_rho = stats.pearsonr(h_v, tau_v)
        print(f"\nPearson r(τ,h) = {rho:.4f}, p={p_rho:.4f}")
        out['pearson_r'] = rho
        out['pearson_p'] = p_rho
    else:
        out.update({'sl_h': np.nan, 'ic_h': np.nan,
                    'r2_h': np.nan, 'p_h': np.nan,
                    'r2_rank': np.nan,
                    'pearson_r': np.nan, 'pearson_p': np.nan})

    return out


# ══════════════════════════════════════════════════════════════════
# 8. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(results, stats_out):
    alg_list = list(results.keys())

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII v3\n"
        r"Testing $\tau \propto h_{\rm Coxeter}$ "
        "across A6, E6, E7, E8\n"
        r"Frequencies: $\omega_i = 2\pi m_i \varphi / h$ "
        r"($\varphi$ = golden ratio, irrational)",
        fontsize=14, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.38)

    # Row 0: ΔS(t) per algebra
    for i, alg_name in enumerate(alg_list):
        res = results[alg_name]
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        t  = res['t_pts']
        dS = res['dS_mean']
        ds = res['dS_std'] / np.sqrt(50)   # SEM
        pv = res['p_vals']
        c  = res['color']

        ax.fill_between(t, dS - ds, dS + ds,
                        alpha=0.25, color=c)
        ax.plot(t, dS, '-', color=c, lw=2.5, zorder=4,
                label=alg_name)
        ax.axhline(0, c='k', lw=1.8, zorder=3)

        for ti_i in range(len(t)):
            if pv[ti_i] < 0.01 and dS[ti_i] < 0:
                ax.scatter(t[ti_i], dS[ti_i], s=70,
                           c='#2E7D32', zorder=6,
                           edgecolors='k', lw=0.8)
            elif pv[ti_i] < 0.05 and dS[ti_i] < 0:
                ax.scatter(t[ti_i], dS[ti_i], s=50,
                           c='#81C784', zorder=6,
                           edgecolors='k', lw=0.8)

        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.axvline(tau, c='red', ls='--', lw=2,
                       label=f'τ={tau:.0f}')
        ax.axvline(res['tau_pred'], c='orange',
                   ls=':', lw=2,
                   label=f"pred={res['tau_pred']}")

        D_str = (f"{res['D_corr']:.2f}"
                 if not np.isnan(res['D_corr']) else "N/A")
        ax.set_title(
            f"{alg_name} (h={res['h']}, rank={res['rank']})\n"
            f"τ={'N/A' if np.isnan(tau) else f'{tau:.0f}'}  "
            f"D_corr={D_str}\n"
            f"Status: {res['status']}",
            fontsize=9, fontweight='bold')
        ax.set_xlabel('t (steps)', fontsize=9)
        ax.set_ylabel('ΔS = S(Cox)−S(rand)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1 left: τ vs h
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    h_line = np.linspace(5, 35, 200)
    ax.plot(h_line, 20*h_line, '--', c='gray', lw=2,
            alpha=0.6, label='τ = 20×h (predicted from E6)')
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.07, color='gray',
                    label='10h – 40h band')

    # Part VI reference
    ax.scatter(12, 237, s=200, marker='*', c='blue',
               zorder=7, label='E6 Part VI (τ=237)',
               edgecolors='k')

    for alg_name, res in results.items():
        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.scatter(res['h'], tau, s=220,
                       c=res['color'], zorder=5,
                       edgecolors='k', lw=2)
            ax.annotate(
                f"{alg_name}\nτ={tau:.0f}\n"
                f"({res['status']})",
                xy=(res['h'], tau),
                xytext=(res['h']+0.5, tau+10),
                fontsize=9, fontweight='bold',
                color=res['color'])

    if (stats_out and
            not np.isnan(stats_out.get('r2_h', np.nan)) and
            stats_out['r2_h'] > 0.3):
        sl = stats_out['sl_h']
        ic = stats_out['ic_h']
        r2 = stats_out['r2_h']
        ax.plot(h_line, sl*h_line + ic, '-',
                c='red', lw=2, alpha=0.8,
                label=f'fit: {sl:.1f}h+{ic:.0f} R²={r2:.3f}')

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Memory time τ (steps)', fontsize=12)
    ax.set_title('KEY RESULT: τ vs h\n'
                 'Prediction: τ ≈ 20h',
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
                        label=f"{alg_name}",
                        zorder=5)

    # Reference line: rank-1 (KAM theory prediction)
    ranks_line = np.linspace(5.5, 9, 50)
    ax.plot(ranks_line, ranks_line - 1, '--',
            c='purple', lw=2, alpha=0.6,
            label='D=rank−1 (KAM)')
    ax.axhline(3.02, c='blue', ls=':', lw=1.5,
               alpha=0.5, label='E6 Part V: 3.02')

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('D_corr', fontsize=12)
    ax.set_title('D_corr vs Rank\n'
                 r'Does $D_{corr}\approx$ rank−1?',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 10])
    ax.set_xlim([5, 9.5])

    # Row 2: Summary
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    # Table
    ax.text(0.01, 0.97, "RESULTS TABLE",
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            color='#1a1a2e')
    hdr = (f"{'Alg':<5} {'h':<4} {'rank':<5} "
           f"{'pred':>6} {'obs':>7} {'obs/h':>7} "
           f"{'D_corr':>12}  {'Status'}")
    ax.text(0.01, 0.87, hdr,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace',
            fontweight='bold', color='#1a1a2e')
    ax.text(0.01, 0.82, "-"*72,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', color='gray')

    y = 0.76
    for alg_name, res in results.items():
        tau = res['tau_obs']
        D   = res['D_corr']
        ts  = f"{tau:>7.0f}" if not np.isnan(tau) else f"{'N/A':>7}"
        rs  = f"{tau/res['h']:>7.1f}" if not np.isnan(tau) else f"{'N/A':>7}"
        ds  = (f"{D:>6.3f}±{res['D_std']:.3f}"
               if not np.isnan(D) else f"{'N/A':>12}")
        st  = res['status']
        c   = ('#2E7D32' if 'CONFIRMS' in st else
               '#C62828' if 'REJECTS'  in st else '#888888')
        row = (f"{alg_name:<5} {res['h']:<4} {res['rank']:<5} "
               f"{res['tau_pred']:>6}{ts}{rs} "
               f"{ds}  {st}")
        ax.text(0.01, y, row, transform=ax.transAxes,
                fontsize=10, fontfamily='monospace', color=c)
        y -= 0.09

    # Stats box
    if stats_out:
        r2h = stats_out.get('r2_h', np.nan)
        k0  = stats_out.get('k0', np.nan)
        r2_0= stats_out.get('r2_h0', np.nan)
        rho = stats_out.get('pearson_r', np.nan)
        pp  = stats_out.get('pearson_p', np.nan)

        if r2h > 0.85:
            verdict = "STRONG: tau PROPORTIONAL to h"
            vc = '#2E7D32'
        elif r2h > 0.5:
            verdict = "MODERATE: tau correlates with h"
            vc = '#FF9800'
        elif not np.isnan(r2h):
            verdict = "WEAK: tau NOT proportional to h"
            vc = '#C62828'
        else:
            verdict = "INSUFFICIENT DATA"
            vc = '#888888'

        slines = [
            f"tau = k*h (forced): k={k0:.1f} [exp:20], R2={r2_0:.3f}",
            f"tau ~ h (free):     R2={r2h:.3f}",
            f"tau ~ rank:         "
            f"R2={stats_out.get('r2_rank', np.nan):.3f}",
            f"Pearson r(tau,h) = {rho:.3f}, p = {pp:.3f}",
            "",
            f"VERDICT: {verdict}",
        ]
        ax.text(0.62, 0.95, "STATISTICS",
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color='#1a1a2e')
        for i, line in enumerate(slines):
            c_line = vc if 'VERDICT' in line else '#1a1a2e'
            fw = 'bold' if 'VERDICT' in line else 'normal'
            ax.text(0.62, 0.87 - i*0.085, line,
                    transform=ax.transAxes,
                    fontsize=10, fontfamily='monospace',
                    color=c_line, fontweight=fw)

    ax.text(0.50, 0.01,
            "Physical prediction: τ ≈ 20h  "
            "(from Part VI: E6 h=12 → τ≈237≈20×12)  "
            "| Null: τ independent of h",
            transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_v3.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_v3.png")


# ══════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VII v3: Physically correct τ ∝ h test\n")
    print(f"Key fix: ω_i = 2π × m_i × φ / h  (φ={PHI:.6f})")
    print("Null model: random frequencies in [ω_min, ω_max]\n")

    results   = run_all(seed=2025)
    stats_out = statistical_analysis(results)
    make_figure(results, stats_out)

    print("\n" + "="*65)
    print("FINAL TABLE")
    print("="*65)
    print(f"{'Alg':<5} {'h':<4} {'pred':>6} "
          f"{'obs':>7} {'obs/h':>7} "
          f"{'D_corr':>10}  status")
    print("-"*65)
    for alg_name, res in results.items():
        tau = res['tau_obs']
        D   = res['D_corr']
        ts  = f"{tau:.0f}" if not np.isnan(tau) else "NaN"
        rs  = f"{tau/res['h']:.1f}" if not np.isnan(tau) else "NaN"
        ds  = f"{D:.3f}" if not np.isnan(D) else "NaN"
        print(f"{alg_name:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6} {ts:>7} {rs:>7} "
              f"{ds:>10}  {res['status']}")
    print("\nDone.")
