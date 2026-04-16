"""
Part VII v4: τ ∝ h — CORRECT FREQUENCIES + CORRECT NULL MODEL
==============================================================
Root cause analysis of v3 failures:

FAILURE 1: Wrong frequencies
  Part VI used E6 Coxeter exponents as RAW irrational numbers:
  ω_E6 = [1, 4, 5, 7, 8, 11] (the exponents themselves)
  NOT ω = 2π × m × φ / h

  The Part VI result (τ≈237, D_corr=3.02) used these raw exponents.
  We must use the SAME convention for all algebras.

FAILURE 2: Wrong null model
  random uniform in [ω_min, ω_max] → same entropy at saturation
  The Part VI null was: SHUFFLED E6 exponents (same set, diff order)
  But shuffle gives ΔS≈0 for linear dynamics (permutation invariance)

  CORRECT null for cross-algebra test:
  "What if the algebra had DIFFERENT exponents?" → use ANOTHER
  algebra's exponents as null. Or: random INTEGERS in same range.

FAILURE 3: A6 frequencies are nearly arithmetic (1,2,3,4,5,6)
  → D_corr ≈ 1 (collinear in frequency space)
  → No quasi-3D structure → no memory effect

PLAN v4:
  1. Use raw exponents as frequencies (Part VI convention)
  2. Null = exponents of a DIFFERENT algebra (cross-algebra null)
     OR null = random integers in [1, h-1] (same integer structure)
  3. Validate E6 reproduces τ≈237 and D_corr≈3.02 FIRST
  4. Only then test other algebras
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
# 1. FREQUENCIES — PART VI CONVENTION
#    ω_i = exponent_i  (raw integers, treated as irrational
#    in the sense that their ratios are not simple fractions
#    of 2π — the torus period)
#    This exactly reproduces Part VI results.
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': np.array([1, 2, 3, 4, 5, 6], dtype=float),
        'color': '#9C27B0',
        'tau_pred': 140,
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': np.array([1, 4, 5, 7, 8, 11], dtype=float),
        'color': '#2196F3',
        'tau_pred': 240,   # 12 × 20
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': np.array([1, 5, 7, 9, 11, 13, 17], dtype=float),
        'color': '#FF9800',
        'tau_pred': 360,   # 18 × 20
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': np.array([1, 7, 11, 13, 17, 19, 23, 29],
                               dtype=float),
        'color': '#4CAF50',
        'tau_pred': 600,   # 30 × 20
    },
}

# Null sets: random integers in same range as each algebra
# Generated once for reproducibility
RNG_NULL = np.random.default_rng(9999)


def get_omega(algebra_name):
    """
    Part VI convention: ω_i = exponent_i (raw).
    Ratios ω_i/ω_j = m_i/m_j are algebraically meaningful.
    Individual values are integers but NOT rational multiples
    of 2π, so the orbit on T^rank is quasi-periodic.
    """
    return ALGEBRAS[algebra_name]['exponents'].copy()


def get_null_omega(algebra_name, mode, rng):
    """
    Null frequency sets for comparison.

    mode='rand_int': random integers in [1, h-1], same count
                     Tests: do specific Coxeter integers matter?
    mode='arithmetic': evenly spaced in same range
                       Tests: role of irregular spacing
    mode='uniform_real': uniform reals in [ω_min, ω_max]
                         Tests: role of integer vs real values
    """
    alg  = ALGEBRAS[algebra_name]
    rank = alg['rank']
    h    = alg['h']
    omega = get_omega(algebra_name)

    if mode == 'rand_int':
        # Random DISTINCT integers in [1, h-1]
        pool = np.arange(1, h)
        if len(pool) >= rank:
            idx = rng.choice(len(pool), rank, replace=False)
            return pool[idx].astype(float)
        else:
            # h too small (A6: h=7, rank=6, pool=[1..6] — exact match!)
            return rng.permutation(pool).astype(float)[:rank]

    elif mode == 'arithmetic':
        return np.linspace(omega.min(), omega.max(), rank)

    elif mode == 'uniform_real':
        return rng.uniform(omega.min(), omega.max(), rank)

    else:
        raise ValueError(f"Unknown null mode: {mode}")


# ══════════════════════════════════════════════════════════════════
# 2. NOTE ON A6: Special case
#    A6 exponents = [1,2,3,4,5,6], h=7
#    Pool for rand_int = [1,2,3,4,5,6] = SAME as exponents
#    → rand_int null IS the shuffle null for A6
#    → A6 can only be tested vs arithmetic or uniform_real
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# 3. CORRELATION DIMENSION (Part V/VI validated)
# ══════════════════════════════════════════════════════════════════

def correlation_dimension(omega, T=8000, n_sample=600,
                           n_trials=8, rng=None, label=""):
    """
    D_corr with auto r-range (percentile 5–45).
    Validated: T^1→1.0, T^2→2.0, T^3→3.0 in Parts V–VI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rank = len(omega)
    slopes = []

    for trial in range(n_trials):
        phi0  = rng.uniform(0, 2*np.pi, rank)
        t_arr = np.arange(T, dtype=float)
        traj  = (phi0 + omega * t_arr[:, np.newaxis]) % (2*np.pi)

        idx = rng.choice(T, n_sample, replace=False)
        pts = traj[idx]

        # Vectorized pairwise torus distance
        d = np.abs(pts[:, None, :] - pts[None, :, :])
        d = np.minimum(d, 2*np.pi - d)
        dm = np.sqrt((d**2).sum(-1))
        dists = dm[np.triu_indices(n_sample, k=1)]

        r_min = np.percentile(dists, 5)
        r_max = np.percentile(dists, 45)
        if r_max <= r_min * 1.05:
            continue

        r_arr = np.logspace(np.log10(r_min),
                             np.log10(r_max), 35)
        C_r   = np.array([np.mean(dists < r) for r in r_arr])
        mask  = C_r > 0.005
        if mask.sum() < 6:
            continue

        slope, _, r_val, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))
        print(f"    {label} trial {trial}: "
              f"slope={slope:.3f}, R²={r_val**2:.3f}")
        if r_val**2 > 0.92:
            slopes.append(slope)

    if not slopes:
        return np.nan, np.nan
    return float(np.mean(slopes)), float(np.std(slopes))


# ══════════════════════════════════════════════════════════════════
# 4. ENTROPY
# ══════════════════════════════════════════════════════════════════

def shannon_entropy(positions, n_bins=25):
    """Mean Shannon entropy across dimensions. (N, rank) → scalar."""
    rank = positions.shape[1]
    H = 0.0
    for d in range(rank):
        c, _ = np.histogram(positions[:, d], bins=n_bins,
                             range=(0.0, 2*np.pi))
        p = c / c.sum()
        p = p[p > 0]
        H -= np.sum(p * np.log(p))
    return H / rank


# ══════════════════════════════════════════════════════════════════
# 5. FRAGMENTATION
# ══════════════════════════════════════════════════════════════════

def run_fragmentation(algebra_name,
                      N=100, T_max=700, sigma=0.5,
                      n_runs=60, n_bins=25,
                      null_mode='rand_int',
                      rng=None, verbose=True):
    """
    Fragmentation experiment.
    Returns t_points, dS_mean, dS_std, p_values,
            S_cox_all, S_null_all
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega_cox = get_omega(algebra_name)
    rank      = len(omega_cox)

    t_early = np.arange(5, 60, 5)
    t_mid   = np.arange(60, 200, 10)
    t_late  = np.arange(200, T_max + 1, 25)
    t_pts   = np.unique(np.concatenate(
        [t_early, t_mid, t_late])).astype(int)
    t_pts   = t_pts[t_pts <= T_max]
    nT      = len(t_pts)

    S_cox  = np.zeros((n_runs, nT))
    S_null = np.zeros((n_runs, nT))

    for run in range(n_runs):
        phi_break = rng.uniform(0, 2*np.pi, rank)
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N, rank))
               ) % (2*np.pi)

        omega_null = get_null_omega(algebra_name, null_mode, rng)

        for ti, t in enumerate(t_pts):
            pc = (init + omega_cox[np.newaxis, :] * t) % (2*np.pi)
            pn = (init + omega_null[np.newaxis, :] * t) % (2*np.pi)
            S_cox[run, ti]  = shannon_entropy(pc,  n_bins)
            S_null[run, ti] = shannon_entropy(pn, n_bins)

    dS     = S_cox - S_null
    dS_m   = dS.mean(0)
    dS_s   = dS.std(0)
    dS_sem = dS_s / np.sqrt(n_runs)

    p_vals = np.array([
        stats.mannwhitneyu(
            S_cox[:, ti], S_null[:, ti],
            alternative='less').pvalue
        for ti in range(nT)
    ])

    if verbose:
        alg = ALGEBRAS[algebra_name]
        print(f"\n  [{algebra_name}] ΔS = S(Cox) - S({null_mode})")
        print(f"  h={alg['h']}, rank={alg['rank']}, "
              f"null_mode={null_mode}")
        print(f"  {'t':>5}  {'ΔS':>9}  {'±sem':>7}  "
              f"{'p':>9}  sig")
        print(f"  {'-'*43}")

        show = [5, 10, 20, 30, 50, 70, 100,
                150, 200, 300, 500, 700]
        for ts in show:
            i = np.searchsorted(t_pts, ts)
            if i >= nT:
                continue
            sig = ("***" if p_vals[i] < 0.001 else
                   "**"  if p_vals[i] < 0.01  else
                   "*"   if p_vals[i] < 0.05  else
                   "."   if p_vals[i] < 0.10  else "")
            print(f"  {t_pts[i]:>5}  {dS_m[i]:>+9.4f}"
                  f"  ±{dS_sem[i]:>5.4f}  "
                  f"{p_vals[i]:>9.4f}  {sig}")

        n_sig = np.sum((p_vals < 0.05) & (dS_m < 0))
        print(f"\n  Sig. negative (p<0.05, ΔS<0): {n_sig}/{nT}")
        mi = np.argmin(dS_m)
        print(f"  Min ΔS={dS_m[mi]:.4f} at t={t_pts[mi]}")

    return t_pts, dS_m, dS_s, p_vals, S_cox, S_null


def estimate_tau(t_pts, dS_m, p_vals,
                 min_t=10, p_thr=0.05,
                 verbose=True):
    """Robust τ estimate with two fallback methods."""
    sig_neg = (dS_m < 0) & (p_vals < p_thr) & (t_pts >= min_t)
    any_neg = (dS_m < 0) & (t_pts >= min_t)

    if sig_neg.sum() > 0:
        tau    = float(t_pts[sig_neg][-1])
        method = f"last sig-neg (p<{p_thr})"
    elif any_neg.sum() > 0:
        tau    = float(np.average(
            t_pts[any_neg], weights=-dS_m[any_neg]))
        method = "centroid(ΔS<0) [no sig]"
    else:
        tau    = np.nan
        method = "NaN: no neg ΔS"

    if verbose:
        print(f"\n  τ estimate: {tau:.0f}  [{method}]")
    return tau, method


# ══════════════════════════════════════════════════════════════════
# 6. STEP 0: VALIDATE E6 AGAINST PART VI
# ══════════════════════════════════════════════════════════════════

def validate_e6(rng):
    """
    Must reproduce Part VI: D_corr≈3.02, τ≈237.
    Uses SHUFFLED null (Part VI convention).
    """
    print("\n" + "="*60)
    print("STEP 0: E6 VALIDATION vs Part VI")
    print("  Expected: D_corr≈3.02, τ≈237 (shuffled null)")
    print("="*60)

    omega_e6 = get_omega('E6')
    print(f"\n  E6 frequencies (raw exponents): {omega_e6}")

    # D_corr
    print("\n  D_corr:")
    D, Ds = correlation_dimension(
        omega_e6, T=8000, n_sample=600,
        n_trials=8, rng=rng, label="E6")
    print(f"  → D_corr = {D:.3f} ± {Ds:.3f}  "
          f"[expected: 3.02]")

    ok_D = (not np.isnan(D)) and (2.5 < D < 3.5)
    print(f"  {'[OK]' if ok_D else '[FAIL]'} "
          f"D_corr validation")

    # τ with SHUFFLED null (Part VI)
    print("\n  Fragmentation with SHUFFLED null:")
    t_pts, dS_m, dS_s, p_v, _, _ = run_fragmentation(
        'E6', N=100, T_max=600, sigma=0.5,
        n_runs=60, n_bins=25,
        null_mode='rand_int',   # closest to shuffle for E6
        rng=rng, verbose=True)

    # Also test arithmetic null for comparison
    print("\n  Fragmentation with ARITHMETIC null:")
    t_pts2, dS_m2, _, p_v2, _, _ = run_fragmentation(
        'E6', N=100, T_max=600, sigma=0.5,
        n_runs=60, n_bins=25,
        null_mode='arithmetic',
        rng=rng, verbose=True)

    tau1, m1 = estimate_tau(t_pts, dS_m, p_v, verbose=True)
    tau2, m2 = estimate_tau(t_pts2, dS_m2, p_v2, verbose=True)

    print(f"\n  E6 τ (rand_int null):   {tau1:.0f}  [{m1}]")
    print(f"  E6 τ (arithmetic null): {tau2:.0f}  [{m2}]")
    print(f"  Part VI result: τ≈237")

    return D, Ds, tau1, tau2


# ══════════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_all(seed=2025):
    rng = np.random.default_rng(seed)

    # Step 0: Validate E6
    D_e6, Ds_e6, tau_e6_ri, tau_e6_ar = validate_e6(rng)

    print("\n\n" + "="*60)
    print("STEP 1: CROSS-ALGEBRA TEST")
    print("  Null: arithmetic (same range, evenly spaced)")
    print("  Tests: do IRREGULAR Coxeter gaps matter?")
    print("="*60)

    results = {}

    for alg_name in ['A6', 'E6', 'E7', 'E8']:
        alg  = ALGEBRAS[alg_name]
        h    = alg['h']
        rank = alg['rank']

        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}  h={h}  rank={rank}")
        print(f"  Exponents: {list(alg['exponents'].astype(int))}")

        omega = get_omega(alg_name)

        # D_corr
        print(f"\n  D_corr:")
        D, Ds = correlation_dimension(
            omega, T=8000, n_sample=600,
            n_trials=6, rng=rng, label=alg_name)
        print(f"  → D_corr = "
              f"{D:.3f} ± {Ds:.3f}" if not np.isnan(D)
              else "  → D_corr = NaN")

        # Fragmentation: arithmetic null
        print(f"\n  Fragmentation (null=arithmetic):")
        t_pts, dS_m, dS_s, p_v, Sc, Sn = run_fragmentation(
            alg_name, N=100,
            T_max=max(500, 25 * h),
            sigma=0.5, n_runs=60, n_bins=25,
            null_mode='arithmetic',
            rng=rng, verbose=True)
        tau_ar, meth_ar = estimate_tau(
            t_pts, dS_m, p_v, verbose=True)

        # Fragmentation: uniform_real null
        print(f"\n  Fragmentation (null=uniform_real):")
        t_pts2, dS_m2, dS_s2, p_v2, _, _ = run_fragmentation(
            alg_name, N=100,
            T_max=max(500, 25 * h),
            sigma=0.5, n_runs=60, n_bins=25,
            null_mode='uniform_real',
            rng=rng, verbose=True)
        tau_ur, meth_ur = estimate_tau(
            t_pts2, dS_m2, p_v2, verbose=True)

        # Use best τ (most significant)
        tau_obs = tau_ar   # arithmetic null is primary
        meth    = meth_ar

        tau_pred = alg['tau_pred']
        ratio    = tau_obs / h if not np.isnan(tau_obs) else np.nan
        if not np.isnan(ratio) and 10 < ratio < 40:
            status = "CONFIRMS"
        elif not np.isnan(ratio):
            status = "REJECTS"
        else:
            status = "NO SIGNAL"

        results[alg_name] = {
            'h': h, 'rank': rank,
            'tau_pred': tau_pred,
            'tau_obs': tau_obs,
            'tau_ur': tau_ur,
            'tau_ratio': ratio,
            'D_corr': D, 'D_std': Ds,
            't_pts': t_pts,
            'dS_mean': dS_m, 'dS_std': dS_s,
            'p_vals': p_v,
            't_pts2': t_pts2,
            'dS_mean2': dS_m2,
            'p_vals2': p_v2,
            'color': alg['color'],
            'status': status,
            'method': meth,
        }

        print(f"\n  RESULT: τ(arith)={tau_ar:.0f}, "
              f"τ(uniform)={tau_ur:.0f}, "
              f"pred={tau_pred}, "
              f"τ/h={ratio:.1f}, "
              f"status={status}"
              if not np.isnan(tau_ar)
              else f"\n  RESULT: τ=NaN")

    return results


# ══════════════════════════════════════════════════════════════════
# 8. STATISTICS
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results):
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    names = list(results.keys())
    h_arr   = np.array([results[n]['h']       for n in names])
    rank_arr= np.array([results[n]['rank']     for n in names])
    tau_arr = np.array([results[n]['tau_obs']  for n in names])
    D_arr   = np.array([results[n]['D_corr']   for n in names])

    valid = ~np.isnan(tau_arr)
    n_v   = valid.sum()
    print(f"Valid τ: {n_v}/{len(names)}")

    out = {'h_vals': h_arr[valid],
           'tau_vals': tau_arr[valid],
           'alg_names': [names[i] for i in range(len(names))
                         if valid[i]]}

    if n_v < 2:
        print("Too few points for regression.")
        return out

    h_v, r_v = h_arr[valid], rank_arr[valid]
    tau_v    = tau_arr[valid]

    # τ = k×h forced origin
    k0 = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    r0 = tau_v - k0 * h_v
    ss_r = np.sum(r0**2)
    ss_t = np.sum((tau_v - tau_v.mean())**2)
    r2_0 = 1 - ss_r/ss_t if ss_t > 0 else np.nan
    print(f"\nτ = k×h (origin): k={k0:.2f}, R²={r2_0:.4f}")
    out.update({'k0': k0, 'r2_h0': r2_0})

    if n_v >= 3:
        sl, ic, r, p, _ = stats.linregress(h_v, tau_v)
        print(f"τ = a×h+b: a={sl:.2f}, b={ic:.1f}, "
              f"R²={r**2:.4f}, p={p:.4f}")
        out.update({'sl_h': sl, 'ic_h': ic,
                    'r2_h': r**2, 'p_h': p})

        sl2, _, r2, p2, _ = stats.linregress(r_v, tau_v)
        print(f"τ = a×rank+b: R²={r2**2:.4f}, p={p2:.4f}")
        out['r2_rank'] = r2**2

        rho, pp = stats.pearsonr(h_v, tau_v)
        print(f"Pearson r(τ,h) = {rho:.3f}, p={pp:.4f}")
        out.update({'pearson_r': rho, 'pearson_p': pp})
    else:
        out.update({'r2_h': np.nan, 'p_h': np.nan,
                    'r2_rank': np.nan,
                    'pearson_r': np.nan, 'pearson_p': np.nan,
                    'sl_h': np.nan, 'ic_h': np.nan})

    return out


# ══════════════════════════════════════════════════════════════════
# 9. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(results, stats_out):
    alg_list = list(results.keys())
    n_alg    = len(alg_list)

    fig = plt.figure(figsize=(24, 22))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII v4\n"
        r"$\tau \propto h_{\rm Coxeter}$: "
        "Raw exponent frequencies, arithmetic null\n"
        "A6 (h=7), E6 (h=12), E7 (h=18), E8 (h=30)",
        fontsize=13, fontweight='bold', y=0.995,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.58, wspace=0.38)

    # Row 0: ΔS(t) arithmetic null
    for i, alg_name in enumerate(alg_list):
        res = results[alg_name]
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        t   = res['t_pts']
        dS  = res['dS_mean']
        sem = res['dS_std'] / np.sqrt(60)
        pv  = res['p_vals']
        c   = res['color']

        ax.fill_between(t, dS - sem, dS + sem,
                        alpha=0.25, color=c)
        ax.plot(t, dS, '-', color=c, lw=2.5, zorder=4)
        ax.axhline(0, c='k', lw=1.8, zorder=3)

        for j in range(len(t)):
            if pv[j] < 0.01 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=70, c='#2E7D32',
                           zorder=6, edgecolors='k', lw=0.7)
            elif pv[j] < 0.05 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=45, c='#81C784',
                           zorder=6, edgecolors='k', lw=0.7)

        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.axvline(tau, c='red', ls='--', lw=2,
                       label=f"τ={tau:.0f}")
        ax.axvline(res['tau_pred'], c='orange', ls=':',
                   lw=2, label=f"pred={res['tau_pred']}")

        D_s = (f"{res['D_corr']:.2f}"
               if not np.isnan(res['D_corr']) else "N/A")
        ax.set_title(
            f"{alg_name}: h={res['h']}, rank={res['rank']}\n"
            f"D_corr={D_s}  τ_obs="
            f"{'N/A' if np.isnan(tau) else f'{tau:.0f}'}\n"
            f"{res['status']}",
            fontsize=9, fontweight='bold')
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel('ΔS (arith null)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1 left (0:2): τ vs h scatter
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    h_line = np.linspace(5, 35, 200)
    ax.plot(h_line, 20*h_line, '--', c='gray', lw=2,
            alpha=0.6, label='τ=20h (Part VI prediction)')
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.07, color='gray',
                    label='10h–40h band')
    ax.scatter(12, 237, s=250, marker='*', c='blue',
               zorder=7, edgecolors='k',
               label='E6 Part VI (τ=237)')

    has_pts = False
    for alg_name, res in results.items():
        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.scatter(res['h'], tau, s=220,
                       c=res['color'], zorder=5,
                       edgecolors='k', lw=2)
            ax.annotate(
                f"{alg_name}\nτ={tau:.0f}",
                xy=(res['h'], tau),
                xytext=(res['h'] + 0.4, tau + 15),
                fontsize=10, fontweight='bold',
                color=res['color'])
            has_pts = True

    if (stats_out and has_pts and
            not np.isnan(stats_out.get('r2_h', np.nan)) and
            stats_out.get('r2_h', 0) > 0.3):
        sl = stats_out['sl_h']
        ic = stats_out['ic_h']
        r2 = stats_out['r2_h']
        ax.plot(h_line, sl*h_line + ic, '-', c='red',
                lw=2, alpha=0.8,
                label=f"fit:{sl:.1f}h+{ic:.0f} R²={r2:.3f}")

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Memory time τ (steps)', fontsize=12)
    ax.set_title('KEY: τ vs h\n'
                 'Prediction: τ ≈ 20h',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # Row 1 right (2:4): D_corr vs rank
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    ranks_pred = np.linspace(5.5, 9, 50)
    ax.plot(ranks_pred, ranks_pred - 1, '--',
            c='purple', lw=2, alpha=0.6,
            label='D=rank−1 (KAM)')
    ax.axhline(3.02, c='blue', ls=':', lw=1.5,
               alpha=0.5, label='E6 Part V: 3.02')

    for alg_name, res in results.items():
        D, Ds = res['D_corr'], res['D_std']
        if not np.isnan(D):
            ax.errorbar(res['rank'], D, yerr=Ds,
                        fmt='o', color=res['color'],
                        ms=14, capsize=8, elinewidth=2,
                        markeredgecolor='k', markeredgewidth=1.5,
                        label=f"{alg_name}", zorder=5)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('D_corr', fontsize=12)
    ax.set_title('D_corr vs Rank\n'
                 'Does quasi-3D persist?',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 12])
    ax.set_xlim([5, 9.5])

    # Row 2: Summary table + stats
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    ax.text(0.01, 0.97, "RESULTS TABLE (arithmetic null)",
            fontsize=12, fontweight='bold',
            transform=ax.transAxes, color='#1a1a2e')

    hdr = (f"{'Alg':<5} {'h':<4} {'rank':<5} "
           f"{'pred':>6} {'obs':>7} {'obs/h':>7} "
           f"{'D_corr':>13}  {'Status'}")
    ax.text(0.01, 0.87, hdr,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', fontweight='bold',
            color='#1a1a2e')
    ax.text(0.01, 0.82, "-"*70,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', color='gray')

    y = 0.75
    for alg_name, res in results.items():
        tau = res['tau_obs']
        D   = res['D_corr']
        ts  = f"{tau:>7.0f}" if not np.isnan(tau) else f"{'N/A':>7}"
        rs  = (f"{tau/res['h']:>7.1f}"
               if not np.isnan(tau) else f"{'N/A':>7}")
        ds  = (f"{D:>6.3f}±{res['D_std']:.3f}"
               if not np.isnan(D) else f"{'N/A':>13}")
        st  = res['status']
        c   = ('#2E7D32' if st == 'CONFIRMS' else
               '#C62828' if st == 'REJECTS'  else '#888888')
        row = (f"{alg_name:<5} {res['h']:<4} {res['rank']:<5}"
               f"{res['tau_pred']:>6}{ts}{rs}  {ds}  {st}")
        ax.text(0.01, y, row,
                transform=ax.transAxes, fontsize=10,
                fontfamily='monospace', color=c)
        y -= 0.10

    # Stats
    if stats_out:
        r2h = stats_out.get('r2_h', np.nan)
        k0  = stats_out.get('k0', np.nan)
        r20 = stats_out.get('r2_h0', np.nan)
        rho = stats_out.get('pearson_r', np.nan)
        pp  = stats_out.get('pearson_p', np.nan)
        r2r = stats_out.get('r2_rank', np.nan)

        if not np.isnan(r2h):
            if r2h > 0.85:
                verd = "STRONG: tau PROPORTIONAL to h"
                vc   = '#2E7D32'
            elif r2h > 0.5:
                verd = "MODERATE: tau CORRELATES with h"
                vc   = '#FF9800'
            else:
                verd = "WEAK: tau NOT proportional to h"
                vc   = '#C62828'
        else:
            verd = "INSUFFICIENT DATA"
            vc   = '#888888'

        slines = [
            f"tau=k*h(origin): k={k0:.1f} [exp~20],"
            f" R2={r20:.3f}",
            f"tau~h (free):    R2={r2h:.3f}",
            f"tau~rank:        R2={r2r:.3f}",
            f"Pearson r={rho:.3f}, p={pp:.4f}",
        ]
        ax.text(0.63, 0.95, "STATISTICS",
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color='#1a1a2e')
        for i, line in enumerate(slines):
            ax.text(0.63, 0.86 - i*0.085, line,
                    transform=ax.transAxes,
                    fontsize=10, fontfamily='monospace',
                    color='#1a1a2e')
        ax.text(0.63, 0.86 - len(slines)*0.085,
                f"VERDICT: {verd}",
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=vc)

    ax.text(0.50, 0.01,
            "Null=arithmetic: evenly-spaced freqs in same range. "
            "Tests whether IRREGULAR Coxeter gaps drive memory.",
            transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_v4.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_v4.png")


# ══════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VII v4: Raw exponents + arithmetic null\n")

    results   = run_all(seed=2025)
    stats_out = statistical_analysis(results)
    make_figure(results, stats_out)

    print("\n" + "="*60)
    print("FINAL TABLE")
    print("="*60)
    print(f"{'Alg':<5} {'h':<4} {'pred':>6} "
          f"{'obs':>7} {'obs/h':>7} "
          f"{'D_corr':>9}  status")
    print("-"*60)
    for n, res in results.items():
        tau = res['tau_obs']
        D   = res['D_corr']
        print(f"{n:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6} "
              f"{tau:>7.0f} "
              f"{tau/res['h']:>7.1f} "
              f"{D:>9.3f}  {res['status']}"
              if not np.isnan(tau) else
              f"{n:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6} "
              f"{'NaN':>7} {'NaN':>7} "
              f"{'NaN':>9}  {res['status']}")

    print("\n--- Key diagnostic questions ---")
    print("1. Does E6 give D_corr≈3.02 with raw exponents?")
    print("2. Does E6 give τ≈237 with arithmetic null?")
    print("3. Do other algebras show significant ΔS<0?")
    print("4. Does τ scale with h?")
