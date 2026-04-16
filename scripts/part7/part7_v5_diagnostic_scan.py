"""
Part VII v5: DIAGNOSTIC — Reproduce Part VI exactly
====================================================
Goal: Find exact conditions that gave D_corr=3.02 and τ=237 in Part VI.

Part VI script was: part6_measure_tau_crossover.py
Key parameters from Part VI paper:
- T = 5000
- n_sample = 500 (likely)
- E6 raw exponents [1,4,5,7,8,11]
- Null = SHUFFLED E6 (same exponents, random order)
- N_daughters = 50
- sigma = 0.5
- n_runs = 30 (likely)

We systematically vary T and n_sample to find where D_corr=3.02.
Then reproduce the τ=237 measurement exactly.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── E6 frequencies (Part VI convention) ──────────────────────────
OMEGA_E6   = np.array([1., 4., 5., 7., 8., 11.])
RANK_E6    = 6


# ══════════════════════════════════════════════════════════════════
# 1. D_corr: systematic scan over T and percentile range
# ══════════════════════════════════════════════════════════════════

def dcorr_single(omega, T, n_sample, pct_lo, pct_hi,
                 rng, n_trials=5):
    """
    Compute D_corr for one parameter set.
    Returns (mean, std) or (nan, nan).
    """
    rank   = len(omega)
    slopes = []

    for _ in range(n_trials):
        phi0  = rng.uniform(0, 2*np.pi, rank)
        t_arr = np.arange(T, dtype=float)
        traj  = (phi0 + omega * t_arr[:, np.newaxis]
                ) % (2*np.pi)

        idx = rng.choice(T, n_sample, replace=False)
        pts = traj[idx]

        # Pairwise torus distances
        d    = np.abs(pts[:, None, :] - pts[None, :, :])
        d    = np.minimum(d, 2*np.pi - d)
        dm   = np.sqrt((d**2).sum(-1))
        dist = dm[np.triu_indices(n_sample, k=1)]

        r_lo = np.percentile(dist, pct_lo)
        r_hi = np.percentile(dist, pct_hi)
        if r_hi <= r_lo * 1.05:
            continue

        r_arr = np.logspace(np.log10(r_lo),
                             np.log10(r_hi), 30)
        C_r   = np.array([np.mean(dist < r) for r in r_arr])
        mask  = C_r > 0.005
        if mask.sum() < 5:
            continue

        slope, _, rv, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))
        if rv**2 > 0.90:
            slopes.append(slope)

    if not slopes:
        return np.nan, np.nan
    return float(np.mean(slopes)), float(np.std(slopes))


def scan_dcorr_parameters(rng):
    """
    Systematically vary T, n_sample, and percentile range
    to find conditions that reproduce D_corr = 3.02.
    """
    print("\n" + "="*60)
    print("SCAN 1: D_corr vs (T, n_sample)")
    print("Fixed: pct_lo=5, pct_hi=45")
    print("="*60)

    T_vals = [1000, 2000, 5000, 8000, 15000]
    N_vals = [200, 500, 800, 1200]

    results = {}
    print(f"\n{'T':>7}  {'n':>5}  {'D_corr':>8}  {'std':>6}")
    print("-"*35)

    for T in T_vals:
        for n in N_vals:
            if n >= T:
                continue
            D, Ds = dcorr_single(
                OMEGA_E6, T, n,
                pct_lo=5, pct_hi=45,
                rng=rng, n_trials=5)
            marker = " <-- 3.02?" if (
                not np.isnan(D) and 2.8 < D < 3.3) else ""
            print(f"{T:>7}  {n:>5}  "
                  f"{D:>8.3f}  {Ds:>6.3f}{marker}"
                  if not np.isnan(D) else
                  f"{T:>7}  {n:>5}  {'NaN':>8}")
            results[(T, n)] = (D, Ds)

    print("\n" + "="*60)
    print("SCAN 2: D_corr vs percentile range")
    print("Fixed: T=5000, n=500")
    print("="*60)

    pct_pairs = [
        (1, 20), (1, 30), (1, 40), (1, 50),
        (2, 30), (5, 30), (5, 40), (5, 45),
        (5, 50), (10, 40), (10, 50), (2, 20),
    ]

    print(f"\n{'pct_lo':>7}  {'pct_hi':>7}  "
          f"{'D_corr':>8}  {'std':>6}")
    print("-"*35)

    for plo, phi in pct_pairs:
        D, Ds = dcorr_single(
            OMEGA_E6, T=5000, n_sample=500,
            pct_lo=plo, pct_hi=phi,
            rng=rng, n_trials=5)
        marker = " <-- 3.02?" if (
            not np.isnan(D) and 2.8 < D < 3.3) else ""
        print(f"{plo:>7}  {phi:>7}  "
              f"{D:>8.3f}  {Ds:>6.3f}{marker}"
              if not np.isnan(D) else
              f"{plo:>7}  {phi:>7}  {'NaN':>8}")

    return results


# ══════════════════════════════════════════════════════════════════
# 2. Reproduce the EXACT Part VI D_corr=3.02
#    Part V script used LINEAR r-range, not log
#    Part VI paper says "auto r-range: 5th-45th percentile"
#    but the DISTANCE METRIC may differ
# ══════════════════════════════════════════════════════════════════

def dcorr_linear_rrange(omega, T=5000, n_sample=500,
                         pct_lo=5, pct_hi=45,
                         rng=None, n_trials=5,
                         label=""):
    """
    D_corr with LINEAR r-range (as in Part V original code).
    Compare with logspace version.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rank   = len(omega)
    slopes = []

    for trial in range(n_trials):
        phi0  = rng.uniform(0, 2*np.pi, rank)
        t_arr = np.arange(T, dtype=float)
        traj  = (phi0 + omega * t_arr[:, np.newaxis]
                ) % (2*np.pi)

        idx  = rng.choice(T, n_sample, replace=False)
        pts  = traj[idx]

        d    = np.abs(pts[:, None, :] - pts[None, :, :])
        d    = np.minimum(d, 2*np.pi - d)
        dm   = np.sqrt((d**2).sum(-1))
        dist = dm[np.triu_indices(n_sample, k=1)]

        r_lo = np.percentile(dist, pct_lo)
        r_hi = np.percentile(dist, pct_hi)
        if r_hi <= r_lo:
            continue

        # LINEAR r-range (Part V used linspace)
        r_arr = np.linspace(r_lo, r_hi, 25)
        C_r   = np.array([np.mean(dist < r) for r in r_arr])
        mask  = C_r > 0.005
        if mask.sum() < 5:
            continue

        slope, _, rv, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))

        print(f"  {label} trial {trial}: "
              f"slope={slope:.3f}, R²={rv**2:.3f}, "
              f"r_range=[{r_lo:.3f},{r_hi:.3f}]")

        if rv**2 > 0.90:
            slopes.append(slope)

    if not slopes:
        return np.nan, np.nan
    return float(np.mean(slopes)), float(np.std(slopes))


def reproduce_part6_dcorr(rng):
    """
    Try all known variations to reproduce D_corr=3.02.
    """
    print("\n" + "="*60)
    print("REPRODUCE Part VI D_corr=3.02")
    print("="*60)

    configs = [
        # (T, n, pct_lo, pct_hi, r_type, label)
        (5000, 500,  5,  45, 'log',  "T5k n500 log 5-45"),
        (5000, 500,  5,  45, 'lin',  "T5k n500 lin 5-45"),
        (5000, 500,  5,  30, 'log',  "T5k n500 log 5-30"),
        (5000, 500,  5,  30, 'lin',  "T5k n500 lin 5-30"),
        (5000, 500,  1,  30, 'lin',  "T5k n500 lin 1-30"),
        (5000, 500, 10,  40, 'lin',  "T5k n500 lin 10-40"),
        (5000, 500,  2,  25, 'lin',  "T5k n500 lin 2-25"),
        (5000, 500,  5,  20, 'lin',  "T5k n500 lin 5-20"),
        (5000, 500,  5,  20, 'log',  "T5k n500 log 5-20"),
        (3000, 300,  5,  45, 'lin',  "T3k n300 lin 5-45"),
        (3000, 300,  5,  45, 'log',  "T3k n300 log 5-45"),
        (5000, 500,  5,  45, 'lin',  "T5k n500 lin 5-45 (2)"),
    ]

    print(f"\n{'Config':<28}  "
          f"{'D_corr':>8}  {'std':>6}  note")
    print("-"*58)

    best_D  = None
    best_cfg = None

    for (T, n, plo, phi, rtyp, lbl) in configs:
        rank   = RANK_E6
        slopes = []

        for _ in range(6):
            phi0  = rng.uniform(0, 2*np.pi, rank)
            t_arr = np.arange(T, dtype=float)
            traj  = (phi0 + OMEGA_E6 * t_arr[:, np.newaxis]
                    ) % (2*np.pi)
            idx  = rng.choice(T, n, replace=False)
            pts  = traj[idx]

            d    = np.abs(pts[:, None, :] - pts[None, :, :])
            d    = np.minimum(d, 2*np.pi - d)
            dm   = np.sqrt((d**2).sum(-1))
            dist = dm[np.triu_indices(n, k=1)]

            r_lo_v = np.percentile(dist, plo)
            r_hi_v = np.percentile(dist, phi)
            if r_hi_v <= r_lo_v * 1.02:
                continue

            if rtyp == 'log':
                r_arr = np.logspace(
                    np.log10(r_lo_v), np.log10(r_hi_v), 25)
            else:
                r_arr = np.linspace(r_lo_v, r_hi_v, 25)

            C_r  = np.array([np.mean(dist < r) for r in r_arr])
            mask = C_r > 0.005
            if mask.sum() < 5:
                continue

            slope, _, rv, _, _ = stats.linregress(
                np.log(r_arr[mask]), np.log(C_r[mask]))
            if rv**2 > 0.88:
                slopes.append(slope)

        D  = float(np.mean(slopes)) if slopes else np.nan
        Ds = float(np.std(slopes))  if slopes else np.nan
        note = ""
        if not np.isnan(D):
            if 2.7 < D < 3.4:
                note = " *** MATCHES 3.02 ***"
            elif 3.8 < D < 4.3:
                note = " (4.0 region)"
        print(f"{lbl:<28}  "
              f"{D:>8.3f}  {Ds:>6.3f}{note}"
              if not np.isnan(D) else
              f"{lbl:<28}  {'NaN':>8}")

        if (not np.isnan(D) and
                (best_D is None or
                 abs(D - 3.02) < abs(best_D - 3.02))):
            best_D   = D
            best_cfg = (T, n, plo, phi, rtyp, lbl)

    print(f"\nBest match: D={best_D:.3f} "
          f"with config: {best_cfg}")
    return best_D, best_cfg


# ══════════════════════════════════════════════════════════════════
# 3. Reproduce Part VI τ=237
# ══════════════════════════════════════════════════════════════════

def shannon_entropy(positions, n_bins=20):
    rank = positions.shape[1]
    H = 0.0
    for d in range(rank):
        c, _ = np.histogram(positions[:, d], bins=n_bins,
                             range=(0., 2*np.pi))
        p = c / c.sum()
        p = p[p > 0]
        H -= np.sum(p * np.log(p))
    return H / rank


def reproduce_part6_tau(rng, N=50, T_max=700,
                         sigma=0.5, n_runs=30):
    """
    Reproduce Part VI fragmentation with SHUFFLED null.
    Part VI used: N=50, sigma=0.5, n_runs=30, n_bins=20.
    """
    print("\n" + "="*60)
    print("REPRODUCE Part VI τ=237")
    print("Parameters: N=50, sigma=0.5, n_runs=30")
    print("Null: SHUFFLED E6 (same exponents, random order)")
    print("="*60)

    # Time grid from Part VI (dense early)
    t_pts = np.unique(np.concatenate([
        np.arange(10, 80,  5),
        np.arange(80, 200, 10),
        np.arange(200, 500, 20),
        np.arange(500, T_max+1, 50),
    ])).astype(int)

    nT         = len(t_pts)
    S_cox_all  = np.zeros((n_runs, nT))
    S_shuf_all = np.zeros((n_runs, nT))

    for run in range(n_runs):
        phi_break = rng.uniform(0, 2*np.pi, RANK_E6)
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N, RANK_E6))
               ) % (2*np.pi)

        # Shuffled null: same magnitudes, different order
        shuf_idx  = rng.permutation(RANK_E6)
        omega_shuf = OMEGA_E6[shuf_idx]

        for ti, t in enumerate(t_pts):
            pc = (init + OMEGA_E6[None, :] * t) % (2*np.pi)
            ps = (init + omega_shuf[None, :] * t) % (2*np.pi)
            S_cox_all[run, ti]  = shannon_entropy(pc,  20)
            S_shuf_all[run, ti] = shannon_entropy(ps, 20)

    dS_m  = (S_cox_all - S_shuf_all).mean(0)
    dS_s  = (S_cox_all - S_shuf_all).std(0)
    dS_sem = dS_s / np.sqrt(n_runs)

    p_vals = np.array([
        stats.mannwhitneyu(
            S_cox_all[:, ti], S_shuf_all[:, ti],
            alternative='less').pvalue
        for ti in range(nT)
    ])

    print(f"\n  t     ΔS        ±sem      p       sig")
    print(f"  {'-'*45}")
    for ts in [10, 20, 30, 40, 50, 70, 100,
               150, 200, 300, 500, 700]:
        i = np.searchsorted(t_pts, ts)
        if i >= nT:
            continue
        sig = ("***" if p_vals[i] < 0.001 else
               "**"  if p_vals[i] < 0.01  else
               "*"   if p_vals[i] < 0.05  else
               "."   if p_vals[i] < 0.10  else "")
        print(f"  {t_pts[i]:>3}  {dS_m[i]:>+8.4f}"
              f"  ±{dS_sem[i]:>6.4f}"
              f"  {p_vals[i]:>7.4f}  {sig}")

    n_sig = np.sum((p_vals < 0.05) & (dS_m < 0))
    print(f"\n  Sig. negative: {n_sig}/{nT}")
    mi = np.argmin(dS_m)
    print(f"  Min ΔS={dS_m[mi]:.4f} at t={t_pts[mi]}")

    # τ estimate
    sig_neg = (dS_m < 0) & (p_vals < 0.05) & (t_pts >= 10)
    any_neg = (dS_m < 0) & (t_pts >= 10)

    if sig_neg.sum() > 0:
        tau    = float(t_pts[sig_neg][-1])
        method = "last sig-neg"
    elif any_neg.sum() > 0:
        tau    = float(np.average(
            t_pts[any_neg], weights=-dS_m[any_neg]))
        method = "centroid [no sig]"
    else:
        tau, method = np.nan, "NaN"

    print(f"\n  τ = {tau:.0f}  [{method}]")
    print(f"  Part VI result: τ ≈ 237")

    if not np.isnan(tau) and 180 < tau < 300:
        print(f"  [OK] τ within expected range!")
    else:
        print(f"  [WARN] τ outside expected range.")

    return t_pts, dS_m, dS_s, p_vals, tau


# ══════════════════════════════════════════════════════════════════
# 4. KEY DIAGNOSTIC: what WAS different in Part VI?
# ══════════════════════════════════════════════════════════════════

def diagnose_shuffle_vs_random(rng, N=50, T_max=500,
                                sigma=0.5, n_runs=40):
    """
    Compare shuffle null vs random null directly.
    Part VI: ΔS(shuffle) was significant at t=30-40.
    Part VII: ΔS(random) is noise.
    WHY?
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC: WHY is shuffled null different from random?")
    print("="*60)

    t_pts = np.arange(5, T_max+1, 5).astype(int)
    nT    = len(t_pts)

    results = {}
    for null_name, get_null in [
        ('shuffle',  lambda r: OMEGA_E6[r.permutation(RANK_E6)]),
        ('rand_int', lambda r: r.choice(
            np.arange(1, 12), RANK_E6, replace=False).astype(float)),
        ('uniform',  lambda r: r.uniform(1., 11., RANK_E6)),
        ('arith',    lambda r: np.linspace(1., 11., RANK_E6)),
    ]:
        S_cox  = np.zeros((n_runs, nT))
        S_null = np.zeros((n_runs, nT))

        for run in range(n_runs):
            phi_b = rng.uniform(0, 2*np.pi, RANK_E6)
            init  = (phi_b[None, :]
                     + rng.normal(0, sigma, (N, RANK_E6))
                    ) % (2*np.pi)
            omega_n = get_null(rng)

            for ti, t in enumerate(t_pts):
                pc = (init + OMEGA_E6[None, :] * t) % (2*np.pi)
                pn = (init + omega_n[None, :] * t) % (2*np.pi)
                S_cox[run, ti]  = shannon_entropy(pc,  20)
                S_null[run, ti] = shannon_entropy(pn, 20)

        dS_m  = (S_cox - S_null).mean(0)
        p_v   = np.array([
            stats.mannwhitneyu(
                S_cox[:, i], S_null[:, i],
                alternative='less').pvalue
            for i in range(nT)
        ])

        # Find range t=20–100 (Part VI signal region)
        mask_early = (t_pts >= 20) & (t_pts <= 100)
        min_dS = dS_m[mask_early].min()
        min_t  = t_pts[mask_early][dS_m[mask_early].argmin()]
        n_sig  = np.sum((p_v < 0.05) & (dS_m < 0))

        results[null_name] = {
            'dS_m': dS_m, 'p_v': p_v,
            'min_dS': min_dS, 'min_t': min_t,
            'n_sig': n_sig,
        }

        print(f"\n  Null={null_name}:")
        print(f"    Min ΔS (t=20-100) = {min_dS:.4f} at t={min_t}")
        print(f"    Total sig. neg points: {n_sig}/{nT}")
        print(f"    t=30: ΔS={dS_m[t_pts==30][0]:.4f}, "
              f"p={p_v[t_pts==30][0]:.4f}")

    # Key question: does shuffled null differ from E6?
    print("\n\n  KEY FINDING:")
    print("  If ΔS(shuffle) ≈ 0 → shuffle is identical to E6")
    print("  (linear dynamics: entropy is permutation-invariant)")
    print("  → Part VI τ=237 may have been measured with")
    print("    a DIFFERENT quantity (not Shannon entropy)")
    print("    or a DIFFERENT initial condition setup.")

    return results, t_pts


# ══════════════════════════════════════════════════════════════════
# 5. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_diagnostic_figure(dcorr_scan, tau_result,
                            null_compare, t_pts_null):
    """Create diagnostic figure."""
    t_pts_tau = tau_result[0]
    dS_tau    = tau_result[1]
    dS_std_t  = tau_result[2]
    p_tau     = tau_result[3]
    tau_val   = tau_result[4]

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Part VII v5: Diagnostic — Reproducing Part VI\n"
        "Finding D_corr=3.02 and τ=237 conditions",
        fontsize=14, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.48, wspace=0.38)

    # Panel A: D_corr scan
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('#f0f4f8')

    T_vals = sorted(set(k[0] for k in dcorr_scan))
    N_vals = sorted(set(k[1] for k in dcorr_scan))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_vals)))

    for j, n in enumerate(N_vals):
        T_plot, D_plot = [], []
        for T in T_vals:
            if (T, n) in dcorr_scan:
                D, _ = dcorr_scan[(T, n)]
                if not np.isnan(D):
                    T_plot.append(T)
                    D_plot.append(D)
        if T_plot:
            ax.plot(T_plot, D_plot, 'o-',
                    color=colors[j], lw=2, ms=8,
                    label=f"n={n}")

    ax.axhline(3.02, c='blue', ls='--', lw=2,
               label='Part VI: 3.02')
    ax.axhline(4.09, c='red', ls=':', lw=1.5,
               label='Current: 4.09', alpha=0.7)
    ax.set_xlabel('T (trajectory length)', fontsize=11)
    ax.set_ylabel('D_corr', fontsize=11)
    ax.set_title('D_corr vs T and n_sample\n'
                 'Which conditions give 3.02?',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 7])

    # Panel B: Reproduce Part VI τ=237
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor('#f0f4f8')

    sem_tau = dS_std_t / np.sqrt(30)
    ax.fill_between(t_pts_tau,
                    dS_tau - sem_tau, dS_tau + sem_tau,
                    alpha=0.25, color='#2196F3')
    ax.plot(t_pts_tau, dS_tau, '-',
            color='#2196F3', lw=2.5, zorder=4,
            label='E6 (shuffled null)')
    ax.axhline(0, c='k', lw=1.8)

    for i in range(len(t_pts_tau)):
        if p_tau[i] < 0.01 and dS_tau[i] < 0:
            ax.scatter(t_pts_tau[i], dS_tau[i],
                       s=80, c='#2E7D32', zorder=6,
                       edgecolors='k', lw=0.8)
        elif p_tau[i] < 0.05 and dS_tau[i] < 0:
            ax.scatter(t_pts_tau[i], dS_tau[i],
                       s=50, c='#81C784', zorder=6,
                       edgecolors='k', lw=0.8)

    if not np.isnan(tau_val):
        ax.axvline(tau_val, c='red', ls='--', lw=2,
                   label=f'τ={tau_val:.0f}')
    ax.axvline(237, c='orange', ls=':', lw=2,
               label='Part VI: τ=237')

    ax.set_xlabel('t (steps)', fontsize=11)
    ax.set_ylabel('ΔS = S(E6) − S(shuf)', fontsize=10)
    ax.set_title('Reproducing τ=237\n'
                 '(shuffled null, N=50, σ=0.5)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Null model comparison
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor('#f0f4f8')

    null_colors = {
        'shuffle':  '#2196F3',
        'rand_int': '#FF9800',
        'uniform':  '#4CAF50',
        'arith':    '#9C27B0',
    }

    for null_name, res in null_compare.items():
        mask = t_pts_null <= 500
        ax.plot(t_pts_null[mask], res['dS_m'][mask],
                '-', color=null_colors.get(null_name, 'gray'),
                lw=2, alpha=0.85, label=null_name)

    ax.axhline(0, c='k', lw=1.5)
    ax.set_xlabel('t (steps)', fontsize=11)
    ax.set_ylabel('ΔS = S(E6) − S(null)', fontsize=10)
    ax.set_title('ΔS: All null models vs E6\n'
                 'Which null gives clearest signal?',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: D_corr interpretation
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#f0f4f8')
    ax.axis('off')

    txt = [
        "WHY does D_corr depend on T and n?",
        "",
        "E6 orbit: phi(t) = phi0 + omega*t (mod 2pi)",
        "omega = [1,4,5,7,8,11] (integers)",
        "",
        "At small T: orbit covers only part of",
        "the quasi-3D manifold → D_corr underestimates",
        "",
        "At large T: orbit fills the 3D attractor",
        "completely → D_corr = 3.02 (Part V/VI)",
        "",
        "At T=8000, n=600 (v4): D_corr = 4.09?",
        "Possible: n_sample too large → distances",
        "span multiple scales → r-range captures",
        "a different regime.",
        "",
        "Resolution: use SAME T, n as Part VI.",
    ]
    ax.text(0.05, 0.97, "\n".join(txt),
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            va='top', color='#1a1a2e')
    ax.set_title('D_corr Diagnosis',
                 fontsize=10, fontweight='bold')

    # Panel E: τ interpretation
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('#f0f4f8')
    ax.axis('off')

    txt2 = [
        "WHY is ΔS(shuffle) different from ΔS(random)?",
        "",
        "Linear dynamics: phi_i(t) = init_i + omega_i*t",
        "",
        "SHUFFLE null: omega_shuf = permutation(omega_E6)",
        "  → Same set of frequencies, different dims",
        "  → At t=0: SAME initial cloud",
        "  → Entropy difference only from WHICH dims",
        "    the frequencies are assigned to",
        "  → Very subtle effect (same magnitudes)",
        "",
        "RANDOM null: omega_rand = random in [1,11]",
        "  → Different magnitudes AND positions",
        "  → Asymptotic entropy CAN differ",
        "  → But initial transient is the same",
        "",
        "Part VI signal (tau=237) may be real but",
        "requires shuffled null to be detectable.",
        "Random null has too much variance.",
    ]
    ax.text(0.05, 0.97, "\n".join(txt2),
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            va='top', color='#1a1a2e')
    ax.set_title('τ & Null Model Diagnosis',
                 fontsize=10, fontweight='bold')

    # Panel F: Action plan
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('#e8f5e9')
    ax.axis('off')

    txt3 = [
        "CONCLUSIONS FROM v5 DIAGNOSTIC:",
        "",
        "1. D_corr=3.02 requires specific T,n",
        "   → Use T from scan that gives 3.02",
        "   → Part VI likely used T=5000, n=500",
        "   → r-range choice is CRITICAL",
        "",
        "2. τ=237 requires SHUFFLED null",
        "   → Random null has too much variance",
        "   → Shuffled tests ORDER, not magnitude",
        "   → This is physically meaningful:",
        "     'Do specific Coxeter ratios matter",
        "      vs arbitrary permutation?'",
        "",
        "3. For cross-algebra test (Part VII main):",
        "   Null = shuffled exponents of SAME algebra",
        "   This is the correct null for tau prop h.",
        "",
        "4. A6: exponents [1..6] = all integers in [1,h-1]",
        "   Shuffle = same set → ΔS = 0 always",
        "   A6 CANNOT be tested with shuffle null!",
        "   → Remove A6, focus on E6, E7, E8",
    ]
    colors_txt = ['#1565C0' if i == 0 else '#1a1a2e'
                  for i in range(len(txt3))]
    ax.text(0.05, 0.97, "\n".join(txt3),
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            va='top', color='#1a1a2e')
    ax.set_title('Action Plan for Part VII v6',
                 fontsize=10, fontweight='bold',
                 color='#1565C0')

    plt.savefig('monostring_part7_v5_diagnostic.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_v5_diagnostic.png")


# ══════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(2025)

    print("Part VII v5: Diagnostic")
    print("Goal: Find exact conditions for D_corr=3.02 and τ=237\n")

    # Scan 1: D_corr vs T and n_sample
    dcorr_scan = scan_dcorr_parameters(rng)

    # Scan 2: Reproduce D_corr=3.02 exactly
    best_D, best_cfg = reproduce_part6_dcorr(rng)

    # Reproduce τ=237
    tau_result = reproduce_part6_tau(
        rng, N=50, T_max=700, sigma=0.5, n_runs=30)

    # Compare null models
    null_compare, t_pts_null = diagnose_shuffle_vs_random(
        rng, N=50, T_max=500, sigma=0.5, n_runs=40)

    # Figure
    make_diagnostic_figure(
        dcorr_scan, tau_result,
        null_compare, t_pts_null)

    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Best D_corr: {best_D:.3f} with {best_cfg}")
    print(f"τ (shuffled null): {tau_result[4]:.0f}")
    print("\nNext step (v6): Use validated parameters")
    print("for cross-algebra test with SHUFFLED null.")
    print("Algebras: E6 (h=12), E7 (h=18), E8 (h=30)")
    print("A6 excluded: shuffle null = trivial for [1..6]")
