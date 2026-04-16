"""
Part VII v6: GROUND TRUTH — Reconstruct Part VI exactly
=========================================================
The diagnostic (v5) showed:
  - D_corr(E6 raw exponents) = 4.09 consistently (not 3.02)
  - ΔS signal = noise for ALL null models (no τ=237)

This means Part VI used DIFFERENT dynamics or metric.
Part VI paper says:
  "fragmented daughter strings retain structural memory"
  "Shannon entropy S(t) of the N-string cloud"
  "daughter strings initialized in σ-ball around φ_break"

BUT Part VI also mentions:
  "E6 Coxeter orbit" with D_corr=3.02 from Part V

Part V used a NONLINEAR map (coupled standard map):
  φ_{n+1} = φ_n + ω + κ·sin(φ_n)  (mod 2π)
  NOT linear: φ(t) = φ_0 + ω·t

The D_corr=3.02 and τ=237 were measured on the
NONLINEAR ATTRACTOR, not on a linear torus flow.

This script tests:
1. Nonlinear standard map with E6 Coxeter frequencies
2. D_corr of the nonlinear attractor
3. τ measurement on nonlinear daughters
4. Cross-algebra comparison: A6, E6, E7, E8
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
# 1. ALGEBRA DEFINITIONS
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': np.array([1., 2., 3., 4., 5., 6.]),
        'color': '#9C27B0', 'tau_pred': 140,
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': np.array([1., 4., 5., 7., 8., 11.]),
        'color': '#2196F3', 'tau_pred': 240,
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': np.array([1., 5., 7., 9., 11., 13., 17.]),
        'color': '#FF9800', 'tau_pred': 360,
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': np.array([1., 7., 11., 13., 17., 19., 23., 29.]),
        'color': '#4CAF50', 'tau_pred': 600,
    },
}

# Standard map nonlinearity (Part VI value)
KAPPA = 0.5


# ══════════════════════════════════════════════════════════════════
# 2. DYNAMICS: Two modes
#    A) Linear torus flow (what v2-v5 used — WRONG for Part VI)
#    B) Nonlinear standard map (what Part VI actually used)
# ══════════════════════════════════════════════════════════════════

def evolve_linear(phi0, omega, T):
    """
    Linear: phi(t) = phi0 + omega*t  (mod 2π)
    phi0: (N, rank), omega: (rank,)
    Returns: (T, N, rank)
    """
    t    = np.arange(T, dtype=float)
    traj = (phi0[np.newaxis, :, :]
            + omega[np.newaxis, np.newaxis, :] * t[:, np.newaxis, np.newaxis]
            ) % (2 * np.pi)
    return traj


def step_standard_map(phi, omega, kappa):
    """
    One step of the coupled standard map:
    φ_{n+1} = φ_n + ω + κ·sin(φ_n)  (mod 2π)
    phi: (N, rank), omega: (rank,)
    """
    return (phi + omega[np.newaxis, :] +
            kappa * np.sin(phi)) % (2 * np.pi)


def evolve_nonlinear(phi0, omega, T, kappa=KAPPA):
    """
    Nonlinear standard map evolution.
    phi0: (N, rank)
    Returns trajectory: (T+1, N, rank)
    """
    rank = phi0.shape[1]
    N    = phi0.shape[0]
    traj = np.zeros((T + 1, N, rank))
    traj[0] = phi0.copy()
    for t in range(T):
        traj[t + 1] = step_standard_map(
            traj[t], omega, kappa)
    return traj


# ══════════════════════════════════════════════════════════════════
# 3. CORRELATION DIMENSION
#    Test both linear and nonlinear to find which gives 3.02
# ══════════════════════════════════════════════════════════════════

def dcorr_from_trajectory(traj, n_sample=500,
                            pct_lo=5, pct_hi=45,
                            r_type='lin', n_trials=5,
                            rng=None):
    """
    Compute D_corr from a precomputed trajectory.
    traj: (T, rank) — single-string orbit
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T    = traj.shape[0]
    slopes = []

    for _ in range(n_trials):
        idx  = rng.choice(T, min(n_sample, T), replace=False)
        pts  = traj[idx]
        n    = len(pts)

        d    = np.abs(pts[:, None, :] - pts[None, :, :])
        d    = np.minimum(d, 2*np.pi - d)
        dm   = np.sqrt((d**2).sum(-1))
        dist = dm[np.triu_indices(n, k=1)]

        r_lo = np.percentile(dist, pct_lo)
        r_hi = np.percentile(dist, pct_hi)
        if r_hi <= r_lo * 1.02:
            continue

        if r_type == 'log':
            r_arr = np.logspace(np.log10(r_lo),
                                np.log10(r_hi), 25)
        else:
            r_arr = np.linspace(r_lo, r_hi, 25)

        C_r  = np.array([np.mean(dist < r) for r in r_arr])
        mask = C_r > 0.005
        if mask.sum() < 5:
            continue

        sl, _, rv, _, _ = stats.linregress(
            np.log(r_arr[mask]), np.log(C_r[mask]))
        if rv**2 > 0.88:
            slopes.append(sl)

    if not slopes:
        return np.nan, np.nan
    return float(np.mean(slopes)), float(np.std(slopes))


def compare_linear_vs_nonlinear_dcorr(rng):
    """
    Key test: does nonlinear standard map give D_corr=3.02?
    """
    print("\n" + "="*60)
    print("D_corr: LINEAR vs NONLINEAR standard map")
    print(f"E6 exponents: [1,4,5,7,8,11], κ={KAPPA}")
    print("="*60)

    omega = ALGEBRAS['E6']['exponents']
    T     = 5000

    # Linear orbit (single string)
    phi0_lin = np.zeros((1, 6))
    traj_lin = evolve_linear(phi0_lin, omega, T)
    traj_lin = traj_lin[:, 0, :]  # (T, rank)

    # Nonlinear orbit (single string)
    phi0_nl  = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    traj_nl  = evolve_nonlinear(phi0_nl, omega, T, KAPPA)
    traj_nl  = traj_nl[:, 0, :]  # (T+1, rank)

    configs = [
        ('lin 5-45', 'lin', 5,  45),
        ('lin 10-40','lin', 10, 40),
        ('log 5-45', 'log', 5,  45),
        ('lin 5-30', 'lin', 5,  30),
    ]

    print(f"\n{'Config':<12}  "
          f"{'D_linear':>10}  {'D_nonlin':>10}")
    print("-"*38)

    results = {}
    for lbl, rtyp, plo, phi in configs:
        D_lin, Dlin_s = dcorr_from_trajectory(
            traj_lin, n_sample=500,
            pct_lo=plo, pct_hi=phi,
            r_type=rtyp, n_trials=8, rng=rng)
        D_nl, Dnl_s = dcorr_from_trajectory(
            traj_nl, n_sample=500,
            pct_lo=plo, pct_hi=phi,
            r_type=rtyp, n_trials=8, rng=rng)

        note_lin = " <--3.02?" if (
            not np.isnan(D_lin) and 2.7 < D_lin < 3.3) else ""
        note_nl  = " <--3.02?" if (
            not np.isnan(D_nl)  and 2.7 < D_nl  < 3.3) else ""

        print(f"{lbl:<12}  "
              f"{D_lin:>10.3f}{note_lin}  "
              f"{D_nl:>10.3f}{note_nl}")
        results[lbl] = (D_lin, D_nl)

    print(f"\nExpected: D_corr = 3.02 (Part V/VI)")
    return results, traj_lin, traj_nl


# ══════════════════════════════════════════════════════════════════
# 4. FRAGMENTATION WITH NONLINEAR MAP
# ══════════════════════════════════════════════════════════════════

def shannon_entropy(positions, n_bins=20):
    """Mean Shannon entropy across dims. (N, rank) → float."""
    rank = positions.shape[1]
    H = 0.0
    for d in range(rank):
        c, _ = np.histogram(positions[:, d], bins=n_bins,
                             range=(0., 2*np.pi))
        p = c / c.sum()
        p = p[p > 0]
        H -= np.sum(p * np.log(p))
    return H / rank


def fragmentation_nonlinear(algebra_name,
                             N=50, T_max=500,
                             sigma=0.5, kappa=KAPPA,
                             n_runs=30, n_bins=20,
                             rng=None, verbose=True):
    """
    Fragmentation with nonlinear standard map.
    Null: SHUFFLED frequencies (same algebra).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega = ALGEBRAS[algebra_name]['exponents']
    rank  = len(omega)

    # Time grid
    t_pts = np.unique(np.concatenate([
        np.arange(5,  60,  5),
        np.arange(60, 200, 10),
        np.arange(200, T_max+1, 25),
    ])).astype(int)
    t_pts = t_pts[t_pts <= T_max]
    nT    = len(t_pts)

    S_cox  = np.zeros((n_runs, nT))
    S_shuf = np.zeros((n_runs, nT))

    for run in range(n_runs):
        # Break point: random point on attractor
        # Warm up for 1000 steps first
        phi_warmup = rng.uniform(0, 2*np.pi, (1, rank))
        for _ in range(1000):
            phi_warmup = step_standard_map(
                phi_warmup, omega, kappa)
        phi_break = phi_warmup[0]

        # Initialize N daughters in σ-ball
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N, rank))
               ) % (2 * np.pi)

        # Shuffled frequencies
        shuf_idx   = rng.permutation(rank)
        omega_shuf = omega[shuf_idx]

        # Evolve both
        pos_cox  = init.copy()
        pos_shuf = init.copy()

        step_idx = 0
        for ti, t in enumerate(t_pts):
            # Evolve from last position
            steps_needed = t - (t_pts[ti-1] if ti > 0 else 0)
            for _ in range(steps_needed):
                pos_cox  = step_standard_map(
                    pos_cox, omega, kappa)
                pos_shuf = step_standard_map(
                    pos_shuf, omega_shuf, kappa)

            S_cox[run, ti]  = shannon_entropy(pos_cox,  n_bins)
            S_shuf[run, ti] = shannon_entropy(pos_shuf, n_bins)

    dS_m  = (S_cox - S_shuf).mean(0)
    dS_s  = (S_cox - S_shuf).std(0)
    dS_sem = dS_s / np.sqrt(n_runs)

    p_vals = np.array([
        stats.mannwhitneyu(
            S_cox[:, i], S_shuf[:, i],
            alternative='less').pvalue
        for i in range(nT)
    ])

    if verbose:
        print(f"\n  [{algebra_name}] ΔS(nonlinear, κ={kappa})")
        print(f"  {'t':>4}  {'ΔS':>9}  {'±sem':>7}"
              f"  {'p':>8}  sig")
        print(f"  {'-'*42}")
        for ts in [10, 20, 30, 40, 50, 70, 100,
                   150, 200, 300, 500]:
            i = np.searchsorted(t_pts, ts)
            if i >= nT:
                continue
            sig = ("***" if p_vals[i] < 0.001 else
                   "**"  if p_vals[i] < 0.01  else
                   "*"   if p_vals[i] < 0.05  else
                   "."   if p_vals[i] < 0.10  else "")
            print(f"  {t_pts[i]:>4}  {dS_m[i]:>+9.4f}"
                  f"  ±{dS_sem[i]:>5.4f}"
                  f"  {p_vals[i]:>8.4f}  {sig}")
        n_sig = np.sum((p_vals < 0.05) & (dS_m < 0))
        print(f"\n  Sig. negative: {n_sig}/{nT}")
        mi = np.argmin(dS_m)
        print(f"  Min ΔS={dS_m[mi]:.4f} at t={t_pts[mi]}")

    return t_pts, dS_m, dS_s, p_vals


def estimate_tau(t_pts, dS_m, p_vals,
                 min_t=10, p_thr=0.05):
    """Estimate τ from ΔS profile."""
    sig_neg = ((dS_m < 0) & (p_vals < p_thr) &
               (t_pts >= min_t))
    any_neg = (dS_m < 0) & (t_pts >= min_t)

    if sig_neg.sum() > 0:
        tau    = float(t_pts[sig_neg][-1])
        method = f"last sig-neg (p<{p_thr})"
    elif any_neg.sum() > 0:
        tau    = float(np.average(
            t_pts[any_neg],
            weights=-dS_m[any_neg]))
        method = "centroid(ΔS<0) [no sig]"
    else:
        tau, method = np.nan, "NaN"
    return tau, method


# ══════════════════════════════════════════════════════════════════
# 5. KAPPA SCAN: find κ that gives τ=237
# ══════════════════════════════════════════════════════════════════

def kappa_scan_e6(rng):
    """
    Scan κ from 0 (linear) to 2.0 to find
    which κ reproduces τ=237 for E6.
    """
    print("\n" + "="*60)
    print("κ SCAN: Finding nonlinearity that gives τ=237")
    print("E6, N=50, σ=0.5, n_runs=30")
    print("="*60)

    kappas = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0,
              1.5, 2.0, 3.0]
    tau_results = {}

    for kappa in kappas:
        print(f"\n  κ = {kappa}:")
        t_pts, dS_m, _, p_v = fragmentation_nonlinear(
            'E6', N=50, T_max=500,
            sigma=0.5, kappa=kappa,
            n_runs=30, n_bins=20,
            rng=rng, verbose=False)

        tau, meth = estimate_tau(t_pts, dS_m, p_v)
        n_sig = np.sum((p_v < 0.05) & (dS_m < 0))
        mi    = np.argmin(dS_m)

        print(f"    Min ΔS={dS_m[mi]:.4f} at t={t_pts[mi]}"
              f",  sig.neg={n_sig}/{len(t_pts)}"
              f",  τ={tau:.0f} [{meth}]"
              if not np.isnan(tau) else
              f"    Min ΔS={dS_m[mi]:.4f} at t={t_pts[mi]}"
              f",  sig.neg={n_sig}/{len(t_pts)}"
              f",  τ=NaN")

        tau_results[kappa] = (tau, n_sig,
                               dS_m[mi], t_pts[mi])

    print("\n  Summary:")
    print(f"  {'κ':>6}  {'τ':>8}  {'n_sig':>6}  "
          f"{'min_ΔS':>8}  note")
    print(f"  {'-'*45}")
    for k, (tau, ns, mdS, mt) in tau_results.items():
        note = ""
        if not np.isnan(tau) and 180 < tau < 300:
            note = " <-- τ≈237?"
        ts = f"{tau:>8.0f}" if not np.isnan(tau) else f"{'NaN':>8}"
        print(f"  {k:>6.2f}  {ts}  {ns:>6}"
              f"  {mdS:>8.4f}  {note}")

    return tau_results


# ══════════════════════════════════════════════════════════════════
# 6. MAIN: Cross-algebra test with best κ
# ══════════════════════════════════════════════════════════════════

def run_cross_algebra(best_kappa, rng):
    """
    Run the τ ∝ h test with the κ that gives
    a signal for E6.
    """
    print("\n" + "="*60)
    print(f"CROSS-ALGEBRA TEST (κ={best_kappa})")
    print("Algebras: A6, E6, E7, E8")
    print("Null: SHUFFLED frequencies")
    print("="*60)

    results = {}

    for alg_name, alg in ALGEBRAS.items():
        h    = alg['h']
        rank = alg['rank']
        print(f"\n{'='*50}")
        print(f"{alg_name}: h={h}, rank={rank}")

        # D_corr (nonlinear attractor)
        omega = alg['exponents']
        phi0_s = np.array([rng.uniform(0, 2*np.pi, rank)])
        # Warmup
        for _ in range(2000):
            phi0_s = step_standard_map(
                phi0_s, omega, best_kappa)
        # Collect orbit
        T_orb = 5000
        orbit = np.zeros((T_orb, rank))
        phi_c = phi0_s[0].copy()
        for t in range(T_orb):
            phi_c    = step_standard_map(
                phi_c[np.newaxis, :], omega, best_kappa)[0]
            orbit[t] = phi_c

        D, Ds = dcorr_from_trajectory(
            orbit, n_sample=500,
            pct_lo=5, pct_hi=45,
            r_type='lin', n_trials=6, rng=rng)
        print(f"  D_corr = "
              f"{D:.3f}±{Ds:.3f}" if not np.isnan(D)
              else "  D_corr = NaN")

        # Fragmentation
        T_max = max(500, 25 * h)
        t_pts, dS_m, dS_s, p_v = fragmentation_nonlinear(
            alg_name, N=50,
            T_max=T_max,
            sigma=0.5, kappa=best_kappa,
            n_runs=30, n_bins=20,
            rng=rng, verbose=True)

        tau, meth = estimate_tau(t_pts, dS_m, p_v)

        ratio  = tau / h if not np.isnan(tau) else np.nan
        status = ("CONFIRMS" if not np.isnan(ratio) and
                  10 < ratio < 40 else
                  "REJECTS" if not np.isnan(ratio) else
                  "NO SIGNAL")

        results[alg_name] = {
            'h': h, 'rank': rank,
            'tau_pred': alg['tau_pred'],
            'tau_obs': tau, 'tau_ratio': ratio,
            'D_corr': D, 'D_std': Ds,
            't_pts': t_pts,
            'dS_mean': dS_m, 'dS_std': dS_s,
            'p_vals': p_v,
            'color': alg['color'],
            'status': status,
        }

        print(f"\n  τ_obs={tau:.0f}, τ_pred={alg['tau_pred']}, "
              f"τ/h={ratio:.1f}, status={status}"
              if not np.isnan(tau) else
              f"\n  τ=NaN, status={status}")

    return results


# ══════════════════════════════════════════════════════════════════
# 7. STATISTICS
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results):
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    names   = list(results.keys())
    h_arr   = np.array([results[n]['h']      for n in names])
    tau_arr = np.array([results[n]['tau_obs'] for n in names])
    rank_arr= np.array([results[n]['rank']    for n in names])

    valid = ~np.isnan(tau_arr)
    n_v   = valid.sum()
    print(f"Valid τ: {n_v}/{len(names)}")

    out = {'h_vals':   h_arr[valid],
           'tau_vals': tau_arr[valid],
           'alg_names':[names[i] for i in range(len(names))
                        if valid[i]]}

    if n_v < 2:
        print("Too few points.")
        return out

    h_v   = h_arr[valid]
    tau_v = tau_arr[valid]
    r_v   = rank_arr[valid]

    k0   = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    res0 = tau_v - k0 * h_v
    r2_0 = (1 - np.sum(res0**2) /
             np.sum((tau_v - tau_v.mean())**2))
    print(f"\nτ=k*h (origin): k={k0:.2f}, R²={r2_0:.4f}")
    out.update({'k0': k0, 'r2_h0': r2_0})

    if n_v >= 3:
        sl, ic, r, p, _ = stats.linregress(h_v, tau_v)
        print(f"τ=a*h+b:  a={sl:.2f}, b={ic:.1f}, "
              f"R²={r**2:.4f}, p={p:.4f}")
        out.update({'sl_h': sl, 'ic_h': ic,
                    'r2_h': r**2, 'p_h': p})
        sl2, _, r2, p2, _ = stats.linregress(r_v, tau_v)
        print(f"τ=a*rank: R²={r2**2:.4f}, p={p2:.4f}")
        out['r2_rank'] = r2**2
        rho, pp = stats.pearsonr(h_v, tau_v)
        print(f"Pearson r={rho:.3f}, p={pp:.4f}")
        out.update({'pearson_r': rho, 'pearson_p': pp})
    else:
        out.update({'r2_h': np.nan, 'p_h': np.nan,
                    'r2_rank': np.nan,
                    'pearson_r': np.nan,
                    'pearson_p': np.nan,
                    'sl_h': np.nan, 'ic_h': np.nan})
    return out


# ══════════════════════════════════════════════════════════════════
# 8. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(kappa_scan, results, stats_out, best_kappa):
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        f"Monostring Hypothesis — Part VII v6\n"
        f"Nonlinear standard map (κ={best_kappa}): "
        r"$\tau \propto h$?"
        f"\nA6 (h=7), E6 (h=12), E7 (h=18), E8 (h=30)",
        fontsize=13, fontweight='bold', y=0.995,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.38)

    # Row 0: ΔS(t) per algebra
    alg_list = list(results.keys())
    for i, alg_name in enumerate(alg_list):
        res = results[alg_name]
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        t   = res['t_pts']
        dS  = res['dS_mean']
        sem = res['dS_std'] / np.sqrt(30)
        pv  = res['p_vals']
        c   = res['color']

        ax.fill_between(t, dS-sem, dS+sem,
                        alpha=0.25, color=c)
        ax.plot(t, dS, '-', color=c, lw=2.5, zorder=4)
        ax.axhline(0, c='k', lw=1.8)

        for j in range(len(t)):
            if pv[j] < 0.01 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=70,
                           c='#2E7D32', zorder=6,
                           edgecolors='k', lw=0.7)
            elif pv[j] < 0.05 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=45,
                           c='#81C784', zorder=6,
                           edgecolors='k', lw=0.7)

        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.axvline(tau, c='red', ls='--',
                       lw=2, label=f"τ={tau:.0f}")
        ax.axvline(res['tau_pred'], c='orange',
                   ls=':', lw=2,
                   label=f"pred={res['tau_pred']}")

        D_s = (f"{res['D_corr']:.2f}"
               if not np.isnan(res['D_corr']) else "N/A")
        ax.set_title(
            f"{alg_name}: h={res['h']}, rank={res['rank']}\n"
            f"D_corr={D_s}  τ="
            f"{'N/A' if np.isnan(tau) else f'{tau:.0f}'}\n"
            f"{res['status']}",
            fontsize=9, fontweight='bold')
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel('ΔS', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1 left: κ scan
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    ks   = sorted(kappa_scan.keys())
    taus = [kappa_scan[k][0] for k in ks]
    nsig = [kappa_scan[k][1] for k in ks]

    ax2 = ax.twinx()
    ax.bar(ks, [n if not np.isnan(t) else 0
                for n, t in zip(nsig, taus)],
           width=0.08, alpha=0.4, color='#90CAF9',
           label='n_sig (right)')
    ax2.plot([k for k, t in zip(ks, taus)
              if not np.isnan(t)],
             [t for t in taus if not np.isnan(t)],
             'o-', c='red', lw=2.5, ms=10, zorder=5,
             label='τ (left)')
    ax2.axhline(237, c='blue', ls='--', lw=2,
                alpha=0.6, label='τ=237 (Part VI)')

    ax.set_xlabel('κ (nonlinearity)', fontsize=11)
    ax.set_ylabel('n_sig negative', fontsize=11)
    ax2.set_ylabel('τ (steps)', fontsize=11, color='red')
    ax.set_title('κ scan: finding Part VI τ=237\n'
                 'bars=sig.neg points, line=τ',
                 fontsize=10, fontweight='bold')
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(best_kappa, c='green', ls='--',
               lw=2, alpha=0.7,
               label=f'best κ={best_kappa}')

    # Row 1 right: τ vs h scatter
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    h_line = np.linspace(5, 35, 200)
    ax.plot(h_line, 20*h_line, '--', c='gray',
            lw=2, alpha=0.5, label='τ=20h (predicted)')
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.06, color='gray')
    ax.scatter(12, 237, s=250, marker='*', c='blue',
               zorder=7, edgecolors='k',
               label='Part VI E6 (τ=237)')

    for alg_name, res in results.items():
        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.scatter(res['h'], tau, s=220,
                       c=res['color'], zorder=5,
                       edgecolors='k', lw=2)
            ax.annotate(
                f"{alg_name}\nτ={tau:.0f}\n{res['status']}",
                xy=(res['h'], tau),
                xytext=(res['h']+0.4, tau+10),
                fontsize=9, fontweight='bold',
                color=res['color'])

    if (stats_out and
            not np.isnan(stats_out.get('r2_h', np.nan)) and
            stats_out.get('r2_h', 0) > 0.3 and
            len(stats_out.get('h_vals', [])) >= 3):
        sl = stats_out['sl_h']
        ic = stats_out['ic_h']
        r2 = stats_out['r2_h']
        ax.plot(h_line, sl*h_line+ic, '-',
                c='red', lw=2,
                label=f"fit:{sl:.1f}h+{ic:.0f} R²={r2:.3f}")

    ax.set_xlabel('Coxeter h', fontsize=12)
    ax.set_ylabel('τ (steps)', fontsize=12)
    ax.set_title('KEY: τ vs h\n'
                 'Prediction: τ≈20h',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # Row 2: Summary
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    # Left: table
    ax.text(0.01, 0.97, "RESULTS (nonlinear map)",
            fontsize=12, fontweight='bold',
            transform=ax.transAxes, color='#1a1a2e')
    hdr = (f"{'Alg':<5} {'h':<4} {'rank':<5} "
           f"{'pred':>6} {'obs':>7} {'obs/h':>7} "
           f"{'D_corr':>13}  status")
    ax.text(0.01, 0.85, hdr,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', fontweight='bold',
            color='#1a1a2e')
    ax.text(0.01, 0.80, "-"*68,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', color='gray')

    y = 0.73
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
               '#C62828' if st == 'REJECTS'  else '#888')
        row = (f"{alg_name:<5} {res['h']:<4} {res['rank']:<5}"
               f"{res['tau_pred']:>6}{ts}{rs}  {ds}  {st}")
        ax.text(0.01, y, row,
                transform=ax.transAxes, fontsize=10,
                fontfamily='monospace', color=c)
        y -= 0.10

    # Right: stats + verdict
    if stats_out:
        r2h = stats_out.get('r2_h', np.nan)
        k0  = stats_out.get('k0', np.nan)
        r20 = stats_out.get('r2_h0', np.nan)
        rho = stats_out.get('pearson_r', np.nan)
        pp  = stats_out.get('pearson_p', np.nan)

        if not np.isnan(r2h):
            if r2h > 0.85:
                verd = "STRONG: tau PROPORTIONAL to h"
                vc = '#2E7D32'
            elif r2h > 0.5:
                verd = "MODERATE: tau correlates with h"
                vc = '#FF9800'
            else:
                verd = "WEAK: tau NOT proportional to h"
                vc = '#C62828'
        else:
            verd = "INSUFFICIENT DATA"
            vc   = '#888'

        slines = [
            f"tau=k*h(origin): k={k0:.1f} [exp~20],"
            f" R2={r20:.3f}",
            f"tau~h (free):    R2={r2h:.3f}",
            f"tau~rank:        R2={stats_out.get('r2_rank',np.nan):.3f}",
            f"Pearson: r={rho:.3f}, p={pp:.4f}",
        ]
        ax.text(0.63, 0.95, "STATISTICS",
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color='#1a1a2e')
        for i, line in enumerate(slines):
            ax.text(0.63, 0.86-i*0.09, line,
                    transform=ax.transAxes, fontsize=10,
                    fontfamily='monospace',
                    color='#1a1a2e')
        ax.text(0.63, 0.86-len(slines)*0.09,
                f"VERDICT: {verd}",
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=vc)

    ax.text(0.50, 0.01,
            f"Nonlinear standard map: φ_{{n+1}} = φ_n + ω + κ·sin(φ_n), "
            f"κ={best_kappa}  |  "
            "Null: shuffled Coxeter exponents",
            transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_v6.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_v6.png")


# ══════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(2025)

    print("Part VII v6: Nonlinear standard map")
    print("Testing whether NONLINEARITY reproduces D_corr=3.02\n")

    # Step 1: linear vs nonlinear D_corr
    dcorr_results, traj_lin, traj_nl = \
        compare_linear_vs_nonlinear_dcorr(rng)

    # Step 2: κ scan for E6
    kappa_scan = kappa_scan_e6(rng)

    # Step 3: find best κ (most sig. neg points or τ≈237)
    best_kappa = 0.5   # default; updated by scan
    best_nsig  = -1
    for k, (tau, ns, mdS, mt) in kappa_scan.items():
        if ns > best_nsig:
            best_nsig  = ns
            best_kappa = k
    print(f"\nBest κ from scan: {best_kappa} "
          f"(n_sig={best_nsig})")

    # Step 4: cross-algebra test
    results = run_cross_algebra(best_kappa, rng)

    # Step 5: statistics
    stats_out = statistical_analysis(results)

    # Step 6: figure
    make_figure(kappa_scan, results, stats_out, best_kappa)

    # Final table
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
              f"{tau:>7.0f} {tau/res['h']:>7.1f} "
              f"{D:>9.3f}  {res['status']}"
              if not np.isnan(tau) else
              f"{n:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6} "
              f"{'NaN':>7} {'NaN':>7} "
              f"{'NaN':>9}  {res['status']}")

    print(f"\nConclusion:")
    r2 = stats_out.get('r2_h', np.nan)
    k0 = stats_out.get('k0',   np.nan)
    if not np.isnan(r2):
        if r2 > 0.85:
            print(f"  CONFIRMED: τ ∝ h (R²={r2:.3f}, k={k0:.1f})")
        elif r2 > 0.5:
            print(f"  PARTIAL: τ correlates with h (R²={r2:.3f})")
        else:
            print(f"  REJECTED: τ not proportional to h (R²={r2:.3f})")
    else:
        print("  INCONCLUSIVE: insufficient valid τ measurements")
