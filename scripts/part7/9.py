"""
Part VII FINAL v2: Memory metrics across algebras
=================================================
Fix: np.trapz → np.trapezoid (NumPy 2.0+)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind, pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')

# ── Compatibility fix ─────────────────────────────────────────────
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid


# ══════════════════════════════════════════════════════════════════
# 1. ALGEBRA DEFINITIONS
# ══════════════════════════════════════════════════════════════════

def coxeter_omega(exponents, h):
    """omega_i = 2*sin(π*m_i/h) — Part VI convention."""
    return 2.0 * np.sin(np.pi * np.array(exponents, float) / h)


ALGEBRAS = {
    'A6': {'rank': 6, 'h': 7,
            'exponents': [1, 2, 3, 4, 5, 6],
            'color': '#9C27B0', 'tau_pred': 140},
    'E6': {'rank': 6, 'h': 12,
            'exponents': [1, 4, 5, 7, 8, 11],
            'color': '#2196F3', 'tau_pred': 240},
    'E7': {'rank': 7, 'h': 18,
            'exponents': [1, 5, 7, 9, 11, 13, 17],
            'color': '#FF9800', 'tau_pred': 360},
    'E8': {'rank': 8, 'h': 30,
            'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
            'color': '#4CAF50', 'tau_pred': 600},
}

KAPPA = 0.05
NOISE = 0.006


# ══════════════════════════════════════════════════════════════════
# 2. DYNAMICS (Part VI exact)
# ══════════════════════════════════════════════════════════════════

def get_phi_break(omega, seed=42, warmup=400):
    """Get a point on the monostring attractor."""
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2 * np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega
               + 0.30 * np.sin(phi)) % (2 * np.pi)
    return phi


def shannon_H(phases, bins=15):
    """
    Part VI: SUM (not mean) of per-dim entropies.
    phases: (N, rank)
    """
    H = 0.0
    for d in range(phases.shape[1]):
        hist, _ = np.histogram(
            phases[:, d], bins=bins,
            range=(0, 2 * np.pi))
        p = hist / (hist.sum() + 1e-15)
        H -= np.sum(p * np.log(p + 1e-15))
    return float(H)


def evolve_cloud(phi, omega, T_steps,
                 kappa=KAPPA, noise=NOISE, rng=None):
    """Evolve N-string cloud for T_steps steps."""
    if rng is None:
        rng = np.random.RandomState(0)
    phi = phi.copy()
    for _ in range(T_steps):
        phi = (phi
               + omega[np.newaxis, :]
               + kappa * np.sin(phi)
               + rng.normal(0, noise, phi.shape)
               ) % (2 * np.pi)
    return phi


def measure_dS_profile(omega_A, omega_B,
                        N=200, sigma=0.5,
                        n_runs=30, n_bins=15,
                        t_points=None,
                        seed_base=42):
    """
    Measure ΔS(t) = S(A) - S(B) at each time point.

    Population A: shared origin phi_break, evolves with omega_A
    Population B: random origin,           evolves with omega_B

    This is EXACT Part VI convention (v10 script).
    """
    rank_A = len(omega_A)
    rank_B = len(omega_B)

    if t_points is None:
        t_points = np.unique(np.round(
            np.logspace(np.log10(10),
                        np.log10(900), 22)
        ).astype(int))

    phi_break = get_phi_break(omega_A, seed=seed_base)

    S_A_all = np.zeros((n_runs, len(t_points)))
    S_B_all = np.zeros((n_runs, len(t_points)))

    for run in range(n_runs):
        rng = np.random.RandomState(
            run * 17 + seed_base)

        # Population A: shared origin phi_break
        init_A = (phi_break[np.newaxis, :]
                  + rng.normal(0, sigma, (N, rank_A))
                  ) % (2 * np.pi)

        # Population B: RANDOM origin (Part VI null)
        phi_B0 = rng.uniform(0, 2 * np.pi, rank_B)
        init_B = (phi_B0[np.newaxis, :]
                  + rng.normal(0, sigma, (N, rank_B))
                  ) % (2 * np.pi)

        pos_A = init_A.copy()
        pos_B = init_B.copy()
        prev_t = 0

        for ti, t in enumerate(t_points):
            steps = int(t - prev_t)
            pos_A = evolve_cloud(
                pos_A, omega_A, steps,
                kappa=KAPPA, noise=NOISE, rng=rng)
            pos_B = evolve_cloud(
                pos_B, omega_B, steps,
                kappa=KAPPA, noise=NOISE, rng=rng)
            prev_t = t

            S_A_all[run, ti] = shannon_H(pos_A, n_bins)
            S_B_all[run, ti] = shannon_H(pos_B, n_bins)

    dS_all  = S_A_all - S_B_all
    dS_mean = dS_all.mean(0)
    dS_std  = dS_all.std(0)

    p_vals = np.array([
        ttest_ind(S_A_all[:, i],
                  S_B_all[:, i]).pvalue
        for i in range(len(t_points))
    ])

    return t_points, dS_mean, dS_std, p_vals


# ══════════════════════════════════════════════════════════════════
# 3. MEMORY METRICS
# ══════════════════════════════════════════════════════════════════

def compute_memory_metrics(t_pts, dS_mean, p_vals,
                            p_thresh=0.05):
    """
    Three robust metrics replacing broken τ_crossover:

    T_ord:     ∫ max(0, -ΔS) dt  — integrated ordering area
    f_neg:     fraction of time points with p<thresh AND ΔS<0
    tau_decay: exponential decay time of -ΔS in early phase
    """
    # Metric 1: integrated ordering (trapezoid rule)
    neg_dS = np.maximum(0.0, -dS_mean)
    T_ord  = float(np.trapz(neg_dS, t_pts))

    # Metric 2: fraction significant-negative
    sig_neg = (dS_mean < 0) & (p_vals < p_thresh)
    f_neg   = float(sig_neg.sum()) / len(t_pts)

    # Metric 3: exponential decay fit (early phase only)
    early    = t_pts <= 300
    neg_early = -dS_mean[early]
    t_early   = t_pts[early]
    pos_mask  = neg_early > 0.02

    tau_decay = np.nan
    if pos_mask.sum() >= 3:
        log_neg = np.log(
            np.abs(neg_early[pos_mask]) + 1e-10)
        slope, _, rv, _, _ = linregress(
            t_early[pos_mask].astype(float),
            log_neg)
        if slope < 0 and rv ** 2 > 0.2:
            tau_decay = float(-1.0 / slope)

    return T_ord, f_neg, tau_decay


# ══════════════════════════════════════════════════════════════════
# 4. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_experiment(seed=2025):
    rng_main = np.random.RandomState(seed)

    print("=" * 65)
    print("PART VII FINAL: Memory metrics across algebras")
    print("=" * 65)

    # Shared log-spaced time grid
    t_points = np.unique(np.round(
        np.logspace(np.log10(10),
                    np.log10(900), 22)
    ).astype(int))

    results = {}

    for alg_name, alg in ALGEBRAS.items():
        h    = alg['h']
        rank = alg['rank']

        print(f"\n{'=' * 55}")
        print(f"ALGEBRA: {alg_name}  h={h}  rank={rank}")

        omega      = coxeter_omega(alg['exponents'], h)
        seed_alg   = int(rng_main.randint(1000))
        rng_shuf   = np.random.RandomState(seed_alg)
        omega_shuf = rng_shuf.permutation(omega.copy())

        print(f"  omega : {np.round(omega, 3).tolist()}")
        print(f"  shuf  : "
              f"{np.round(omega_shuf, 3).tolist()}")

        # ΔS profile
        t_pts, dS_m, dS_s, p_v = measure_dS_profile(
            omega_A=omega,
            omega_B=omega_shuf,
            N=200, sigma=0.5,
            n_runs=30, n_bins=15,
            t_points=t_points,
            seed_base=seed_alg + 100)

        # Memory metrics
        T_ord, f_neg, tau_decay = compute_memory_metrics(
            t_pts, dS_m, p_v)

        # Print profile (selective)
        print(f"\n  {'t':>5}  {'ΔS':>9}  "
              f"{'p':>9}  sig")
        print(f"  {'-' * 38}")
        for i in range(len(t_pts)):
            sig = ("***" if p_v[i] < 0.001 else
                   "**"  if p_v[i] < 0.01  else
                   "*"   if p_v[i] < 0.05  else
                   "."   if p_v[i] < 0.10  else "")
            if p_v[i] < 0.10 or i % 4 == 0 or i < 5:
                print(f"  {t_pts[i]:>5}  "
                      f"{dS_m[i]:>+9.4f}  "
                      f"{p_v[i]:>9.4f}  {sig}")

        n_sig_neg = int(np.sum((p_v < 0.05)
                               & (dS_m < 0)))
        n_sig_pos = int(np.sum((p_v < 0.05)
                               & (dS_m > 0)))

        print(f"\n  n_sig-={n_sig_neg}  n_sig+={n_sig_pos}")
        print(f"  T_ord    = {T_ord:.1f}")
        print(f"  f_neg    = {f_neg:.3f}")
        print(f"  tau_decay= "
              f"{'N/A' if np.isnan(tau_decay) else f'{tau_decay:.1f}'}")

        results[alg_name] = {
            'h': h, 'rank': rank,
            'tau_pred':   alg['tau_pred'],
            'omega':      omega,
            't_pts':      t_pts,
            'dS_mean':    dS_m,
            'dS_std':     dS_s,
            'p_vals':     p_v,
            'T_ord':      T_ord,
            'f_neg':      f_neg,
            'tau_decay':  tau_decay,
            'n_sig_neg':  n_sig_neg,
            'n_sig_pos':  n_sig_pos,
            'color':      alg['color'],
        }

    return results


# ══════════════════════════════════════════════════════════════════
# 5. STATISTICS
# ══════════════════════════════════════════════════════════════════

def analyse_metrics(results):
    print("\n" + "=" * 65)
    print("METRIC ANALYSIS: Which metric ∝ h?")
    print("=" * 65)

    names  = list(results.keys())
    h_arr  = np.array([results[n]['h']        for n in names])
    r_arr  = np.array([results[n]['rank']      for n in names])
    To_arr = np.array([results[n]['T_ord']     for n in names])
    fn_arr = np.array([results[n]['f_neg']     for n in names])
    td_arr = np.array([results[n]['tau_decay'] for n in names])
    nn_arr = np.array([results[n]['n_sig_neg'] for n in names],
                      dtype=float)

    print(f"\n{'Alg':<5} {'h':>4} "
          f"{'T_ord':>9} {'f_neg':>7} "
          f"{'tau_d':>8} {'n_sig-':>7}")
    print("-" * 50)
    for i, n in enumerate(names):
        td_s = (f"{td_arr[i]:>8.1f}"
                if not np.isnan(td_arr[i])
                else f"{'N/A':>8}")
        print(f"{n:<5} {h_arr[i]:>4} "
              f"{To_arr[i]:>9.1f} "
              f"{fn_arr[i]:>7.3f}"
              f"{td_s} "
              f"{nn_arr[i]:>7.0f}")

    out = {}
    print(f"\n{'Metric':<15}  "
          f"{'r(M,h)':>8}  {'p_h':>8}  "
          f"{'r(M,rank)':>10}  verdict")
    print("-" * 62)

    for metric_name, m_arr in [
            ('T_ord',     To_arr),
            ('f_neg',     fn_arr),
            ('n_sig_neg', nn_arr),
            ('tau_decay', td_arr)]:

        valid = np.isfinite(m_arr)
        if valid.sum() < 3:
            print(f"{metric_name:<15}  "
                  f"{'N/A':>8}  (too few points)")
            continue

        h_v = h_arr[valid]
        r_v = r_arr[valid]
        m_v = m_arr[valid]

        rho_h, p_h = pearsonr(h_v, m_v)
        rho_r, p_r = pearsonr(r_v, m_v)

        if abs(rho_h) > 0.85 and p_h < 0.05:
            verdict = "STRONG ∝h"
        elif abs(rho_h) > 0.6 and p_h < 0.15:
            verdict = "moderate ∝h"
        elif abs(rho_r) > abs(rho_h):
            verdict = f"∝rank (r={rho_r:+.2f})"
        else:
            verdict = "none"

        print(f"{metric_name:<15}  "
              f"{rho_h:>+8.3f}  {p_h:>8.4f}  "
              f"{rho_r:>+10.3f}  {verdict}")

        out[metric_name] = {
            'rho_h': rho_h, 'p_h':   p_h,
            'rho_r': rho_r, 'p_r':   p_r,
            'vals':  m_v,   'h_vals': h_v,
        }

    return out


# ══════════════════════════════════════════════════════════════════
# 6. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(results, metric_stats):
    alg_list = list(results.keys())

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII Final\n"
        r"Memory metrics: $T_{\rm ord}$, $f_{\rm neg}$, "
        r"$n_{\rm sig-}$ across A6, E6, E7, E8"
        "\n"
        r"$\omega_i=2\sin(\pi m_i/h)$ | "
        r"$\phi_{n+1}=\phi_n+\omega+0.05\sin\phi"
        r"+\mathcal{N}(0,0.006)$ | null=random-origin",
        fontsize=12, fontweight='bold', y=0.995,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.38)

    # ── Row 0: ΔS(t) per algebra ──────────────────────
    for i, alg_name in enumerate(alg_list):
        res = results[alg_name]
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        t  = res['t_pts']
        dS = res['dS_mean']
        pv = res['p_vals']
        c  = res['color']

        # Colour regions
        for j in range(len(t) - 1):
            mid = (dS[j] + dS[j + 1]) / 2.0
            col = '#c8e6c9' if mid < 0 else '#ffcdd2'
            ax.fill_between(t[j:j + 2],
                            dS[j:j + 2], 0,
                            alpha=0.4, color=col)

        ax.plot(t, dS, '-', color=c, lw=2.2, zorder=4)
        ax.axhline(0, c='k', lw=1.8)

        for j in range(len(t)):
            if pv[j] < 0.01 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=70,
                           c='#2E7D32', zorder=6,
                           edgecolors='k', lw=0.7)
            elif pv[j] < 0.01 and dS[j] > 0:
                ax.scatter(t[j], dS[j], s=70,
                           c='#C62828', zorder=6,
                           edgecolors='k', lw=0.7)
            elif pv[j] < 0.05:
                ax.scatter(t[j], dS[j], s=40,
                           c='#FF9800', zorder=5,
                           edgecolors='k', lw=0.5)

        ax.set_title(
            f"{alg_name}: h={res['h']}, rank={res['rank']}\n"
            f"T_ord={res['T_ord']:.0f}  "
            f"f_neg={res['f_neg']:.2f}  "
            f"n_sig-={res['n_sig_neg']}",
            fontsize=9, fontweight='bold')
        ax.set_xlabel('t (log scale)', fontsize=9)
        ax.set_ylabel('ΔS = S(Cox)−S(Shuf)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    # ── Row 1: three metrics vs h ──────────────────────
    metric_list = [
        ('T_ord',     'T_ord (integrated ordering)'),
        ('f_neg',     'f_neg (frac sig-neg)'),
        ('n_sig_neg', 'n_sig- (count sig-neg)'),
    ]

    for col, (mkey, mlabel) in enumerate(metric_list):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#f0f4f8')

        h_vals = np.array([results[n]['h']   for n in alg_list])
        m_vals = np.array([results[n][mkey]  for n in alg_list],
                          dtype=float)
        clrs   = [results[n]['color'] for n in alg_list]

        for h_v, m_v, c, n in zip(h_vals, m_vals,
                                    clrs, alg_list):
            if np.isfinite(m_v):
                ax.scatter(h_v, m_v, s=220, c=c,
                           zorder=5, edgecolors='k',
                           lw=2)
                ax.annotate(
                    n, xy=(h_v, m_v),
                    xytext=(h_v + 0.4, m_v),
                    fontsize=10, fontweight='bold',
                    color=c)

        # Fit line if moderate correlation
        if mkey in metric_stats:
            ms  = metric_stats[mkey]
            rho = ms['rho_h']
            p_h = ms['p_h']
            if abs(rho) > 0.5:
                sl, ic, _, _, _ = linregress(
                    ms['h_vals'].astype(float),
                    ms['vals'])
                h_l = np.linspace(5, 33, 100)
                ax.plot(h_l, sl * h_l + ic,
                        '--', c='red', lw=2,
                        label=f"r={rho:+.3f}, "
                              f"p={p_h:.3f}")
                ax.legend(fontsize=9)

        ax.set_xlabel('Coxeter number h', fontsize=11)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f'{mlabel} vs h',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([4, 34])

    # ── Row 1 col 3: stats summary ─────────────────────
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor('#eef2ff')
    ax.axis('off')

    lines = ["PEARSON r(metric, h)", "=" * 26, ""]
    best_rho = 0.0
    best_key = None

    for mkey in ['T_ord', 'f_neg',
                 'n_sig_neg', 'tau_decay']:
        if mkey in metric_stats:
            ms  = metric_stats[mkey]
            rh  = ms['rho_h']
            ph  = ms['p_h']
            sig = ("***" if ph < 0.001 else
                   "**"  if ph < 0.01  else
                   "*"   if ph < 0.05  else
                   "."   if ph < 0.10  else "ns")
            lines.append(
                f"{mkey:<12}: {rh:+.3f} {sig}")
            if abs(rh) > abs(best_rho):
                best_rho = rh
                best_key = mkey

    lines += ["", "Prediction: τ∝h → r≈+1.0", ""]

    if abs(best_rho) > 0.85:
        verdict_txt = "STRONG: memory ∝ h"
        vc = '#2E7D32'
    elif abs(best_rho) > 0.6:
        verdict_txt = "MODERATE correlation"
        vc = '#FF9800'
    else:
        verdict_txt = "WEAK: τ∝h not confirmed"
        vc = '#C62828'

    lines += [f"Best: {best_key}",
              f"r = {best_rho:+.3f}",
              "",
              verdict_txt]

    ax.text(0.05, 0.97, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace',
            va='top', color='#1a1a2e')
    ax.set_title('Statistics', fontsize=10,
                 fontweight='bold', color=vc)

    # ── Row 2: final table ─────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    ax.text(0.01, 0.97,
            "PART VII FINAL — RESULTS TABLE",
            fontsize=12, fontweight='bold',
            transform=ax.transAxes,
            color='#1a1a2e')

    hdr = (f"{'Alg':<5} {'h':<4} {'rank':<5} "
           f"{'pred':>6} {'T_ord':>9} "
           f"{'f_neg':>7} {'n_sig-':>7} "
           f"{'n_sig+':>7}  verdict")
    ax.text(0.01, 0.85, hdr,
            transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            fontweight='bold', color='#1a1a2e')
    ax.text(0.01, 0.80, "-" * 72,
            transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            color='gray')

    y = 0.73
    for alg_name, res in results.items():
        nn = res['n_sig_neg']
        np_ = res['n_sig_pos']
        note = ("ordering" if nn > np_ else
                "ANTI-ord!" if np_ > nn else "mixed")
        row = (f"{alg_name:<5} {res['h']:<4} "
               f"{res['rank']:<5} "
               f"{res['tau_pred']:>6} "
               f"{res['T_ord']:>9.0f} "
               f"{res['f_neg']:>7.3f} "
               f"{nn:>7} "
               f"{np_:>7}  {note}")
        ax.text(0.01, y, row,
                transform=ax.transAxes,
                fontsize=10,
                fontfamily='monospace',
                color=res['color'])
        y -= 0.10

    # Verdict box
    ax.text(0.65, 0.90,
            f"FINAL VERDICT:\n{verdict_txt}\n\n"
            f"Best metric: {best_key}\n"
            f"r(metric, h) = {best_rho:+.3f}\n\n"
            "Physical note:\n"
            "ΔS oscillates quasi-periodically\n"
            "due to near-resonances in\n"
            "omega=2*sin(pi*m/h).\n"
            "Integrated metrics (T_ord)\n"
            "are more robust than τ_crossover.",
            transform=ax.transAxes,
            fontsize=10.5, fontfamily='monospace',
            va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#fff9c4',
                      alpha=0.9,
                      edgecolor=vc,
                      linewidth=2))

    plt.savefig('monostring_part7_final.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_final.png")


# ══════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VII Final v2: Memory metrics\n")
    print(f"NumPy version: {np.__version__}")
    print(f"trapz available: {hasattr(np, 'trapz')}\n")

    results      = run_experiment(seed=2025)
    metric_stats = analyse_metrics(results)
    make_figure(results, metric_stats)

    # ── Final summary ──────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    print(f"\n{'Alg':<5} {'h':>4} "
          f"{'T_ord':>9} {'f_neg':>7} "
          f"{'n_sig-':>7} {'n_sig+':>7}")
    print("-" * 45)
    for n, res in results.items():
        print(f"{n:<5} {res['h']:>4} "
              f"{res['T_ord']:>9.1f} "
              f"{res['f_neg']:>7.3f} "
              f"{res['n_sig_neg']:>7} "
              f"{res['n_sig_pos']:>7}")

    print("\nCorrelations with h:")
    best_rho = 0.0
    best_key = None
    for mkey in ['T_ord', 'f_neg', 'n_sig_neg']:
        if mkey in metric_stats:
            ms  = metric_stats[mkey]
            rho = ms['rho_h']
            ph  = ms['p_h']
            print(f"  {mkey:<12}: r={rho:+.3f}, p={ph:.4f}")
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_key = mkey

    print(f"\nConclusion:")
    if best_key and abs(best_rho) > 0.85:
        print(f"  CONFIRMED: {best_key} ∝ h  "
              f"(r={best_rho:+.3f})")
    elif best_key and abs(best_rho) > 0.6:
        print(f"  PARTIAL: {best_key} correlates with h  "
              f"(r={best_rho:+.3f})")
    else:
        print(f"  REJECTED: no memory metric scales with h")
        print(f"  Best r={best_rho:+.3f} ({best_key})")

    print("\nDone.")