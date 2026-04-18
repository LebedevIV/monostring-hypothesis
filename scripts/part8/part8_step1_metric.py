"""
Part VIII — Step 1: Field Metric from Attractor Geometry
=========================================================
Question: What geometry does the E6 orbit define?

Method: Measure the covariance matrix of the orbit point cloud.
Eigenvalues λ₁...λ₆ define the "shape" of the attractor.

If λ₁≈λ₂≈...≈λ₆: isotropic (T⁶, no preferred directions)
If 3 large + 3 small: quasi-3D (matches D_corr≈3)
If ratio λ_max/λ_min differs between algebras: algebra matters

Physical interpretation:
The covariance matrix IS the effective metric tensor of the
emergent space. Its eigenvalues are the "stiffness" of the
field in each direction.

Null hypothesis H0: all eigenvalues equal (isotropic)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# 1. ALGEBRA DEFINITIONS
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': [1, 2, 3, 4, 5, 6],
        'color': '#9C27B0',
        'label': 'A6 (arithmetic)',
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': [1, 4, 5, 7, 8, 11],
        'color': '#2196F3',
        'label': 'E6 (Coxeter)',
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': [1, 5, 7, 9, 11, 13, 17],
        'color': '#FF9800',
        'label': 'E7 (Coxeter)',
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
        'color': '#4CAF50',
        'label': 'E8 (Coxeter)',
    },
    'Random': {
        'rank': 6, 'h': None,
        'exponents': None,
        'color': '#9E9E9E',
        'label': 'Random (null)',
    },
}


def get_omega(alg_name, rng=None):
    """Get frequencies for algebra."""
    alg = ALGEBRAS[alg_name]
    if alg['exponents'] is None:
        # Random null: uniform in [0.5, 2.0]
        if rng is None:
            rng = np.random.default_rng(42)
        return rng.uniform(0.5, 2.0, alg['rank'])
    # Part VI convention: omega_i = 2*sin(π*m_i/h)
    m = np.array(alg['exponents'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])


# ══════════════════════════════════════════════════════════════════
# 2. ORBIT GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_orbit(omega, T=10000, kappa=0.05,
                   warmup=500, seed=42):
    """
    Generate monostring orbit.
    phi_{n+1} = phi_n + omega + kappa*sin(phi_n)  (mod 2π)
    Returns: (T, rank) trajectory array.
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    # Warmup
    for _ in range(warmup):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)

    traj = np.zeros((T, len(omega)))
    for t in range(T):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)
        traj[t] = phi

    return traj


# ══════════════════════════════════════════════════════════════════
# 3. COVARIANCE METRIC ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyse_metric(orbit, alg_name):
    """
    Compute the effective metric tensor from the orbit.

    The covariance matrix C = <φᵢφⱼ> - <φᵢ><φⱼ>
    is the metric tensor of the attractor geometry.

    Returns:
    - eigenvalues (sorted descending)
    - eigenvectors
    - anisotropy ratio λ_max/λ_min
    - isotropy test p-value
    - effective dimension (from eigenvalue spectrum)
    """
    T, rank = orbit.shape

    # Center the orbit
    mean = orbit.mean(axis=0)
    centered = orbit - mean[np.newaxis, :]

    # Covariance matrix (metric tensor)
    C = (centered.T @ centered) / T

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Anisotropy ratio
    ratio = eigenvalues[0] / eigenvalues[-1]

    # Isotropy test: are all eigenvalues equal?
    # Under H0 (isotropic), eigenvalues follow
    # chi² distribution scaled by σ²/rank
    # We use coefficient of variation as test stat
    cv = eigenvalues.std() / eigenvalues.mean()

    # Effective dimension: participation ratio
    # D_eff = (Σλᵢ)² / Σλᵢ²
    # = rank if isotropic, = 1 if rank-1 degenerate
    D_eff = (eigenvalues.sum())**2 / (eigenvalues**2).sum()

    # Signature analysis: how many eigenvalues are
    # significantly above/below median?
    med = np.median(eigenvalues)
    n_large = np.sum(eigenvalues > med * 1.5)
    n_small = np.sum(eigenvalues < med * 0.67)

    return {
        'eigenvalues':  eigenvalues,
        'eigenvectors': eigenvectors,
        'covariance':   C,
        'mean':         mean,
        'ratio':        ratio,
        'cv':           cv,
        'D_eff':        D_eff,
        'n_large':      n_large,
        'n_small':      n_small,
        'rank':         rank,
    }


def print_metric_report(alg_name, result):
    """Print formatted metric analysis."""
    alg = ALGEBRAS[alg_name]
    ev  = result['eigenvalues']

    print(f"\n{'='*55}")
    print(f"ALGEBRA: {alg_name}  "
          f"rank={result['rank']}  "
          f"h={alg['h']}")
    print(f"{'='*55}")
    print(f"\n  Eigenvalues of metric tensor (λ₁ ≥ λ₂ ≥ ...):")
    for i, e in enumerate(ev):
        bar_len = int(40 * e / ev[0])
        bar = '█' * bar_len
        print(f"  λ{i+1} = {e:.6f}  {bar}")

    print(f"\n  Anisotropy ratio λ_max/λ_min = "
          f"{result['ratio']:.3f}")
    print(f"  Coefficient of variation (CV) = "
          f"{result['cv']:.4f}")
    print(f"  Effective dimension D_eff = "
          f"{result['D_eff']:.3f}")
    print(f"  Large eigenvalues (>1.5×median): "
          f"{result['n_large']}")
    print(f"  Small eigenvalues (<0.67×median): "
          f"{result['n_small']}")

    # Interpretation
    print(f"\n  Geometry interpretation:")
    if result['cv'] < 0.05:
        print(f"    → ISOTROPIC: uniform T^rank geometry")
        print(f"    → No preferred directions")
    elif result['D_eff'] < result['rank'] * 0.6:
        print(f"    → ANISOTROPIC: quasi-"
              f"{result['D_eff']:.0f}D submanifold")
        print(f"    → Strong dimensional reduction")
    else:
        print(f"    → MILDLY ANISOTROPIC")

    # Compare eigenvalue ratios to known metrics
    ev_norm = ev / ev.sum()
    print(f"\n  Normalized eigenvalues "
          f"(sum=1): {np.round(ev_norm, 4).tolist()}")


# ══════════════════════════════════════════════════════════════════
# 4. COMPARE ACROSS ALGEBRAS
# ══════════════════════════════════════════════════════════════════

def run_all_algebras(T=10000, kappa=0.05, seed=42):
    """Run metric analysis for all algebras."""
    rng = np.random.default_rng(seed)
    results = {}

    print("=" * 60)
    print("PART VIII Step 1: Field Metric from Attractor")
    print("=" * 60)
    print(f"T={T}, κ={kappa}")

    for alg_name in ALGEBRAS:
        omega = get_omega(alg_name, rng=rng)
        orbit = generate_orbit(
            omega, T=T, kappa=kappa,
            warmup=500, seed=seed)
        result = analyse_metric(orbit, alg_name)
        result['omega'] = omega
        results[alg_name] = result
        print_metric_report(alg_name, result)

    return results


# ══════════════════════════════════════════════════════════════════
# 5. STATISTICAL COMPARISON
# ══════════════════════════════════════════════════════════════════

def statistical_comparison(results):
    """
    Compare metric properties across algebras.

    Key questions:
    1. Is E6 more anisotropic than Random?
    2. Does anisotropy correlate with h?
    3. Do Coxeter algebras form a distinct cluster?
    """
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)

    names  = list(results.keys())
    ratios = np.array([results[n]['ratio']  for n in names])
    d_effs = np.array([results[n]['D_eff']  for n in names])
    cvs    = np.array([results[n]['cv']     for n in names])
    ranks  = np.array([ALGEBRAS[n]['rank']  for n in names])
    h_vals = np.array([
        ALGEBRAS[n]['h'] if ALGEBRAS[n]['h'] else 0
        for n in names])

    print(f"\n{'Algebra':<10} {'λ_max/λ_min':>12} "
          f"{'D_eff':>8} {'CV':>8}")
    print("-" * 42)
    for i, n in enumerate(names):
        print(f"{n:<10} {ratios[i]:>12.3f} "
              f"{d_effs[i]:>8.3f} {cvs[i]:>8.4f}")

    # E6 vs Random comparison
    e6_ratio  = results['E6']['ratio']
    rnd_ratio = results['Random']['ratio']
    e6_deff   = results['E6']['D_eff']
    rnd_deff  = results['Random']['D_eff']

    print(f"\n  E6 vs Random:")
    print(f"    Anisotropy ratio: E6={e6_ratio:.3f}, "
          f"Random={rnd_ratio:.3f}")
    print(f"    D_eff:  E6={e6_deff:.3f}, "
          f"Random={rnd_deff:.3f}")

    if e6_ratio > rnd_ratio:
        print(f"    → E6 MORE anisotropic than Random")
    else:
        print(f"    → E6 LESS anisotropic than Random")

    # Coxeter algebras only
    cox_names  = ['A6', 'E6', 'E7', 'E8']
    cox_ratios = np.array([results[n]['ratio']
                           for n in cox_names])
    cox_h      = np.array([ALGEBRAS[n]['h']
                           for n in cox_names])
    cox_deff   = np.array([results[n]['D_eff']
                           for n in cox_names])

    if len(cox_h) >= 3:
        r_rh, p_rh = stats.pearsonr(cox_h, cox_ratios)
        r_dh, p_dh = stats.pearsonr(cox_h, cox_deff)
        print(f"\n  Coxeter algebras: correlations with h")
        print(f"    r(λ_ratio, h) = {r_rh:+.3f}, "
              f"p = {p_rh:.4f}")
        print(f"    r(D_eff, h)   = {r_dh:+.3f}, "
              f"p = {p_dh:.4f}")

    return {
        'ratios': ratios, 'd_effs': d_effs,
        'cvs': cvs, 'names': names,
        'cox_h': cox_h, 'cox_ratios': cox_ratios,
        'cox_deff': cox_deff,
    }


# ══════════════════════════════════════════════════════════════════
# 6. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(results, stats_out):
    """Create the Part VIII Step 1 figure."""
    alg_list = list(results.keys())

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Part VIII — Step 1: Field Metric from Attractor\n"
        "Eigenvalue spectrum of the covariance tensor\n"
        r"$\omega_i = 2\sin(\pi m_i/h)$, "
        r"$\phi_{n+1}=\phi_n+\omega+0.05\sin\phi$",
        fontsize=13, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.38)

    # ── Row 0: eigenvalue spectra ──────────────────────
    for i, alg_name in enumerate(alg_list):
        if i >= 4:
            break
        res = results[alg_name]
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        ev   = res['eigenvalues']
        rank = res['rank']
        clr  = ALGEBRAS[alg_name]['color']
        x    = np.arange(1, rank + 1)

        ax.bar(x, ev, color=clr, alpha=0.8,
               edgecolor='k', lw=1.2)

        # Isotropic reference line
        iso_val = ev.mean()
        ax.axhline(iso_val, c='red', ls='--',
                   lw=2, alpha=0.7,
                   label=f'isotropic={iso_val:.4f}')

        ax.set_xlabel('Eigenvalue index', fontsize=9)
        ax.set_ylabel('λᵢ', fontsize=9)
        ax.set_title(
            f"{alg_name}: h={ALGEBRAS[alg_name]['h']}\n"
            f"D_eff={res['D_eff']:.2f}  "
            f"ratio={res['ratio']:.2f}\n"
            f"CV={res['cv']:.4f}",
            fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    # ── Row 0 col 3: Random null ───────────────────────
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor('#f5f5f5')
    res = results['Random']
    ev  = res['eigenvalues']
    x   = np.arange(1, res['rank'] + 1)

    ax.bar(x, ev,
           color=ALGEBRAS['Random']['color'],
           alpha=0.8, edgecolor='k', lw=1.2)
    ax.axhline(ev.mean(), c='red', ls='--', lw=2,
               alpha=0.7, label=f'iso={ev.mean():.4f}')
    ax.set_xlabel('Eigenvalue index', fontsize=9)
    ax.set_ylabel('λᵢ', fontsize=9)
    ax.set_title(
        f"Random (null)\n"
        f"D_eff={res['D_eff']:.2f}  "
        f"ratio={res['ratio']:.2f}\n"
        f"CV={res['cv']:.4f}",
        fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Row 1: Anisotropy ratio vs h ──────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    cox_names = ['A6', 'E6', 'E7', 'E8']
    h_vals    = [ALGEBRAS[n]['h'] for n in cox_names]
    ratios    = [results[n]['ratio'] for n in cox_names]
    clrs      = [ALGEBRAS[n]['color'] for n in cox_names]

    for h, r, c, n in zip(h_vals, ratios, clrs, cox_names):
        ax.scatter(h, r, s=220, c=c, zorder=5,
                   edgecolors='k', lw=2)
        ax.annotate(n, xy=(h, r),
                    xytext=(h + 0.4, r + 0.01),
                    fontsize=10, fontweight='bold',
                    color=c)

    # Random null as horizontal band
    rnd_r = results['Random']['ratio']
    ax.axhline(rnd_r, c='gray', ls='--', lw=2,
               alpha=0.7, label=f'Random={rnd_r:.3f}')
    ax.fill_between([5, 33],
                    [rnd_r * 0.9, rnd_r * 0.9],
                    [rnd_r * 1.1, rnd_r * 1.1],
                    alpha=0.1, color='gray',
                    label='±10% band')

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Anisotropy ratio λ_max/λ_min',
                  fontsize=11)
    ax.set_title('Metric anisotropy vs h\n'
                 'Higher → more dimensional reduction',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # ── Row 1: D_eff vs h ─────────────────────────────
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    d_effs = [results[n]['D_eff'] for n in cox_names]
    ranks  = [ALGEBRAS[n]['rank'] for n in cox_names]

    for h, d, r, c, n in zip(h_vals, d_effs, ranks,
                               clrs, cox_names):
        ax.scatter(h, d, s=220, c=c, zorder=5,
                   edgecolors='k', lw=2)
        ax.annotate(f"{n}\n(rank={r})",
                    xy=(h, d),
                    xytext=(h + 0.4, d + 0.05),
                    fontsize=9, fontweight='bold',
                    color=c)

    # Reference lines
    ax.plot([5, 33], [5, 33], ':', c='gray',
            lw=1.5, alpha=0.5, label='D_eff=rank')
    ax.axhline(3.0, c='blue', ls='--', lw=2,
               alpha=0.6, label='D_eff=3 (Part V)')
    rnd_d = results['Random']['D_eff']
    ax.axhline(rnd_d, c='gray', ls='-.',
               lw=2, alpha=0.6,
               label=f'Random D_eff={rnd_d:.2f}')

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Effective dimension D_eff',
                  fontsize=11)
    ax.set_title('Effective dimension vs h\n'
                 'Participation ratio of eigenvalues',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # ── Row 2: Normalized eigenvalue profiles ──────────
    ax = fig.add_subplot(gs[2, 0:3])
    ax.set_facecolor('#f0f4f8')

    for alg_name in ['A6', 'E6', 'E7', 'E8', 'Random']:
        res = results[alg_name]
        ev  = res['eigenvalues']
        # Normalize to sum=1
        ev_n = ev / ev.sum()
        # Pad to max rank for comparison
        max_rank = max(results[n]['rank']
                       for n in results)
        ev_padded = np.zeros(max_rank)
        ev_padded[:len(ev_n)] = ev_n

        x = np.arange(1, max_rank + 1)
        ax.plot(x[:len(ev_n)], ev_n,
                'o-',
                color=ALGEBRAS[alg_name]['color'],
                lw=2.5, ms=10,
                label=ALGEBRAS[alg_name]['label'],
                alpha=0.85)

    ax.set_xlabel('Eigenvalue index (rank order)',
                  fontsize=12)
    ax.set_ylabel('λᵢ / Σλ  (normalized)',
                  fontsize=11)
    ax.set_title(
        'Normalized eigenvalue spectrum: '
        'shape of the effective metric\n'
        'Flat → isotropic (T^rank)  |  '
        'Steep drop → dimensional reduction',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 2 col 3: Interpretation ───────────────────
    ax = fig.add_subplot(gs[2, 3])
    ax.set_facecolor('#e3f2fd')
    ax.axis('off')

    txt = [
        "PHYSICAL INTERPRETATION",
        "=" * 24,
        "",
        "Covariance matrix C:",
        "= metric tensor g_ij",
        "of emergent space",
        "",
        "Eigenvalues λᵢ:",
        "= 'stiffness' of field",
        "in each direction",
        "",
        "Flat spectrum → T⁶",
        "(no preferred direction)",
        "",
        "Steep spectrum →",
        "quasi-3D submanifold",
        "(dimensional reduction)",
        "",
        "D_eff ≈ rank → isotropic",
        "D_eff < rank → reduced",
        "",
        "NEXT STEP:",
        "Do eigenvalue ratios",
        "match Minkowski",
        "signature (+---)?",
    ]
    ax.text(0.05, 0.97, "\n".join(txt),
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            va='top', color='#1a1a2e')

    plt.savefig('part8_step1_metric.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: part8_step1_metric.png")


# ══════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VIII Step 1: Field Metric from Attractor\n")

    results   = run_all_algebras(T=10000,
                                  kappa=0.05,
                                  seed=42)
    stats_out = statistical_comparison(results)
    make_figure(results, stats_out)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Algebra':<10} {'D_eff':>8} "
          f"{'λ_ratio':>9} {'CV':>8}  geometry")
    print("-" * 52)
    for n in results:
        res = results[n]
        geom = ("isotropic" if res['cv'] < 0.05
                else f"quasi-{res['D_eff']:.1f}D")
        print(f"{n:<10} {res['D_eff']:>8.3f} "
              f"{res['ratio']:>9.3f} "
              f"{res['cv']:>8.4f}  {geom}")

    print("\nNull hypothesis H0: all λᵢ equal (isotropic)")
    print("CV < 0.05 → fail to reject H0 (isotropic)")
    print("CV > 0.05 → reject H0 (anisotropic = "
          "dimensional structure)")

    print("\nNext: Step 2 — Collective modes "
          "and dispersion relation")
