"""
Rescue Experiment v5: Omega Scan
=================================
Key insight from v4: omega (frequencies) explain 66% of d_s variance.
Strategy:
  1) Fix spectral dimension: use FULL eigendecomposition
  2) Scan omega from E6 to uniform at K=0 (clean, stable)
  3) Find omega* where d_s = 4.0
  4) Test if E6 exponents are special
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  FREQUENCY VECTORS
# ═══════════════════════════════════════════════════════════════

def omega_E6():
    """E6 exponents {1,4,5,7,8,11}, Coxeter h=12"""
    return 2.0 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)

def omega_uniform():
    """All frequencies equal"""
    return np.full(6, np.mean(omega_E6()))

def omega_interpolated(beta):
    """beta=0 → E6,  beta=1 → uniform"""
    return (1 - beta) * omega_E6() + beta * omega_uniform()

def omega_from_exponents(exponents, h):
    """General: ω_i = 2·sin(π·m_i / h)"""
    return 2.0 * np.sin(np.pi * np.array(exponents) / h)


# ═══════════════════════════════════════════════════════════════
#  CARTAN MATRICES
# ═══════════════════════════════════════════════════════════════

def cartan_E6():
    K = np.diag([2.0] * 6)
    for i, j in [(0,1), (1,2), (2,3), (3,4), (2,5)]:
        K[i, j] = K[j, i] = -1.0
    return K


# ═══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def evolve_phases(N, K, kappa, omega, seed=0):
    """φ_{n+1} = φ_n + ω + κ·K·sin(φ_n) mod 2π"""
    rng = np.random.RandomState(seed)
    D = len(omega)
    phases = np.zeros((N, D))
    phases[0] = rng.uniform(0, 2 * np.pi, D)
    for n in range(N - 1):
        coupling = kappa * K @ np.sin(phases[n])
        phases[n + 1] = (phases[n] + omega + coupling) % (2 * np.pi)
    return phases


def build_graph(phases, eps=1.5, max_conn=5, n_cand=80, seed=42):
    """Chronological chain + phase-proximity shortcuts"""
    N = phases.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N - 1):
        G.add_edge(i, i + 1)

    rng = np.random.RandomState(seed)
    degrees = np.zeros(N, dtype=np.float64)
    for i in range(N - 1):
        degrees[i] += 1.0
        degrees[i + 1] += 1.0

    for i in range(2, N):
        w = degrees[:i] + 1.0
        w /= w.sum()
        pool = min(i, n_cand)
        cands = rng.choice(i, pool, p=w, replace=False)

        diffs = np.abs(phases[i] - phases[cands])
        diffs = np.minimum(diffs, 2 * np.pi - diffs)
        dists = np.linalg.norm(diffs, axis=1)
        close = cands[dists < eps]

        added = 0
        for j in close:
            if added >= max_conn:
                break
            if not G.has_edge(i, int(j)):
                G.add_edge(i, int(j))
                degrees[i] += 1.0
                degrees[int(j)] += 1.0
                added += 1
    return G


def spectral_dimension_full(G, t_lo=0.1, t_hi=100, n_t=300):
    """
    Heat-kernel spectral dimension using FULL eigendecomposition.
    d_s(t) = 2t · Σ λ_i e^{-λ_i t} / Σ e^{-λ_i t}
    
    Fix vs v4: uses ALL eigenvalues, not just 200 smallest.
    """
    N = G.number_of_nodes()
    if N < 30:
        return 0.0, np.array([]), np.array([])

    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    eigs = eigs[eigs > 1e-10]  # drop zero modes

    if len(eigs) < 10:
        return 0.0, np.array([]), np.array([])

    t_range = np.logspace(np.log10(t_lo), np.log10(t_hi), n_t)
    ds_curve = np.zeros(n_t)

    for idx, t in enumerate(t_range):
        exps = np.exp(-eigs * t)
        Z = exps.sum()
        if Z > 1e-30:
            ds_curve[idx] = 2.0 * t * np.sum(eigs * exps) / Z

    # Plateau detection: find flattest window
    valid = (ds_curve > 0.3) & (ds_curve < 15)
    if np.sum(valid) < 10:
        pos = ds_curve[ds_curve > 0]
        val = float(np.median(pos)) if len(pos) else 0.0
        return val, t_range, ds_curve

    dsv = ds_curve[valid]
    tv  = t_range[valid]
    w = min(30, len(dsv) // 3)
    if w < 3:
        return float(np.median(dsv)), t_range, ds_curve

    best_std, best_center = np.inf, len(dsv) // 2
    for s in range(len(dsv) - w):
        std = np.std(dsv[s:s + w])
        if std < best_std:
            best_std = std
            best_center = s + w // 2

    lo = max(0, best_center - w)
    hi = min(len(dsv), best_center + w)
    plateau_val = float(np.median(dsv[lo:hi]))

    return plateau_val, t_range, ds_curve


def frequency_spread(omega):
    """Measure how spread out the frequencies are (0=uniform, 1=maximally spread)"""
    return np.std(omega) / (np.mean(omega) + 1e-10)


def phase_recurrence(phases, eps=1.5, n_pairs=20000, seed=42):
    """Fraction of random phase-pair distances below eps"""
    N = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, N, (n_pairs, 2))
    diffs = np.abs(phases[idx[:, 0]] - phases[idx[:, 1]])
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.mean(dists < eps))


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 1: OMEGA SCAN (K=0, clean measurements)
# ═══════════════════════════════════════════════════════════════

def experiment_omega_scan(N=1500, n_beta=20, n_runs=8):
    """Scan omega from E6 to uniform, K=0"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: OMEGA INTERPOLATION SCAN (K=0)")
    print("  ω(β) = (1-β)·ω_E6 + β·ω_uniform")
    print("  β=0 → E6 frequencies,  β=1 → uniform frequencies")
    print(f"  N={N}, {n_runs} runs per point, {n_beta} points")
    print("=" * 72)

    K = np.zeros((6, 6))  # K=0: no coupling, clean measurements
    kappa = 0.0            # irrelevant when K=0

    betas = np.linspace(0, 1, n_beta)
    results = []

    hdr = f"  {'beta':>6s} | {'d_s':>8s} {'+-':>5s} | {'recur':>8s} | {'<k>':>6s} | {'spread':>7s}"
    print(hdr)
    print("  " + "-" * 56)

    for beta in betas:
        om = omega_interpolated(beta)
        ds_list, rec_list, deg_list = [], [], []

        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds, _, _ = spectral_dimension_full(G)
            ds_list.append(ds)
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'beta': beta,
            'omega': om.copy(),
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'ds_sem': np.std(ds_list) / np.sqrt(n_runs),
            'rec': np.mean(rec_list),
            'deg': np.mean(deg_list),
            'spread': frequency_spread(om),
            'ds_raw': ds_list,
        }
        results.append(row)

        print(f"  {beta:6.3f} | {row['ds_m']:7.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}"
              f" | {row['spread']:7.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 2: OMEGA SCAN (K=E6, with coupling)
# ═══════════════════════════════════════════════════════════════

def experiment_omega_scan_coupled(N=1500, n_beta=12, n_runs=8, kappa=0.5):
    """Same scan but with E6 coupling"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: OMEGA SCAN WITH E6 COUPLING (K=K_E6)")
    print(f"  kappa={kappa}, N={N}, {n_runs} runs, {n_beta} points")
    print("=" * 72)

    K = cartan_E6()
    betas = np.linspace(0, 1, n_beta)
    results = []

    hdr = f"  {'beta':>6s} | {'d_s':>8s} {'+-':>5s} | {'recur':>8s} | {'<k>':>6s}"
    print(hdr)
    print("  " + "-" * 45)

    for beta in betas:
        om = omega_interpolated(beta)
        ds_list, rec_list, deg_list = [], [], []

        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds, _, _ = spectral_dimension_full(G)
            ds_list.append(ds)
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'beta': beta,
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'ds_sem': np.std(ds_list) / np.sqrt(n_runs),
            'rec': np.mean(rec_list),
            'deg': np.mean(deg_list),
            'ds_raw': ds_list,
        }
        results.append(row)

        print(f"  {beta:6.3f} | {row['ds_m']:7.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 3: ALTERNATIVE ALGEBRAS
# ═══════════════════════════════════════════════════════════════

def experiment_other_algebras(N=1500, n_runs=8):
    """Test non-E6 exponent systems at K=0"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: FREQUENCY SYSTEMS FROM DIFFERENT ALGEBRAS (K=0)")
    print("=" * 72)

    # Exponents and Coxeter numbers for rank-6 algebras
    algebras = [
        ('E6',  [1, 4, 5, 7, 8, 11], 12),
        ('A6',  [1, 2, 3, 4, 5, 6],   7),
        ('D6',  [1, 3, 5, 7, 9, 5],  10),  # exponents of D6
        ('B6',  [1, 3, 5, 7, 9, 11], 12),
        ('C6',  [1, 3, 5, 7, 9, 11], 12),  # same exponents as B6
    ]

    # Also test some hand-crafted frequency sets
    special = [
        ('Golden',  2.0 * np.sin(np.pi * np.array([1, 2, 3, 5, 8, 13]) / 21)),
        ('Primes',  2.0 * np.sin(np.pi * np.array([2, 3, 5, 7, 11, 13]) / 17)),
        ('Linear',  np.linspace(0.5, 2.0, 6)),
        ('Uniform', omega_uniform()),
    ]

    K = np.zeros((6, 6))
    kappa = 0.0
    results = []

    hdr = f"  {'Name':<12s} | {'d_s':>8s} {'+-':>5s} | {'recur':>8s} | {'<k>':>6s} | {'spread':>7s} | freqs"
    print(hdr)
    print("  " + "-" * 80)

    # Algebra-derived frequencies
    for name, exps, h in algebras:
        om = omega_from_exponents(exps, h)
        ds_list = []
        rec_list = []
        deg_list = []
        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds, _, _ = spectral_dimension_full(G)
            ds_list.append(ds)
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'name': name, 'omega': om,
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'rec': np.mean(rec_list), 'deg': np.mean(deg_list),
            'spread': frequency_spread(om),
        }
        results.append(row)
        freq_str = ','.join(f'{x:.2f}' for x in om)
        print(f"  {name:<12s} | {row['ds_m']:7.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}"
              f" | {row['spread']:7.4f} | [{freq_str}]")

    # Special frequency sets
    for name, om in special:
        ds_list = []
        rec_list = []
        deg_list = []
        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds, _, _ = spectral_dimension_full(G)
            ds_list.append(ds)
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'name': name, 'omega': om,
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'rec': np.mean(rec_list), 'deg': np.mean(deg_list),
            'spread': frequency_spread(om),
        }
        results.append(row)
        freq_str = ','.join(f'{x:.2f}' for x in om)
        print(f"  {name:<12s} | {row['ds_m']:7.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}"
              f" | {row['spread']:7.4f} | [{freq_str}]")

    return results


# ═══════════════════════════════════════════════════════════════
#  PRECISION MEASUREMENT
# ═══════════════════════════════════════════════════════════════

def precision_at_crossing(results_scan, N=1500, n_precision=20, K=None, kappa=0.0):
    """Find d_s=4.0 crossing and do precision measurement"""
    print("\n" + "=" * 72)
    print("  PRECISION MEASUREMENT AT d_s = 4.0 CROSSING")
    print("=" * 72)

    if K is None:
        K = np.zeros((6, 6))

    betas = np.array([r['beta'] for r in results_scan])
    ds_arr = np.array([r['ds_m'] for r in results_scan])

    target = 4.0
    crossing = None
    for i in range(len(ds_arr) - 1):
        if (ds_arr[i] - target) * (ds_arr[i + 1] - target) < 0:
            frac = (target - ds_arr[i]) / (ds_arr[i + 1] - ds_arr[i])
            crossing = betas[i] + frac * (betas[i + 1] - betas[i])
            break

    if crossing is None:
        print(f"  d_s = 4.0 NOT found in scan range")
        print(f"  d_s range: [{ds_arr.min():.3f}, {ds_arr.max():.3f}]")
        if ds_arr.max() < 4.0:
            print("  All d_s < 4.0")
        elif ds_arr.min() > 4.0:
            print("  All d_s > 4.0")
        return None

    print(f"  Crossing detected at beta* ~ {crossing:.4f}")
    om_star = omega_interpolated(crossing)
    print(f"  omega* = [{', '.join(f'{x:.4f}' for x in om_star)}]")
    print(f"  spread = {frequency_spread(om_star):.4f}")

    # Precision measurement
    print(f"\n  Running {n_precision} precision measurements...")
    ds_vals = []
    for r in range(n_precision):
        ph = evolve_phases(N, K, kappa, om_star, seed=r * 997 + 31)
        G = build_graph(ph, seed=r * 1009 + 37)
        ds, _, _ = spectral_dimension_full(G)
        ds_vals.append(ds)
        if (r + 1) % 5 == 0:
            running_m = np.mean(ds_vals)
            running_s = np.std(ds_vals)
            print(f"    runs 1-{r+1}: d_s = {running_m:.3f} +- {running_s:.3f}")

    vp = np.array(ds_vals)
    sem = vp.std() / np.sqrt(len(vp))
    ci = 1.96 * sem
    hit = abs(vp.mean() - 4.0) < ci

    print(f"\n  PRECISION RESULT:")
    print(f"    d_s = {vp.mean():.4f} +- {vp.std():.4f}")
    print(f"    SEM = {sem:.4f}")
    print(f"    95% CI: [{vp.mean() - ci:.4f}, {vp.mean() + ci:.4f}]")
    print(f"    d_s = 4.0 within 95% CI: {'YES' if hit else 'NO'}")
    print(f"    |d_s - 4.0| = {abs(vp.mean() - 4.0):.4f}")

    result = {
        'beta_star': crossing,
        'omega_star': om_star,
        'ds_mean': vp.mean(), 'ds_std': vp.std(),
        'sem': sem, 'ci': ci, 'hit': hit,
        'ds_raw': vp,
    }
    return result


# ═══════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════

def make_plots(res1, res2, res3, precision):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── (0,0) Omega scan K=0 ──
    ax = axes[0, 0]
    betas1 = [r['beta'] for r in res1]
    ds1_m = [r['ds_m'] for r in res1]
    ds1_s = [r['ds_s'] for r in res1]
    ax.errorbar(betas1, ds1_m, yerr=ds1_s, fmt='o-', color='#2ca02c',
                capsize=3, ms=5, lw=2, label='K=0 (no coupling)')
    ax.axhline(4.0, color='red', ls=':', lw=1.5, alpha=0.8, label='d_s = 4.0')
    ax.fill_between(betas1, 3.8, 4.2, color='gold', alpha=0.15)
    if precision is not None:
        ax.axvline(precision['beta_star'], color='green', ls='--', lw=2,
                   alpha=0.8, label=f'beta* = {precision["beta_star"]:.3f}')
        ax.plot(precision['beta_star'], precision['ds_mean'], '*',
                color='green', ms=15, zorder=5)
    ax.set_xlabel('beta  (0=E6, 1=uniform)', fontsize=11)
    ax.set_ylabel('Spectral dimension d_s', fontsize=11)
    ax.set_title('Exp 1: Omega scan (K=0, clean)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # ── (0,1) Omega scan K=E6 ──
    ax = axes[0, 1]
    if res2:
        betas2 = [r['beta'] for r in res2]
        ds2_m = [r['ds_m'] for r in res2]
        ds2_s = [r['ds_s'] for r in res2]
        ax.errorbar(betas2, ds2_m, yerr=ds2_s, fmt='D-', color='#d62728',
                    capsize=3, ms=5, lw=2, label='K=E6 (coupled)')
        # Overlay K=0 for comparison
        ax.plot(betas1, ds1_m, 'o--', color='#2ca02c', ms=3, lw=1,
                alpha=0.5, label='K=0 (ref)')
    ax.axhline(4.0, color='red', ls=':', lw=1.5, alpha=0.8)
    ax.set_xlabel('beta  (0=E6, 1=uniform)', fontsize=11)
    ax.set_ylabel('Spectral dimension d_s', fontsize=11)
    ax.set_title('Exp 2: Omega scan (K=E6, coupled)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # ── (1,0) d_s vs frequency spread ──
    ax = axes[1, 0]
    # Combine all results
    all_names = [r['name'] for r in res3]
    all_ds = [r['ds_m'] for r in res3]
    all_ds_e = [r['ds_s'] for r in res3]
    all_spread = [r['spread'] for r in res3]

    ax.errorbar(all_spread, all_ds, yerr=all_ds_e, fmt='s', color='indigo',
                capsize=4, ms=7, lw=1.5)
    for i, nm in enumerate(all_names):
        ax.annotate(nm, (all_spread[i], all_ds[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.8)
    ax.axhline(4.0, color='red', ls=':', lw=1.5, alpha=0.8, label='d_s = 4.0')

    if len(all_spread) > 2:
        sp = np.array(all_spread)
        ds = np.array(all_ds)
        valid = np.isfinite(sp) & np.isfinite(ds)
        if np.sum(valid) > 2 and np.std(sp[valid]) > 1e-10:
            corr = np.corrcoef(sp[valid], ds[valid])[0, 1]
            ax.set_title(f'Exp 3: d_s vs frequency spread (r={corr:+.3f})',
                         fontsize=12)
        else:
            ax.set_title('Exp 3: d_s vs frequency spread', fontsize=12)
    else:
        ax.set_title('Exp 3: d_s vs frequency spread', fontsize=12)

    ax.set_xlabel('Frequency spread (std/mean)', fontsize=11)
    ax.set_ylabel('Spectral dimension d_s', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # ── (1,1) d_s vs recurrence ──
    ax = axes[1, 1]
    all_rec = [r['rec'] for r in res3]
    ax.errorbar(all_rec, all_ds, yerr=all_ds_e, fmt='o', color='teal',
                capsize=4, ms=7, lw=1.5)
    for i, nm in enumerate(all_names):
        ax.annotate(nm, (all_rec[i], all_ds[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, alpha=0.8)
    ax.axhline(4.0, color='red', ls=':', lw=1.5, alpha=0.8, label='d_s = 4.0')

    if len(all_rec) > 2:
        rc = np.array(all_rec)
        valid = np.isfinite(rc) & np.isfinite(ds)
        if np.sum(valid) > 2 and np.std(rc[valid]) > 1e-10:
            corr = np.corrcoef(rc[valid], ds[valid])[0, 1]
            ax.set_title(f'd_s vs recurrence rate (r={corr:+.3f})',
                         fontsize=12)
        else:
            ax.set_title('d_s vs recurrence rate', fontsize=12)
    else:
        ax.set_title('d_s vs recurrence rate', fontsize=12)

    ax.set_xlabel('Phase recurrence fraction', fontsize=11)
    ax.set_ylabel('Spectral dimension d_s', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('Rescue v5: What Frequency Structure Gives d_s = 4?',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('rescue_v5_omega_scan.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  [+] Figure saved: rescue_v5_omega_scan.png")


# ═══════════════════════════════════════════════════════════════
#  VERDICT
# ═══════════════════════════════════════════════════════════════

def print_verdict(res1, res2, res3, precision):
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    # 1. K=0 scan
    ds_K0 = np.array([r['ds_m'] for r in res1])
    print(f"\n  1. OMEGA SCAN (K=0):")
    print(f"     d_s range: [{ds_K0.min():.3f}, {ds_K0.max():.3f}]")
    print(f"     d_s at beta=0 (E6): {ds_K0[0]:.3f}")
    print(f"     d_s at beta=1 (uniform): {ds_K0[-1]:.3f}")

    if ds_K0.min() < 4.0 < ds_K0.max():
        print("     d_s = 4.0 IS in range -> crossing exists")
    else:
        print("     d_s = 4.0 NOT in range")

    # 2. Coupling effect
    if res2:
        ds_KE6 = np.array([r['ds_m'] for r in res2])
        ds_K0_matched = np.interp(
            [r['beta'] for r in res2],
            [r['beta'] for r in res1],
            [r['ds_m'] for r in res1]
        )
        coupling_effect = np.mean(np.abs(ds_KE6 - ds_K0_matched))
        print(f"\n  2. COUPLING EFFECT (K=E6 vs K=0):")
        print(f"     Mean |d_s(K=E6) - d_s(K=0)|: {coupling_effect:.3f}")
        print(f"     d_s(K=E6) range: [{ds_KE6.min():.3f}, {ds_KE6.max():.3f}]")
        if coupling_effect < 0.5:
            print("     Coupling has NEGLIGIBLE effect -> omega is everything")
        elif coupling_effect < 2.0:
            print("     Coupling has MODERATE effect")
        else:
            print("     Coupling has STRONG effect")

    # 3. Algebra comparison
    print(f"\n  3. ALGEBRA COMPARISON (K=0):")
    for r in res3:
        marker = " <-- closest to 4.0" if abs(r['ds_m'] - 4.0) < 1.0 else ""
        print(f"     {r['name']:<12s}: d_s = {r['ds_m']:.3f}"
              f" +- {r['ds_s']:.3f}{marker}")

    # Find which algebra gives closest to 4.0
    closest = min(res3, key=lambda r: abs(r['ds_m'] - 4.0))
    print(f"\n     Closest to d_s=4.0: {closest['name']}"
          f" (d_s = {closest['ds_m']:.3f})")

    e6_row = next((r for r in res3 if r['name'] == 'E6'), None)
    if e6_row:
        if closest['name'] == 'E6':
            print("     E6 IS the best algebra for d_s = 4.0")
        else:
            print(f"     E6 is NOT the best: d_s(E6) = {e6_row['ds_m']:.3f}"
                  f" vs d_s({closest['name']}) = {closest['ds_m']:.3f}")

    # 4. Precision
    print(f"\n  4. PRECISION d_s = 4.0:")
    if precision is not None:
        print(f"     beta* = {precision['beta_star']:.4f}")
        print(f"     d_s = {precision['ds_mean']:.4f} +- {precision['ds_std']:.4f}")
        print(f"     95% CI: [{precision['ds_mean']-precision['ci']:.4f},"
              f" {precision['ds_mean']+precision['ci']:.4f}]")
        print(f"     d_s = 4.0 in CI: {'YES' if precision['hit'] else 'NO'}")

        if precision['hit']:
            print(f"\n     omega* = [{', '.join(f'{x:.4f}' for x in precision['omega_star'])}]")
            print(f"     This is {precision['beta_star']*100:.1f}% of the way from E6 to uniform")
    else:
        print("     No crossing found -> d_s = 4.0 not achievable")

    # 5. Key mechanism
    spreads = np.array([r['spread'] for r in res3])
    ds_vals = np.array([r['ds_m'] for r in res3])
    recs = np.array([r['rec'] for r in res3])
    valid = np.isfinite(spreads) & np.isfinite(ds_vals)

    print(f"\n  5. KEY MECHANISM:")
    if np.sum(valid) > 2 and np.std(spreads[valid]) > 1e-10:
        r_spread = np.corrcoef(spreads[valid], ds_vals[valid])[0, 1]
        r_rec = np.corrcoef(recs[valid], ds_vals[valid])[0, 1]
        print(f"     r(d_s, frequency_spread) = {r_spread:+.3f}")
        print(f"     r(d_s, recurrence)        = {r_rec:+.3f}")

        if abs(r_spread) > abs(r_rec):
            print("     -> Frequency SPREAD is the primary driver")
            print("        Low spread (uniform) -> many recurrences -> high d_s")
            print("        High spread (diverse) -> few recurrences -> low d_s")
        else:
            print("     -> RECURRENCE rate is the primary driver")
    else:
        print("     Insufficient data for correlation analysis")

    # Final assessment
    print("\n" + "=" * 72)
    print("  REFORMULATED HYPOTHESIS")
    print("=" * 72)

    if precision is not None and precision['hit']:
        print(f"""
  PARTIALLY RESCUED:

  The spectral dimension d_s = 4.0 is achievable at omega* which is
  {precision['beta_star']*100:.1f}% interpolated from E6 to uniform frequencies.

  The mechanism is NUMBER-THEORETIC:
    - E6 exponents {{1,4,5,7,8,11}} define frequency ratios
    - These ratios control quasi-periodic recurrence on 6D torus
    - Recurrence rate determines shortcut density in the graph
    - Shortcut density determines spectral dimension

  What this means:
    - The E6 algebra contributes through its HARMONIC STRUCTURE
      (exponents/Coxeter number), not its CARTAN MATRIX
    - d_s = 4.0 requires a specific balance of frequency diversity
    - This balance involves E6 frequencies mixed with uniformity

  What remains to show:
    1) Why this specific mixing ratio has physical significance
    2) Whether d_s = 4.0 is an attractor under some dynamics
    3) Connection to actual spacetime physics
""")
    elif precision is not None:
        print(f"""
  CLOSE BUT NOT RESCUED:

  d_s = {precision['ds_mean']:.3f} at beta* = {precision['beta_star']:.3f}
  The 95% CI does not include 4.0.
  
  The E6 frequency structure influences d_s but does not naturally
  produce d_s = 4.0 without parameter tuning.
""")
    else:
        print("""
  NOT RESCUED:

  d_s = 4.0 was not achievable in the omega interpolation range.
  The hypothesis that E6 produces 4D spacetime is not supported.
""")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 72)
    print("  RESCUE v5: WHAT FREQUENCY STRUCTURE GIVES d_s = 4?")
    print("  Key insight from v4: omega explains 66% of d_s variance")
    print("  Strategy: scan omega with FULL spectral dimension")
    print("=" * 72)

    # Experiment 1: Clean omega scan (K=0)
    res1 = experiment_omega_scan(N=1500, n_beta=20, n_runs=8)

    # Experiment 2: Same scan with E6 coupling
    res2 = experiment_omega_scan_coupled(N=1500, n_beta=12, n_runs=8, kappa=0.5)

    # Experiment 3: Different algebra frequency systems
    res3 = experiment_other_algebras(N=1500, n_runs=8)

    # Precision measurement at d_s=4.0 crossing (if exists)
    precision = precision_at_crossing(res1, N=1500, n_precision=20)

    # Plots
    print("\n[+] Building plots...")
    make_plots(res1, res2, res3, precision)

    # Verdict
    print_verdict(res1, res2, res3, precision)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total:.0f}s ({total/60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
