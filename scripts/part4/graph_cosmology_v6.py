"""
Rescue v6: Fixed Spectral Dimension + Key Experiments
======================================================
Critical fix: adaptive t-range based on eigenvalue bounds
Validation: benchmarks on path, 2D grid, 3D grid
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  FIXED SPECTRAL DIMENSION
# ═══════════════════════════════════════════════════════════════

def spectral_dimension(G, n_t=500, return_curve=False):
    """
    Heat-kernel spectral dimension with ADAPTIVE time range.
    
    Fixes vs v5:
      1. t_range adapts to [0.01/λ_max, 100/λ_min] — covers full plateau
      2. n_eff filter: only trust d_s where ≥5 eigenvalues contribute
      3. Peak detection instead of minimum-variance (avoids low-d_s trap)
    """
    N = G.number_of_nodes()
    if N < 20:
        if return_curve:
            return 0.0, np.array([]), np.array([]), np.array([])
        return 0.0

    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    eigs = eigs[eigs > 1e-10]

    if len(eigs) < 10:
        if return_curve:
            return 0.0, np.array([]), np.array([]), np.array([])
        return 0.0

    lam_min = eigs[0]
    lam_max = eigs[-1]

    # ADAPTIVE time range: covers the FULL scaling regime
    t_lo = max(0.01 / lam_max, 1e-3)
    t_hi = min(100.0 / lam_min, 1e7)
    t_range = np.logspace(np.log10(t_lo), np.log10(t_hi), n_t)

    ds_curve = np.zeros(n_t)
    neff_curve = np.zeros(n_t)

    for idx, t in enumerate(t_range):
        exp_lt = np.exp(-eigs * t)
        Z = np.sum(exp_lt)
        Z2 = np.sum(exp_lt ** 2)

        if Z > 1e-30 and Z2 > 1e-30:
            ds_curve[idx] = 2.0 * t * np.sum(eigs * exp_lt) / Z
            neff_curve[idx] = Z ** 2 / Z2  # effective number of modes

    # VALID REGION: enough modes contributing
    min_active = max(5, int(0.005 * len(eigs)))
    valid = (neff_curve >= min_active) & (ds_curve > 0.1)

    if valid.sum() < 3:
        valid = (neff_curve >= 3) & (ds_curve > 0.1)

    if valid.sum() < 3:
        pos = ds_curve > 0.1
        val = float(np.max(ds_curve[pos])) if pos.sum() > 0 else 0.0
        if return_curve:
            return val, t_range, ds_curve, neff_curve
        return val

    ds_valid = ds_curve[valid]

    # Smooth to reduce noise
    if len(ds_valid) > 7:
        kernel = np.ones(5) / 5
        ds_smooth = np.convolve(ds_valid, kernel, mode='valid')
    else:
        ds_smooth = ds_valid

    # Find peak in smoothed valid region
    peak_val = float(np.max(ds_smooth))

    # Plateau = region within 30% of peak
    plateau_mask = ds_smooth > 0.7 * peak_val
    if plateau_mask.sum() >= 3:
        result = float(np.median(ds_smooth[plateau_mask]))
    else:
        result = peak_val

    if return_curve:
        return result, t_range, ds_curve, neff_curve
    return result


# ═══════════════════════════════════════════════════════════════
#  BENCHMARKS
# ═══════════════════════════════════════════════════════════════

def run_benchmarks():
    """Validate spectral dimension on graphs with KNOWN d_s."""
    print("=" * 72)
    print("  BENCHMARKS: Validating spectral dimension measurement")
    print("=" * 72)

    benchmarks = [
        ("Path (d_s=1)",    nx.path_graph(200)),
        ("2D Grid (d_s=2)", nx.grid_2d_graph(15, 15)),
        ("3D Grid (d_s=3)", nx.grid_graph(dim=[6, 6, 6])),
        ("Complete (d_s→∞)", nx.complete_graph(100)),
    ]

    results = []
    fig_b, axes_b = plt.subplots(1, len(benchmarks), figsize=(20, 4))

    for idx, (name, G) in enumerate(benchmarks):
        t0 = time.time()
        ds, t_range, ds_curve, neff_curve = spectral_dimension(
            G, return_curve=True
        )
        dt = time.time() - t0
        N = G.number_of_nodes()
        E = G.number_of_edges()

        results.append({'name': name, 'ds': ds, 'N': N, 'E': E})
        print(f"  {name:<20s}: d_s = {ds:.3f}  "
              f"(N={N}, E={E}, {dt:.1f}s)")

        # Plot d_s(t) curve
        ax = axes_b[idx]
        mask = ds_curve > 0
        if mask.sum() > 0:
            ax.semilogx(t_range[mask], ds_curve[mask], '-', color='teal', lw=2)
        ax.axhline(ds, color='red', ls='--', lw=1.5,
                   label=f'measured = {ds:.2f}')

        # Expected value
        expected = {'Path': 1, '2D Grid': 2, '3D Grid': 3, 'Complete': None}
        for key, val in expected.items():
            if key in name and val is not None:
                ax.axhline(val, color='green', ls=':', lw=1.5,
                           label=f'expected = {val}')
                break

        ax.set_title(name, fontsize=11)
        ax.set_xlabel('t')
        ax.set_ylabel('d_s(t)')
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_ylim(0, max(ds * 1.5, 5))

    fig_b.suptitle('Spectral Dimension Benchmarks', fontsize=14)
    plt.tight_layout()
    plt.savefig('benchmark_ds.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [+] Saved: benchmark_ds.png")

    # Validation check
    path_ds = results[0]['ds']
    grid2d_ds = results[1]['ds']
    grid3d_ds = results[2]['ds']

    print(f"\n  Validation:")
    print(f"    Path:    |d_s - 1| = {abs(path_ds - 1):.3f}  "
          f"{'PASS' if abs(path_ds - 1) < 0.3 else 'FAIL'}")
    print(f"    2D Grid: |d_s - 2| = {abs(grid2d_ds - 2):.3f}  "
          f"{'PASS' if abs(grid2d_ds - 2) < 0.5 else 'FAIL'}")
    print(f"    3D Grid: |d_s - 3| = {abs(grid3d_ds - 3):.3f}  "
          f"{'PASS' if abs(grid3d_ds - 3) < 0.5 else 'FAIL'}")

    all_pass = (abs(path_ds - 1) < 0.3 and
                abs(grid2d_ds - 2) < 0.5 and
                abs(grid3d_ds - 3) < 0.5)

    if all_pass:
        print("  >>> ALL BENCHMARKS PASS — measurement is reliable <<<")
    else:
        print("  >>> SOME BENCHMARKS FAIL — interpret results with caution <<<")

    return results, all_pass


# ═══════════════════════════════════════════════════════════════
#  FREQUENCY AND PHASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def omega_E6():
    return 2.0 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)

def omega_uniform():
    return np.full(6, np.mean(omega_E6()))

def omega_interpolated(beta):
    return (1 - beta) * omega_E6() + beta * omega_uniform()

def omega_from_exponents(exponents, h):
    return 2.0 * np.sin(np.pi * np.array(exponents, dtype=float) / h)

def evolve_phases(N, K, kappa, omega, seed=0):
    rng = np.random.RandomState(seed)
    D = len(omega)
    phases = np.zeros((N, D))
    phases[0] = rng.uniform(0, 2 * np.pi, D)
    for n in range(N - 1):
        coupling = kappa * K @ np.sin(phases[n])
        phases[n + 1] = (phases[n] + omega + coupling) % (2 * np.pi)
    return phases

def build_graph(phases, eps=1.5, max_conn=5, n_cand=80, seed=42):
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

def phase_recurrence(phases, eps=1.5, n_pairs=20000, seed=42):
    N = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, N, (n_pairs, 2))
    diffs = np.abs(phases[idx[:, 0]] - phases[idx[:, 1]])
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.mean(dists < eps))

def frequency_spread(omega):
    return np.std(omega) / (np.mean(omega) + 1e-10)


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 1: ALGEBRA FREQUENCY SCAN (K=0)
# ═══════════════════════════════════════════════════════════════

def experiment_algebras(N=1000, n_runs=5):
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: FREQUENCY SYSTEMS FROM DIFFERENT ALGEBRAS (K=0)")
    print(f"  N={N}, {n_runs} runs each")
    print("=" * 72)

    K = np.zeros((6, 6))
    kappa = 0.0

    configs = [
        ('E6',      [1, 4, 5, 7, 8, 11],  12),
        ('A6',      [1, 2, 3, 4, 5, 6],    7),
        ('D6',      [1, 3, 5, 7, 9, 5],   10),
        ('B6',      [1, 3, 5, 7, 9, 11],  12),
    ]

    specials = [
        ('Golden',  2.0 * np.sin(np.pi * np.array([1,2,3,5,8,13]) / 21)),
        ('Primes',  2.0 * np.sin(np.pi * np.array([2,3,5,7,11,13]) / 17)),
        ('Linear',  np.linspace(0.5, 2.0, 6)),
        ('Uniform', omega_uniform()),
    ]

    results = []

    hdr = (f"  {'Name':<12s} | {'d_s':>7s} {'+-':>5s} | {'recur':>8s}"
           f" | {'<k>':>6s} | {'spread':>7s}")
    print(hdr)
    print("  " + "-" * 62)

    for name, exps, h in configs:
        om = omega_from_exponents(exps, h)
        ds_list, rec_list, deg_list = [], [], []
        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds_list.append(spectral_dimension(G))
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'name': name, 'omega': om,
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'rec': np.mean(rec_list), 'deg': np.mean(deg_list),
            'spread': frequency_spread(om),
        }
        results.append(row)
        print(f"  {name:<12s} | {row['ds_m']:6.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}"
              f" | {row['spread']:7.4f}")

    for name, om in specials:
        ds_list, rec_list, deg_list = [], [], []
        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds_list.append(spectral_dimension(G))
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'name': name, 'omega': om,
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'rec': np.mean(rec_list), 'deg': np.mean(deg_list),
            'spread': frequency_spread(om),
        }
        results.append(row)
        print(f"  {name:<12s} | {row['ds_m']:6.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}"
              f" | {row['spread']:7.4f}")

    # Null: random phases
    ds_null = []
    for r in range(n_runs):
        ph = np.random.RandomState(r * 333).uniform(0, 2*np.pi, (N, 6))
        G = build_graph(ph, seed=r * 444)
        ds_null.append(spectral_dimension(G))
    null_m, null_s = np.mean(ds_null), np.std(ds_null)
    print(f"  {'Null':<12s} | {null_m:6.3f} +{null_s:5.3f}"
          f" |       -- |     -- |      --")

    return results, null_m


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 2: OMEGA INTERPOLATION (K=0)
# ═══════════════════════════════════════════════════════════════

def experiment_interpolation(N=1000, n_beta=20, n_runs=5):
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: OMEGA INTERPOLATION (K=0)")
    print(f"  omega(beta) = (1-beta)*omega_E6 + beta*omega_uniform")
    print(f"  N={N}, {n_runs} runs, {n_beta} points")
    print("=" * 72)

    K = np.zeros((6, 6))
    kappa = 0.0
    betas = np.linspace(0, 1, n_beta)
    results = []

    hdr = f"  {'beta':>6s} | {'d_s':>7s} {'+-':>5s} | {'recur':>8s} | {'<k>':>6s}"
    print(hdr)
    print("  " + "-" * 45)

    for beta in betas:
        om = omega_interpolated(beta)
        ds_list, rec_list, deg_list = [], [], []

        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds_list.append(spectral_dimension(G))
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        row = {
            'beta': beta, 'omega': om.copy(),
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'rec': np.mean(rec_list), 'deg': np.mean(deg_list),
        }
        results.append(row)
        print(f"  {beta:6.3f} | {row['ds_m']:6.3f} +{row['ds_s']:5.3f}"
              f" | {row['rec']:8.5f} | {row['deg']:6.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  PRECISION AT d_s = 4.0 CROSSING
# ═══════════════════════════════════════════════════════════════

def find_crossing_and_measure(results, target=4.0, N=1000, n_prec=20):
    print("\n" + "=" * 72)
    print(f"  PRECISION MEASUREMENT AT d_s = {target} CROSSING")
    print("=" * 72)

    betas = np.array([r['beta'] for r in results])
    ds_arr = np.array([r['ds_m'] for r in results])

    crossing = None
    for i in range(len(ds_arr) - 1):
        if (ds_arr[i] - target) * (ds_arr[i + 1] - target) < 0:
            frac = (target - ds_arr[i]) / (ds_arr[i + 1] - ds_arr[i])
            crossing = betas[i] + frac * (betas[i + 1] - betas[i])
            break

    if crossing is None:
        print(f"  d_s = {target} NOT found in scan range")
        print(f"  d_s range: [{ds_arr.min():.3f}, {ds_arr.max():.3f}]")
        return None

    print(f"  Crossing detected at beta* ~ {crossing:.4f}")
    om_star = omega_interpolated(crossing)
    print(f"  omega* = [{', '.join(f'{x:.4f}' for x in om_star)}]")

    K = np.zeros((6, 6))
    kappa = 0.0
    ds_vals = []

    for r in range(n_prec):
        ph = evolve_phases(N, K, kappa, om_star, seed=r * 997 + 31)
        G = build_graph(ph, seed=r * 1009 + 37)
        ds_vals.append(spectral_dimension(G))
        if (r + 1) % 5 == 0:
            rm, rs = np.mean(ds_vals), np.std(ds_vals)
            print(f"    runs 1-{r+1}: d_s = {rm:.3f} +- {rs:.3f}")

    vp = np.array(ds_vals)
    sem = vp.std() / np.sqrt(len(vp))
    ci = 1.96 * sem
    hit = abs(vp.mean() - target) < ci

    print(f"\n  PRECISION RESULT:")
    print(f"    d_s = {vp.mean():.4f} +- {vp.std():.4f}")
    print(f"    SEM = {sem:.4f}")
    print(f"    95% CI: [{vp.mean()-ci:.4f}, {vp.mean()+ci:.4f}]")
    print(f"    d_s = {target} within 95% CI: {'YES' if hit else 'NO'}")

    return {
        'beta_star': crossing, 'omega_star': om_star,
        'ds_mean': vp.mean(), 'ds_std': vp.std(),
        'sem': sem, 'ci': ci, 'hit': hit,
    }


# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════

def make_plots(alg_results, null_m, interp_results, precision):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (0) Algebra comparison: d_s vs spread
    ax = axes[0]
    names = [r['name'] for r in alg_results]
    ds_vals = [r['ds_m'] for r in alg_results]
    ds_errs = [r['ds_s'] for r in alg_results]
    spreads = [r['spread'] for r in alg_results]

    ax.errorbar(spreads, ds_vals, yerr=ds_errs, fmt='o', color='teal',
                capsize=4, ms=8, lw=1.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (spreads[i], ds_vals[i]),
                    textcoords="offset points", xytext=(5, 6), fontsize=9)
    ax.axhline(4.0, color='red', ls=':', lw=1.5, label='d_s = 4.0')
    ax.axhline(null_m, color='gray', ls='--', lw=1,
               label=f'Null = {null_m:.2f}')
    ax.set_xlabel('Frequency spread (std/mean)', fontsize=11)
    ax.set_ylabel('Spectral dimension d_s', fontsize=11)
    ax.set_title('Exp 1: d_s vs frequency diversity', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # (1) Interpolation
    ax = axes[1]
    betas = [r['beta'] for r in interp_results]
    ds_m = [r['ds_m'] for r in interp_results]
    ds_s = [r['ds_s'] for r in interp_results]

    ax.errorbar(betas, ds_m, yerr=ds_s, fmt='D-', color='#d62728',
                capsize=3, ms=5, lw=2)
    ax.axhline(4.0, color='black', ls=':', lw=1.5, label='d_s = 4.0')
    ax.fill_between(betas, 3.5, 4.5, color='gold', alpha=0.15)

    if precision is not None:
        ax.axvline(precision['beta_star'], color='green', ls='--', lw=2,
                   label=f'beta* = {precision["beta_star"]:.3f}')
        ax.plot(precision['beta_star'], precision['ds_mean'], '*',
                color='green', ms=15, zorder=5)

    ax.set_xlabel('beta  (0=E6, 1=uniform)', fontsize=11)
    ax.set_ylabel('d_s', fontsize=11)
    ax.set_title('Exp 2: E6 -> Uniform interpolation', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # (2) d_s vs recurrence
    ax = axes[2]
    recs = [r['rec'] for r in alg_results]
    ax.errorbar(recs, ds_vals, yerr=ds_errs, fmt='s', color='indigo',
                capsize=4, ms=8, lw=1.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (recs[i], ds_vals[i]),
                    textcoords="offset points", xytext=(5, 6), fontsize=9)
    ax.axhline(4.0, color='red', ls=':', lw=1.5, label='d_s = 4.0')

    if len(recs) > 2:
        r_arr = np.array(recs)
        d_arr = np.array(ds_vals)
        if np.std(r_arr) > 1e-10:
            corr = np.corrcoef(r_arr, d_arr)[0, 1]
            ax.set_title(f'd_s vs recurrence  (r={corr:+.3f})', fontsize=12)
        else:
            ax.set_title('d_s vs recurrence', fontsize=12)
    ax.set_xlabel('Phase recurrence fraction', fontsize=11)
    ax.set_ylabel('d_s', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('Rescue v6: Fixed Spectral Dimension Results', fontsize=14,
                 y=1.02)
    plt.tight_layout()
    plt.savefig('rescue_v6.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  [+] Saved: rescue_v6.png")


# ═══════════════════════════════════════════════════════════════
#  VERDICT
# ═══════════════════════════════════════════════════════════════

def print_verdict(alg_results, null_m, interp_results, precision,
                  benchmarks_pass):
    print("\n" + "=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)

    if not benchmarks_pass:
        print("\n  WARNING: Benchmarks did not all pass.")
        print("  Results below should be interpreted with caution.\n")

    # 1. Which algebra is closest to d_s = 4?
    print("  1. ALGEBRA RANKING (closest to d_s = 4.0):")
    sorted_alg = sorted(alg_results, key=lambda r: abs(r['ds_m'] - 4.0))
    for i, r in enumerate(sorted_alg[:5]):
        marker = " <<<" if i == 0 else ""
        print(f"     {r['name']:<12s}: d_s = {r['ds_m']:.3f} "
              f"(|d_s-4| = {abs(r['ds_m']-4):.3f}){marker}")

    e6_row = next((r for r in alg_results if r['name'] == 'E6'), None)
    best = sorted_alg[0]
    if e6_row:
        if best['name'] == 'E6':
            print("     -> E6 IS the closest to d_s = 4.0")
        else:
            print(f"     -> E6 is NOT closest: {best['name']} is closer")

    # 2. Interpolation result
    print("\n  2. INTERPOLATION E6 -> UNIFORM:")
    ds_interp = np.array([r['ds_m'] for r in interp_results])
    print(f"     d_s range: [{ds_interp.min():.3f}, {ds_interp.max():.3f}]")

    if precision is not None:
        print(f"     Crossing at beta* = {precision['beta_star']:.4f}")
        print(f"     d_s = {precision['ds_mean']:.3f} +- {precision['ds_std']:.3f}")
        print(f"     95% CI includes 4.0: {'YES' if precision['hit'] else 'NO'}")
    else:
        if ds_interp.max() < 4.0:
            print("     d_s = 4.0 NOT reachable (all below)")
        elif ds_interp.min() > 4.0:
            print("     d_s = 4.0 NOT reachable (all above)")
        else:
            print("     d_s = 4.0 in range but crossing not detected cleanly")

    # 3. Key correlations
    print("\n  3. KEY CORRELATIONS:")
    spreads = np.array([r['spread'] for r in alg_results])
    recs = np.array([r['rec'] for r in alg_results])
    ds_vals = np.array([r['ds_m'] for r in alg_results])
    degs = np.array([r['deg'] for r in alg_results])

    for metric_name, metric_vals in [('spread', spreads), ('recurrence', recs),
                                      ('degree', degs)]:
        if np.std(metric_vals) > 1e-10 and np.std(ds_vals) > 1e-10:
            corr = np.corrcoef(metric_vals, ds_vals)[0, 1]
            sig = " <<<" if abs(corr) > 0.7 else ""
            print(f"     r(d_s, {metric_name:<12s}) = {corr:+.3f}{sig}")

    # 4. Final assessment
    print("\n" + "=" * 72)
    print("  HYPOTHESIS STATUS")
    print("=" * 72)

    rescued = (precision is not None and precision['hit'] and
               e6_row is not None and best['name'] == 'E6')

    if rescued:
        print(f"""
  RESCUED:
    E6 frequencies produce d_s = 4.0 at beta* = {precision['beta_star']:.3f}
    E6 is the closest algebra to d_s = 4.0
    The mechanism operates through frequency structure, not Cartan matrix
""")
    elif precision is not None and precision['hit']:
        print(f"""
  PARTIALLY RESCUED:
    d_s = 4.0 achievable at beta* = {precision['beta_star']:.3f}
    But E6 is not the closest algebra (best: {best['name']})
    E6 participates but is not uniquely special
""")
    else:
        e6_ds = e6_row['ds_m'] if e6_row else 0
        print(f"""
  NOT RESCUED:
    d_s(E6) = {e6_ds:.3f}
    d_s = 4.0 {'not found in interpolation' if precision is None
               else f'found but imprecise: {precision["ds_mean"]:.3f}'}
    
  Salvage options:
    a) Try different N (current: graph size effects)
    b) Scan kappa with K=E6 (coupling strength)
    c) Test E7, E8 (higher rank)
    d) Accept that d_s = 4 is not a natural prediction of this model
""")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 72)
    print("  RESCUE v6: FIXED SPECTRAL DIMENSION")
    print("  Key fix: adaptive t-range covers full eigenvalue spectrum")
    print("=" * 72)

    # Step 0: Benchmarks
    bench_results, bench_pass = run_benchmarks()

    if not bench_pass:
        print("\n  !!! BENCHMARKS FAILED — stopping here !!!")
        print("  Fix spectral_dimension() before proceeding.")
        return

    # Step 1: Algebra scan
    alg_results, null_m = experiment_algebras(N=1000, n_runs=5)

    # Step 2: Omega interpolation
    interp_results = experiment_interpolation(N=1000, n_beta=20, n_runs=5)

    # Step 3: Precision at crossing
    precision = find_crossing_and_measure(interp_results, target=4.0,
                                          N=1000, n_prec=20)

    # Step 4: Plots
    print("\n[+] Building plots...")
    make_plots(alg_results, null_m, interp_results, precision)

    # Step 5: Verdict
    print_verdict(alg_results, null_m, interp_results, precision, bench_pass)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total:.0f}s ({total/60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
