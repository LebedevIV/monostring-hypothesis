"""
Rescue v7: Fine scan around the d_s ≈ 4.0 resonance
=====================================================
Key finding from v6:
  beta=0.053 gives d_s = 4.067 ± 0.064 — within 2% of 4.0!
  
Strategy:
  1) Fine scan beta ∈ [0, 0.12] with 40 points, 10 runs each
  2) Precision measurement at the minimum
  3) Test: is this a NUMBER-THEORETIC resonance specific to E6?
  4) Compare: does the same resonance exist for A6, D6, B6?
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  SPECTRAL DIMENSION (proven correct in v6 benchmarks)
# ═══════════════════════════════════════════════════════════════

def spectral_dimension(G):
    N = G.number_of_nodes()
    if N < 20:
        return 0.0

    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    eigs = eigs[eigs > 1e-10]

    if len(eigs) < 10:
        return 0.0

    lam_min = eigs[0]
    lam_max = eigs[-1]

    t_lo = max(0.01 / lam_max, 1e-3)
    t_hi = min(100.0 / lam_min, 1e7)
    t_range = np.logspace(np.log10(t_lo), np.log10(t_hi), 500)

    ds_curve = np.zeros(len(t_range))
    neff_curve = np.zeros(len(t_range))

    for idx, t in enumerate(t_range):
        exp_lt = np.exp(-eigs * t)
        Z = np.sum(exp_lt)
        Z2 = np.sum(exp_lt ** 2)
        if Z > 1e-30 and Z2 > 1e-30:
            ds_curve[idx] = 2.0 * t * np.sum(eigs * exp_lt) / Z
            neff_curve[idx] = Z ** 2 / Z2

    min_active = max(5, int(0.005 * len(eigs)))
    valid = (neff_curve >= min_active) & (ds_curve > 0.1)
    if valid.sum() < 3:
        valid = (neff_curve >= 3) & (ds_curve > 0.1)
    if valid.sum() < 3:
        pos = ds_curve > 0.1
        return float(np.max(ds_curve[pos])) if pos.sum() > 0 else 0.0

    ds_valid = ds_curve[valid]
    if len(ds_valid) > 7:
        kernel = np.ones(5) / 5
        ds_smooth = np.convolve(ds_valid, kernel, mode='valid')
    else:
        ds_smooth = ds_valid

    peak_val = float(np.max(ds_smooth))
    plateau_mask = ds_smooth > 0.7 * peak_val
    if plateau_mask.sum() >= 3:
        return float(np.median(ds_smooth[plateau_mask]))
    return peak_val


# ═══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def omega_E6():
    return 2.0 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)

def omega_A6():
    return 2.0 * np.sin(np.pi * np.array([1, 2, 3, 4, 5, 6]) / 7)

def omega_D6():
    return 2.0 * np.sin(np.pi * np.array([1, 3, 5, 7, 9, 5]) / 10)

def omega_B6():
    return 2.0 * np.sin(np.pi * np.array([1, 3, 5, 7, 9, 11]) / 12)

def omega_uniform():
    return np.full(6, np.mean(omega_E6()))

def omega_interp(base_omega, beta):
    """Interpolate between base_omega and uniform"""
    target = np.full(6, np.mean(base_omega))
    return (1 - beta) * base_omega + beta * target

def evolve_phases(N, omega, seed=0):
    """Phase evolution WITHOUT coupling (K=0, cleanest signal)"""
    rng = np.random.RandomState(seed)
    D = len(omega)
    phases = np.zeros((N, D))
    phases[0] = rng.uniform(0, 2 * np.pi, D)
    for n in range(N - 1):
        phases[n + 1] = (phases[n] + omega + 0.1 * np.sin(phases[n])) % (2 * np.pi)
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


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 1: FINE SCAN AROUND beta=0.053
# ═══════════════════════════════════════════════════════════════

def experiment_fine_scan(N=1000, n_beta=40, n_runs=10):
    """Fine scan beta ∈ [0, 0.12] near the d_s ≈ 4.0 resonance"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: FINE SCAN beta in [0, 0.12]")
    print(f"  N={N}, {n_runs} runs, {n_beta} points")
    print("  Looking for d_s = 4.0 resonance found at beta~0.053 in v6")
    print("=" * 72)

    betas = np.linspace(0, 0.12, n_beta)
    results = []

    hdr = f"  {'beta':>7s} | {'d_s':>7s} {'+-':>5s} {'SEM':>5s} | {'recur':>8s} | {'<k>':>6s}"
    print(hdr)
    print("  " + "-" * 55)

    for beta in betas:
        om = omega_interp(omega_E6(), beta)
        ds_list = []
        rec_list = []
        deg_list = []

        for r in range(n_runs):
            ph = evolve_phases(N, om, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds_list.append(spectral_dimension(G))
            rec_list.append(phase_recurrence(ph, seed=r * 311))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        ds_arr = np.array(ds_list)
        sem = ds_arr.std() / np.sqrt(n_runs)

        row = {
            'beta': beta,
            'omega': om.copy(),
            'ds_m': ds_arr.mean(), 'ds_s': ds_arr.std(), 'ds_sem': sem,
            'rec': np.mean(rec_list), 'deg': np.mean(deg_list),
            'ds_raw': ds_list,
        }
        results.append(row)

        # Mark points near 4.0
        marker = ""
        if abs(row['ds_m'] - 4.0) < 0.5:
            marker = " <<<"
        if abs(row['ds_m'] - 4.0) < row['ds_s']:
            marker = " <<<*"

        print(f"  {beta:7.4f} | {row['ds_m']:6.3f} +{row['ds_s']:5.3f}"
              f" {sem:5.3f} | {row['rec']:8.5f} | {row['deg']:6.2f}{marker}")

    return results


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 2: SAME SCAN FOR OTHER ALGEBRAS
# ═══════════════════════════════════════════════════════════════

def experiment_algebra_comparison(N=1000, n_beta=20, n_runs=8):
    """Does the d_s ≈ 4.0 resonance exist for A6, D6, B6?"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: RESONANCE SEARCH FOR OTHER ALGEBRAS")
    print(f"  Scanning beta in [0, 0.15] for E6, A6, D6, B6")
    print(f"  N={N}, {n_runs} runs, {n_beta} points each")
    print("=" * 72)

    algebras = [
        ('E6', omega_E6()),
        ('A6', omega_A6()),
        ('D6', omega_D6()),
        ('B6', omega_B6()),
    ]

    betas = np.linspace(0, 0.15, n_beta)
    all_results = {}

    for alg_name, base_omega in algebras:
        print(f"\n  --- {alg_name} ---")
        print(f"  omega = [{', '.join(f'{x:.3f}' for x in base_omega)}]")

        results = []
        for beta in betas:
            om = omega_interp(base_omega, beta)
            ds_list = []

            for r in range(n_runs):
                ph = evolve_phases(N, om, seed=r * 137 + 7)
                G = build_graph(ph, seed=r * 251 + 13)
                ds_list.append(spectral_dimension(G))

            ds_arr = np.array(ds_list)
            row = {
                'beta': beta,
                'ds_m': ds_arr.mean(), 'ds_s': ds_arr.std(),
                'ds_sem': ds_arr.std() / np.sqrt(n_runs),
            }
            results.append(row)

        all_results[alg_name] = results

        # Find minimum and check if d_s = 4.0 is reachable
        ds_mins = [(r['beta'], r['ds_m'], r['ds_s']) for r in results]
        ds_mins.sort(key=lambda x: abs(x[1] - 4.0))
        best = ds_mins[0]
        print(f"  Closest to d_s=4.0: beta={best[0]:.4f},"
              f" d_s={best[1]:.3f} +- {best[2]:.3f}"
              f" (|d_s-4|={abs(best[1]-4):.3f})")

        # Check if 4.0 is within 95% CI at best point
        best_row = next(r for r in results
                        if abs(r['beta'] - best[0]) < 1e-6)
        ci = 1.96 * best_row['ds_sem']
        in_ci = abs(best_row['ds_m'] - 4.0) < ci
        print(f"  95% CI: [{best_row['ds_m']-ci:.3f},"
              f" {best_row['ds_m']+ci:.3f}]"
              f" -> d_s=4.0 {'IN' if in_ci else 'NOT IN'} CI")

    return all_results


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 3: PRECISION AT BEST POINT
# ═══════════════════════════════════════════════════════════════

def experiment_precision(fine_results, N=1000, n_precision=30):
    """Precision measurement at the beta closest to d_s = 4.0"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: PRECISION MEASUREMENT")
    print("=" * 72)

    # Find the point closest to d_s = 4.0
    sorted_by_dist = sorted(fine_results, key=lambda r: abs(r['ds_m'] - 4.0))
    best = sorted_by_dist[0]
    beta_star = best['beta']
    om_star = omega_interp(omega_E6(), beta_star)

    print(f"  Best point: beta = {beta_star:.5f}")
    print(f"  Initial estimate: d_s = {best['ds_m']:.3f} +- {best['ds_s']:.3f}")
    print(f"  omega = [{', '.join(f'{x:.4f}' for x in om_star)}]")
    print(f"\n  Running {n_precision} precision measurements...")

    ds_vals = []
    rec_vals = []
    deg_vals = []

    for r in range(n_precision):
        ph = evolve_phases(N, om_star, seed=r * 997 + 31)
        G = build_graph(ph, seed=r * 1009 + 37)
        ds_vals.append(spectral_dimension(G))
        rec_vals.append(phase_recurrence(ph, seed=r * 1013))
        deg_vals.append(2 * G.number_of_edges() / G.number_of_nodes())

        if (r + 1) % 5 == 0:
            rm = np.mean(ds_vals)
            rs = np.std(ds_vals)
            sem = rs / np.sqrt(r + 1)
            print(f"    runs 1-{r+1:2d}: d_s = {rm:.4f} +- {rs:.4f}"
                  f" (SEM = {sem:.4f})")

    vp = np.array(ds_vals)
    sem = vp.std() / np.sqrt(len(vp))
    ci = 1.96 * sem
    hit = abs(vp.mean() - 4.0) < ci

    print(f"\n  PRECISION RESULT:")
    print(f"    beta* = {beta_star:.5f}")
    print(f"    d_s   = {vp.mean():.4f} +- {vp.std():.4f}")
    print(f"    SEM   = {sem:.4f}")
    print(f"    95% CI: [{vp.mean()-ci:.4f}, {vp.mean()+ci:.4f}]")
    print(f"    |d_s - 4.0| = {abs(vp.mean()-4.0):.4f}")
    print(f"    d_s = 4.0 within 95% CI: {'YES' if hit else 'NO'}")

    # Frequency analysis
    om_E6 = omega_E6()
    om_unif = omega_uniform()
    print(f"\n  Frequency comparison:")
    print(f"    {'dim':>4s} | {'omega_E6':>9s} | {'omega*':>9s}"
          f" | {'uniform':>9s} | {'shift':>8s}")
    print(f"    {'':>4s}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    for d in range(6):
        shift = om_star[d] - om_E6[d]
        print(f"    {d:4d} | {om_E6[d]:9.5f} | {om_star[d]:9.5f}"
              f" | {om_unif[d]:9.5f} | {shift:+8.5f}")

    # Recurrence and degree
    print(f"\n  Graph properties at beta*:")
    print(f"    Recurrence:     {np.mean(rec_vals):.5f}")
    print(f"    Mean degree:    {np.mean(deg_vals):.2f}")

    return {
        'beta_star': beta_star,
        'omega_star': om_star,
        'ds_mean': vp.mean(), 'ds_std': vp.std(),
        'sem': sem, 'ci': ci, 'hit': hit,
        'ds_raw': vp,
        'rec': np.mean(rec_vals), 'deg': np.mean(deg_vals),
    }


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 4: SIZE DEPENDENCE
# ═══════════════════════════════════════════════════════════════

def experiment_size_dependence(beta_star, n_runs=5):
    """Check if d_s at beta* depends on graph size N"""
    print("\n" + "=" * 72)
    print("  EXPERIMENT 4: SIZE DEPENDENCE AT beta*")
    print(f"  beta* = {beta_star:.5f}")
    print("=" * 72)

    sizes = [300, 500, 700, 1000, 1300]
    om_star = omega_interp(omega_E6(), beta_star)
    results = []

    hdr = f"  {'N':>6s} | {'d_s':>7s} {'+-':>5s} {'SEM':>5s} | {'<k>':>6s}"
    print(hdr)
    print("  " + "-" * 42)

    for N in sizes:
        ds_list = []
        deg_list = []
        for r in range(n_runs):
            ph = evolve_phases(N, om_star, seed=r * 137 + 7)
            G = build_graph(ph, seed=r * 251 + 13)
            ds_list.append(spectral_dimension(G))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        ds_arr = np.array(ds_list)
        sem = ds_arr.std() / np.sqrt(n_runs)
        row = {
            'N': N,
            'ds_m': ds_arr.mean(), 'ds_s': ds_arr.std(), 'ds_sem': sem,
            'deg': np.mean(deg_list),
        }
        results.append(row)
        print(f"  {N:6d} | {row['ds_m']:6.3f} +{row['ds_s']:5.3f}"
              f" {sem:5.3f} | {row['deg']:6.2f}")

    # Check convergence
    ds_vals = [r['ds_m'] for r in results]
    if len(ds_vals) >= 3:
        trend = np.polyfit(sizes, ds_vals, 1)
        print(f"\n  Linear trend: d_s = {trend[1]:.3f} + {trend[0]:.6f} * N")
        if abs(trend[0]) < 0.001:
            print("  Size dependence: NEGLIGIBLE (d_s converged)")
        else:
            extrapolated = trend[1] + trend[0] * 2000
            print(f"  Extrapolated d_s(N=2000): {extrapolated:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════

def make_plots(fine_results, alg_results, precision, size_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── (0,0) Fine scan ──
    ax = axes[0, 0]
    betas = [r['beta'] for r in fine_results]
    ds_m = [r['ds_m'] for r in fine_results]
    ds_s = [r['ds_s'] for r in fine_results]

    ax.errorbar(betas, ds_m, yerr=ds_s, fmt='o-', color='teal',
                capsize=2, ms=4, lw=1.5, label='E6 fine scan')
    ax.axhline(4.0, color='red', ls=':', lw=2, label='d_s = 4.0')
    ax.fill_between(betas, 3.7, 4.3, color='gold', alpha=0.15)

    if precision is not None:
        ax.axvline(precision['beta_star'], color='green', ls='--', lw=2,
                   label=f'beta* = {precision["beta_star"]:.4f}')
        ax.plot(precision['beta_star'], precision['ds_mean'], '*',
                color='green', ms=15, zorder=5)

    ax.set_xlabel('beta (0=E6, beta=uniform fraction)', fontsize=11)
    ax.set_ylabel('d_s', fontsize=11)
    ax.set_title('Fine scan: E6 -> Uniform (K=0)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # ── (0,1) Algebra comparison ──
    ax = axes[0, 1]
    colors = {'E6': '#d62728', 'A6': '#1f77b4', 'D6': '#2ca02c', 'B6': '#ff7f0e'}

    for alg_name, results in alg_results.items():
        b = [r['beta'] for r in results]
        d = [r['ds_m'] for r in results]
        s = [r['ds_s'] for r in results]
        ax.errorbar(b, d, yerr=s, fmt='o-', color=colors.get(alg_name, 'gray'),
                    capsize=2, ms=3, lw=1.5, label=alg_name)

    ax.axhline(4.0, color='black', ls=':', lw=2, label='d_s = 4.0')
    ax.fill_between([0, 0.15], 3.7, 4.3, color='gold', alpha=0.1)
    ax.set_xlabel('beta', fontsize=11)
    ax.set_ylabel('d_s', fontsize=11)
    ax.set_title('Algebra comparison: all -> uniform', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # ── (1,0) Precision histogram ──
    ax = axes[1, 0]
    if precision is not None:
        ax.hist(precision['ds_raw'], bins=15, color='teal', alpha=0.7,
                edgecolor='black', density=True)
        ax.axvline(4.0, color='red', ls=':', lw=2, label='d_s = 4.0')
        ax.axvline(precision['ds_mean'], color='green', ls='--', lw=2,
                   label=f'mean = {precision["ds_mean"]:.3f}')

        # CI band
        lo = precision['ds_mean'] - precision['ci']
        hi = precision['ds_mean'] + precision['ci']
        ax.axvspan(lo, hi, color='green', alpha=0.1, label='95% CI')

        ax.set_xlabel('d_s', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Precision at beta* = {precision["beta_star"]:.4f}'
                     f' (n={len(precision["ds_raw"])})', fontsize=12)
        ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # ── (1,1) Size dependence ──
    ax = axes[1, 1]
    if size_results:
        Ns = [r['N'] for r in size_results]
        ds = [r['ds_m'] for r in size_results]
        ds_e = [r['ds_s'] for r in size_results]

        ax.errorbar(Ns, ds, yerr=ds_e, fmt='D-', color='#d62728',
                    capsize=4, ms=7, lw=2)
        ax.axhline(4.0, color='black', ls=':', lw=1.5, label='d_s = 4.0')
        ax.set_xlabel('Graph size N', fontsize=11)
        ax.set_ylabel('d_s', fontsize=11)
        ax.set_title('Size dependence at beta*', fontsize=12)
        ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    fig.suptitle('Rescue v7: Resonance Analysis', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('rescue_v7.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  [+] Saved: rescue_v7.png")


# ═══════════════════════════════════════════════════════════════
#  VERDICT
# ═══════════════════════════════════════════════════════════════

def print_verdict(fine_results, alg_results, precision, size_results):
    print("\n" + "=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)

    # 1. Is the resonance real?
    print("\n  1. IS THE d_s ~ 4.0 RESONANCE REAL?")
    if precision is not None:
        print(f"     beta* = {precision['beta_star']:.5f}")
        print(f"     d_s = {precision['ds_mean']:.4f} +- {precision['ds_std']:.4f}")
        print(f"     95% CI: [{precision['ds_mean']-precision['ci']:.4f},"
              f" {precision['ds_mean']+precision['ci']:.4f}]")

        if precision['hit']:
            print("     -> YES: d_s = 4.0 IS within the 95% CI")
        else:
            print(f"     -> NO: |d_s - 4.0| = {abs(precision['ds_mean']-4.0):.4f}"
                  f" > {precision['ci']:.4f}")

    # 2. Is it unique to E6?
    print("\n  2. IS THE RESONANCE UNIQUE TO E6?")
    for alg_name, results in alg_results.items():
        best = min(results, key=lambda r: abs(r['ds_m'] - 4.0))
        ci = 1.96 * best['ds_sem']
        in_ci = abs(best['ds_m'] - 4.0) < ci
        print(f"     {alg_name}: closest d_s = {best['ds_m']:.3f}"
              f" at beta={best['beta']:.4f}"
              f" (in CI: {'YES' if in_ci else 'NO'})")

    algebras_with_4 = []
    for alg_name, results in alg_results.items():
        best = min(results, key=lambda r: abs(r['ds_m'] - 4.0))
        ci = 1.96 * best['ds_sem']
        if abs(best['ds_m'] - 4.0) < ci:
            algebras_with_4.append(alg_name)

    if len(algebras_with_4) == 0:
        print("     -> No algebra achieves d_s = 4.0 in this scan")
    elif len(algebras_with_4) == 1 and 'E6' in algebras_with_4:
        print("     -> E6 IS uniquely special!")
    elif 'E6' in algebras_with_4:
        print(f"     -> E6 achieves d_s=4.0 but so do: "
              f"{', '.join(a for a in algebras_with_4 if a != 'E6')}")
    else:
        print(f"     -> E6 does NOT achieve d_s=4.0."
              f" Achievers: {', '.join(algebras_with_4)}")

    # 3. Size dependence
    print("\n  3. SIZE DEPENDENCE:")
    if size_results:
        ds_vals = [r['ds_m'] for r in size_results]
        sizes = [r['N'] for r in size_results]
        trend = np.polyfit(sizes, ds_vals, 1)
        print(f"     Slope: {trend[0]:.6f} per node")
        if abs(trend[0]) < 0.001:
            print("     -> d_s is CONVERGED (size-independent)")
        else:
            print(f"     -> d_s has size dependence"
                  f" ({'+' if trend[0] > 0 else ''}{trend[0]*1000:.3f} per 1000 nodes)")

    # 4. Final assessment
    print("\n" + "=" * 72)
    print("  HYPOTHESIS STATUS")
    print("=" * 72)

    e6_unique = (len(algebras_with_4) == 1 and 'E6' in algebras_with_4)
    precision_hit = precision is not None and precision['hit']
    size_ok = True
    if size_results:
        trend = np.polyfit([r['N'] for r in size_results],
                           [r['ds_m'] for r in size_results], 1)
        size_ok = abs(trend[0]) < 0.002

    if precision_hit and e6_unique and size_ok:
        print(f"""
  *** RESCUED ***

  E6 frequencies, perturbed by {precision['beta_star']*100:.1f}% toward uniformity,
  uniquely produce d_s = {precision['ds_mean']:.3f} +/- {precision['ds_std']:.3f}
  (4.0 within 95% CI), and this is size-independent.

  Physical interpretation:
    The E6 exponents {{1,4,5,7,8,11}} define frequency ratios that create
    a specific quasi-periodic pattern on the 6D torus. A small perturbation
    toward uniformity tunes the recurrence structure to produce exactly
    the shortcut density needed for d_s = 4.0.

  Remaining questions:
    1) Why {precision['beta_star']*100:.1f}% perturbation specifically?
    2) Is this an attractor under some dynamics?
    3) Does this connect to actual GR?
""")
    elif precision_hit:
        others = [a for a in algebras_with_4 if a != 'E6']
        print(f"""
  PARTIALLY RESCUED:

  d_s = 4.0 is achievable near E6 frequencies (beta*={precision['beta_star']:.4f})
  {'But other algebras also achieve it: ' + ', '.join(others) if others else ''}
  {'Size dependence detected — result may not converge' if not size_ok else ''}

  The mechanism (number-theoretic resonance in frequency ratios)
  is genuine but may not be unique to E6.
""")
    else:
        print(f"""
  NOT RESCUED:

  E6 gives d_s = {fine_results[0]['ds_m']:.3f} at beta=0 (pure E6).
  d_s = 4.0 {'is achievable but not precisely at E6' 
              if any(abs(r['ds_m']-4.0) < 1.0 for r in fine_results)
              else 'is not achievable in this parameter range'}.

  The hypothesis that E6 naturally produces d_s = 4.0 is not supported.

  Options:
    a) Accept d_s(E6) ~ {fine_results[0]['ds_m']:.1f} and find its meaning
    b) Try adding K=E6 coupling (combined effect)
    c) Try E7/E8 (higher rank algebras)
    d) Reframe: "E6 constrains d_s to [X, Y]" instead of "d_s = 4"
""")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 72)
    print("  RESCUE v7: RESONANCE ANALYSIS")
    print("  Target: d_s = 4.0 near beta ~ 0.053 (from v6)")
    print("=" * 72)

    # Experiment 1: Fine scan
    fine = experiment_fine_scan(N=1000, n_beta=40, n_runs=10)

    # Experiment 2: Other algebras
    alg = experiment_algebra_comparison(N=1000, n_beta=20, n_runs=8)

    # Experiment 3: Precision
    precision = experiment_precision(fine, N=1000, n_precision=30)

    # Experiment 4: Size dependence
    beta_star = precision['beta_star'] if precision else 0.053
    sizes = experiment_size_dependence(beta_star, n_runs=5)

    # Plots
    print("\n[+] Building plots...")
    make_plots(fine, alg, precision, sizes)

    # Verdict
    print_verdict(fine, alg, precision, sizes)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total:.0f}s ({total/60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
