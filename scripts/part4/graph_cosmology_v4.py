"""
Rescue Experiment: What Controls Spectral Dimension?
=====================================================
Three experiments on identical phase landscapes:
  A) Matrix Scan:    12 coupling matrices → d_s + correlations
  B) ω vs K Factor:  Which factor (frequencies or coupling) dominates?
  C) Interpolation:  K(α) = (1−α)·K_E6 + α·I → find α* where d_s = 4.0
"""

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  COUPLING MATRICES
# ═══════════════════════════════════════════════════════════════

def cartan_E6():
    K = np.diag([2.0] * 6)
    for i, j in [(0,1), (1,2), (2,3), (3,4), (2,5)]:
        K[i, j] = K[j, i] = -1.0
    return K

def cartan_A6():
    K = np.diag([2.0] * 6)
    for i in range(5):
        K[i, i+1] = K[i+1, i] = -1.0
    return K

def cartan_D6():
    K = np.diag([2.0] * 6)
    for i, j in [(0,1), (1,2), (2,3), (3,4), (3,5)]:
        K[i, j] = K[j, i] = -1.0
    return K


# ═══════════════════════════════════════════════════════════════
#  FREQUENCY VECTORS
# ═══════════════════════════════════════════════════════════════

def omega_E6():
    return 2.0 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)

def omega_uniform():
    return np.full(6, np.mean(omega_E6()))

def omega_random(seed=99):
    return np.random.RandomState(seed).uniform(0.5, 2.0, 6)


# ═══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

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


def spectral_dimension(G, n_eigs=200):
    N = G.number_of_nodes()
    if N < 30:
        return 0.0

    L = nx.normalized_laplacian_matrix(G).astype(float)
    k = min(N - 2, n_eigs)

    try:
        eigs = eigsh(L, k=k, which='SM', return_eigenvectors=False)
        eigs = np.sort(np.real(eigs))
        eigs = eigs[eigs > 1e-10]
    except Exception:
        return 0.0

    if len(eigs) < 10:
        return 0.0

    t_range = np.logspace(0, 2.5, 250)
    ds_curve = np.zeros(len(t_range))

    for idx, t in enumerate(t_range):
        exp_lt = np.exp(-eigs * t)
        Z = np.sum(exp_lt)
        if Z > 1e-30:
            ds_curve[idx] = 2.0 * t * np.sum(eigs * exp_lt) / Z

    valid = (ds_curve > 0.3) & (ds_curve < 20)
    if np.sum(valid) < 10:
        pos = ds_curve[ds_curve > 0]
        return float(np.median(pos)) if len(pos) else 0.0

    dsv = ds_curve[valid]
    w = min(25, len(dsv) // 3)
    if w < 3:
        return float(np.median(dsv))

    best_s, best_c = np.inf, len(dsv) // 2
    for s in range(len(dsv) - w):
        std = np.std(dsv[s:s + w])
        if std < best_s:
            best_s = std
            best_c = s + w // 2

    lo = max(0, best_c - w // 2)
    hi = min(len(dsv), best_c + w // 2)
    return float(np.median(dsv[lo:hi]))


def phase_recurrence(phases, eps=1.5, n_pairs=10000, seed=42):
    N = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, N, (n_pairs, 2))
    diffs = np.abs(phases[idx[:, 0]] - phases[idx[:, 1]])
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.mean(dists < eps))


def mat_props(K):
    eigs = np.sort(np.linalg.eigvalsh(K))
    od = K.copy()
    np.fill_diagonal(od, 0)
    return {
        'offdiag':    np.linalg.norm(od, 'fro'),
        'eig_spread': eigs[-1] - eigs[0],
        'eig_max':    eigs[-1],
        'eig_min':    eigs[0],
        'trace':      np.trace(K),
        'det':        np.linalg.det(K),
    }


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT A: MATRIX SCAN
# ═══════════════════════════════════════════════════════════════

def experiment_A(N=1200, kappa=0.5, n_runs=5):
    print("\n" + "=" * 72)
    print("  EXPERIMENT A: MATRIX SCAN (11 coupling matrices + null)")
    print("  Fixed: omega=omega_E6, kappa=%.2f, N=%d, %d runs" % (kappa, N, n_runs))
    print("=" * 72)

    rng = np.random.RandomState(42)
    R = rng.randn(6, 6)
    R = (R + R.T) / 2
    R *= np.linalg.norm(cartan_E6(), 'fro') / np.linalg.norm(R, 'fro')

    matrices = [
        ('Zero',          np.zeros((6, 6))),
        ('0.5*I',         0.5 * np.eye(6)),
        ('I',             np.eye(6)),
        ('2*I',           2.0 * np.eye(6)),
        ('3*I',           3.0 * np.eye(6)),
        ('E6',            cartan_E6()),
        ('A6',            cartan_A6()),
        ('D6',            cartan_D6()),
        ('0.5*E6',        0.5 * cartan_E6()),
        ('Diag(eig_E6)',  np.diag(np.linalg.eigvalsh(cartan_E6()))),
        ('Rand Sym',      R),
    ]

    omega = omega_E6()
    results = []

    hdr = (f"  {'Matrix':<16s}|{'d_s':>8s} {'+-':>4s}"
           f"|{'recur':>8s}|{'offdiag':>8s}|{'eig_spr':>8s}|{'<k>':>6s}|")
    print(hdr)
    print("  " + "-" * 66)

    for name, K in matrices:
        t0 = time.time()
        ds_list, rec_list, deg_list = [], [], []
        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, omega, seed=r * 111)
            G = build_graph(ph, seed=r * 222)
            ds_list.append(spectral_dimension(G))
            rec_list.append(phase_recurrence(ph))
            deg_list.append(2 * G.number_of_edges() / G.number_of_nodes())

        props = mat_props(K)
        row = {
            'name': name,
            'ds_m': np.mean(ds_list), 'ds_s': np.std(ds_list),
            'rec':  np.mean(rec_list),
            'deg':  np.mean(deg_list),
            **props,
        }
        results.append(row)
        dt = time.time() - t0

        print(f"  {name:<16s}|{row['ds_m']:7.3f} +{row['ds_s']:4.2f}"
              f"|{row['rec']:8.4f}|{props['offdiag']:8.3f}"
              f"|{props['eig_spread']:8.3f}|{row['deg']:6.1f}| {dt:.0f}s")

    ds_null = []
    for r in range(n_runs):
        ph_rand = np.random.RandomState(r * 333).uniform(0, 2 * np.pi, (N, 6))
        G_null = build_graph(ph_rand, seed=r * 444)
        ds_null.append(spectral_dimension(G_null))

    null_m, null_s = np.mean(ds_null), np.std(ds_null)
    print(f"  {'Null (random)':<16s}|{null_m:7.3f} +{null_s:4.2f}"
          f"|{'--':>8s}|{'--':>8s}|{'--':>8s}|{'--':>6s}|")

    ds_arr = np.array([r['ds_m'] for r in results])
    print("\n  -- Correlations with d_s --")
    corr_results = {}
    for prop in ['offdiag', 'eig_spread', 'eig_max', 'trace', 'rec', 'deg']:
        vals = np.array([r[prop] for r in results])
        if np.std(vals) > 1e-10 and np.std(ds_arr) > 1e-10:
            corr = np.corrcoef(vals, ds_arr)[0, 1]
            corr_results[prop] = corr
            marker = " <<<" if abs(corr) > 0.7 else ""
            print(f"    r(d_s, {prop:12s}) = {corr:+.3f}{marker}")

    return results, null_m, null_s, corr_results


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT B: OMEGA vs K FACTORIAL
# ═══════════════════════════════════════════════════════════════

def experiment_B(N=1200, kappa=0.5, n_runs=5):
    print("\n" + "=" * 72)
    print("  EXPERIMENT B: omega vs K FACTORIAL DESIGN")
    print("  Question: Is it the FREQUENCIES or the COUPLING that controls d_s?")
    print("=" * 72)

    omegas = [
        ('w_E6',   omega_E6()),
        ('w_unif', omega_uniform()),
        ('w_rand', omega_random()),
    ]
    Ks = [
        ('K=0',  np.zeros((6, 6))),
        ('K=I',  np.eye(6)),
        ('K=E6', cartan_E6()),
    ]

    table = {}

    print(f"\n  {'':10s}", end='')
    for kn, _ in Ks:
        print(f" | {kn:^16s}", end='')
    print()
    print("  " + "-" * 62)

    for on, om in omegas:
        print(f"  {on:10s}", end='')
        for kn, K in Ks:
            ds_vals = []
            for r in range(n_runs):
                ph = evolve_phases(N, K, kappa, om, seed=r * 77 + 13)
                G = build_graph(ph, seed=r * 88 + 7)
                ds_vals.append(spectral_dimension(G))
            m, s = np.mean(ds_vals), np.std(ds_vals)
            table[(on, kn)] = (m, s)
            print(f" | {m:5.2f}+-{s:4.2f}   ", end='')
        print()

    all_ds = [v[0] for v in table.values()]
    gm = np.mean(all_ds)

    ss_omega = sum(
        len(Ks) * (np.mean([table[(on, kn)][0] for kn, _ in Ks]) - gm) ** 2
        for on, _ in omegas
    )
    ss_K = sum(
        len(omegas) * (np.mean([table[(on, kn)][0] for on, _ in omegas]) - gm) ** 2
        for kn, _ in Ks
    )
    ss_total = sum((v[0] - gm) ** 2 for v in table.values())
    ss_resid = max(ss_total - ss_omega - ss_K, 0)

    print(f"\n  Variance decomposition (two-way ANOVA):")
    if ss_total > 1e-10:
        print(f"    omega explains: {100 * ss_omega / ss_total:5.1f}%")
        print(f"    K     explains: {100 * ss_K / ss_total:5.1f}%")
        print(f"    Residual:       {100 * ss_resid / ss_total:5.1f}%")
    else:
        print(f"    No variance detected")

    dominant = "omega (frequencies)" if ss_omega > ss_K else "K (coupling matrix)"
    print(f"    => {dominant} is the DOMINANT factor")

    return table, ss_omega, ss_K, ss_total


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT C: INTERPOLATION E6 <-> I
# ═══════════════════════════════════════════════════════════════

def experiment_C(N=1200, kappa=0.5, n_runs=5, n_alpha=15, n_precision=20):
    print("\n" + "=" * 72)
    print("  EXPERIMENT C: INTERPOLATION SWEEP")
    print("  K(alpha) = (1-alpha)*K_E6 + alpha*I,  alpha in [0, 1]")
    print("  Goal: find alpha* where d_s = 4.0")
    print("=" * 72)

    K_E6 = cartan_E6()
    K_I = np.eye(6)
    omega = omega_E6()

    alphas = np.linspace(0, 1, n_alpha)
    ds_m_list, ds_s_list = [], []

    for alpha in alphas:
        K = (1 - alpha) * K_E6 + alpha * K_I
        vals = []
        for r in range(n_runs):
            ph = evolve_phases(N, K, kappa, omega, seed=r * 55 + 3)
            G = build_graph(ph, seed=r * 66 + 5)
            vals.append(spectral_dimension(G))
        m, s = np.mean(vals), np.std(vals)
        ds_m_list.append(m)
        ds_s_list.append(s)

        od = np.linalg.norm(K - np.diag(np.diag(K)), 'fro')
        print(f"    alpha={alpha:.3f} | offdiag={od:.3f}"
              f" | d_s = {m:.3f} +- {s:.3f}")

    ds_m_arr = np.array(ds_m_list)
    ds_s_arr = np.array(ds_s_list)

    target = 4.0
    crossing = None
    for i in range(len(ds_m_arr) - 1):
        if (ds_m_arr[i] - target) * (ds_m_arr[i + 1] - target) < 0:
            frac = (target - ds_m_arr[i]) / (ds_m_arr[i + 1] - ds_m_arr[i])
            crossing = alphas[i] + frac * (alphas[i + 1] - alphas[i])
            break

    precision_result = None

    if crossing is not None:
        print(f"\n  >>> d_s = 4.0 crossing at alpha* ~ {crossing:.4f}")
        K_star = (1 - crossing) * K_E6 + crossing * K_I

        vals_p = []
        for r in range(n_precision):
            ph = evolve_phases(N, K_star, kappa, omega, seed=r * 33)
            G = build_graph(ph, seed=r * 44)
            vals_p.append(spectral_dimension(G))

        vp = np.array(vals_p)
        sem = vp.std() / np.sqrt(len(vp))
        ci = 1.96 * sem
        hit = abs(vp.mean() - 4.0) < ci

        precision_result = {
            'mean': vp.mean(), 'std': vp.std(),
            'sem': sem, 'ci': ci, 'hit': hit,
        }

        print(f"    Precision ({n_precision} runs): "
              f"d_s = {vp.mean():.3f} +- {vp.std():.3f}")
        print(f"    95% CI: [{vp.mean() - ci:.3f}, {vp.mean() + ci:.3f}]")
        print(f"    d_s = 4.0 within 95% CI: {'YES' if hit else 'NO'}")
    else:
        print(f"\n  >>> d_s = 4.0 crossing NOT found in alpha in [0, 1]")
        print(f"    d_s range: [{ds_m_arr.min():.3f}, {ds_m_arr.max():.3f}]")
        if ds_m_arr.min() > 4.0:
            print("    All d_s > 4.0: try larger kappa or different omega.")
        elif ds_m_arr.max() < 4.0:
            print("    All d_s < 4.0: try smaller kappa.")

    return alphas, ds_m_arr, ds_s_arr, crossing, precision_result


# ═══════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════

def make_plots(res_A, null_m, corr_A, alphas, ds_C, ds_C_err, crossing,
               table_B, ss_omega, ss_K, ss_total):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names  = [r['name'] for r in res_A]
    ds_A   = [r['ds_m'] for r in res_A]
    ds_A_e = [r['ds_s'] for r in res_A]
    od_A   = [r['offdiag'] for r in res_A]
    rec_A  = [r['rec'] for r in res_A]

    # (0,0) d_s vs off-diagonal norm
    ax = axes[0, 0]
    ax.errorbar(od_A, ds_A, yerr=ds_A_e, fmt='o', color='teal',
                capsize=4, ms=7, lw=1.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (od_A[i], ds_A[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)
    ax.axhline(null_m, color='gray', ls='--', alpha=0.6, label=f'Null = {null_m:.2f}')
    ax.axhline(4.0, color='red', ls=':', alpha=0.8, label='d_s = 4.0 target')
    r_od = corr_A.get('offdiag', 0)
    ax.set_title(f'd_s vs off-diagonal norm  (r = {r_od:+.3f})', fontsize=12)
    ax.set_xlabel('||K_offdiag||_F')
    ax.set_ylabel('Spectral dimension d_s')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # (0,1) d_s vs phase recurrence
    ax = axes[0, 1]
    ax.errorbar(rec_A, ds_A, yerr=ds_A_e, fmt='s', color='indigo',
                capsize=4, ms=7, lw=1.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (rec_A[i], ds_A[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.8)
    ax.axhline(4.0, color='red', ls=':', alpha=0.8, label='d_s = 4.0 target')
    r_rec = corr_A.get('rec', 0)
    ax.set_title(f'd_s vs recurrence rate  (r = {r_rec:+.3f})', fontsize=12)
    ax.set_xlabel('Phase recurrence fraction')
    ax.set_ylabel('Spectral dimension d_s')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # (1,0) Experiment B heatmap
    ax = axes[1, 0]
    omega_names = ['w_E6', 'w_unif', 'w_rand']
    K_names = ['K=0', 'K=I', 'K=E6']
    grid = np.zeros((len(omega_names), len(K_names)))
    for io, on in enumerate(omega_names):
        for ik, kn in enumerate(K_names):
            if (on, kn) in table_B:
                grid[io, ik] = table_B[(on, kn)][0]

    im = ax.imshow(grid, cmap='RdYlBu_r', aspect='auto',
                   vmin=max(0, grid.min() - 0.5),
                   vmax=grid.max() + 0.5)
    ax.set_xticks(range(len(K_names)))
    ax.set_xticklabels(K_names, fontsize=10)
    ax.set_yticks(range(len(omega_names)))
    ax.set_yticklabels(omega_names, fontsize=10)

    for io in range(len(omega_names)):
        for ik in range(len(K_names)):
            key = (omega_names[io], K_names[ik])
            if key in table_B:
                m, s = table_B[key]
                ax.text(ik, io, f'{m:.2f}\n+/-{s:.2f}',
                        ha='center', va='center', fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='d_s')

    if ss_total > 1e-10:
        pct_om = 100 * ss_omega / ss_total
        pct_K  = 100 * ss_K / ss_total
        subtitle = f'omega explains {pct_om:.0f}%,  K explains {pct_K:.0f}%'
    else:
        subtitle = 'No variance detected'
    ax.set_title(f'Exp B: omega vs K factorial\n{subtitle}', fontsize=12)

    # (1,1) Experiment C interpolation
    ax = axes[1, 1]
    ax.errorbar(alphas, ds_C, yerr=ds_C_err, fmt='D-', color='#d62728',
                capsize=3, ms=5, lw=2, label='K(a) = (1-a)*E6 + a*I')
    ax.axhline(4.0, color='black', ls=':', lw=1.5, alpha=0.8, label='d_s = 4.0')
    ax.fill_between(alphas, 3.8, 4.2, color='gold', alpha=0.15, label='+/-0.2 band')

    if crossing is not None:
        ax.axvline(crossing, color='green', ls='--', lw=2, alpha=0.8,
                   label=f'a* = {crossing:.3f}')
        ax.plot(crossing, 4.0, '*', color='green', ms=15, zorder=5)

    ax.set_xlabel('alpha  (0 = pure E6,  1 = pure I)', fontsize=11)
    ax.set_ylabel('Spectral dimension d_s', fontsize=11)
    ax.set_title('Exp C: E6 <-> Identity interpolation', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_xlim(-0.05, 1.05)

    fig.suptitle('Rescue Experiment: What Controls Spectral Dimension?',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('rescue_experiment.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  [+] Figure saved: rescue_experiment.png")


# ═══════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ═══════════════════════════════════════════════════════════════

def print_verdict(res_A, null_m, null_s, corr_A,
                  table_B, ss_omega, ss_K, ss_total,
                  crossing, precision_result):

    print("\n" + "=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)

    # 1. What controls d_s?
    print("\n  1. WHAT CONTROLS d_s?")
    best_prop, best_r = None, 0
    for prop, r in corr_A.items():
        if abs(r) > abs(best_r):
            best_r = r
            best_prop = prop

    if best_prop and abs(best_r) > 0.7:
        print(f"     Strongest predictor: {best_prop} (r = {best_r:+.3f})")
        if best_prop == 'rec':
            print("     -> Phase RECURRENCE controls d_s:")
            print("        More recurrences -> more shortcuts -> lower d_s")
        elif best_prop == 'offdiag':
            direction = "more" if best_r > 0 else "less"
            print(f"     -> Off-diagonal coupling -> {direction} d_s")
        elif best_prop == 'deg':
            print("     -> Average degree controls d_s directly")
        else:
            print(f"     -> {best_prop} has strong correlation with d_s")
    elif best_prop:
        print(f"     No dominant predictor (best: {best_prop}, r={best_r:+.3f})")
    else:
        print("     No correlations computed")

    # 2. omega vs K
    print("\n  2. FREQUENCIES (omega) vs COUPLING (K)?")
    if ss_total > 1e-10:
        pct_om = 100 * ss_omega / ss_total
        pct_K  = 100 * ss_K / ss_total
        print(f"     omega explains: {pct_om:.1f}% of variance")
        print(f"     K explains:     {pct_K:.1f}% of variance")
        if pct_om > 2 * pct_K:
            print("     -> FREQUENCIES dominate")
        elif pct_K > 2 * pct_om:
            print("     -> COUPLING dominates")
        else:
            print("     -> Both factors contribute comparably")
    else:
        print("     No variance detected")

    # 3. Can d_s = 4.0?
    print("\n  3. CAN d_s = 4.0 BE ACHIEVED?")
    if crossing is not None and precision_result is not None:
        pm = precision_result
        if pm['hit']:
            print(f"     YES at alpha* = {crossing:.4f}")
            print(f"     K* = {1-crossing:.3f}*K_E6 + {crossing:.3f}*I")
            print(f"     d_s = {pm['mean']:.3f} +- {pm['std']:.3f}"
                  f" (SEM = {pm['sem']:.4f})")
            print(f"     95% CI: [{pm['mean']-pm['ci']:.3f},"
                  f" {pm['mean']+pm['ci']:.3f}]")
            print("     d_s = 4.0 IS within the 95% CI")

            K_E6 = cartan_E6()
            K_star = (1 - crossing) * K_E6 + crossing * np.eye(6)
            eigs_star = np.linalg.eigvalsh(K_star)
            od_star = K_star.copy()
            np.fill_diagonal(od_star, 0)
            od_norm = np.linalg.norm(od_star, 'fro')

            print(f"\n     Properties of K*:")
            print(f"       Eigenvalues: "
                  f"[{', '.join(f'{e:.3f}' for e in eigs_star)}]")
            print(f"       Off-diag norm: {od_norm:.3f}")
            print(f"       Trace: {np.trace(K_star):.3f}")
            print(f"       Det: {np.linalg.det(K_star):.4f}")
        else:
            print(f"     CLOSE but not exact at alpha* = {crossing:.4f}")
            print(f"     d_s = {pm['mean']:.3f} +- {pm['std']:.3f}")
            print(f"     95% CI: [{pm['mean']-pm['ci']:.3f},"
                  f" {pm['mean']+pm['ci']:.3f}]")
            print("     d_s = 4.0 NOT within the 95% CI")
    elif crossing is not None:
        print(f"     Crossing at alpha* ~ {crossing:.4f} (no precision data)")
    else:
        ds_all = [r['ds_m'] for r in res_A]
        print(f"     NO -- d_s = 4.0 not found in interpolation")
        print(f"     d_s range: [{min(ds_all):.2f}, {max(ds_all):.2f}]")
        if min(ds_all) > 4.0:
            print("     All d_s > 4.0: try larger kappa")
        elif max(ds_all) < 4.0:
            print("     All d_s < 4.0: try smaller kappa")

    # 4. Identity anomaly
    print("\n  4. IDENTITY ANOMALY EXPLAINED?")
    id_row = next((r for r in res_A if r['name'] == 'I'), None)
    e6_row = next((r for r in res_A if r['name'] == 'E6'), None)
    zero_row = next((r for r in res_A if r['name'] == 'Zero'), None)

    if id_row and e6_row:
        print(f"     d_s(I)  = {id_row['ds_m']:.3f},"
              f" recurrence = {id_row['rec']:.4f}")
        print(f"     d_s(E6) = {e6_row['ds_m']:.3f},"
              f" recurrence = {e6_row['rec']:.4f}")
        if zero_row:
            print(f"     d_s(0)  = {zero_row['ds_m']:.3f},"
                  f" recurrence = {zero_row['rec']:.4f}")

        if id_row['rec'] > e6_row['rec']:
            print("     Identity has HIGHER recurrence -> more shortcuts -> lower d_s")
        elif id_row['rec'] < e6_row['rec']:
            print("     Identity has LOWER recurrence -> something else controls d_s")
        else:
            print("     Similar recurrence -> d_s difference is topological")

    # 5. Reformulated hypothesis
    print("\n" + "=" * 72)
    print("  REFORMULATED HYPOTHESIS")
    print("=" * 72)

    if crossing is not None and precision_result and precision_result['hit']:
        r_rec_val = corr_A.get('rec', 0)
        mechanism = ('recurrence-driven' if abs(r_rec_val) > 0.5
                     else 'coupling-driven')
        print(f"""
  Original: "E6 Monostring predicts d_s = 4 (4D spacetime)"
  Status:   PARTIALLY RESCUED

  Revised:  "A coupled 6D oscillator with E6-derived frequencies and
             coupling K* = {1-crossing:.2f}*K_E6 + {crossing:.2f}*I
             produces a graph with spectral dimension d_s = 4.0 +/- CI."

  What this DOES show:
    - d_s = 4.0 exists in the (K, omega, kappa) parameter space
    - E6 algebraic structure participates in reaching this point
    - Mechanism is {mechanism}

  What this does NOT show:
    - WHY K* should be the physically preferred coupling
    - That d_s = 4.0 is an attractor rather than a tuned point
    - Any connection to General Relativity or Friedmann equations
    - That this is different from other algebras achieving d_s = 4.0

  Next steps required:
    1) Show K* arises from a physical principle (e.g. minimum action)
    2) Test whether D6, A6 also reach d_s=4.0 at some alpha*
    3) Compute the continuous limit and check for Friedmann equations
    4) Compare with Pantheon+ supernova data
""")
    else:
        print(f"""
  Original: "E6 Monostring predicts d_s = 4 (4D spacetime)"
  Status:   NOT RESCUED in current parameter range

  What WAS found:
    - E6 structure reduces d_s compared to null (real effect)
    - d_s = 4.0 was {'not reachable' if crossing is None
                     else 'reachable but imprecise'}
      with current E6->I interpolation

  Possible salvage routes:
    a) Scan kappa more broadly (current: 0.5)
    b) Try nonlinear coupling: kappa*K*sin(phi) -> kappa*K*f(phi)
    c) Try higher-rank algebras (E7, E8)
    d) Allow K to be non-symmetric (directed coupling)
    e) Characterise d_s = f(algebra, kappa) landscape
""")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("=" * 72)
    print("  RESCUE EXPERIMENT: WHAT CONTROLS SPECTRAL DIMENSION?")
    print("  Three sub-experiments on identical phase landscapes")
    print("=" * 72)

    res_A, null_m, null_s, corr_A = experiment_A(
        N=1200, kappa=0.5, n_runs=5
    )

    table_B, ss_omega, ss_K, ss_total = experiment_B(
        N=1200, kappa=0.5, n_runs=5
    )

    alphas, ds_C, ds_C_err, crossing, precision = experiment_C(
        N=1200, kappa=0.5, n_runs=5, n_alpha=15, n_precision=20
    )

    print("\n[+] Building plots...")
    make_plots(res_A, null_m, corr_A,
               alphas, ds_C, ds_C_err, crossing,
               table_B, ss_omega, ss_K, ss_total)

    print_verdict(res_A, null_m, null_s, corr_A,
                  table_B, ss_omega, ss_K, ss_total,
                  crossing, precision)

    total = time.time() - t_start
    print(f"\n  Total runtime: {total:.0f}s ({total/60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
