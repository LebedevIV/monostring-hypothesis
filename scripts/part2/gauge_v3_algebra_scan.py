import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# CARTAN MATRICES FOR ALL RANK-6 SIMPLE ALGEBRAS
# ================================================================
def cartan_An(n):
    C = np.zeros((n, n))
    for i in range(n): C[i, i] = 2
    for i in range(n - 1): C[i, i+1] = -1; C[i+1, i] = -1
    return C

def cartan_Bn(n):
    C = cartan_An(n)
    C[n-2, n-1] = -2
    return C

def cartan_Cn(n):
    C = cartan_An(n)
    C[n-1, n-2] = -2
    return C

def cartan_Dn(n):
    C = np.zeros((n, n))
    for i in range(n): C[i, i] = 2
    for i in range(n - 2): C[i, i+1] = -1; C[i+1, i] = -1
    C[n-3, n-1] = -1; C[n-1, n-3] = -1
    return C

def cartan_E6():
    return np.array([
        [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
        [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
    ], dtype=np.float64)

def coxeter_omega(algebra_name, rank):
    coxeter_data = {
        'A6': (7, [1,2,3,4,5,6]),
        'B6': (11, [1,3,5,7,9,11]),
        'C6': (12, [1,3,5,7,9,11]),
        'D6': (10, [1,3,5,5,7,9]),
        'E6': (12, [1,4,5,7,8,11]),
    }
    if algebra_name in coxeter_data:
        h, exp = coxeter_data[algebra_name]
        return 2 * np.sin(np.pi * np.array(exp[:rank], dtype=float) / h)
    return np.sqrt(np.arange(2, 2 + rank, dtype=float))

def get_algebras():
    return {
        'A6_SU7':  (cartan_An(6), 'A6'),
        'B6_SO13': (cartan_Bn(6), 'B6'),
        'C6_Sp12': (cartan_Cn(6), 'C6'),
        'D6_SO12': (cartan_Dn(6), 'D6'),
        'E6':      (cartan_E6(),  'E6'),
    }

# ================================================================
# CORE
# ================================================================
def gen_traj(N, kappa, T, cartan, omega):
    D = cartan.shape[0]
    ph = np.zeros((N, D))
    ph[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N - 1):
        ph[n+1] = (ph[n] + omega + kappa * cartan @ np.sin(ph[n])
                   + np.random.normal(0, T, D)) % (2 * np.pi)
    return ph

def build_graph(phases, target_deg=25, delta_min=5):
    N = len(phases)
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    lo, hi = 0.01, 8.0
    best_eps, best_deg, best_pf = 1.0, 0, None
    for _ in range(25):
        mid = (lo + hi) / 2
        pairs = tree.query_pairs(r=mid, output_type='ndarray')
        pf = pairs[np.abs(pairs[:,0]-pairs[:,1]) > delta_min] if len(pairs) > 0 \
            else np.zeros((0,2), dtype=int)
        actual = (2*len(pf) + 2*(N-1)) / N
        if actual < target_deg: lo = mid
        else: hi = mid
        if abs(actual - target_deg) < abs(best_deg - target_deg):
            best_deg, best_eps, best_pf = actual, mid, pf.copy()
        if abs(actual - target_deg) / target_deg < 0.05: break
    G = nx.Graph(); G.add_nodes_from(range(N))
    for i in range(N-1): G.add_edge(i, i+1)
    if best_pf is not None:
        for a, b in best_pf: G.add_edge(int(a), int(b))
    fd = sum(dict(G.degree()).values()) / N
    if fd > target_deg * 1.1:
        re = [(u,v) for u,v in G.edges() if abs(u-v) > delta_min]
        nr = int((fd - target_deg) * N / 2)
        if 0 < nr < len(re):
            G.remove_edges_from([re[i] for i in np.random.choice(len(re), nr, replace=False)])
    return G, sum(dict(G.degree()).values()) / N

def torus_diff(a, b):
    return ((a - b) + np.pi) % (2*np.pi) - np.pi

def kuramoto(phases):
    r = np.zeros(phases.shape[1])
    for d in range(phases.shape[1]):
        r[d] = np.abs(np.mean(np.exp(1j * phases[:, d])))
    return r

def edge_variance(G, phases):
    D = phases.shape[1]
    var = np.zeros(D)
    diffs = {d: [] for d in range(D)}
    for u, v in G.edges():
        for d in range(D):
            diffs[d].append(torus_diff(phases[v, d], phases[u, d]))
    for d in range(D):
        var[d] = np.var(diffs[d])
    return var

# ================================================================
# NULL MODEL: Random phases, artificially synchronized
# ================================================================
def null_model_synced(N, synced_dims, target_deg=25):
    """
    Null model: random graph on T^6 with random phases,
    but dims in synced_dims are SET to constant (artificial sync).
    Tests whether edge variance ratio is trivial.
    """
    D = 6
    phases = np.random.uniform(0, 2*np.pi, (N, D))
    for d in synced_dims:
        phases[:, d] = np.random.normal(0, 0.1, N) % (2*np.pi)
    G, deg = build_graph(phases, target_deg=target_deg)
    return G, phases, deg

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_algebra_scan():
    print("=" * 70)
    print("  GAUGE HIGGS: ALL RANK-6 ALGEBRAS + NULL MODEL")
    print("=" * 70)

    N = 8000
    kappa = 0.5
    T_cold = 0.02
    T_hot = 3.0
    TARGET_DEG = 25

    algebras = get_algebras()
    results = {}

    print("\n  {:15s} | {:>5s} {:>5s} | {:>7s} {:>7s} {:>7s} | {:>6s} | {:>5s}".format(
        "Algebra", "r_max", "r_min", "Var_s", "Var_u", "Ratio", "Patt.", "deg"))
    print("  " + "-" * 75)

    for name, (cartan, cox_name) in algebras.items():
        t0 = time.time()
        omega = coxeter_omega(cox_name, 6)

        # Cold phase
        phases = gen_traj(N, kappa, T_cold, cartan, omega)
        G, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_per = kuramoto(phases)
        var = edge_variance(G, phases)

        synced = [d for d in range(6) if r_per[d] > 0.5]
        unsynced = [d for d in range(6) if r_per[d] <= 0.5]

        if len(synced) > 0 and len(unsynced) > 0:
            Vs = np.mean(var[synced])
            Vu = np.mean(var[unsynced])
            ratio = Vu / Vs if Vs > 1e-10 else np.nan
        else:
            Vs = Vu = ratio = np.nan

        pattern = "{}+{}".format(len(synced), len(unsynced))

        results[name] = {
            'r_per': r_per, 'synced': synced, 'unsynced': unsynced,
            'Var_s': Vs, 'Var_u': Vu, 'ratio': ratio,
            'pattern': pattern, 'deg': deg
        }

        r_max = np.max(r_per)
        r_min = np.min(r_per)
        print("  {:15s} | {:>5.3f} {:>5.3f} | {:>7.4f} {:>7.4f} {:>7.2f} | {:>6s} | {:>5.1f}  ({:.1f}s)".format(
            name, r_max, r_min,
            Vs if not np.isnan(Vs) else 0,
            Vu if not np.isnan(Vu) else 0,
            ratio if not np.isnan(ratio) else 0,
            pattern, deg, time.time() - t0))

    # ============================================================
    # NULL MODEL
    # ============================================================
    print("\n  NULL MODELS:")
    print("  " + "-" * 75)

    # Null 1: Random phases, dims 0,5 artificially synced
    G_null, ph_null, deg_null = null_model_synced(N, [0, 5], target_deg=TARGET_DEG)
    var_null = edge_variance(G_null, ph_null)
    Vs_null = np.mean(var_null[[0, 5]])
    Vu_null = np.mean(var_null[[1, 2, 3, 4]])
    ratio_null = Vu_null / Vs_null if Vs_null > 1e-10 else np.nan

    print("  {:15s} | {:>5s} {:>5s} | {:>7.4f} {:>7.4f} {:>7.2f} | {:>6s} | {:>5.1f}".format(
        "Null (art.sync)", "-", "-", Vs_null, Vu_null, ratio_null, "2+4", deg_null))

    # Null 2: Random phases, no sync, random split
    G_null2, ph_null2, deg_null2 = null_model_synced(N, [], target_deg=TARGET_DEG)
    var_null2 = edge_variance(G_null2, ph_null2)
    Vs_null2 = np.mean(var_null2[[0, 5]])
    Vu_null2 = np.mean(var_null2[[1, 2, 3, 4]])
    ratio_null2 = Vu_null2 / Vs_null2 if Vs_null2 > 1e-10 else np.nan

    print("  {:15s} | {:>5s} {:>5s} | {:>7.4f} {:>7.4f} {:>7.2f} | {:>6s} | {:>5.1f}".format(
        "Null (no sync)", "-", "-", Vs_null2, Vu_null2, ratio_null2, "0+6", deg_null2))

    # ============================================================
    # WEINBERG ANGLE ANALOG
    # ============================================================
    print("\n  WEINBERG ANGLE ANALOG (for algebras with 2+ synced dims):")
    print("  SM value: cos(theta_W) = {:.4f}".format(np.cos(np.arctan(
        np.sqrt(3/5) * 0.3574 / 0.6517))))  # approximate
    print("  {:15s} | {:>10s} {:>10s} {:>10s}".format(
        "Algebra", "Var(d_s1)", "Var(d_s2)", "ratio"))
    print("  " + "-" * 50)

    for name, data in results.items():
        if len(data['synced']) >= 2:
            # Get the two synced dimensions
            phases_alg = gen_traj(N, kappa, T_cold,
                                  algebras[name][0],
                                  coxeter_omega(algebras[name][1], 6))
            G_alg, _ = build_graph(phases_alg, target_deg=TARGET_DEG)
            var_alg = edge_variance(G_alg, phases_alg)

            d1, d2 = data['synced'][0], data['synced'][1]
            v1, v2 = var_alg[d1], var_alg[d2]
            weinberg = v1 / v2 if v2 > 1e-10 else np.nan

            print("  {:15s} | {:>10.6f} {:>10.6f} {:>10.4f}".format(
                name, v1, v2, weinberg))

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  VERDICT: ALGEBRA COMPARISON")
    print("=" * 70)

    # Is E6 unique?
    e6_ratio = results['E6']['ratio'] if not np.isnan(results.get('E6', {}).get('ratio', np.nan)) else 0
    other_ratios = [results[n]['ratio'] for n in results if n != 'E6' and not np.isnan(results[n]['ratio'])]

    if other_ratios:
        mean_other = np.mean(other_ratios)
        print("  E6 ratio: {:.2f}".format(e6_ratio))
        print("  Other algebras mean ratio: {:.2f}".format(mean_other))
        print("  Null (artificial sync) ratio: {:.2f}".format(ratio_null))
        print("  Null (no sync) ratio: {:.2f}".format(ratio_null2))

        if abs(e6_ratio - mean_other) / max(mean_other, 1) > 0.3:
            print("\n  E6 IS SPECIAL for gauge Higgs!")
        else:
            print("\n  E6 is NOT special — all algebras give similar ratio")

        if ratio_null > 5:
            print("  WARNING: Null model also gives high ratio — result may be trivial!")
        else:
            print("  Null model ratio is LOW — E6 dynamics is essential")

    # Breaking patterns
    print("\n  Breaking patterns:")
    for name, data in results.items():
        print("    {}: {} (synced dims: {})".format(name, data['pattern'], data['synced']))

    print("=" * 70)

    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Ratio comparison
    ax = axes[0]
    names = list(results.keys()) + ['Null(sync)', 'Null(none)']
    ratios = [results[n]['ratio'] for n in results] + [ratio_null, ratio_null2]
    colors = ['green' if r > 5 else 'orange' if r > 1.5 else 'red' for r in ratios]
    ax.barh(range(len(names)), ratios, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(1, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('Edge variance ratio (unsync/sync)')
    ax.set_title('Gauge Higgs: All Rank-6 Algebras')
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Breaking patterns
    ax = axes[1]
    for i, (name, data) in enumerate(results.items()):
        r_per = data['r_per']
        for d in range(6):
            c = 'red' if d in data['synced'] else 'blue'
            ax.plot(d, i, 'o', color=c, markersize=10 + 10*r_per[d])
    ax.set_xticks(range(6))
    ax.set_xticklabels(['d{}'.format(d+1) for d in range(6)])
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(list(results.keys()))
    ax.set_title('Synchronization pattern\n(red=synced, blue=free, size=r)')
    ax.grid(True, alpha=0.3)

    # 3. Per-dim variance (E6 vs null)
    ax = axes[2]
    var_e6_cold = edge_variance(
        build_graph(gen_traj(N, kappa, T_cold, cartan_E6(),
                              coxeter_omega('E6', 6)),
                     target_deg=TARGET_DEG)[0],
        gen_traj(N, kappa, T_cold, cartan_E6(), coxeter_omega('E6', 6))
    )
    ax.bar(np.arange(6) - 0.15, var_e6_cold, 0.3, color='teal', label='E6', alpha=0.7)
    ax.bar(np.arange(6) + 0.15, var_null, 0.3, color='gray', label='Null(sync)', alpha=0.7)
    ax.set_xticks(range(6))
    ax.set_xticklabels(['d{}'.format(d+1) for d in range(6)])
    ax.set_ylabel('Edge variance')
    ax.set_title('E6 vs Null: per-dimension')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Gauge Higgs Mechanism: Algebra Comparison + Null Model',
                 fontsize=14, fontweight='bold')
    plt.savefig('gauge_algebra_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: gauge_algebra_comparison.png")
    plt.close()

    print("\n  NEXT STEPS:")
    print("  1. If E6 is unique → focus on E6 representation theory")
    print("  2. If all algebras work → the result is about Kuramoto + Lie, not E6")
    print("  3. If null model also works → result is trivial (just sync geometry)")
    print("  4. Look for 3+1 breaking pattern (Standard Model)")

if __name__ == "__main__":
    run_algebra_scan()
