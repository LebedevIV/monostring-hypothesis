import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import eye, diags, coo_matrix
from scipy.sparse.linalg import eigsh
import time
import warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

def cartan_An(n):
    C = np.zeros((n,n))
    for i in range(n): C[i,i] = 2
    for i in range(n-1): C[i,i+1] = -1; C[i+1,i] = -1
    return C

def cartan_Dn(n):
    C = np.zeros((n,n))
    for i in range(n): C[i,i] = 2
    for i in range(n-2): C[i,i+1] = -1; C[i+1,i] = -1
    if n >= 3: C[n-3,n-1] = -1; C[n-1,n-3] = -1
    return C

def gen_traj(N, kappa, T, n_phases, cartan=None):
    primes = [2,3,5,7,11,13,17,19,23,29,31,37]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])
    if cartan is None:
        if n_phases <= 6:
            cartan = C_E6[:n_phases,:n_phases].copy()
        else:
            cartan = np.zeros((n_phases,n_phases))
            cartan[:6,:6] = C_E6
            for i in range(6, n_phases):
                cartan[i,i] = 2
                cartan[i,i-1] = -1; cartan[i-1,i] = -1
    ph = np.zeros((N, n_phases))
    ph[0] = np.random.uniform(0, 2*np.pi, n_phases)
    for n in range(N-1):
        noise = np.random.normal(0, T, n_phases) if T > 0 else 0
        ph[n+1] = (ph[n] + omega + kappa * cartan @ np.sin(ph[n]) + noise) % (2*np.pi)
    return ph

def build_graph_sparse(phases, target_deg=25, delta_min=5):
    N = len(phases)
    if phases.shape[1] == 0:
        rows, cols = [], []
        for i in range(N-1):
            rows.extend([i,i+1]); cols.extend([i+1,i])
        A = coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N)).tocsc()
        degrees = np.array(A.sum(axis=1)).flatten()
        return A, degrees, np.mean(degrees)
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    lo, hi = 0.01, 8.0
    best_eps, best_deg, best_pf = 1.0, 0, None
    for _ in range(25):
        mid = (lo+hi)/2
        pairs = tree.query_pairs(r=mid, output_type='ndarray')
        pf = pairs[np.abs(pairs[:,0]-pairs[:,1])>delta_min] if len(pairs)>0 \
            else np.zeros((0,2),dtype=int)
        actual = (2*len(pf)+2*(N-1))/N
        if actual < target_deg: lo = mid
        else: hi = mid
        if abs(actual-target_deg) < abs(best_deg-target_deg):
            best_deg, best_eps, best_pf = actual, mid, pf.copy()
        if abs(actual-target_deg)/target_deg < 0.05: break
    rows, cols = [], []
    for i in range(N-1):
        rows.extend([i,i+1]); cols.extend([i+1,i])
    if best_pf is not None:
        for a,b in best_pf:
            rows.extend([int(a),int(b)]); cols.extend([int(b),int(a)])
    A = coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N)).tocsc()
    degrees = np.array(A.sum(axis=1)).flatten()
    avg_deg = np.mean(degrees)
    if avg_deg > target_deg*1.1 and best_pf is not None:
        nr = int((avg_deg-target_deg)*N/2)
        if nr > 0 and len(best_pf) > nr:
            keep = np.random.choice(len(best_pf), len(best_pf)-nr, replace=False)
            rows2, cols2 = [], []
            for i in range(N-1):
                rows2.extend([i,i+1]); cols2.extend([i+1,i])
            for idx in keep:
                a,b = int(best_pf[idx,0]),int(best_pf[idx,1])
                rows2.extend([a,b]); cols2.extend([b,a])
            A = coo_matrix((np.ones(len(rows2)),(rows2,cols2)),shape=(N,N)).tocsc()
            degrees = np.array(A.sum(axis=1)).flatten()
            avg_deg = np.mean(degrees)
    return A, degrees, avg_deg

def spectral_dim_weyl(A, degrees, N, k_eig=300):
    D_inv_sqrt = diags(1.0/np.sqrt(np.maximum(degrees,1)))
    L = eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    k_actual = min(k_eig, N-2)
    evals, _ = eigsh(L, k=k_actual, which='SM', tol=1e-4, maxiter=5000)
    evals = np.sort(evals)
    nz = evals[evals > 1e-6]
    if len(nz) < 10:
        return 0, 0
    N_lam = np.arange(1, len(nz)+1, dtype=float)
    slopes = []
    for frac in [0.2, 0.3, 0.5]:
        nf = max(5, int(len(nz)*frac))
        try:
            s, _ = np.polyfit(np.log(nz[:nf]), np.log(N_lam[:nf]), 1)
            slopes.append(2*s)
        except:
            pass
    d = np.median(slopes) if slopes else 0
    d_err = np.std(slopes) if len(slopes) > 1 else 0
    return d, d_err

def kuramoto(phases):
    r = np.zeros(phases.shape[1])
    for d in range(phases.shape[1]):
        r[d] = np.abs(np.mean(np.exp(1j*phases[:,d])))
    return r

def run_clarification():
    print("=" * 70)
    print("  CLARIFICATION EXPERIMENTS (FIXED)")
    print("  Resolving three remaining ambiguities")
    print("=" * 70)

    N = 8000
    TARGET_DEG = 25
    K_EIG = 300

    # ============================================================
    # TEST 1: Is d_s(n=4) = 4.45 trivial?
    # ============================================================
    print("\n" + "=" * 60)
    print("  TEST 1: Is d_s ~ 4 at n_phases=4 trivial?")
    print("  Compare E6-Monostring vs null at EACH n_phases")
    print("=" * 60)

    print("\n  Cold phase (T=0.02, kappa=0.5):")
    print("  {:>6s} | {:>7s} {:>9s} {:>8s} | {:>10s}".format(
        "n_ph", "d_s(E6)", "d_s(null)", "d_s(hot)", "reduction%"))
    print("  " + "-" * 55)

    test1_cold = []
    test1_null = []
    test1_hot = []
    test1_nphs = [3, 4, 5, 6]

    for n_ph in test1_nphs:
        results_mono = []
        results_null = []
        results_hot = []

        for run in range(5):
            np.random.seed(run * 71 + n_ph * 13)
            phases = gen_traj(N, kappa=0.5, T=0.02, n_phases=n_ph)
            A, deg, _ = build_graph_sparse(phases, target_deg=TARGET_DEG)
            d, _ = spectral_dim_weyl(A, deg, N, K_EIG)
            results_mono.append(d)

            ph_null = np.random.uniform(0, 2*np.pi, (N, n_ph))
            A_n, deg_n, _ = build_graph_sparse(ph_null, target_deg=TARGET_DEG)
            d_n, _ = spectral_dim_weyl(A_n, deg_n, N, K_EIG)
            results_null.append(d_n)

            phases_h = gen_traj(N, kappa=0.5, T=3.0, n_phases=n_ph)
            A_h, deg_h, _ = build_graph_sparse(phases_h, target_deg=TARGET_DEG)
            d_h, _ = spectral_dim_weyl(A_h, deg_h, N, K_EIG)
            results_hot.append(d_h)

        m_mono = np.mean(results_mono)
        m_null = np.mean(results_null)
        m_hot = np.mean(results_hot)
        reduction = (1 - m_mono / m_null) * 100 if m_null > 0 else 0

        test1_cold.append(m_mono)
        test1_null.append(m_null)
        test1_hot.append(m_hot)

        print("  {:>6d} | {:>7.2f} {:>9.2f} {:>8.2f} | {:>9.1f}%".format(
            n_ph, m_mono, m_null, m_hot, reduction))

    # ============================================================
    # TEST 2: What is compactified?
    # ============================================================
    print("\n" + "=" * 60)
    print("  TEST 2: What is compactified?")
    print("  Spectral dimension of PROJECTIONS onto synced/unsynced dims")
    print("=" * 60)

    np.random.seed(42)
    phases_cold = gen_traj(N, kappa=0.5, T=0.005, n_phases=6)
    r_per = kuramoto(phases_cold)

    sorted_dims = np.argsort(r_per)[::-1]
    synced = list(sorted_dims[:2])
    unsynced = list(sorted_dims[2:])

    print("\n  Per-dim r: {}".format(
        " ".join("{:.3f}".format(r) for r in r_per)))
    print("  Synced dims (top 2 by r): {} (r = {:.3f}, {:.3f})".format(
        synced, r_per[synced[0]], r_per[synced[1]]))
    print("  Unsynced dims: {} (r = {})".format(
        unsynced,
        ", ".join("{:.3f}".format(r_per[d]) for d in unsynced)))

    d_s_synced = 0
    d_s_unsynced = 0
    d_s_full = 0

    if len(synced) >= 1 and len(unsynced) >= 1:
        phases_synced_only = phases_cold[:, synced]
        A_s, deg_s, avgd_s = build_graph_sparse(phases_synced_only, target_deg=TARGET_DEG)
        d_s_synced, _ = spectral_dim_weyl(A_s, deg_s, N, K_EIG)

        phases_unsynced_only = phases_cold[:, unsynced]
        A_u, deg_u, avgd_u = build_graph_sparse(phases_unsynced_only, target_deg=TARGET_DEG)
        d_s_unsynced, _ = spectral_dim_weyl(A_u, deg_u, N, K_EIG)

        A_full, deg_full, avgd_full = build_graph_sparse(phases_cold, target_deg=TARGET_DEG)
        d_s_full, _ = spectral_dim_weyl(A_full, deg_full, N, K_EIG)

        print("\n  d_s from synced dims only ({} dims):     {:.2f}".format(
            len(synced), d_s_synced))
        print("  d_s from unsynced dims only ({} dims):   {:.2f}".format(
            len(unsynced), d_s_unsynced))
        print("  d_s from ALL dims (6 dims):              {:.2f}".format(d_s_full))
        print("  Sum: d_s(sync) + d_s(unsync) =           {:.2f}".format(
            d_s_synced + d_s_unsynced))

        if d_s_synced < d_s_unsynced * 0.5:
            print("\n  *** Synced dimensions are PARTIALLY COLLAPSED ***")
            print("  Effective dimension ~ d_s(unsynced) = {:.2f}".format(d_s_unsynced))
        else:
            print("\n  Synced dimensions are NOT collapsed (d_s similar)")
    else:
        print("\n  WARNING: Cannot separate synced/unsynced. Skipping decomposition.")

    # ============================================================
    # TEST 3: Does d_s depend on the Lie algebra?
    # ============================================================
    print("\n" + "=" * 60)
    print("  TEST 3: Does d_s depend on the Lie algebra?")
    print("  All at n_phases=6, T=0.005, kappa=0.5")
    print("=" * 60)

    algebras = {
        'E6': C_E6,
        'A6 (SU7)': cartan_An(6),
        'D6 (SO12)': cartan_Dn(6),
        'Identity': np.eye(6) * 2,
    }

    print("\n  {:>15s} | {:>7s} {:>6s} | {:>5s} {:>5s} {:>5s} | {:>5s}".format(
        "Algebra", "d_s", "+/-", "r_max", "r_min", "r_K", "deg"))
    print("  " + "-" * 60)

    algebra_results = {}
    for name, C in algebras.items():
        ds_runs = []
        last_r_per = None
        last_avgd = 0
        for run in range(5):
            np.random.seed(run * 37 + hash(name) % 1000)
            phases = gen_traj(N, kappa=0.5, T=0.005, n_phases=6, cartan=C)
            A, deg, avgd = build_graph_sparse(phases, target_deg=TARGET_DEG)
            d, _ = spectral_dim_weyl(A, deg, N, K_EIG)
            ds_runs.append(d)
            last_r_per = kuramoto(phases)
            last_avgd = avgd

        r_k = np.prod(last_r_per)**(1./6)
        mean_d = np.mean(ds_runs)
        std_d = np.std(ds_runs)

        algebra_results[name] = {
            'd_s': mean_d, 'd_err': std_d,
            'r_max': np.max(last_r_per), 'r_min': np.min(last_r_per),
            'r_kura': r_k
        }

        print("  {:>15s} | {:>7.2f} {:>6.2f} | {:>5.3f} {:>5.3f} {:>5.3f} | {:>5.1f}".format(
            name, mean_d, std_d,
            np.max(last_r_per), np.min(last_r_per), r_k, last_avgd))

    # ============================================================
    # TEST 4: d_s vs kappa at n_phases=4
    # ============================================================
    print("\n" + "=" * 60)
    print("  TEST 4: d_s vs kappa at n_phases=4 (where d_s ~ 4)")
    print("=" * 60)

    print("\n  {:>8s} | {:>7s} {:>6s} | {:>7s} | {:>5s} {:>5s}".format(
        "kappa", "d_s", "+/-", "d_null", "r_max", "r_K"))
    print("  " + "-" * 50)

    for kappa in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]:
        ds_runs = []
        last_r_per = None
        for run in range(3):
            np.random.seed(run * 53 + int(kappa * 100))
            phases = gen_traj(N, kappa=kappa, T=0.02, n_phases=4)
            A, deg, _ = build_graph_sparse(phases, target_deg=TARGET_DEG)
            d, _ = spectral_dim_weyl(A, deg, N, K_EIG)
            ds_runs.append(d)
            last_r_per = kuramoto(phases)

        r_k = np.prod(last_r_per)**(1./4)

        np.random.seed(999)
        ph_n = np.random.uniform(0, 2*np.pi, (N, 4))
        A_n, deg_n, _ = build_graph_sparse(ph_n, target_deg=TARGET_DEG)
        d_n, _ = spectral_dim_weyl(A_n, deg_n, N, K_EIG)

        print("  {:>8.2f} | {:>7.2f} {:>6.2f} | {:>7.2f} | {:>5.3f} {:>5.3f}".format(
            kappa, np.mean(ds_runs), np.std(ds_runs), d_n,
            np.max(last_r_per), r_k))

    # ============================================================
    # TEST 5: Precision d_s at n_phases=4, kappa=0.5
    # ============================================================
    print("\n" + "=" * 60)
    print("  TEST 5: Precision measurement at n_phases=4, kappa=0.5")
    print("  20 runs with different initial conditions")
    print("=" * 60)

    ds_4ph = []
    ds_4ph_null = []
    for run in range(20):
        np.random.seed(run * 97 + 7)
        phases = gen_traj(N, kappa=0.5, T=0.02, n_phases=4)
        A, deg, _ = build_graph_sparse(phases, target_deg=TARGET_DEG)
        d, _ = spectral_dim_weyl(A, deg, N, K_EIG)
        ds_4ph.append(d)

        ph_n = np.random.uniform(0, 2*np.pi, (N, 4))
        A_n, deg_n, _ = build_graph_sparse(ph_n, target_deg=TARGET_DEG)
        d_n, _ = spectral_dim_weyl(A_n, deg_n, N, K_EIG)
        ds_4ph_null.append(d_n)

        if (run+1) % 5 == 0:
            print("  Runs 1-{}: d_s = {:.3f} +/- {:.3f}, null = {:.3f} +/- {:.3f}".format(
                run+1, np.mean(ds_4ph), np.std(ds_4ph),
                np.mean(ds_4ph_null), np.std(ds_4ph_null)))

    mean_4 = np.mean(ds_4ph)
    std_4 = np.std(ds_4ph)
    sem_4 = std_4 / np.sqrt(len(ds_4ph))
    mean_4n = np.mean(ds_4ph_null)

    print("\n  PRECISION RESULT (n_phases=4):")
    print("  d_s = {:.3f} +/- {:.3f} (SEM = {:.3f})".format(mean_4, std_4, sem_4))
    print("  95% CI: [{:.3f}, {:.3f}]".format(mean_4 - 1.96*sem_4, mean_4 + 1.96*sem_4))
    print("  Null: {:.3f} +/- {:.3f}".format(mean_4n, np.std(ds_4ph_null)))
    print("  Reduction vs null: {:.1f}%".format((1 - mean_4/mean_4n)*100))
    print("  |d_s - 4.0| = {:.3f}".format(abs(mean_4 - 4.0)))
    print("  d_s = 4.0 within 95% CI: {}".format(
        "YES" if mean_4 - 1.96*sem_4 <= 4.0 <= mean_4 + 1.96*sem_4 else "NO"))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Test 1: d_s per n_phases
    ax = axes[0, 0]
    ax.plot(test1_nphs, test1_cold, 'o-', color='blue', lw=2.5, markersize=10,
            label='E6 cold (T=0.02)')
    ax.plot(test1_nphs, test1_null, 's--', color='gray', lw=2, markersize=8,
            label='Null (random)')
    ax.plot(test1_nphs, test1_hot, '^--', color='orange', lw=2, markersize=8,
            label='E6 hot (T=3.0)')
    ax.plot([2,7], [2,7], 'k:', alpha=0.3, label='d_s = n')
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='d_s = 4')
    ax.set_xlabel('n_phases', fontsize=12)
    ax.set_ylabel('d_s (Weyl)', fontsize=12)
    ax.set_title('Spectral dim: E6 vs Null\nGap = synchronization effect')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Test 2: Subgraph decomposition
    ax = axes[0, 1]
    if d_s_synced > 0 or d_s_unsynced > 0:
        labels_sub = ['Synced\n({} dims)'.format(len(synced)),
                      'Unsynced\n({} dims)'.format(len(unsynced)),
                      'Full\n(6 dims)',
                      'Sum\n(S+U)']
        values_sub = [d_s_synced, d_s_unsynced, d_s_full,
                      d_s_synced + d_s_unsynced]
        colors_sub = ['red', 'blue', 'green', 'purple']
        ax.bar(range(4), values_sub, color=colors_sub, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_sub)
        ax.axhline(4, ls='--', color='red', alpha=0.5)
        ax.set_ylabel('d_s')
        ax.set_title('Dimensional decomposition')
    else:
        ax.text(0.5, 0.5, 'Test 2 skipped\n(no sync detected)',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Test 3: Algebra comparison
    ax = axes[0, 2]
    alg_names = list(algebra_results.keys())
    alg_ds = [algebra_results[n]['d_s'] for n in alg_names]
    alg_err = [algebra_results[n]['d_err'] for n in alg_names]
    ax.barh(range(len(alg_names)), alg_ds, xerr=alg_err,
            color='teal', alpha=0.7, edgecolor='black', capsize=3)
    ax.set_yticks(range(len(alg_names)))
    ax.set_yticklabels(alg_names)
    ax.axvline(4, ls='--', color='red', alpha=0.5)
    ax.set_xlabel('d_s')
    ax.set_title('d_s by Lie algebra\n(n=6, T=0.005)')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Test 5: Precision histogram
    ax = axes[1, 0]
    ax.hist(ds_4ph, bins=10, color='navy', alpha=0.7, edgecolor='white',
            label='Monostring (n=4)')
    ax.hist(ds_4ph_null, bins=10, color='gray', alpha=0.5, edgecolor='white',
            label='Null (n=4)')
    ax.axvline(4.0, ls='--', color='red', lw=2, label='d_s = 4')
    ax.axvline(mean_4, ls='-', color='green', lw=2,
               label='mean = {:.2f}'.format(mean_4))
    ax.set_xlabel('d_s')
    ax.set_ylabel('Count')
    ax.set_title('Precision: n_phases=4 (20 runs)\n'
                 '{:.2f} +/- {:.2f}'.format(mean_4, std_4))
    ax.legend(fontsize=8)

    # 5. Reduction percentage
    ax = axes[1, 1]
    reductions_pct = [(1 - c/n)*100 for c, n in zip(test1_cold, test1_null)]
    ax.bar(test1_nphs, reductions_pct, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('n_phases')
    ax.set_ylabel('Reduction vs null (%)')
    ax.set_title('E6 synchronization effect\n(% reduction in d_s)')
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Summary
    ax = axes[1, 2]
    summary = (
        "SUMMARY\n"
        "=" * 35 + "\n\n"
        "Test 1: d_s(n=4, E6) vs null\n"
        "  E6 cold: {:.2f}\n"
        "  Null:    {:.2f}\n"
        "  Reduction: {:.0f}%\n\n"
        "Test 2: Subgraph decomposition\n"
        "  Synced:   {:.2f}\n"
        "  Unsynced: {:.2f}\n"
        "  Full:     {:.2f}\n\n"
        "Test 5: Precision (n=4)\n"
        "  d_s = {:.3f} +/- {:.3f}\n"
        "  4.0 in 95% CI: {}"
    ).format(
        test1_cold[1], test1_null[1],
        (1 - test1_cold[1]/test1_null[1])*100,
        d_s_synced, d_s_unsynced, d_s_full,
        mean_4, std_4,
        "YES" if mean_4 - 1.96*sem_4 <= 4.0 <= mean_4 + 1.96*sem_4 else "NO"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.suptitle('Clarification Experiments: Resolving Ambiguities',
                 fontsize=14, fontweight='bold')
    plt.savefig('clarification_results.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: clarification_results.png")
    plt.close()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  CLARIFICATION VERDICT")
    print("=" * 70)

    reductions_pct_all = [(1 - c/n)*100 for c, n in zip(test1_cold, test1_null)]

    alg_ds_values = [algebra_results[n]['d_s'] for n in algebra_results]

    tests = [
        ("d_s(E6, n=4) < d_s(null, n=4) by > 1.0",
         test1_cold[1] < test1_null[1] - 1.0,
         "E6={:.2f}, null={:.2f}".format(test1_cold[1], test1_null[1])),
        ("d_s(E6, n=4) ~ 4.0 (within 95% CI)",
         mean_4 - 1.96*sem_4 <= 4.0 <= mean_4 + 1.96*sem_4,
         "{:.3f} +/- {:.3f}".format(mean_4, sem_4)),
        ("Synced subgraph has lower d_s than unsynced",
         d_s_synced < d_s_unsynced if d_s_synced > 0 else False,
         "sync={:.2f}, unsync={:.2f}".format(d_s_synced, d_s_unsynced)),
        ("d_s depends on algebra (range > 1.0)",
         max(alg_ds_values) - min(alg_ds_values) > 1.0 if alg_ds_values else False,
         "range: {:.2f} to {:.2f}".format(
             min(alg_ds_values), max(alg_ds_values)) if alg_ds_values else "N/A"),
        ("Reduction > 30% vs null for all n tested",
         all(r > 30 for r in reductions_pct_all),
         "min reduction = {:.0f}%".format(min(reductions_pct_all))),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print("  {} {}  {:50s} {}".format(
            "[+]" if passed else "[-]",
            "PASS" if passed else "FAIL",
            name, detail))

    print("\n  Score: {}/{}".format(n_pass, len(tests)))
    print("=" * 70)

if __name__ == "__main__":
    run_clarification()
