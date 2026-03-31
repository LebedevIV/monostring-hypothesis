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

def gen_traj(N, kappa, T, n_phases):
    primes = [2,3,5,7,11,13,17,19,23,29,31,37]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])
    if n_phases <= 6:
        C = C_E6[:n_phases,:n_phases].copy()
    else:
        C = np.zeros((n_phases, n_phases))
        C[:6,:6] = C_E6
        for i in range(6, n_phases):
            C[i,i] = 2
            C[i,i-1] = -1; C[i-1,i] = -1
    ph = np.zeros((N, n_phases))
    ph[0] = np.random.uniform(0, 2*np.pi, n_phases)
    for n in range(N-1):
        noise = np.random.normal(0, T, n_phases) if T > 0 else 0
        ph[n+1] = (ph[n] + omega + kappa * C @ np.sin(ph[n]) + noise) % (2*np.pi)
    return ph

def build_graph_sparse(phases, target_deg=25, delta_min=5):
    N = len(phases)
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
                a,b = int(best_pf[idx,0]), int(best_pf[idx,1])
                rows2.extend([a,b]); cols2.extend([b,a])
            A = coo_matrix((np.ones(len(rows2)),(rows2,cols2)),shape=(N,N)).tocsc()
            degrees = np.array(A.sum(axis=1)).flatten()
            avg_deg = np.mean(degrees)
    return A, degrees, avg_deg

def spectral_dimension_weyl(A, degrees, N, k_eig=300):
    D_inv_sqrt = diags(1.0/np.sqrt(np.maximum(degrees,1)))
    L = eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    k_actual = min(k_eig, N-2)
    evals, _ = eigsh(L, k=k_actual, which='SM', tol=1e-4, maxiter=5000)
    evals = np.sort(evals)
    nz = evals[evals > 1e-6]
    if len(nz) < 10:
        return 0, 0, nz
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
    return d, d_err, nz

def kuramoto(phases):
    r = np.zeros(phases.shape[1])
    for d in range(phases.shape[1]):
        r[d] = np.abs(np.mean(np.exp(1j*phases[:,d])))
    return r

def run_final_test():
    print("=" * 70)
    print("  FINAL TEST: SPECTRAL DIMENSION vs NUMBER OF PHASES")
    print("  Question: is d_s ≈ n_phases (trivial) or d_s < n_phases?")
    print("=" * 70)

    N = 8000
    TARGET_DEG = 25
    K_EIG = 300

    # ============================================================
    # PART 1: d_s vs n_phases at FIXED temperature and kappa
    # ============================================================
    print("\n[1/4] d_s vs n_phases (T=0.02, kappa=0.5)")
    print("      {:>8s} | {:>8s} {:>6s} | {:>8s} {:>8s} | {:>5s}".format(
        "n_phases", "d_s(Weyl)", "±err", "d_s(null)", "ratio", "deg"))
    print("      " + "-" * 60)

    phase_results = []
    for n_ph in [2, 3, 4, 5, 6, 8, 10]:
        t0 = time.time()

        # Monostring
        phases = gen_traj(N, kappa=0.5, T=0.02, n_phases=n_ph)
        A, degrees, avg_deg = build_graph_sparse(phases, target_deg=TARGET_DEG)
        d_s, d_err, nz = spectral_dimension_weyl(A, degrees, N, k_eig=K_EIG)
        r_per = kuramoto(phases)

        # Null model (same n_phases, random)
        phases_null = np.random.uniform(0, 2*np.pi, (N, n_ph))
        A_null, deg_null, _ = build_graph_sparse(phases_null, target_deg=TARGET_DEG)
        d_null, _, _ = spectral_dimension_weyl(A_null, deg_null, N, k_eig=K_EIG)

        ratio = d_s / d_null if d_null > 0.1 else np.nan

        phase_results.append({
            'n_ph': n_ph, 'd_s': d_s, 'd_err': d_err,
            'd_null': d_null, 'ratio': ratio, 'deg': avg_deg,
            'r_per': r_per
        })

        print("      {:>8d} | {:>8.2f} {:>6.2f} | {:>8.2f} {:>8.3f} | {:>5.1f}  ({:.1f}s)".format(
            n_ph, d_s, d_err, d_null, ratio, avg_deg, time.time()-t0))

    # ============================================================
    # PART 2: d_s vs n_phases at HIGH temperature (symmetric)
    # ============================================================
    print("\n[2/4] d_s vs n_phases (T=3.0, kappa=0.5) — symmetric phase")
    print("      {:>8s} | {:>8s} {:>6s} | {:>8s} {:>8s}".format(
        "n_phases", "d_s(hot)", "±err", "d_s(null)", "ratio"))
    print("      " + "-" * 50)

    hot_results = []
    for n_ph in [2, 3, 4, 5, 6, 8, 10]:
        t0 = time.time()
        phases = gen_traj(N, kappa=0.5, T=3.0, n_phases=n_ph)
        A, degrees, _ = build_graph_sparse(phases, target_deg=TARGET_DEG)
        d_s, d_err, _ = spectral_dimension_weyl(A, degrees, N, k_eig=K_EIG)

        phases_null = np.random.uniform(0, 2*np.pi, (N, n_ph))
        A_null, deg_null, _ = build_graph_sparse(phases_null, target_deg=TARGET_DEG)
        d_null, _, _ = spectral_dimension_weyl(A_null, deg_null, N, k_eig=K_EIG)

        ratio = d_s / d_null if d_null > 0.1 else np.nan
        hot_results.append({
            'n_ph': n_ph, 'd_s': d_s, 'd_err': d_err,
            'd_null': d_null, 'ratio': ratio
        })

        print("      {:>8d} | {:>8.2f} {:>6.2f} | {:>8.2f} {:>8.3f}  ({:.1f}s)".format(
            n_ph, d_s, d_err, d_null, ratio, time.time()-t0))

    # ============================================================
    # PART 3: REDUCTION FACTOR = d_s(cold) / d_s(hot) vs n_phases
    # ============================================================
    print("\n[3/4] REDUCTION FACTOR: d_s(cold) / d_s(hot)")
    print("      {:>8s} | {:>8s} {:>8s} {:>10s}".format(
        "n_phases", "d_s(cold)", "d_s(hot)", "reduction"))
    print("      " + "-" * 40)

    for cold, hot in zip(phase_results, hot_results):
        if cold['n_ph'] == hot['n_ph']:
            red = cold['d_s'] / hot['d_s'] if hot['d_s'] > 0 else np.nan
            print("      {:>8d} | {:>8.2f} {:>8.2f} {:>10.3f}".format(
                cold['n_ph'], cold['d_s'], hot['d_s'], red))

    # ============================================================
    # PART 4: REPRODUCIBILITY at n_phases=6
    # ============================================================
    print("\n[4/4] REPRODUCIBILITY: 10 runs at n_phases=6, T=0.02")

    d_values = []
    d_null_values = []
    for run in range(10):
        np.random.seed(run * 42 + 13)
        phases = gen_traj(N, kappa=0.5, T=0.02, n_phases=6)
        A, degrees, _ = build_graph_sparse(phases, target_deg=TARGET_DEG)
        d_s, _, _ = spectral_dimension_weyl(A, degrees, N, k_eig=K_EIG)
        d_values.append(d_s)

        phases_null = np.random.uniform(0, 2*np.pi, (N, 6))
        A_null, deg_null, _ = build_graph_sparse(phases_null, target_deg=TARGET_DEG)
        d_null, _, _ = spectral_dimension_weyl(A_null, deg_null, N, k_eig=K_EIG)
        d_null_values.append(d_null)

        if (run+1) % 5 == 0:
            print("      Runs 1-{}: d_s = {:.2f} ± {:.2f}, null = {:.2f} ± {:.2f}".format(
                run+1, np.mean(d_values), np.std(d_values),
                np.mean(d_null_values), np.std(d_null_values)))

    mean_d = np.mean(d_values)
    std_d = np.std(d_values)
    mean_null = np.mean(d_null_values)

    print("\n      FINAL: d_s = {:.3f} ± {:.3f}".format(mean_d, std_d))
    print("      NULL:  d_s = {:.3f} ± {:.3f}".format(mean_null, np.std(d_null_values)))
    print("      Difference: {:.3f}".format(mean_d - mean_null))
    print("      Reduction vs null: {:.1f}%".format(
        (1 - mean_d/mean_null) * 100 if mean_null > 0 else 0))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. d_s vs n_phases (cold vs hot vs null)
    ax = axes[0, 0]
    nphs = [r['n_ph'] for r in phase_results]
    ds_cold = [r['d_s'] for r in phase_results]
    ds_null = [r['d_null'] for r in phase_results]
    ds_hot = [r['d_s'] for r in hot_results]

    ax.plot(nphs, ds_cold, 'o-', color='blue', lw=2, markersize=8,
            label='Monostring (T=0.02, cold)')
    ax.plot(nphs, ds_hot, 's-', color='orange', lw=2, markersize=8,
            label='Monostring (T=3.0, hot)')
    ax.plot(nphs, ds_null, '^--', color='gray', lw=2, markersize=8,
            label='Null (random)')
    ax.plot(nphs, nphs, 'k:', lw=1.5, alpha=0.5, label='d_s = n_phases (trivial)')
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='d_s = 4')
    ax.set_xlabel('Number of internal phases', fontsize=12)
    ax.set_ylabel('Spectral dimension d_s (Weyl)', fontsize=12)
    ax.set_title('KEY TEST: d_s vs n_phases\n'
                 'Below diagonal = non-trivial reduction')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Reduction factor vs n_phases
    ax = axes[0, 1]
    reductions = [c['d_s']/h['d_s'] if h['d_s'] > 0 else 1
                  for c, h in zip(phase_results, hot_results)]
    ax.plot(nphs, reductions, 'D-', color='darkgreen', lw=2.5, markersize=10)
    ax.axhline(1, ls='--', color='black', alpha=0.5, label='No reduction')
    ax.set_xlabel('Number of phases')
    ax.set_ylabel('d_s(cold) / d_s(hot)')
    ax.set_title('Reduction factor\n<1 = synchronization reduces d_s')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. d_s(mono) / d_s(null) vs n_phases
    ax = axes[1, 0]
    ratios = [r['ratio'] for r in phase_results]
    ax.plot(nphs, ratios, 'o-', color='purple', lw=2.5, markersize=10)
    ax.axhline(1, ls='--', color='black', alpha=0.5, label='= null (trivial)')
    ax.set_xlabel('Number of phases')
    ax.set_ylabel('d_s(Monostring) / d_s(null)')
    ax.set_title('Monostring vs Null\n<1 = E₆ dynamics reduces d_s')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Reproducibility histogram
    ax = axes[1, 1]
    ax.hist(d_values, bins=8, color='navy', alpha=0.7, edgecolor='white',
            label='Monostring')
    ax.hist(d_null_values, bins=8, color='gray', alpha=0.5, edgecolor='white',
            label='Null')
    ax.axvline(4, ls='--', color='red', lw=2, label='d_s = 4')
    ax.set_xlabel('Spectral dimension d_s')
    ax.set_ylabel('Count')
    ax.set_title('Reproducibility (10 runs, n_phases=6)\n'
                 'Mono: {:.1f}±{:.1f}, Null: {:.1f}±{:.1f}'.format(
                     mean_d, std_d, mean_null, np.std(d_null_values)))
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Final Test: Is spectral dimension reduction non-trivial?',
                 fontsize=14, fontweight='bold')
    plt.savefig('quantum_walk_final.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: quantum_walk_final.png")
    plt.close()

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  FINAL VERDICT: SPECTRAL DIMENSION")
    print("=" * 70)

    # Key test: is d_s < n_phases at low T?
    ds_6_cold = [r for r in phase_results if r['n_ph'] == 6][0]['d_s']
    ds_6_hot = [r for r in hot_results if r['n_ph'] == 6][0]['d_s']
    ds_6_null = [r for r in phase_results if r['n_ph'] == 6][0]['d_null']

    tests = [
        ("d_s(cold) < d_s(hot) for n=6",
         ds_6_cold < ds_6_hot - 0.5,
         "cold={:.2f}, hot={:.2f}".format(ds_6_cold, ds_6_hot)),
        ("d_s(cold) < d_s(null) for n=6",
         ds_6_cold < ds_6_null - 0.5,
         "mono={:.2f}, null={:.2f}".format(ds_6_cold, ds_6_null)),
        ("d_s(cold) < n_phases for n=6",
         ds_6_cold < 6.0 - 0.5,
         "d_s={:.2f} < 6?".format(ds_6_cold)),
        ("Reduction factor < 0.8 for most n_phases",
         np.mean([r < 0.8 for r in reductions]) > 0.5,
         "factors: {}".format(["{:.2f}".format(r) for r in reductions])),
        ("d_s DOES NOT scale linearly with n_phases",
         True,  # Will check below
         ""),
        ("Reproducible (std < 10% of mean)",
         std_d / mean_d < 0.1,
         "std/mean = {:.1%}".format(std_d/mean_d)),
    ]

    # Check linearity
    nphs_arr = np.array(nphs, dtype=float)
    ds_cold_arr = np.array(ds_cold)
    corr_linear = np.corrcoef(nphs_arr, ds_cold_arr)[0, 1]
    slope, intercept = np.polyfit(nphs_arr, ds_cold_arr, 1)
    tests[4] = (
        "d_s does NOT scale as d_s = n_phases",
        slope < 0.8,  # If slope ≈ 1, it's trivial (d_s ≈ n_phases)
        "slope={:.2f}, corr={:.3f}".format(slope, corr_linear)
    )

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print("  {} {}  {:50s} {}".format(
            "[+]" if passed else "[-]",
            "PASS" if passed else "FAIL",
            name, detail))

    print("\n  Score: {}/{}".format(n_pass, len(tests)))

    if ds_6_cold < 5.0 and ds_6_cold < ds_6_null - 1.0:
        print("\n  *** NON-TRIVIAL SPECTRAL DIMENSION REDUCTION ***")
        print("  E₆ synchronization genuinely reduces spectral dimension")
        print("  below both null model and phase space dimension.")
    elif ds_6_cold < ds_6_null - 0.5:
        print("\n  * Significant reduction vs null, but d_s > 5 *")
        print("  * Synchronization helps but doesn't reach d_s = 4 *")
    else:
        print("\n  No significant spectral dimension reduction.")

    print("\n  OVERALL INTERPRETATION:")
    if slope < 0.5:
        print("  d_s is WEAKLY dependent on n_phases → non-trivial")
    elif slope < 0.8:
        print("  d_s is MODERATELY dependent on n_phases → partially trivial")
    else:
        print("  d_s ≈ n_phases → TRIVIAL (graph dimension = input dimension)")

    print("=" * 70)

if __name__ == "__main__":
    run_final_test()
