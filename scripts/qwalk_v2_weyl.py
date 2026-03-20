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

# ================================================================
# GRAPH CONSTRUCTION
# ================================================================
def gen_traj(N, kappa, T, n_phases=6):
    primes = [2,3,5,7,11,13]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])
    C = C_E6[:n_phases,:n_phases].copy() if n_phases <= 6 else np.eye(n_phases)*2
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
        rows.extend([i, i+1]); cols.extend([i+1, i])
    if best_pf is not None:
        for a,b in best_pf:
            rows.extend([int(a), int(b)]); cols.extend([int(b), int(a)])

    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N)).tocsc()

    degrees = np.array(A.sum(axis=1)).flatten()
    avg_deg = np.mean(degrees)

    # Prune if too dense
    if avg_deg > target_deg * 1.1 and best_pf is not None:
        nr = int((avg_deg - target_deg) * N / 2)
        if nr > 0 and len(best_pf) > nr:
            keep = np.random.choice(len(best_pf), len(best_pf) - nr, replace=False)
            rows2, cols2 = [], []
            for i in range(N-1):
                rows2.extend([i, i+1]); cols2.extend([i+1, i])
            for idx in keep:
                a, b = int(best_pf[idx, 0]), int(best_pf[idx, 1])
                rows2.extend([a, b]); cols2.extend([b, a])
            A = coo_matrix((np.ones(len(rows2)), (rows2, cols2)), shape=(N,N)).tocsc()
            degrees = np.array(A.sum(axis=1)).flatten()
            avg_deg = np.mean(degrees)

    return A, degrees, avg_deg

def build_lattice_sparse(dim, L):
    N = L**dim
    rows, cols = [], []
    for idx in range(N):
        cs = []
        temp = idx
        for d in range(dim):
            cs.append(temp % L); temp //= L
        for d in range(dim):
            nc = cs.copy()
            nc[d] = (cs[d] + 1) % L
            nb = sum(c * L**dd for dd, c in enumerate(nc))
            rows.extend([idx, nb]); cols.extend([nb, idx])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N,N)).tocsc()
    degrees = np.array(A.sum(axis=1)).flatten()
    return A, degrees, N

def kuramoto(phases):
    r = np.zeros(phases.shape[1])
    for d in range(phases.shape[1]):
        r[d] = np.abs(np.mean(np.exp(1j * phases[:, d])))
    return r

# ================================================================
# SPECTRAL DIMENSION VIA EIGENVALUE COUNTING
# ================================================================
def spectral_dimension_eigenvalue(A, degrees, N, k_eig=200):
    """
    Spectral dimension from the EIGENVALUE DENSITY of the Laplacian.

    The integrated density of states N(λ) = #{eigenvalues ≤ λ}
    scales as N(λ) ~ λ^{d_s/2} for small λ (Weyl's law on manifolds).

    This is MORE ROBUST than return probability because:
    1. No time evolution needed (no finite-time artifacts)
    2. Eigenvalues are computed once, then analyzed
    3. The scaling region is in λ → 0, not t → ∞

    We compute the first k eigenvalues of the normalized Laplacian
    and fit the scaling of N(λ) vs λ.
    """
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1)))
    L = eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    k_actual = min(k_eig, N - 2)
    evals, evecs = eigsh(L, k=k_actual, which='SM', tol=1e-4, maxiter=5000)
    evals = np.sort(evals)

    # Remove zero mode(s)
    nonzero = evals[evals > 1e-6]

    if len(nonzero) < 10:
        return 0, 0, evals, nonzero

    # Integrated density of states: N(λ) = index / total
    N_lambda = np.arange(1, len(nonzero) + 1, dtype=float)

    # Fit log N(λ) = (d_s/2) * log λ + const
    # Use the bottom 50% of eigenvalues (small λ regime)
    n_fit = len(nonzero) // 2
    if n_fit < 5:
        n_fit = min(len(nonzero), 20)

    log_lambda = np.log(nonzero[:n_fit])
    log_N = np.log(N_lambda[:n_fit])

    try:
        slope, intercept = np.polyfit(log_lambda, log_N, 1)
        d_spectral = 2 * slope  # N(λ) ~ λ^{d/2}
    except:
        d_spectral = 0

    # Also try different fitting ranges for robustness
    slopes = []
    for frac in [0.2, 0.3, 0.5, 0.7]:
        nf = max(5, int(len(nonzero) * frac))
        try:
            s, _ = np.polyfit(np.log(nonzero[:nf]), np.log(N_lambda[:nf]), 1)
            slopes.append(2 * s)
        except:
            pass

    d_robust = np.median(slopes) if slopes else d_spectral
    d_err = np.std(slopes) if len(slopes) > 1 else 0

    return d_robust, d_err, evals, nonzero


def spectral_dimension_heat_kernel(evals_nonzero, N,
                                    t_range=(0.01, 10), n_times=80):
    """
    Spectral dimension from the HEAT KERNEL trace.

    K(t) = Σ exp(-λ_k * t) / N

    For a d-dimensional manifold: K(t) ~ t^{-d/2} for small t.

    Using precomputed eigenvalues (no time evolution needed).
    This is exact (no approximation from expm_multiply).
    """
    times = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_times)

    K = np.zeros(n_times)
    for t_idx, t in enumerate(times):
        K[t_idx] = np.mean(np.exp(-evals_nonzero * t))

    # Local spectral dimension: d_s(t) = -2 * d(ln K)/d(ln t)
    log_t = np.log(times)
    log_K = np.log(np.maximum(K, 1e-30))

    d_local = np.zeros(n_times - 1)
    t_mid = np.zeros(n_times - 1)
    for i in range(n_times - 1):
        if K[i] > 1e-20 and K[i+1] > 1e-20:
            d_local[i] = -2 * (log_K[i+1] - log_K[i]) / (log_t[i+1] - log_t[i])
        t_mid[i] = np.sqrt(times[i] * times[i+1])

    return d_local, t_mid, K, times


# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_qw_v2():
    print("=" * 70)
    print("  QUANTUM WALK v2: EIGENVALUE COUNTING + HEAT KERNEL")
    print("  Fix: use Weyl's law N(λ)~λ^{d/2} instead of P(t)~t^{-d/2}")
    print("  Fix: heat kernel from exact eigenvalues (no finite-time error)")
    print("=" * 70)

    # ============================================================
    # PART 1: CONTROL — Known lattices
    # ============================================================
    print("\n[1/5] CONTROL: Known lattices (eigenvalue counting)")
    print("      {:>10s} {:>6s} | {:>8s} {:>6s} | {:>8s}".format(
        "Lattice", "N", "d_s(Weyl)", "±err", "k_eig"))
    print("      " + "-" * 50)

    control_data = {}
    for dim, L, k in [(2, 30, 300), (3, 12, 300), (4, 7, 300), (5, 5, 200)]:
        t0 = time.time()
        A, degrees, N = build_lattice_sparse(dim, L)
        d_s, d_err, evals, nz = spectral_dimension_eigenvalue(A, degrees, N, k_eig=k)

        control_data[dim] = {'evals': evals, 'nonzero': nz, 'N': N, 'd_s': d_s}

        print("      {:>2d}D (L={:>2d}) {:>6d} | {:>8.2f} {:>6.2f} | {:>8d}  ({:.1f}s)".format(
            dim, L, N, d_s, d_err, k, time.time()-t0))

    # Heat kernel for controls
    print("\n      CONTROL: Heat kernel d_s(t)")
    print("      {:>10s} | {:>8s} {:>8s} {:>8s}".format(
        "Lattice", "d_s(t=0.1)", "d_s(t=1)", "d_s(t=5)"))
    print("      " + "-" * 40)

    for dim in control_data:
        nz = control_data[dim]['nonzero']
        d_local, t_mid, K, times = spectral_dimension_heat_kernel(
            nz, control_data[dim]['N'], t_range=(0.01, 20))

        # Find d_s at specific times
        ds_01 = d_local[np.argmin(np.abs(t_mid - 0.1))] if len(d_local) > 0 else 0
        ds_1 = d_local[np.argmin(np.abs(t_mid - 1.0))] if len(d_local) > 0 else 0
        ds_5 = d_local[np.argmin(np.abs(t_mid - 5.0))] if len(d_local) > 0 else 0

        control_data[dim]['d_local'] = d_local
        control_data[dim]['t_mid'] = t_mid
        control_data[dim]['K'] = K
        control_data[dim]['times'] = times

        print("      {:>2d}D        | {:>8.2f} {:>8.2f} {:>8.2f}".format(
            dim, ds_01, ds_1, ds_5))

    # ============================================================
    # PART 2: MONOSTRING — Temperature scan
    # ============================================================
    print("\n[2/5] MONOSTRING: Spectral dimension vs temperature")

    N_mono = 8000
    kappa = 0.5
    TARGET_DEG = 25
    K_EIG = 300

    temperatures = [3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

    print("      {:>6s} | {:>8s} {:>6s} | {:>8s} {:>8s} {:>8s} | {:>5s} {:>5s}".format(
        "T", "d_s(Weyl)", "±err", "d_s(0.1)", "d_s(1.0)", "d_s(5.0)",
        "r_K", "deg"))
    print("      " + "-" * 75)

    mono_data = []

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N_mono, kappa, T)
        A, degrees, avg_deg = build_graph_sparse(phases, target_deg=TARGET_DEG)
        r_per = kuramoto(phases)
        r_k = np.prod(r_per)**(1./6)

        d_s, d_err, evals, nz = spectral_dimension_eigenvalue(
            A, degrees, N_mono, k_eig=K_EIG)

        d_local, t_mid, K, times = spectral_dimension_heat_kernel(
            nz, N_mono, t_range=(0.01, 20))

        ds_01 = d_local[np.argmin(np.abs(t_mid - 0.1))] if len(d_local) > 0 else 0
        ds_1 = d_local[np.argmin(np.abs(t_mid - 1.0))] if len(d_local) > 0 else 0
        ds_5 = d_local[np.argmin(np.abs(t_mid - 5.0))] if len(d_local) > 0 else 0

        mono_data.append({
            'T': T, 'd_weyl': d_s, 'd_err': d_err,
            'ds_01': ds_01, 'ds_1': ds_1, 'ds_5': ds_5,
            'r_kura': r_k, 'deg': avg_deg,
            'd_local': d_local, 't_mid': t_mid,
            'K': K, 'times': times, 'evals': evals, 'nz': nz
        })

        print("      {:>6.3f} | {:>8.2f} {:>6.2f} | {:>8.2f} {:>8.2f} {:>8.2f} | "
              "{:>5.3f} {:>5.1f}  ({:.1f}s)".format(
            T, d_s, d_err, ds_01, ds_1, ds_5, r_k, avg_deg, time.time()-t0))

    # ============================================================
    # PART 3: NULL MODEL
    # ============================================================
    print("\n[3/5] NULL MODEL: Random T^6")

    t0 = time.time()
    phases_null = np.random.uniform(0, 2*np.pi, (N_mono, 6))
    A_null, deg_null, avg_deg_null = build_graph_sparse(
        phases_null, target_deg=TARGET_DEG)

    d_null, d_null_err, evals_null, nz_null = spectral_dimension_eigenvalue(
        A_null, deg_null, N_mono, k_eig=K_EIG)

    d_local_null, t_mid_null, K_null, times_null = spectral_dimension_heat_kernel(
        nz_null, N_mono, t_range=(0.01, 20))

    ds_null_01 = d_local_null[np.argmin(np.abs(t_mid_null - 0.1))]
    ds_null_1 = d_local_null[np.argmin(np.abs(t_mid_null - 1.0))]
    ds_null_5 = d_local_null[np.argmin(np.abs(t_mid_null - 5.0))]

    print("      d_s(Weyl) = {:.2f} ± {:.2f}".format(d_null, d_null_err))
    print("      d_s(t=0.1) = {:.2f}, d_s(t=1) = {:.2f}, d_s(t=5) = {:.2f}".format(
        ds_null_01, ds_null_1, ds_null_5))
    print("      ({:.1f}s)".format(time.time()-t0))

    # ============================================================
    # PART 4: EIGENVALUE SPACING (spectral statistics)
    # ============================================================
    print("\n[4/5] EIGENVALUE SPACING STATISTICS")
    print("      GUE (quantum chaos): Wigner-Dyson spacing P(s)~s*exp(-s²)")
    print("      Poisson (integrable): P(s) = exp(-s)")

    for label, nz_data in [("Monostring T=0.02", mono_data[-1]['nz']),
                            ("Null (random)", nz_null)]:
        if len(nz_data) < 20:
            continue
        # Unfolded spacing
        spacings = np.diff(nz_data)
        mean_s = np.mean(spacings)
        if mean_s > 0:
            s_normalized = spacings / mean_s
            # Level spacing ratio <r> = <min(s_i, s_{i+1})/max(s_i, s_{i+1})>
            if len(s_normalized) > 2:
                ratios = np.minimum(s_normalized[:-1], s_normalized[1:]) / \
                         np.maximum(s_normalized[:-1], s_normalized[1:])
                r_mean = np.mean(ratios)
                # GUE: <r> ≈ 0.5307, Poisson: <r> ≈ 0.3863
                print("      {}: <r> = {:.4f} (GUE=0.531, Poisson=0.386)".format(
                    label, r_mean))

    # ============================================================
    # PART 5: KAPPA SCAN
    # ============================================================
    print("\n[5/5] KAPPA SCAN: d_s vs coupling strength")
    print("      {:>8s} | {:>8s} {:>6s} | {:>8s} {:>8s}".format(
        "kappa", "d_s(Weyl)", "±err", "d_s(0.1)", "d_s(1.0)"))
    print("      " + "-" * 45)

    kappa_data = []
    for kappa_val in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        t0 = time.time()
        phases_k = gen_traj(N_mono, kappa_val, T=0.02)
        A_k, deg_k, avg_k = build_graph_sparse(phases_k, target_deg=TARGET_DEG)
        d_k, d_k_err, _, nz_k = spectral_dimension_eigenvalue(
            A_k, deg_k, N_mono, k_eig=K_EIG)

        d_loc_k, t_mid_k, _, _ = spectral_dimension_heat_kernel(
            nz_k, N_mono, t_range=(0.01, 20))
        ds_k_01 = d_loc_k[np.argmin(np.abs(t_mid_k - 0.1))] if len(d_loc_k) > 0 else 0
        ds_k_1 = d_loc_k[np.argmin(np.abs(t_mid_k - 1.0))] if len(d_loc_k) > 0 else 0

        kappa_data.append({
            'kappa': kappa_val, 'd_weyl': d_k, 'd_err': d_k_err,
            'ds_01': ds_k_01, 'ds_1': ds_k_1
        })

        print("      {:>8.2f} | {:>8.2f} {:>6.2f} | {:>8.2f} {:>8.2f}  ({:.1f}s)".format(
            kappa_val, d_k, d_k_err, ds_k_01, ds_k_1, time.time()-t0))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Control: heat kernel d_s(t)
    ax = axes[0, 0]
    for dim in sorted(control_data.keys()):
        cd = control_data[dim]
        if len(cd.get('d_local', [])) > 0:
            ax.semilogx(cd['t_mid'], cd['d_local'], '-', lw=2,
                        label='{}D lattice'.format(dim))
            ax.axhline(dim, ls=':', alpha=0.3)
    ax.set_xlabel('Time scale t')
    ax.set_ylabel('Heat kernel d_s(t)')
    ax.set_title('CONTROL: Heat kernel spectral dimension\n(should plateau at true D)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)

    # 2. Monostring: d_s(t) at different T
    ax = axes[0, 1]
    for i in [0, len(mono_data)//2, -1]:
        md = mono_data[i]
        if len(md['d_local']) > 0:
            ax.semilogx(md['t_mid'], md['d_local'], '-', lw=2,
                        label='T={}'.format(md['T']))
    # Null
    if len(d_local_null) > 0:
        ax.semilogx(t_mid_null, d_local_null, '--', color='gray', lw=2,
                    label='Null (random)')
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='d_s=4')
    ax.set_xlabel('Time scale t')
    ax.set_ylabel('d_s(t)')
    ax.set_title('Monostring: Heat kernel d_s(t)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)

    # 3. Weyl d_s vs T
    ax = axes[0, 2]
    Ts = [md['T'] for md in mono_data]
    ds_weyl = [md['d_weyl'] for md in mono_data]
    ds_err = [md['d_err'] for md in mono_data]
    ax.errorbar(Ts, ds_weyl, yerr=ds_err, fmt='o-', color='navy',
                lw=2, capsize=3, markersize=8)
    ax.axhline(d_null, ls='--', color='gray', alpha=0.7,
               label='Null d_s={:.1f}'.format(d_null))
    ax.axhline(4, ls=':', color='red', alpha=0.5, label='d_s=4')
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel('Temperature T (cooling →)')
    ax.set_ylabel('d_s (Weyl)')
    ax.set_title('Spectral dimension vs T\n(Weyl law: N(λ)~λ^{d/2})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Eigenvalue density N(λ) — log-log
    ax = axes[1, 0]
    md_cold = mono_data[-1]
    nz_cold = md_cold['nz']
    if len(nz_cold) > 5:
        N_lam = np.arange(1, len(nz_cold)+1, dtype=float)
        ax.loglog(nz_cold, N_lam, 'o', color='navy', markersize=2,
                  label='Monostring (T=0.02)')
    if len(nz_null) > 5:
        N_lam_null = np.arange(1, len(nz_null)+1, dtype=float)
        ax.loglog(nz_null, N_lam_null, 's', color='gray', markersize=2,
                  label='Null')
    # Reference slopes
    lam_ref = np.logspace(-3, 0, 100)
    for d_ref in [2, 4, 6]:
        ax.loglog(lam_ref, lam_ref**(d_ref/2) * 50, ':', alpha=0.3,
                  label='d_s={}'.format(d_ref))
    ax.set_xlabel('λ (eigenvalue)')
    ax.set_ylabel('N(λ) (integrated DOS)')
    ax.set_title('Weyl law: N(λ) ~ λ^{d_s/2}')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. Kappa scan
    ax = axes[1, 1]
    ks = [kd['kappa'] for kd in kappa_data]
    ds_k = [kd['d_weyl'] for kd in kappa_data]
    ds_k_err = [kd['d_err'] for kd in kappa_data]
    ax.errorbar(ks, ds_k, yerr=ds_k_err, fmt='o-', color='darkred',
                lw=2, capsize=3, markersize=8)
    ax.axhline(d_null, ls='--', color='gray', alpha=0.7, label='Null')
    ax.axhline(4, ls=':', color='red', alpha=0.5, label='d_s=4')
    ax.set_xlabel('Coupling κ')
    ax.set_ylabel('d_s (Weyl)')
    ax.set_title('Spectral dimension vs κ')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. d_s vs Kuramoto r
    ax = axes[1, 2]
    r_kuras = [md['r_kura'] for md in mono_data]
    ax.scatter(r_kuras, ds_weyl, c=Ts, cmap='coolwarm', s=80, edgecolors='black')
    ax.axhline(d_null, ls='--', color='gray', alpha=0.5, label='Null')
    ax.axhline(4, ls=':', color='red', alpha=0.5, label='d_s=4')
    ax.set_xlabel('Kuramoto order parameter r')
    ax.set_ylabel('d_s (Weyl)')
    ax.set_title('d_s vs synchronization')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Quantum Walk v2: Eigenvalue Counting + Heat Kernel',
                 fontsize=14, fontweight='bold')
    plt.savefig('quantum_walk_v2_results.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: quantum_walk_v2_results.png")
    plt.close()

    # ============================================================
    # VERDICT
    # ============================================================
    d_cold_weyl = mono_data[-1]['d_weyl']
    d_hot_weyl = mono_data[0]['d_weyl']

    print("\n" + "=" * 70)
    print("  QUANTUM WALK v2 VERDICT")
    print("=" * 70)

    tests = [
        ("Control: 2D lattice gives d_s ≈ 2",
         abs(control_data[2]['d_s'] - 2) < 1.0,
         "d_s = {:.2f}".format(control_data[2]['d_s'])),
        ("Control: 4D lattice gives d_s ≈ 4",
         abs(control_data[4]['d_s'] - 4) < 1.5,
         "d_s = {:.2f}".format(control_data[4]['d_s'])),
        ("Monostring d_s changes with T",
         abs(d_cold_weyl - d_hot_weyl) > 0.3,
         "hot={:.2f}, cold={:.2f}".format(d_hot_weyl, d_cold_weyl)),
        ("Monostring d_s differs from null",
         abs(d_cold_weyl - d_null) > 0.3,
         "mono={:.2f}, null={:.2f}".format(d_cold_weyl, d_null)),
        ("Monostring d_s ≈ 4 at some T",
         any(abs(md['d_weyl'] - 4.0) < 1.0 for md in mono_data),
         "closest: {:.2f}".format(
             min((md['d_weyl'] for md in mono_data), key=lambda d: abs(d-4)))),
        ("d_s correlates with Kuramoto r",
         abs(np.corrcoef(r_kuras, ds_weyl)[0,1]) > 0.5 if len(r_kuras) > 2 else False,
         "corr = {:.3f}".format(np.corrcoef(r_kuras, ds_weyl)[0,1]) if len(r_kuras) > 2 else "N/A"),
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
    run_qw_v2()
