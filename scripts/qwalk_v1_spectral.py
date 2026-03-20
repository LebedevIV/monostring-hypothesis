import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, eye, diags
from scipy.sparse.linalg import eigsh, expm_multiply
import time
import warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# GRAPH CONSTRUCTION (from established code)
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

def build_graph(phases, target_deg=25, delta_min=5):
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

    # Build adjacency
    adj = {i: set() for i in range(N)}
    for i in range(N-1):
        adj[i].add(i+1); adj[i+1].add(i)
    if best_pf is not None:
        for a,b in best_pf:
            adj[int(a)].add(int(b)); adj[int(b)].add(int(a))

    fd = sum(len(adj[v]) for v in adj) / N
    if fd > target_deg * 1.1:
        res_edges = [(u,v) for u in adj for v in adj[u] if v > u and abs(u-v) > delta_min]
        nr = int((fd - target_deg) * N / 2)
        if 0 < nr < len(res_edges):
            for u,v in [res_edges[i] for i in np.random.choice(len(res_edges), nr, replace=False)]:
                adj[u].discard(v); adj[v].discard(u)

    return adj, sum(len(adj[v]) for v in adj) / N

def adj_to_sparse(adj, N):
    rows, cols = [], []
    for i in range(N):
        for j in adj[i]:
            rows.append(i); cols.append(j)
    data = np.ones(len(rows))
    from scipy.sparse import coo_matrix
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()

def kuramoto(phases):
    r = np.zeros(phases.shape[1])
    for d in range(phases.shape[1]):
        r[d] = np.abs(np.mean(np.exp(1j * phases[:, d])))
    return r

# ================================================================
# QUANTUM WALK: SPECTRAL DIMENSION
# ================================================================
def spectral_dimension(adj, N, t_range=(0.1, 100), n_times=50, n_starts=30):
    """
    Spectral dimension from return probability of quantum walk.

    P(t) = |<x|e^{-iHt}|x>|^2

    For a d-dimensional lattice: <P(t)> ~ t^{-d/2}
    → d_s = -2 * d(ln P)/d(ln t)

    This is the SPECTRAL dimension — determined by the spectrum
    of the Hamiltonian, not by graph construction tricks.
    """
    A = adj_to_sparse(adj, N)
    degrees = np.array([len(adj[i]) for i in range(N)], dtype=float)
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1)))

    # Normalized Laplacian as Hamiltonian
    H = eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    times = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_times)

    avg_return_prob = np.zeros(n_times)

    starts = np.random.choice(N, min(n_starts, N), replace=False)

    for start in starts:
        psi_0 = np.zeros(N, dtype=np.complex128)
        psi_0[start] = 1.0

        for t_idx, t in enumerate(times):
            psi_t = expm_multiply(-1j * t * H, psi_0)
            return_prob = np.abs(psi_t[start])**2
            avg_return_prob[t_idx] += return_prob

    avg_return_prob /= len(starts)

    # Fit: ln P = -(d_s/2) * ln t + const
    # Local spectral dimension at each scale
    log_t = np.log(times)
    log_P = np.log(np.maximum(avg_return_prob, 1e-30))

    # Compute local slope
    d_spectral = np.zeros(n_times - 1)
    t_mid = np.zeros(n_times - 1)
    for i in range(n_times - 1):
        if avg_return_prob[i] > 1e-20 and avg_return_prob[i+1] > 1e-20:
            d_spectral[i] = -2 * (log_P[i+1] - log_P[i]) / (log_t[i+1] - log_t[i])
        t_mid[i] = np.sqrt(times[i] * times[i+1])

    # Also fit globally over the middle range
    mid_start = n_times // 4
    mid_end = 3 * n_times // 4
    valid = avg_return_prob[mid_start:mid_end] > 1e-20
    if np.sum(valid) > 3:
        coeffs = np.polyfit(log_t[mid_start:mid_end][valid],
                            log_P[mid_start:mid_end][valid], 1)
        d_global = -2 * coeffs[0]
    else:
        d_global = 0

    return {
        'times': times,
        'return_prob': avg_return_prob,
        'd_spectral_local': d_spectral,
        't_mid': t_mid,
        'd_global': d_global
    }

# ================================================================
# QUANTUM WALK: DISPERSION RELATION
# ================================================================
def dispersion_relation(adj, N, k_max=60):
    """
    Dispersion relation from Laplacian eigenvalues.

    E_k = eigenvalue_k of H (Hamiltonian = normalized Laplacian)

    For a relativistic particle: E ~ |k| (linear, massless)
    For a non-relativistic: E ~ k² (quadratic, massive)
    For a massive relativistic: E² = m² + k² (gap at k=0)

    The shape of E(k) tells us the physics.
    """
    A = adj_to_sparse(adj, N)
    degrees = np.array([len(adj[i]) for i in range(N)], dtype=float)
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1)))
    H = eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    k_actual = min(k_max, N - 2)
    evals, evecs = eigsh(H, k=k_actual, which='SM', tol=1e-4, maxiter=5000)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    return evals, evecs

# ================================================================
# QUANTUM WALK: PROPAGATOR AND MASS
# ================================================================
def measure_propagator(adj, N, H_sparse, n_starts=20,
                        t_max=20, n_times=40):
    """
    Two-point correlator G(r, t) = <0|e^{-iHt}|r>

    For massive particle: G decays exponentially in r at fixed t
    For massless: G decays as power law

    Measure G(r,t) and extract mass from decay rate.
    """
    times = np.linspace(0.5, t_max, n_times)

    # BFS distance bins
    max_dist = 10
    G_rt = np.zeros((max_dist + 1, n_times))
    counts = np.zeros(max_dist + 1)

    starts = np.random.choice(N, min(n_starts, N), replace=False)

    for start in starts:
        # BFS distances
        from collections import deque
        dist = {start: 0}
        queue = deque([start])
        while queue:
            v = queue.popleft()
            if dist[v] >= max_dist: continue
            for u in adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    queue.append(u)

        # Initial state
        psi_0 = np.zeros(N, dtype=np.complex128)
        psi_0[start] = 1.0

        for t_idx, t in enumerate(times):
            psi_t = expm_multiply(-1j * t * H_sparse, psi_0)

            for node, d in dist.items():
                if d <= max_dist:
                    G_rt[d, t_idx] += np.abs(psi_t[node])**2
                    if t_idx == 0:
                        counts[d] += 1

    # Normalize
    for d in range(max_dist + 1):
        if counts[d] > 0:
            G_rt[d, :] /= counts[d]

    return G_rt, times, max_dist

# ================================================================
# CONTROL: KNOWN LATTICES
# ================================================================
def build_lattice(dim, L):
    """Build a d-dimensional cubic lattice of side L."""
    N = L**dim
    adj = {i: set() for i in range(N)}

    for idx in range(N):
        coords = []
        temp = idx
        for d in range(dim):
            coords.append(temp % L)
            temp //= L

        for d in range(dim):
            # Positive neighbor
            new_coords = coords.copy()
            new_coords[d] = (coords[d] + 1) % L
            neighbor = sum(c * L**d for d, c in enumerate(new_coords))
            adj[idx].add(neighbor)
            adj[neighbor].add(idx)

    return adj, N

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_quantum_walk():
    print("=" * 70)
    print("  MONOSTRING QUANTUM WALK LABORATORY")
    print("  Spectral dimension + Dispersion relation + Propagator")
    print("=" * 70)

    # ============================================================
    # PART 1: CONTROL — Known lattices
    # ============================================================
    print("\n[1/4] CONTROL: Spectral dimension of known lattices")
    print("      (should give d_s = lattice dimension)")
    print("      {:>10s} {:>5s} | {:>8s} {:>8s}".format(
        "Lattice", "N", "d_s(glob)", "d_s(mid)"))
    print("      " + "-" * 40)

    control_results = {}

    for dim, L in [(2, 20), (3, 10), (4, 6)]:
        t0 = time.time()
        adj, N = build_lattice(dim, L)
        result = spectral_dimension(adj, N, t_range=(0.1, 50),
                                     n_times=40, n_starts=20)

        # Mid-range spectral dimension
        ds_local = result['d_spectral_local']
        mid = len(ds_local) // 2
        d_mid = np.mean(ds_local[max(0,mid-3):mid+3])

        control_results[dim] = result

        print("      {:>2d}D (L={:>2d}) {:>5d} | {:>8.2f} {:>8.2f}  ({:.1f}s)".format(
            dim, L, N, result['d_global'], d_mid, time.time()-t0))

    # ============================================================
    # PART 2: MONOSTRING GRAPHS — Spectral dimension
    # ============================================================
    print("\n[2/4] MONOSTRING: Spectral dimension vs temperature")

    N_mono = 5000
    kappa = 0.5
    TARGET_DEG = 25

    temperatures = [3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

    print("      {:>6s} | {:>8s} {:>8s} {:>8s} | {:>5s} {:>5s} {:>5s}".format(
        "T", "d_s(glob)", "d_s(mid)", "d_s(short)", "r_K", "deg", ""))
    print("      " + "-" * 60)

    mono_results = []

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N_mono, kappa, T)
        adj, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_per = kuramoto(phases)
        r_k = np.prod(r_per)**(1./6)

        result = spectral_dimension(adj, N_mono, t_range=(0.1, 30),
                                     n_times=35, n_starts=20)

        ds_local = result['d_spectral_local']
        # Short scale (early times)
        d_short = np.mean(ds_local[:5]) if len(ds_local) >= 5 else 0
        # Mid scale
        mid = len(ds_local) // 2
        d_mid = np.mean(ds_local[max(0,mid-3):mid+3])

        mono_results.append({
            'T': T, 'd_global': result['d_global'],
            'd_mid': d_mid, 'd_short': d_short,
            'r_kura': r_k, 'deg': deg,
            'result': result
        })

        print("      {:>6.3f} | {:>8.2f} {:>8.2f} {:>8.2f} | {:>5.3f} {:>5.1f}  ({:.1f}s)".format(
            T, result['d_global'], d_mid, d_short, r_k, deg, time.time()-t0))

    # ============================================================
    # PART 3: DISPERSION RELATION
    # ============================================================
    print("\n[3/4] DISPERSION RELATION: E(k) for hot and cold")

    for label, T in [("HOT (T=3.0)", 3.0), ("COLD (T=0.02)", 0.02)]:
        t0 = time.time()
        phases = gen_traj(N_mono, kappa, T)
        adj, deg = build_graph(phases, target_deg=TARGET_DEG)

        evals, evecs = dispersion_relation(adj, N_mono, k_max=50)

        # Analyze low-energy spectrum
        nonzero = evals[evals > 1e-5]
        if len(nonzero) > 0:
            gap = nonzero[0]
            # Check linearity: E ~ k or E ~ k²?
            k_indices = np.arange(1, min(30, len(nonzero) + 1))
            E_vals = np.sqrt(nonzero[:len(k_indices)])

            # Fit E = a*k^alpha
            log_k = np.log(k_indices)
            log_E = np.log(E_vals)
            alpha, _ = np.polyfit(log_k, log_E, 1)

            print("\n      {}:".format(label))
            print("        Spectral gap: {:.6f}".format(gap))
            print("        E ~ k^{:.3f}".format(alpha))
            print("        alpha=1: relativistic, alpha=0.5: non-relativistic")
            print("        First 10 eigenvalues: {}".format(
                " ".join("{:.4f}".format(e) for e in evals[1:11])))
            print("        ({:.1f}s)".format(time.time()-t0))

    # ============================================================
    # PART 4: NULL MODEL COMPARISON
    # ============================================================
    print("\n[4/4] NULL MODEL: Random graph vs Monostring")

    # Build null model
    phases_null = np.random.uniform(0, 2*np.pi, (N_mono, 6))
    adj_null, deg_null = build_graph(phases_null, target_deg=TARGET_DEG)

    # Spectral dimension
    result_null = spectral_dimension(adj_null, N_mono, t_range=(0.1, 30),
                                      n_times=35, n_starts=20)

    ds_null_local = result_null['d_spectral_local']
    mid = len(ds_null_local) // 2
    d_null_mid = np.mean(ds_null_local[max(0,mid-3):mid+3])

    # Compare with cold Monostring
    d_mono_mid = mono_results[-1]['d_mid']
    d_mono_glob = mono_results[-1]['d_global']

    print("      Monostring (T=0.02): d_s = {:.2f} (global), {:.2f} (mid)".format(
        d_mono_glob, d_mono_mid))
    print("      Null (random T^6):   d_s = {:.2f} (global), {:.2f} (mid)".format(
        result_null['d_global'], d_null_mid))
    print("      Difference: {:.2f}".format(d_mono_mid - d_null_mid))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Control: return probability P(t)
    ax = axes[0, 0]
    for dim in [2, 3, 4]:
        r = control_results[dim]
        ax.loglog(r['times'], r['return_prob'], 'o-', markersize=3,
                  label='{}D lattice'.format(dim))
    ax.set_xlabel('Time t')
    ax.set_ylabel('Return probability P(t)')
    ax.set_title('Control: P(t) ~ t^{-d/2}')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Control: local spectral dimension
    ax = axes[0, 1]
    for dim in [2, 3, 4]:
        r = control_results[dim]
        ax.semilogx(r['t_mid'], r['d_spectral_local'], 'o-', markersize=3,
                    label='{}D (expect d_s={})'.format(dim, dim))
        ax.axhline(dim, ls=':', alpha=0.3)
    ax.set_xlabel('Time scale t')
    ax.set_ylabel('Local spectral dimension d_s(t)')
    ax.set_title('Control: d_s should converge to true D')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    # 3. Monostring: d_s vs T
    ax = axes[0, 2]
    Ts = [r['T'] for r in mono_results]
    ds_glob = [r['d_global'] for r in mono_results]
    ds_mid = [r['d_mid'] for r in mono_results]
    ds_short = [r['d_short'] for r in mono_results]
    ax.plot(Ts, ds_glob, 'o-', color='navy', lw=2, label='d_s (global)')
    ax.plot(Ts, ds_mid, 's-', color='crimson', lw=2, label='d_s (mid)')
    ax.plot(Ts, ds_short, '^-', color='green', lw=2, label='d_s (short)')
    ax.axhline(4, ls='--', color='black', alpha=0.5, label='d_s=4')
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel('Temperature T (cooling →)')
    ax.set_ylabel('Spectral dimension d_s')
    ax.set_title('Monostring: d_s vs Temperature')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Monostring: P(t) at different T
    ax = axes[1, 0]
    for i in [0, len(mono_results)//2, -1]:
        r = mono_results[i]['result']
        ax.loglog(r['times'], r['return_prob'], 'o-', markersize=3,
                  label='T={}'.format(mono_results[i]['T']))
    # Add null
    ax.loglog(result_null['times'], result_null['return_prob'],
              's--', color='gray', markersize=3, label='Null (random)')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Return probability P(t)')
    ax.set_title('Monostring: P(t) at different T')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. d_s(t) for cold Monostring vs null
    ax = axes[1, 1]
    r_cold = mono_results[-1]['result']
    ax.semilogx(r_cold['t_mid'], r_cold['d_spectral_local'], 'o-',
                color='navy', markersize=4, label='Monostring (T=0.02)')
    ax.semilogx(result_null['t_mid'], result_null['d_spectral_local'],
                's--', color='gray', markersize=4, label='Null (random)')
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='d_s=4')
    ax.axhline(2, ls=':', color='orange', alpha=0.5, label='d_s=2')
    ax.set_xlabel('Time scale t')
    ax.set_ylabel('Local d_s(t)')
    ax.set_title('Spectral dimension: Monostring vs Null')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 8)

    # 6. d_s vs Kuramoto order parameter
    ax = axes[1, 2]
    r_kuras = [r['r_kura'] for r in mono_results]
    ax.scatter(r_kuras, ds_mid, c=Ts, cmap='coolwarm', s=80, edgecolors='black')
    cb = plt.colorbar(ax.scatter(r_kuras, ds_mid, c=Ts, cmap='coolwarm', s=0), ax=ax)
    cb.set_label('Temperature')
    ax.set_xlabel('Kuramoto order parameter r')
    ax.set_ylabel('Spectral dimension d_s (mid)')
    ax.set_title('d_s vs Phase synchronization')
    ax.axhline(4, ls='--', color='red', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring Quantum Walk Laboratory\n'
                 'Spectral dimension from return probability',
                 fontsize=14, fontweight='bold')
    plt.savefig('quantum_walk_results.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: quantum_walk_results.png")
    plt.close()

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  QUANTUM WALK VERDICT")
    print("=" * 70)

    d_cold = mono_results[-1]['d_mid']
    d_hot = mono_results[0]['d_mid']
    d_null = d_null_mid

    tests = [
        ("d_s changes with T (phase transition effect)",
         abs(d_cold - d_hot) > 0.3,
         "d_s(hot)={:.2f}, d_s(cold)={:.2f}".format(d_hot, d_cold)),
        ("d_s(Monostring) differs from d_s(null)",
         abs(d_cold - d_null) > 0.3,
         "mono={:.2f}, null={:.2f}".format(d_cold, d_null)),
        ("d_s ≈ 4 at some temperature",
         any(abs(d - 4.0) < 0.5 for d in ds_mid),
         "closest: {:.2f}".format(min(ds_mid, key=lambda d: abs(d-4)))),
        ("d_s correlates with Kuramoto r",
         abs(np.corrcoef(r_kuras, ds_mid)[0,1]) > 0.5 if len(r_kuras) > 2 else False,
         "corr={:.3f}".format(np.corrcoef(r_kuras, ds_mid)[0,1]) if len(r_kuras) > 2 else "N/A"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print("  {} {}  {:50s} {}".format(
            "[+]" if passed else "[-]",
            "PASS" if passed else "FAIL",
            name, detail))

    print("\n  Score: {}/{}".format(n_pass, len(tests)))

    if d_cold > 3.5 and d_cold < 4.5 and abs(d_cold - d_null) > 0.3:
        print("\n  *** SPECTRAL DIMENSION d_s ≈ 4 FROM QUANTUM WALK ***")
        print("  AND it differs from the null model!")
    elif any(abs(d - 4.0) < 0.5 for d in ds_mid):
        print("\n  d_s ≈ 4 at some T, but need to check null model")
    else:
        print("\n  d_s ≠ 4. Spectral dimension of Monostring graph ≠ 4.")

    print("=" * 70)

if __name__ == "__main__":
    run_quantum_walk()
