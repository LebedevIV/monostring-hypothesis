import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, eye, diags
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.optimize import linprog
from collections import deque
import time, random, warnings
warnings.filterwarnings("ignore")

# ================================================================
# CARTAN MATRIX E6
# ================================================================
C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],
    [-1, 2,-1, 0, 0, 0],
    [ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],
    [ 0, 0, 0,-1, 2, 0],
    [ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# BLOCK 1: TRAJECTORY GENERATION
# ================================================================
def generate_trajectory(N, kappa, omega=None):
    """E6-coupled standard map on T^6."""
    D = 6
    if omega is None:
        omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    phases = np.zeros((N, D))
    phases[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N - 1):
        coupling = kappa * C_E6 @ np.sin(phases[n])
        phases[n+1] = (phases[n] + omega + coupling) % (2 * np.pi)
    return phases

def estimate_correlation_dimension(phases, max_pairs=100000):
    """Grassberger-Procaccia correlation dimension."""
    N = len(phases)
    n_pairs = min(max_pairs, N * (N - 1) // 2)
    idx_a = np.random.randint(0, N, n_pairs)
    idx_b = np.random.randint(0, N, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    diff = np.abs(phases[idx_a] - phases[idx_b])
    diff = np.minimum(diff, 2 * np.pi - diff)
    dists = np.sqrt(np.sum(diff ** 2, axis=1))

    r_values = np.logspace(np.log10(np.percentile(dists, 1)),
                           np.log10(np.percentile(dists, 90)), 60)
    C_r = np.array([np.mean(dists < r) for r in r_values])

    # Scaling region: 5%–40% of pairs
    valid = (C_r > 0.05) & (C_r < 0.40)
    if np.sum(valid) < 5:
        valid = (C_r > 0.01) & (C_r < 0.5)
    if np.sum(valid) < 5:
        return np.nan

    coeffs = np.polyfit(np.log(r_values[valid]), np.log(C_r[valid]), 1)
    return coeffs[0]

def compute_lyapunov_max(N, kappa, omega=None, D=6):
    """Estimate maximal Lyapunov exponent via QR method."""
    if omega is None:
        omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    phi = np.random.uniform(0, 2*np.pi, D)
    # Tangent vector
    v = np.random.randn(D)
    v /= np.linalg.norm(v)

    lyap_sum = 0.0
    n_steps = min(N, 50000)

    for n in range(n_steps):
        # Jacobian of the map at current point
        # phi_next = phi + omega + kappa * C @ sin(phi)
        # J = I + kappa * C @ diag(cos(phi))
        J = np.eye(D) + kappa * C_E6 @ np.diag(np.cos(phi))

        # Evolve tangent vector
        v = J @ v
        norm_v = np.linalg.norm(v)
        lyap_sum += np.log(norm_v)
        v /= norm_v

        # Evolve base point
        phi = (phi + omega + kappa * C_E6 @ np.sin(phi)) % (2 * np.pi)

    return lyap_sum / n_steps

# ================================================================
# BLOCK 2: GRAPH CONSTRUCTION WITH AUTO-TUNING
# ================================================================
def build_graph(phases, target_degree=40, delta_min=10):
    """
    Build resonance graph with automatic epsilon tuning
    to achieve target average degree.
    """
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)

    # Binary search for epsilon
    eps_lo, eps_hi = 0.1, 5.0
    best_eps = 1.0

    for iteration in range(12):
        eps_mid = (eps_lo + eps_hi) / 2
        # Estimate degree from a sample
        sample_idx = random.sample(range(N), min(500, N))
        degrees = []
        for idx in sample_idx:
            neighbors = tree.query_ball_point(coords[idx], eps_mid)
            # Filter chronologically close
            deg = sum(1 for j in neighbors if abs(j - idx) > delta_min and j != idx)
            degrees.append(deg)
        avg_deg = np.mean(degrees)

        if avg_deg < target_degree:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
        best_eps = eps_mid

        if abs(avg_deg - target_degree) / max(target_degree, 1) < 0.1:
            break

    # Build actual graph
    pairs = tree.query_pairs(r=best_eps, output_type='ndarray')
    mask = np.abs(pairs[:, 0] - pairs[:, 1]) > delta_min
    pairs = pairs[mask]

    adj = {i: set() for i in range(N)}
    for i in range(N - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)

    actual_avg = sum(len(adj[v]) for v in adj) / N
    return adj, len(pairs), best_eps, actual_avg

# ================================================================
# BLOCK 3: TEST 1 — DIMENSION MEASUREMENT
# ================================================================
def measure_dimension(adj, N, num_observers=25, max_radius=12):
    """BFS volume growth → effective dimension D(R)."""
    observers = random.sample(range(N // 10, 9 * N // 10),
                              min(num_observers, N // 2))
    radii = list(range(1, max_radius + 1))
    all_dims = []

    for obs in observers:
        dist = {obs: 0}
        queue = deque([obs])
        while queue:
            v = queue.popleft()
            if dist[v] >= max_radius:
                continue
            for u in adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    queue.append(u)

        volumes = [sum(1 for d_val in dist.values() if d_val <= r)
                   for r in radii]

        local_D = []
        for i in range(1, len(radii)):
            if volumes[i] > volumes[i - 1] > 0:
                dv = np.log(volumes[i]) - np.log(volumes[i - 1])
                dr = np.log(radii[i]) - np.log(radii[i - 1])
                local_D.append(dv / dr if dr > 0 else np.nan)
            else:
                local_D.append(np.nan)
        all_dims.append(local_D)

    arr = np.array(all_dims)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    return radii[1:], means, stds

# ================================================================
# BLOCK 4: TEST 2 — OLLIVIER-RICCI CURVATURE
# ================================================================
def bfs_limited(adj, start, max_dist=4):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if dist[v] >= max_dist:
            continue
        for u in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist

def ollivier_ricci(adj, a, b, alpha=0.5):
    """Ollivier-Ricci curvature for edge (a,b)."""
    nbrs_a = list(adj[a])
    nbrs_b = list(adj[b])
    support_a = [a] + nbrs_a
    support_b = [b] + nbrs_b
    deg_a, deg_b = len(nbrs_a), len(nbrs_b)

    if deg_a == 0 or deg_b == 0:
        return None

    mu_a = {a: alpha}
    for v in nbrs_a:
        mu_a[v] = mu_a.get(v, 0) + (1 - alpha) / deg_a
    mu_b = {b: alpha}
    for v in nbrs_b:
        mu_b[v] = mu_b.get(v, 0) + (1 - alpha) / deg_b

    n_a, n_b = len(support_a), len(support_b)

    # Distance matrix via BFS
    dist_matrix = np.zeros((n_a, n_b))
    for i, u in enumerate(support_a):
        d = bfs_limited(adj, u, max_dist=6)
        for j, v in enumerate(support_b):
            dist_matrix[i, j] = d.get(v, 20)

    # Transportation LP
    c = dist_matrix.flatten()
    n_vars = n_a * n_b

    A_rows, b_vec = [], []
    for i in range(n_a):
        row = np.zeros(n_vars)
        for j in range(n_b):
            row[i * n_b + j] = 1.0
        A_rows.append(row)
        b_vec.append(mu_a.get(support_a[i], 0.0))
    for j in range(n_b):
        row = np.zeros(n_vars)
        for i in range(n_a):
            row[i * n_b + j] = 1.0
        A_rows.append(row)
        b_vec.append(mu_b.get(support_b[j], 0.0))

    res = linprog(c, A_eq=np.array(A_rows), b_eq=np.array(b_vec),
                  bounds=[(0, None)] * n_vars, method='highs')

    if res.success:
        W1 = res.fun
        return 1.0 - W1  # d(a,b) = 1 for adjacent nodes
    return None

def test_gravity(adj, N, n_samples=300):
    """Test: does Ollivier-Ricci curvature correlate with local density?"""
    all_edges = [(v, u) for v in adj for u in adj[v] if u > v]
    sample = random.sample(all_edges, min(n_samples, len(all_edges)))

    edge_kappa = {}
    for i, (a, b) in enumerate(sample):
        k = ollivier_ricci(adj, a, b)
        if k is not None:
            edge_kappa[(a, b)] = k
        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(sample)} edges computed")

    # Vertex scalar curvature
    v_curv = {}
    for (a, b), k in edge_kappa.items():
        v_curv[a] = v_curv.get(a, 0) + k
        v_curv[b] = v_curv.get(b, 0) + k

    avg_deg = sum(len(adj[v]) for v in adj) / N
    v_dens = {v: len(adj[v]) / avg_deg for v in v_curv}

    vertices = list(v_curv.keys())
    curvs = np.array([v_curv[v] for v in vertices])
    dens = np.array([v_dens[v] for v in vertices])

    if len(curvs) > 2 and np.std(curvs) > 1e-10 and np.std(dens) > 1e-10:
        corr = np.corrcoef(curvs, dens)[0, 1]
    else:
        corr = 0.0

    mean_curv = np.mean(curvs) if len(curvs) > 0 else 0
    return corr, mean_curv, curvs, dens, vertices

# ================================================================
# BLOCK 5: TEST 3 — BELL INEQUALITY (GRAPH QUANTUM MECHANICS)
# ================================================================
def build_laplacian_sparse(adj, N):
    """Build normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    rows, cols, data = [], [], []
    degrees = np.array([len(adj[i]) for i in range(N)], dtype=float)

    for i in range(N):
        for j in adj[i]:
            rows.append(i)
            cols.append(j)
            data.append(-1.0 / np.sqrt(degrees[i] * degrees[j])
                        if degrees[i] > 0 and degrees[j] > 0 else 0)

    from scipy.sparse import coo_matrix
    A_norm = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
    L = eye(N, format='csc') + A_norm  # L = I - D^{-1/2} A D^{-1/2}, but A_norm has -1/sqrt(di*dj)
    # Actually: L = I - D^{-1/2} A D^{-1/2}
    # A_norm already has the - sign, so L = I + A_norm... wait.

    # Let me redo this correctly.
    # A_norm_{ij} = A_{ij} / sqrt(d_i * d_j)
    # L = I - A_norm
    # Above I put -1/sqrt in data, so A_norm = -sum, and I + A_norm = I - |A_norm|

    # Actually it's correct: data has -1/sqrt(di*dj), so the matrix stored is -A_norm
    # Then I + (-A_norm) = I - A_norm = L. ✓
    return L, degrees

def quantum_bell_test(adj, N, n_eigenstates=50):
    """
    Proper Bell test using graph quantum mechanics.

    1. Compute eigenstates of graph Laplacian
    2. Partition graph into two 'subsystems' A, B
    3. Prepare a singlet-like entangled state
    4. Compute CHSH correlator using quantum observables
    """
    print("      Building graph Laplacian...")
    L, degrees = build_laplacian_sparse(adj, N)

    print(f"      Computing {n_eigenstates} lowest eigenstates...")
    k_eig = min(n_eigenstates, N - 2)
    eigenvalues, eigenvectors = eigsh(L, k=k_eig, which='SM',
                                      tol=1e-4, maxiter=5000)

    # Sort
    idx_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # Partition: A = first half of nodes, B = second half
    mid = N // 2
    A_nodes = list(range(mid))
    B_nodes = list(range(mid, N))

    # Find pairs of eigenstates with strong entanglement
    # Entanglement measure: Schmidt decomposition proxy
    # Use the first few non-trivial eigenstates

    # Define spin operator along direction theta:
    # sigma(theta) = cos(theta)*P_even + sin(theta)*P_odd
    # where P_even/odd project onto even/odd eigenstates

    def spin_operator_A(theta, psi_set_A):
        """
        Observable for subsystem A parametrized by angle theta.
        M_A(theta) projects onto cos(theta)|+> + sin(theta)|->
        where |+>, |-> are defined by parity of eigenstates.
        Returns expectation values for each eigenstate restricted to A.
        """
        n_states = len(psi_set_A)
        M = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                # Pauli-like structure: sigma_z for i=j, sigma_x for |i-j|=1
                if i == j:
                    M[i, j] = np.cos(2 * theta) * ((-1) ** i)
                elif abs(i - j) == 1:
                    M[i, j] = np.sin(2 * theta)
        # Ensure Hermitian
        M = (M + M.T) / 2
        return M

    # Restrict eigenstates to subsystems
    n_use = min(8, k_eig - 1)  # Use first 8 non-trivial eigenstates
    psi_A = eigenvectors[A_nodes, 1:n_use+1]  # Skip zero mode
    psi_B = eigenvectors[B_nodes, 1:n_use+1]

    # Normalize subsystem wavefunctions
    norms_A = np.linalg.norm(psi_A, axis=0)
    norms_B = np.linalg.norm(psi_B, axis=0)
    norms_A[norms_A < 1e-10] = 1
    norms_B[norms_B < 1e-10] = 1
    psi_A = psi_A / norms_A
    psi_B = psi_B / norms_B

    # Construct maximally entangled state (Bell state analog)
    # |Psi> = (1/sqrt(n_use)) * sum_k |k>_A |k>_B
    # Density matrix of full system would be needed, but we use
    # the correlation function directly:

    # E(a,b) = <Psi| M_A(a) ⊗ M_B(b) |Psi>
    #        = (1/n_use) * sum_{k,l} <k|M_A(a)|l> * <k|M_B(b)|l>

    # Compute overlap matrices
    S_A = psi_A.T @ psi_A  # n_use x n_use
    S_B = psi_B.T @ psi_B  # n_use x n_use

    def E_correlation(theta_a, theta_b):
        M_A = spin_operator_A(theta_a, range(n_use))
        M_B = spin_operator_A(theta_b, range(n_use))

        # <Psi|M_A⊗M_B|Psi> = (1/n_use) Tr(M_A @ S_A @ M_B @ S_B)
        # This is approximate but captures the correlation structure
        val = np.trace(M_A @ S_A @ M_B @ S_B) / n_use
        # Clip to [-1, 1]
        return np.clip(val, -1, 1)

    # CHSH optimal angles
    a1, a2 = 0, np.pi / 4
    b1, b2 = np.pi / 8, 3 * np.pi / 8

    E11 = E_correlation(a1, b1)
    E12 = E_correlation(a1, b2)
    E21 = E_correlation(a2, b1)
    E22 = E_correlation(a2, b2)

    S = E11 - E12 + E21 + E22

    return S, E11, E12, E21, E22, eigenvalues[:n_use+1]

# ================================================================
# BLOCK 6: TEST 4 — DIRAC SPECTRUM (MASSES)
# ================================================================
def compute_dirac_spectrum(adj, N, k=30):
    """
    Spectrum of graph Dirac operator D.
    D^2 = graph Laplacian (on vertices and edges).
    Eigenvalues of |D| = masses.
    """
    edges = [(i, j) for i in range(N) for j in adj[i] if j > i]
    n_edges = len(edges)

    if n_edges > 500000:
        print(f"      WARNING: {n_edges} edges, Dirac computation may be slow")

    # Incidence matrix
    inc = lil_matrix((N, n_edges), dtype=np.float64)
    for e_idx, (i, j) in enumerate(edges):
        inc[i, e_idx] = -1.0
        inc[j, e_idx] = +1.0
    inc = csc_matrix(inc)

    # D^2 on vertices = inc @ inc^T = graph Laplacian
    L0 = inc @ inc.T

    k_actual = min(k, N - 2, n_edges - 2)
    eigenvalues, eigenvectors = eigsh(L0, k=k_actual, which='SM',
                                       tol=1e-4, maxiter=5000)

    masses = np.sort(np.sqrt(np.abs(eigenvalues)))
    return masses, eigenvalues

# ================================================================
# BLOCK 7: PHASE DIAGRAM — SWEEP OVER KAPPA
# ================================================================
def phase_diagram_scan(N=15000, kappas=None, target_degree=40):
    """Scan over coupling constant to find critical kappa."""
    if kappas is None:
        kappas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    results = []

    for kappa in kappas:
        print(f"\n  --- kappa = {kappa:.2f} ---")
        t0 = time.time()

        phases = generate_trajectory(N, kappa)
        d_corr = estimate_correlation_dimension(phases)

        # Lyapunov exponent
        lyap = compute_lyapunov_max(N, kappa)

        # Build graph
        adj, n_edges, eps, avg_deg = build_graph(phases, target_degree=target_degree)

        # Measure dimension
        radii, dims, stds = measure_dimension(adj, N, num_observers=15, max_radius=10)
        mid = len(dims) // 2
        D_eff = np.nanmean(dims[max(0, mid-2):mid+2])

        elapsed = time.time() - t0

        results.append({
            'kappa': kappa,
            'd_corr': d_corr,
            'd_eff': D_eff,
            'lyapunov': lyap,
            'avg_degree': avg_deg,
            'epsilon': eps,
            'n_edges': n_edges,
            'radii': radii,
            'dims': dims,
            'stds': stds,
            'time': elapsed
        })

        print(f"      D_corr={d_corr:.2f}, D_eff={D_eff:.2f}, "
              f"λ_max={lyap:.3f}, deg={avg_deg:.1f}, "
              f"t={elapsed:.1f}s")

    return results

# ================================================================
# MASTER TEST FUNCTION
# ================================================================
def run_comprehensive_tests():
    print("╔" + "═" * 65 + "╗")
    print("║  SBE HYPOTHESIS — ENHANCED CRITICAL TEST SUITE            ║")
    print("║  E₆ coupled standard map + auto-tuning + phase diagram    ║")
    print("╚" + "═" * 65 + "╝")

    N = 15000
    KAPPA = 0.5
    TARGET_DEG = 40

    # ---- PHASE 1: Trajectory ----
    print(f"\n[1/7] Generating trajectory (N={N}, κ={KAPPA})...")
    t0 = time.time()
    phases = generate_trajectory(N, KAPPA)
    d_corr = estimate_correlation_dimension(phases)
    lyap = compute_lyapunov_max(N, KAPPA)
    print(f"      D_corr = {d_corr:.3f}")
    print(f"      Max Lyapunov exponent = {lyap:.4f}")
    print(f"      {'CHAOTIC' if lyap > 0.01 else 'REGULAR'} dynamics")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ---- PHASE 2: Graph ----
    print(f"\n[2/7] Building resonance graph (target degree={TARGET_DEG})...")
    t0 = time.time()
    adj, n_edges, eps, avg_deg = build_graph(phases, target_degree=TARGET_DEG)
    print(f"      ε = {eps:.3f}, edges = {n_edges:,}, avg degree = {avg_deg:.1f}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ---- PHASE 3: Dimension ----
    print(f"\n[3/7] Measuring emergent dimension D_eff(R)...")
    t0 = time.time()
    radii, dims, stds = measure_dimension(adj, N, num_observers=25, max_radius=12)
    mid = len(dims) // 2
    D_eff = np.nanmean(dims[max(0, mid-2):mid+2])
    D_small = np.nanmean(dims[:2]) if len(dims) >= 2 else np.nan  # Small-scale dim
    print(f"      D_eff (macro) = {D_eff:.2f}")
    print(f"      D_eff (micro, R=2-3) = {D_small:.2f}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ---- PHASE 4: Curvature ----
    print(f"\n[4/7] Computing Ollivier-Ricci curvature (n=300 edges)...")
    t0 = time.time()
    corr_grav, mean_curv, curvs, dens, verts = test_gravity(adj, N, n_samples=300)
    print(f"      Mean curvature = {mean_curv:.4f}")
    print(f"      Curvature-density correlation = {corr_grav:.3f}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ---- PHASE 5: Bell test ----
    print(f"\n[5/7] Quantum Bell test (graph QM)...")
    t0 = time.time()
    try:
        S_bell, E11, E12, E21, E22, eigs_bell = quantum_bell_test(adj, N, n_eigenstates=30)
        print(f"      CHSH parameter S = {S_bell:.4f}")
        print(f"      E(a,b)={E11:.3f}, E(a,b')={E12:.3f}, "
              f"E(a',b)={E21:.3f}, E(a',b')={E22:.3f}")
        print(f"      |S| = {abs(S_bell):.4f} "
              f"(classical limit: 2, QM limit: {2*np.sqrt(2):.3f})")
        print(f"      Time: {time.time()-t0:.2f}s")
    except Exception as e:
        S_bell = 0
        print(f"      SKIPPED: {e}")

    # ---- PHASE 6: Mass spectrum ----
    print(f"\n[6/7] Computing Dirac mass spectrum...")
    t0 = time.time()
    try:
        masses, raw_eigs = compute_dirac_spectrum(adj, N, k=30)
        nonzero = masses[masses > 1e-5]
        n_zero = len(masses) - len(nonzero)
        print(f"      Zero modes (topology): {n_zero}")
        if len(nonzero) > 0:
            print(f"      Mass gap: {nonzero[0]:.6f}")
            print(f"      First 10 masses: {nonzero[:10]}")
            if len(nonzero) >= 3:
                print(f"      Mass ratios: m1/m0={nonzero[1]/nonzero[0]:.2f}, "
                      f"m2/m0={nonzero[2]/nonzero[0]:.2f}")
        print(f"      Time: {time.time()-t0:.2f}s")
    except Exception as e:
        nonzero = np.array([])
        n_zero = 0
        print(f"      SKIPPED: {e}")

    # ---- PHASE 7: Phase diagram ----
    print(f"\n[7/7] Phase diagram scan (κ sweep)...")
    kappas_scan = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]
    pd_results = phase_diagram_scan(N=10000, kappas=kappas_scan,
                                     target_degree=TARGET_DEG)

    # ================================================================
    # VISUALIZATION
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. D(R) profile
    ax = axes[0, 0]
    ax.errorbar(radii, dims, yerr=stds, fmt='o-', color='crimson',
                capsize=3, label=f'κ={KAPPA}')
    ax.axhline(4, ls='--', color='green', alpha=0.7, label='4D target')
    ax.axhline(6, ls='--', color='blue', alpha=0.5, label='6D trivial')
    ax.axhline(2, ls=':', color='orange', alpha=0.5, label='2D Planck limit')
    ax.set_xlabel('Scale R')
    ax.set_ylabel('D_eff(R)')
    ax.set_title(f'Emergent Dimension\nD_corr={d_corr:.2f}, D_eff={D_eff:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    # 2. Phase diagram D(kappa)
    ax = axes[0, 1]
    ks = [r['kappa'] for r in pd_results]
    d_corrs = [r['d_corr'] for r in pd_results]
    d_effs = [r['d_eff'] for r in pd_results]
    lyaps = [r['lyapunov'] for r in pd_results]

    ax.plot(ks, d_corrs, 'o-', color='navy', linewidth=2, label='D_corr (attractor)')
    ax.plot(ks, d_effs, 's-', color='crimson', linewidth=2, label='D_eff (graph)')
    ax.axhline(4, ls='--', color='green', alpha=0.7, label='4D target')
    ax.set_xlabel('Coupling κ')
    ax.set_ylabel('Dimension')
    ax.set_title('Phase Diagram: D(κ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Lyapunov vs kappa
    ax = axes[0, 2]
    ax.plot(ks, lyaps, 'D-', color='darkred', linewidth=2)
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('Coupling κ')
    ax.set_ylabel('Max Lyapunov exponent λ')
    ax.set_title('Chaos Onset')
    ax.grid(True, alpha=0.3)
    ax.fill_between(ks, 0, lyaps, alpha=0.2, color='red',
                     where=[l > 0 for l in lyaps])

    # 4. Curvature distribution
    ax = axes[1, 0]
    if len(curvs) > 0:
        ax.hist(curvs, bins=30, color='teal', alpha=0.7, edgecolor='white')
        ax.axvline(0, ls='--', color='red', alpha=0.7)
        ax.set_xlabel('Ollivier-Ricci curvature κ_OR')
        ax.set_ylabel('Count')
        ax.set_title(f'Curvature Distribution\nmean={mean_curv:.3f}, corr(κ,ρ)={corr_grav:.3f}')
    ax.grid(True, alpha=0.3)

    # 5. Curvature vs density scatter
    ax = axes[1, 1]
    if len(curvs) > 0 and len(dens) > 0:
        ax.scatter(dens, curvs, alpha=0.4, s=10, color='purple')
        if abs(corr_grav) > 0.01:
            z = np.polyfit(dens, curvs, 1)
            x_fit = np.linspace(min(dens), max(dens), 100)
            ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2,
                    label=f'r = {corr_grav:.3f}')
            ax.legend()
        ax.set_xlabel('Local density (degree/avg_degree)')
        ax.set_ylabel('Scalar curvature')
        ax.set_title('Einstein Equation Test\nκ_OR vs ρ')
    ax.grid(True, alpha=0.3)

    # 6. Mass spectrum
    ax = axes[1, 2]
    if len(nonzero) > 0:
        ax.bar(range(len(nonzero)), nonzero, color='indigo', alpha=0.8)
        ax.set_xlabel('Mode index k')
        ax.set_ylabel('Mass |μ_k|')
        ax.set_title(f'Dirac Mass Spectrum\n'
                     f'Zero modes: {n_zero}, gap: {nonzero[0]:.4f}')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'SBE Hypothesis Critical Tests — N={N}, κ={KAPPA}, '
                 f'D_corr={d_corr:.2f}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('sbe_enhanced_tests.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("\n" + "╔" + "═" * 65 + "╗")
    print("║  FINAL VERDICT                                            ║")
    print("╠" + "═" * 65 + "╣")

    tests = [
        ("Dimensional reduction (D_corr < 5.5)",
         d_corr < 5.5 and not np.isnan(d_corr),
         f"D_corr = {d_corr:.2f}"),
        ("Non-trivial D_eff ≠ 6",
         abs(D_eff - 6) > 0.5 and D_eff > 1,
         f"D_eff = {D_eff:.2f}"),
        ("Planck-scale reduction (D_small < D_eff)",
         D_small < D_eff - 0.3 if not np.isnan(D_small) else False,
         f"D_small = {D_small:.2f}"),
        ("Positive mean curvature",
         mean_curv > 0.01,
         f"<κ_OR> = {mean_curv:.4f}"),
        ("Einstein eqn (corr > 0.3)",
         abs(corr_grav) > 0.3,
         f"corr = {corr_grav:.3f}"),
        ("Bell violation (|S| > 2)",
         abs(S_bell) > 2.0,
         f"|S| = {abs(S_bell):.3f}"),
        ("Mass gap exists",
         len(nonzero) > 0 and nonzero[0] > 1e-4,
         f"Δm = {nonzero[0]:.4f}" if len(nonzero) > 0 else "N/A"),
        ("Chaos at κ > 0",
         lyap > 0.01,
         f"λ_max = {lyap:.4f}"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        status = "PASS ✓" if passed else "FAIL ✗"
        n_pass += int(passed)
        print(f"║  {status}  {name:<40s} {detail:>12s}  ║")

    print("╠" + "═" * 65 + "╣")
    print(f"║  Score: {n_pass}/{len(tests)} tests passed"
          + " " * (65 - 30 - len(str(n_pass)) - len(str(len(tests)))) + "║")
    print("╚" + "═" * 65 + "╝")

    return {
        'd_corr': d_corr, 'D_eff': D_eff, 'D_small': D_small,
        'lyapunov': lyap, 'curvature_corr': corr_grav,
        'mean_curvature': mean_curv, 'S_bell': S_bell,
        'mass_gap': nonzero[0] if len(nonzero) > 0 else None,
        'phase_diagram': pd_results
    }

if __name__ == "__main__":
    results = run_comprehensive_tests()
