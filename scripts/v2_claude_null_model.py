import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, eye, diags, coo_matrix, bmat
from scipy.sparse.linalg import eigsh
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
# TRAJECTORY + ATTRACTOR ANALYSIS
# ================================================================
def generate_trajectory(N, kappa, omega=None):
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
    N = len(phases)
    n_pairs = min(max_pairs, N*(N-1)//2)
    idx_a = np.random.randint(0, N, n_pairs)
    idx_b = np.random.randint(0, N, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    diff = np.abs(phases[idx_a] - phases[idx_b])
    diff = np.minimum(diff, 2*np.pi - diff)
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_values = np.logspace(np.log10(np.percentile(dists, 1)),
                           np.log10(np.percentile(dists, 90)), 60)
    C_r = np.array([np.mean(dists < r) for r in r_values])
    valid = (C_r > 0.05) & (C_r < 0.40)
    if np.sum(valid) < 5:
        valid = (C_r > 0.01) & (C_r < 0.5)
    if np.sum(valid) < 5:
        return np.nan
    coeffs = np.polyfit(np.log(r_values[valid]), np.log(C_r[valid]), 1)
    return coeffs[0]

def compute_lyapunov_max(N_steps, kappa, omega=None, D=6):
    if omega is None:
        omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    phi = np.random.uniform(0, 2*np.pi, D)
    v = np.random.randn(D)
    v /= np.linalg.norm(v)
    lyap_sum = 0.0
    n_steps = min(N_steps, 50000)
    for n in range(n_steps):
        J = np.eye(D) + kappa * C_E6 @ np.diag(np.cos(phi))
        v = J @ v
        norm_v = np.linalg.norm(v)
        lyap_sum += np.log(norm_v)
        v /= norm_v
        phi = (phi + omega + kappa * C_E6 @ np.sin(phi)) % (2 * np.pi)
    return lyap_sum / n_steps

# ================================================================
# GRAPH CONSTRUCTION WITH AUTO-TUNING
# ================================================================
def build_graph(phases, target_degree=40, delta_min=10):
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    eps_lo, eps_hi = 0.1, 5.0
    best_eps = 1.0
    for iteration in range(15):
        eps_mid = (eps_lo + eps_hi) / 2
        sample_idx = random.sample(range(N), min(500, N))
        degrees = []
        for idx in sample_idx:
            neighbors = tree.query_ball_point(coords[idx], eps_mid)
            deg = sum(1 for j in neighbors
                      if abs(j - idx) > delta_min and j != idx)
            degrees.append(deg)
        avg_deg = np.mean(degrees)
        if avg_deg < target_degree:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
        best_eps = eps_mid
        if abs(avg_deg - target_degree) / max(target_degree, 1) < 0.05:
            break

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
# TEST 1: DIMENSION — FIXED FOR FINITE-SIZE SATURATION
# ================================================================
def measure_dimension_fixed(adj, N, num_observers=25, max_radius=12):
    """
    BFS volume growth with finite-size correction.
    Only use radii where V(R) < 0.5 * N to avoid saturation.
    Also compute dimension at each scale separately.
    """
    observers = random.sample(range(N // 10, 9 * N // 10),
                              min(num_observers, N // 2))
    radii = list(range(1, max_radius + 1))
    all_volumes = []

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
        all_volumes.append(volumes)

    avg_volumes = np.mean(all_volumes, axis=0)

    # Find the saturation radius: V(R) > 0.3 * N
    saturation_R = max_radius
    for i, vol in enumerate(avg_volumes):
        if vol > 0.3 * N:
            saturation_R = radii[i]
            break

    # Compute local dimension only below saturation
    local_D = []
    valid_radii = []
    for i in range(1, len(radii)):
        if radii[i] > saturation_R:
            break
        if avg_volumes[i] > avg_volumes[i-1] > 0:
            dv = np.log(avg_volumes[i]) - np.log(avg_volumes[i-1])
            dr = np.log(radii[i]) - np.log(radii[i-1])
            local_D.append(dv / dr if dr > 0 else np.nan)
            valid_radii.append(radii[i])

    # Also compute via log-log regression (more robust)
    valid_mask = np.array(radii) <= saturation_R
    valid_r = np.array(radii)[valid_mask]
    valid_v = avg_volumes[valid_mask]

    if len(valid_r) >= 3 and all(v > 0 for v in valid_v):
        coeffs = np.polyfit(np.log(valid_r), np.log(valid_v), 1)
        D_regression = coeffs[0]
    else:
        D_regression = np.nan

    # Dimension at different scales
    D_micro = np.nanmean(local_D[:2]) if len(local_D) >= 2 else np.nan
    D_meso = np.nanmean(local_D[2:4]) if len(local_D) >= 4 else np.nan
    D_macro = np.nanmean(local_D[4:]) if len(local_D) >= 5 else np.nan

    return {
        'valid_radii': valid_radii,
        'local_D': local_D,
        'D_regression': D_regression,
        'D_micro': D_micro,
        'D_meso': D_meso,
        'D_macro': D_macro,
        'saturation_R': saturation_R,
        'avg_volumes': avg_volumes,
        'radii': radii
    }

# ================================================================
# TEST 2: OLLIVIER-RICCI CURVATURE
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
    nbrs_a = list(adj[a])
    nbrs_b = list(adj[b])
    if len(nbrs_a) == 0 or len(nbrs_b) == 0:
        return None
    support_a = [a] + nbrs_a
    support_b = [b] + nbrs_b
    deg_a, deg_b = len(nbrs_a), len(nbrs_b)

    mu_a = {a: alpha}
    for v in nbrs_a:
        mu_a[v] = mu_a.get(v, 0) + (1 - alpha) / deg_a
    mu_b = {b: alpha}
    for v in nbrs_b:
        mu_b[v] = mu_b.get(v, 0) + (1 - alpha) / deg_b

    n_a, n_b = len(support_a), len(support_b)
    dist_matrix = np.zeros((n_a, n_b))
    for i, u in enumerate(support_a):
        d = bfs_limited(adj, u, max_dist=6)
        for j, v in enumerate(support_b):
            dist_matrix[i, j] = d.get(v, 20)

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
        return 1.0 - res.fun
    return None

def test_gravity(adj, N, n_samples=300):
    all_edges = [(v, u) for v in adj for u in adj[v] if u > v]
    sample = random.sample(all_edges, min(n_samples, len(all_edges)))
    edge_kappa = {}
    for i, (a, b) in enumerate(sample):
        k = ollivier_ricci(adj, a, b)
        if k is not None:
            edge_kappa[(a, b)] = k
        if (i + 1) % 100 == 0:
            print(f"        {i+1}/{len(sample)} edges computed")

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
# TEST 3: BELL INEQUALITY — CORRECTED IMPLEMENTATION
# ================================================================
def build_adjacency_sparse(adj, N):
    """Build sparse adjacency matrix."""
    rows, cols = [], []
    for i in range(N):
        for j in adj[i]:
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
    return A

def proper_bell_test(adj, N, n_eig=60):
    """
    Correct CHSH test using graph quantum mechanics.

    Key fixes vs. previous version:
    1. Observables have eigenvalues ±1 (proper dichotomic)
    2. Entangled state from ground state of coupled system
    3. Correlator is proper quantum expectation value
    4. Result provably satisfies |S| ≤ 2√2
    """
    print("      Constructing Hamiltonian...")

    # Build graph Laplacian as Hamiltonian
    degrees = np.array([len(adj[i]) for i in range(N)], dtype=float)
    A_sparse = build_adjacency_sparse(adj, N)
    D_inv_sqrt = diags(1.0 / np.sqrt(np.maximum(degrees, 1)))
    L_norm = eye(N) - D_inv_sqrt @ A_sparse @ D_inv_sqrt

    # Get low-energy eigenstates
    k_eig = min(n_eig, N - 2)
    print(f"      Computing {k_eig} lowest eigenstates...")
    eigenvalues, eigenvectors = eigsh(L_norm, k=k_eig, which='SM',
                                       tol=1e-4, maxiter=5000)
    idx_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # ---- SUBSYSTEM PARTITION ----
    # Split graph into A and B by graph distance from center
    center = N // 2
    dists_from_center = bfs_limited(adj, center, max_dist=100)

    # A = nodes with even distance, B = nodes with odd distance
    # This gives a natural bipartition respecting graph structure
    A_nodes = [v for v, d in dists_from_center.items() if d % 2 == 0]
    B_nodes = [v for v, d in dists_from_center.items() if d % 2 == 1]

    if len(A_nodes) < 10 or len(B_nodes) < 10:
        # Fallback to simple split
        A_nodes = list(range(N // 2))
        B_nodes = list(range(N // 2, N))

    print(f"      Subsystems: |A|={len(A_nodes)}, |B|={len(B_nodes)}")

    # ---- REDUCED HILBERT SPACE ----
    # Work in the space spanned by the first n_use eigenstates
    # This is an n_use-dimensional Hilbert space
    n_use = min(16, k_eig - 1)  # Skip zero mode

    # Restrict eigenstates to subsystems
    psi_A = eigenvectors[np.array(A_nodes), 1:n_use+1]  # |A| x n_use
    psi_B = eigenvectors[np.array(B_nodes), 1:n_use+1]  # |B| x n_use

    # SVD of the joint state to get Schmidt decomposition
    # The state |Ψ⟩ restricted to modes 1..n_use,
    # projected onto subsystems A,B
    # Cross-correlation matrix: C_{kl} = Σ_i ψ_k(i_A) * ψ_l(i_B)
    # But this doesn't define an entangled state properly.

    # CORRECT APPROACH: Define a simple spin-1/2 model
    # Use the two lowest non-trivial eigenstates as |0⟩ and |1⟩
    # for each subsystem.

    # For subsystem A: take eigenstates restricted to A
    # and orthonormalize
    psi_A_0 = eigenvectors[np.array(A_nodes), 1]  # first excited
    psi_A_1 = eigenvectors[np.array(A_nodes), 2]  # second excited

    # Normalize
    norm_A0 = np.linalg.norm(psi_A_0)
    norm_A1 = np.linalg.norm(psi_A_1)
    if norm_A0 < 1e-10 or norm_A1 < 1e-10:
        return 0, 0, 0, 0, 0, eigenvalues[:n_use+1]
    psi_A_0 /= norm_A0
    psi_A_1 /= norm_A1

    # Gram-Schmidt on A
    overlap = np.dot(psi_A_0, psi_A_1)
    psi_A_1 = psi_A_1 - overlap * psi_A_0
    norm = np.linalg.norm(psi_A_1)
    if norm < 1e-10:
        return 0, 0, 0, 0, 0, eigenvalues[:n_use+1]
    psi_A_1 /= norm

    # Same for B
    psi_B_0 = eigenvectors[np.array(B_nodes), 1]
    psi_B_1 = eigenvectors[np.array(B_nodes), 2]
    norm_B0 = np.linalg.norm(psi_B_0)
    norm_B1 = np.linalg.norm(psi_B_1)
    if norm_B0 < 1e-10 or norm_B1 < 1e-10:
        return 0, 0, 0, 0, 0, eigenvalues[:n_use+1]
    psi_B_0 /= norm_B0
    psi_B_1 /= norm_B1
    overlap = np.dot(psi_B_0, psi_B_1)
    psi_B_1 = psi_B_1 - overlap * psi_B_0
    norm = np.linalg.norm(psi_B_1)
    if norm < 1e-10:
        return 0, 0, 0, 0, 0, eigenvalues[:n_use+1]
    psi_B_1 /= norm

    # ---- ENTANGLED STATE ----
    # Singlet state: |Ψ⟩ = (|01⟩ - |10⟩) / √2
    # In the 4D basis {|00⟩, |01⟩, |10⟩, |11⟩}:
    psi_singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)

    # But we need to check: does the graph Hamiltonian
    # actually produce this state?
    # Compute the overlap of the actual low-energy state
    # with the singlet

    # Cross-subsystem coupling from the graph
    # H_AB = Σ_{i∈A, j∈B, (i,j)∈E} |i⟩⟨j| + h.c.
    # In the {|0⟩,|1⟩}⊗{|0⟩,|1⟩} basis:
    H_AB = np.zeros((2, 2))  # H_AB[k_A, k_B]
    for i_idx, i_node in enumerate(A_nodes):
        for j_node in adj[i_node]:
            if j_node in set(B_nodes):
                j_idx = B_nodes.index(j_node) if j_node in B_nodes else -1
                if j_idx >= 0 and j_idx < len(B_nodes):
                    # Coupling element ⟨k_A| i⟩⟨j |l_B⟩
                    for k in range(2):
                        psi_Ak = [psi_A_0, psi_A_1][k]
                        for l in range(2):
                            psi_Bl = [psi_B_0, psi_B_1][l]
                            jj = B_nodes.index(j_node)
                            H_AB[k, l] += psi_Ak[i_idx] * psi_Bl[jj]

    # Full 4x4 Hamiltonian in the qubit subspace
    # H = H_A ⊗ I + I ⊗ H_B + H_int
    E_A = np.array([eigenvalues[1], eigenvalues[2]])
    E_B = np.array([eigenvalues[1], eigenvalues[2]])

    H_full = np.zeros((4, 4))
    # Diagonal: E_A[i] + E_B[j]
    for i in range(2):
        for j in range(2):
            H_full[2*i+j, 2*i+j] = E_A[i] + E_B[j]
    # Off-diagonal: coupling
    coupling_strength = np.max(np.abs(H_AB)) if np.max(np.abs(H_AB)) > 0 else 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i, j) != (k, l):
                        H_full[2*i+j, 2*k+l] += H_AB[i, l] * (1 if j == k else 0)
                        H_full[2*i+j, 2*k+l] += H_AB[k, j] * (1 if i == l else 0)

    # Diagonalize
    evals_full, evecs_full = np.linalg.eigh(H_full)
    ground_state = evecs_full[:, 0]

    print(f"      Ground state: [{ground_state[0]:.3f}, {ground_state[1]:.3f}, "
          f"{ground_state[2]:.3f}, {ground_state[3]:.3f}]")

    # Check entanglement via concurrence
    # For a 2-qubit pure state |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
    # Concurrence = 2|ad - bc|
    a, b, c, d = ground_state
    concurrence = 2 * abs(a*d - b*c)
    print(f"      Concurrence = {concurrence:.4f} "
          f"({'ENTANGLED' if concurrence > 0.01 else 'SEPARABLE'})")

    # ---- DICHOTOMIC OBSERVABLES ----
    # σ(θ) = cos(θ)σ_z + sin(θ)σ_x
    # This has eigenvalues ±1 BY CONSTRUCTION
    sigma_z = np.array([[1, 0], [0, -1]], dtype=float)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=float)

    def sigma_theta(theta):
        return np.cos(theta) * sigma_z + np.sin(theta) * sigma_x

    # ---- CHSH CORRELATOR ----
    # E(a,b) = ⟨ψ| σ_A(a) ⊗ σ_B(b) |ψ⟩
    def E_correlator(theta_a, theta_b):
        M_A = sigma_theta(theta_a)  # 2x2
        M_B = sigma_theta(theta_b)  # 2x2
        # Tensor product: 4x4
        M_AB = np.kron(M_A, M_B)
        return float(ground_state @ M_AB @ ground_state)

    # CHSH optimal angles
    a1, a2 = 0, np.pi / 4
    b1, b2 = np.pi / 8, 3 * np.pi / 8

    E11 = E_correlator(a1, b1)
    E12 = E_correlator(a1, b2)
    E21 = E_correlator(a2, b1)
    E22 = E_correlator(a2, b2)

    S = E11 - E12 + E21 + E22

    # Verify Tsirelson bound (sanity check)
    assert abs(S) <= 2*np.sqrt(2) + 0.01, \
        f"Tsirelson bound violated: |S|={abs(S):.4f} > 2√2={2*np.sqrt(2):.4f}"

    return S, E11, E12, E21, E22, concurrence

# ================================================================
# TEST 4: DIRAC SPECTRUM
# ================================================================
def compute_dirac_spectrum(adj, N, k=30):
    edges = [(i, j) for i in range(N) for j in adj[i] if j > i]
    n_edges = len(edges)
    inc = lil_matrix((N, n_edges), dtype=np.float64)
    for e_idx, (i, j) in enumerate(edges):
        inc[i, e_idx] = -1.0
        inc[j, e_idx] = +1.0
    inc = csc_matrix(inc)
    L0 = inc @ inc.T
    k_actual = min(k, N - 2)
    eigenvalues, _ = eigsh(L0, k=k_actual, which='SM',
                            tol=1e-4, maxiter=5000)
    masses = np.sort(np.sqrt(np.abs(eigenvalues)))
    return masses, eigenvalues

# ================================================================
# NULL MODEL: RANDOM GEOMETRIC GRAPH ON T^6 (CONTROL)
# ================================================================
def build_null_model(N, target_degree=40, D=6):
    """
    Control experiment: RGG on flat T^6 (no dynamics, no E6).
    Points uniformly distributed on T^6.
    """
    phases_random = np.random.uniform(0, 2*np.pi, (N, D))
    adj, n_edges, eps, avg_deg = build_graph(phases_random,
                                              target_degree=target_degree)
    return adj, phases_random, n_edges, avg_deg

# ================================================================
# PHASE DIAGRAM SCAN
# ================================================================
def phase_diagram_scan(N=10000, kappas=None, target_degree=40):
    if kappas is None:
        kappas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
                  1.0, 1.5, 2.0, 3.0]
    results = []
    for kappa in kappas:
        print(f"    κ={kappa:.2f}...", end=" ", flush=True)
        t0 = time.time()
        phases = generate_trajectory(N, kappa)
        d_corr = estimate_correlation_dimension(phases)
        lyap = compute_lyapunov_max(N, kappa)
        adj, n_edges, eps, avg_deg = build_graph(phases,
                                                  target_degree=target_degree)
        dim_data = measure_dimension_fixed(adj, N, num_observers=15,
                                            max_radius=10)
        results.append({
            'kappa': kappa,
            'd_corr': d_corr,
            'D_regression': dim_data['D_regression'],
            'D_micro': dim_data['D_micro'],
            'lyapunov': lyap,
            'avg_degree': avg_deg,
            'time': time.time() - t0
        })
        print(f"D_corr={d_corr:.2f}, D_reg={dim_data['D_regression']:.2f}, "
              f"λ={lyap:.3f}, t={time.time()-t0:.1f}s")
    return results

# ================================================================
# MASTER FUNCTION
# ================================================================
def run_comprehensive_tests():
    print("╔" + "═" * 65 + "╗")
    print("║  SBE HYPOTHESIS — CORRECTED CRITICAL TEST SUITE v2        ║")
    print("║  Fixes: D_eff saturation, Bell test, null model           ║")
    print("╚" + "═" * 65 + "╝")

    N = 15000
    KAPPA = 0.5
    TARGET_DEG = 40

    # ======== PHASE 1: Trajectory ========
    print(f"\n[1/8] Generating trajectory (N={N}, κ={KAPPA})...")
    t0 = time.time()
    phases = generate_trajectory(N, KAPPA)
    d_corr = estimate_correlation_dimension(phases)
    lyap = compute_lyapunov_max(N, KAPPA)
    print(f"      D_corr = {d_corr:.3f}")
    print(f"      λ_max = {lyap:.4f} ({'CHAOTIC' if lyap > 0.01 else 'REGULAR'})")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 2: Graph ========
    print(f"\n[2/8] Building resonance graph (target deg={TARGET_DEG})...")
    t0 = time.time()
    adj, n_edges, eps, avg_deg = build_graph(phases, target_degree=TARGET_DEG)
    print(f"      ε={eps:.3f}, edges={n_edges:,}, avg_deg={avg_deg:.1f}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 3: Null model (control) ========
    print(f"\n[3/8] Building NULL MODEL (random T^6, same degree)...")
    t0 = time.time()
    adj_null, phases_null, n_edges_null, avg_deg_null = \
        build_null_model(N, target_degree=TARGET_DEG)
    d_corr_null = estimate_correlation_dimension(phases_null)
    print(f"      D_corr(null) = {d_corr_null:.3f}")
    print(f"      edges={n_edges_null:,}, avg_deg={avg_deg_null:.1f}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 4: Dimension (both models) ========
    print(f"\n[4/8] Measuring dimension (SBE vs null)...")
    t0 = time.time()
    dim_sbe = measure_dimension_fixed(adj, N, num_observers=25, max_radius=12)
    dim_null = measure_dimension_fixed(adj_null, N, num_observers=25, max_radius=12)

    print(f"      SBE:  D_reg={dim_sbe['D_regression']:.2f}, "
          f"D_micro={dim_sbe['D_micro']:.2f}, "
          f"D_meso={dim_sbe['D_meso']:.2f}, "
          f"sat_R={dim_sbe['saturation_R']}")
    print(f"      NULL: D_reg={dim_null['D_regression']:.2f}, "
          f"D_micro={dim_null['D_micro']:.2f}, "
          f"D_meso={dim_null['D_meso']:.2f}, "
          f"sat_R={dim_null['saturation_R']}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 5: Curvature ========
    print(f"\n[5/8] Ollivier-Ricci curvature (n=300)...")
    t0 = time.time()
    corr_grav, mean_curv, curvs, dens, verts = test_gravity(adj, N, n_samples=300)
    print(f"      <κ_OR>={mean_curv:.4f}, corr(κ,ρ)={corr_grav:.3f}")

    # Also for null model (subsample)
    corr_null, mean_curv_null, _, _, _ = test_gravity(adj_null, N, n_samples=100)
    print(f"      NULL: <κ_OR>={mean_curv_null:.4f}, corr={corr_null:.3f}")
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 6: Bell test ========
    print(f"\n[6/8] Quantum Bell test (corrected)...")
    t0 = time.time()
    try:
        S_bell, E11, E12, E21, E22, concurrence = \
            proper_bell_test(adj, N, n_eig=40)
        print(f"      S_CHSH = {S_bell:.4f}")
        print(f"      E(a,b)={E11:.3f}, E(a,b')={E12:.3f}, "
              f"E(a',b)={E21:.3f}, E(a',b')={E22:.3f}")
        print(f"      |S|={abs(S_bell):.4f}, "
              f"classical limit: 2, QM: {2*np.sqrt(2):.3f}")
        bell_ok = True
    except Exception as e:
        print(f"      ERROR: {e}")
        S_bell, concurrence = 0, 0
        bell_ok = False
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 7: Mass spectrum ========
    print(f"\n[7/8] Dirac mass spectrum...")
    t0 = time.time()
    try:
        masses, raw_eigs = compute_dirac_spectrum(adj, N, k=40)
        nonzero = masses[masses > 1e-5]
        n_zero = len(masses) - len(nonzero)
        print(f"      Zero modes: {n_zero}")
        if len(nonzero) >= 5:
            print(f"      Mass gap: {nonzero[0]:.6f}")
            print(f"      First 5: {nonzero[:5]}")
            ratios = nonzero[:5] / nonzero[0]
            print(f"      Ratios: {ratios}")
    except Exception as e:
        print(f"      ERROR: {e}")
        nonzero = np.array([])
        n_zero = 0
    print(f"      Time: {time.time()-t0:.2f}s")

    # ======== PHASE 8: Phase diagram ========
    print(f"\n[8/8] Phase diagram κ-sweep...")
    kappas_fine = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    pd = phase_diagram_scan(N=10000, kappas=kappas_fine, target_degree=TARGET_DEG)

    # ================================================================
    # VISUALIZATION
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. D(R) — SBE vs Null
    ax = axes[0, 0]
    if dim_sbe['valid_radii']:
        ax.plot(dim_sbe['valid_radii'], dim_sbe['local_D'],
                'o-', color='crimson', linewidth=2, label='SBE (E₆)')
    if dim_null['valid_radii']:
        ax.plot(dim_null['valid_radii'], dim_null['local_D'],
                's--', color='gray', linewidth=2, label='Null (random T⁶)')
    ax.axhline(4, ls=':', color='green', alpha=0.7, label='4D')
    ax.axhline(6, ls=':', color='blue', alpha=0.5, label='6D')
    ax.set_xlabel('Scale R')
    ax.set_ylabel('D_eff(R)')
    ax.set_title(f'Dimension: SBE vs Null Model\n'
                 f'SBE D_reg={dim_sbe["D_regression"]:.2f}, '
                 f'Null D_reg={dim_null["D_regression"]:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    # 2. Phase diagram
    ax = axes[0, 1]
    ks = [r['kappa'] for r in pd]
    dc = [r['d_corr'] for r in pd]
    dr = [r['D_regression'] for r in pd]
    dm = [r['D_micro'] for r in pd]
    ax.plot(ks, dc, 'o-', color='navy', linewidth=2, label='D_corr (attractor)')
    ax.plot(ks, dr, 's-', color='crimson', linewidth=2, label='D_reg (graph)')
    ax.plot(ks, dm, '^-', color='orange', linewidth=2, label='D_micro (R=2-3)')
    ax.axhline(4, ls=':', color='green', alpha=0.7)
    ax.axhline(6, ls=':', color='blue', alpha=0.5)
    ax.set_xlabel('Coupling κ')
    ax.set_ylabel('Dimension')
    ax.set_title('Phase Diagram: D(κ)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Lyapunov
    ax = axes[0, 2]
    ll = [r['lyapunov'] for r in pd]
    ax.plot(ks, ll, 'D-', color='darkred', linewidth=2)
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.fill_between(ks, 0, ll, alpha=0.15, color='red',
                     where=[l > 0 for l in ll])
    ax.set_xlabel('Coupling κ')
    ax.set_ylabel('λ_max')
    ax.set_title('Lyapunov Exponent\n(Chaos threshold)')
    ax.grid(True, alpha=0.3)

    # 4. Curvature histogram
    ax = axes[1, 0]
    if len(curvs) > 0:
        ax.hist(curvs, bins=30, color='teal', alpha=0.7, edgecolor='white',
                label='SBE')
        ax.axvline(0, ls='--', color='red', alpha=0.7)
        ax.set_xlabel('κ_OR')
        ax.set_ylabel('Count')
        ax.set_title(f'Curvature: <κ>={mean_curv:.3f}\n'
                     f'corr(κ,ρ)={corr_grav:.3f}')
    ax.grid(True, alpha=0.3)

    # 5. Curvature vs density
    ax = axes[1, 1]
    if len(curvs) > 0 and len(dens) > 0:
        ax.scatter(dens, curvs, alpha=0.4, s=10, color='purple')
        if len(dens) > 2:
            z = np.polyfit(dens, curvs, 1)
            x_fit = np.linspace(min(dens), max(dens), 100)
            ax.plot(x_fit, np.polyval(z, x_fit), 'r-', lw=2,
                    label=f'r={corr_grav:.3f}')
            ax.legend()
        ax.set_xlabel('Density ρ = deg/⟨deg⟩')
        ax.set_ylabel('Scalar curvature')
        ax.set_title('Einstein Test: κ_OR vs ρ')
    ax.grid(True, alpha=0.3)

    # 6. Mass spectrum
    ax = axes[1, 2]
    if len(nonzero) > 0:
        ax.bar(range(min(30, len(nonzero))), nonzero[:30],
               color='indigo', alpha=0.8)
        ax.set_xlabel('Mode k')
        ax.set_ylabel('Mass |μ_k|')
        ax.set_title(f'Dirac Masses\n'
                     f'gap={nonzero[0]:.4f}, zero modes={n_zero}')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'SBE Critical Tests v2 — N={N}, κ={KAPPA}\n'
                 f'D_corr={d_corr:.2f}, λ_max={lyap:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sbe_v2_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║  FINAL VERDICT v2 (with null model comparison)               ║")
    print("╠" + "═" * 70 + "╣")

    tests = [
        ("Chaos at κ=0.5 (λ > 0)",
         lyap > 0.01,
         f"λ={lyap:.4f}",
         "Prerequisite for non-trivial geometry"),

        ("D_corr < 5.5 (attractor compressed)",
         d_corr < 5.5 and not np.isnan(d_corr),
         f"D_corr={d_corr:.2f}",
         "E₆ coupling reduces attractor dimension"),

        ("D_reg(SBE) < D_reg(null) (non-trivial)",
         dim_sbe['D_regression'] < dim_null['D_regression'] - 0.3
         if not (np.isnan(dim_sbe['D_regression']) or
                 np.isnan(dim_null['D_regression'])) else False,
         f"SBE={dim_sbe['D_regression']:.2f} vs null={dim_null['D_regression']:.2f}",
         "Graph geometry differs from trivial RGG"),

        ("Positive mean curvature",
         mean_curv > 0.01,
         f"<κ>={mean_curv:.4f}",
         "Positive = de Sitter-like spacetime"),

        ("Einstein eqn: corr(κ,ρ) > 0.3",
         abs(corr_grav) > 0.3,
         f"r={corr_grav:.3f}",
         "Curvature tracks matter density"),

        ("Entanglement (concurrence > 0.1)" if bell_ok else "Bell test ran",
         concurrence > 0.1 if bell_ok else False,
         f"C={concurrence:.3f}" if bell_ok else "N/A",
         "Graph Hamiltonian produces entanglement"),

        ("Bell violation (|S| > 2)" if bell_ok else "Bell test ran",
         abs(S_bell) > 2.0 if bell_ok else False,
         f"|S|={abs(S_bell):.3f}" if bell_ok else "N/A",
         "Quantum correlations from graph"),

        ("Mass gap exists",
         len(nonzero) > 0 and nonzero[0] > 1e-4,
         f"Δm={nonzero[0]:.4f}" if len(nonzero) > 0 else "N/A",
         "Discrete spectrum = quantized masses"),
    ]

    n_pass = 0
    for name, passed, detail, meaning in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        n_pass += int(passed)
        print(f"║  {status}  {name:<42s} {detail:>14s}  ║")
        print(f"║         {meaning:<60s} ║")

    print("╠" + "═" * 70 + "╣")
    score_pct = 100 * n_pass / len(tests)
    color = "PROMISING" if score_pct >= 50 else "NEEDS WORK"
    print(f"║  Score: {n_pass}/{len(tests)} ({score_pct:.0f}%) — {color}"
          + " " * (70 - 25 - len(str(n_pass)) - len(str(len(tests)))
                   - len(f"{score_pct:.0f}") - len(color)) + "║")
    print("╚" + "═" * 70 + "╝")

    return {
        'd_corr': d_corr, 'lyap': lyap,
        'dim_sbe': dim_sbe, 'dim_null': dim_null,
        'gravity_corr': corr_grav, 'mean_curvature': mean_curv,
        'S_bell': S_bell, 'concurrence': concurrence,
        'masses': nonzero, 'phase_diagram': pd
    }

if __name__ == "__main__":
    results = run_comprehensive_tests()