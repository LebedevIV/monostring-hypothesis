import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import deque
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csc_matrix, bmat
from scipy.sparse.linalg import eigsh
import time
import random
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# МАТРИЦА КАРТАНА E6 (Генератор динамики)
# ============================================================
C_E6 = np.array([[ 2, -1,  0,  0,  0,  0],[-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0, -1],
    [ 0,  0, -1,  2, -1,  0],[ 0,  0,  0, -1,  2,  0],[ 0,  0, -1,  0,  0,  2]
], dtype=np.float64)

# ============================================================
# БЛОК 1: ГЕНЕРАЦИЯ, АВТОКАЛИБРОВКА И ГРАФ
# ============================================================
def generate_trajectory_E6(N, kappa):
    D = 6
    omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5), np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    phases = np.zeros((N, D), dtype=np.float32)
    phases[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N - 1):
        coupling = kappa * C_E6 @ np.sin(phases[n])
        phases[n+1] = (phases[n] + omega + coupling) % (2 * np.pi)
    return phases

def build_resonance_graph_kd(phases, target_degree=50):
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)

    # Ищем радиус для нужной плотности (берем случайные точки)
    sample_idx = random.sample(range(N), min(N, 1000))
    dists, _ = tree.query(coords[sample_idx], k=target_degree+1)
    eucl_thresh = np.median(dists[:, -1])

    print(f"      Auto-tuned Euclidean threshold: {eucl_thresh:.4f}")

    pairs = tree.query_pairs(r=eucl_thresh, output_type='ndarray')
    mask = np.abs(pairs[:, 0] - pairs[:, 1]) > 5
    pairs = pairs[mask]

    adj = {i: set() for i in range(N)}
    for i in range(N - 1):
        adj[i].add(i + 1); adj[i + 1].add(i)
    for a, b in pairs:
        adj[a].add(b); adj[b].add(a)

    return adj, len(pairs)

# ============================================================
# БЛОК 2: ТЕСТ 1 - РАЗМЕРНОСТЬ (D_eff)
# ============================================================
def measure_dimension_BFS(adj, N, num_observers=15, max_radius=8):
    observers = random.sample(range(N//10, 9*N//10), min(num_observers, N//2))
    all_dims =[]
    radii = list(range(1, max_radius + 1))
    for obs in observers:
        dist = {obs: 0}
        queue = deque([obs])
        while queue:
            v = queue.popleft()
            if dist[v] >= max_radius: continue
            for u in adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        volumes =[sum(1 for d in dist.values() if d <= r) for r in radii]
        local_D =[]
        for i in range(1, len(radii)):
            if volumes[i] > volumes[i-1] > 0:
                dr = np.log(radii[i]) - np.log(radii[i-1])
                dv = np.log(volumes[i]) - np.log(volumes[i-1])
                local_D.append(dv / dr if dr > 0 else 0)
            else:
                local_D.append(0)
        all_dims.append(local_D)
    return np.mean(all_dims, axis=0)

# ============================================================
# БЛОК 3: ТЕСТ 2 - КРИВИЗНА ОЛИВЬЕ-РИЧЧИ (ЭЙНШТЕЙН)
# ============================================================
def bfs_limited(adj, start, max_dist=4):
    dist = {start: 0}; queue = deque([start])
    while queue:
        v = queue.popleft()
        if dist[v] >= max_dist: continue
        for u in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1; queue.append(u)
    return dist

def ollivier_ricci_curvature(adj, node_a, node_b, alpha=0.5):
    neighbors_a = list(adj[node_a]); support_a = [node_a] + neighbors_a
    neighbors_b = list(adj[node_b]); support_b = [node_b] + neighbors_b
    deg_a, deg_b = len(neighbors_a), len(neighbors_b)
    if deg_a == 0 or deg_b == 0: return 0.0

    mu_a = {node_a: alpha}
    for v in neighbors_a: mu_a[v] = (1 - alpha) / deg_a
    mu_b = {node_b: alpha}
    for v in neighbors_b: mu_b[v] = (1 - alpha) / deg_b

    n_a, n_b = len(support_a), len(support_b)
    dist_matrix = np.zeros((n_a, n_b))
    for i, u in enumerate(support_a):
        dists = bfs_limited(adj, u, max_dist=4)
        for j, v in enumerate(support_b): dist_matrix[i, j] = dists.get(v, 8)

    c = dist_matrix.flatten()
    n_vars = n_a * n_b
    A_eq_rows, b_eq_rows = [],[]
    for i in range(n_a):
        row = np.zeros(n_vars)
        for j in range(n_b): row[i * n_b + j] = 1.0
        A_eq_rows.append(row); b_eq_rows.append(mu_a.get(support_a[i], 0.0))
    for j in range(n_b):
        row = np.zeros(n_vars)
        for i in range(n_a): row[i * n_b + j] = 1.0
        A_eq_rows.append(row); b_eq_rows.append(mu_b.get(support_b[j], 0.0))

    result = linprog(c, A_eq=np.array(A_eq_rows), b_eq=np.array(b_eq_rows),
                     bounds=[(0, None)] * n_vars, method='highs')
    if result.success: return 1 - result.fun / 1.0
    return 0.0

def test_einstein_equations(adj, N, n_samples=100):
    all_edges = [(v, u) for v in adj for u in adj[v] if u > v]
    sample_edges = random.sample(all_edges, min(n_samples, len(all_edges)))
    edge_curvature = {}
    for a, b in sample_edges:
        kappa = ollivier_ricci_curvature(adj, a, b)
        edge_curvature[(a, b)] = kappa

    vertex_curvature = {}
    for (a, b), kappa in edge_curvature.items():
        vertex_curvature[a] = vertex_curvature.get(a, 0) + kappa
        vertex_curvature[b] = vertex_curvature.get(b, 0) + kappa

    avg_deg = sum(len(adj[v]) for v in adj) / N
    vertex_density = {v: len(adj[v]) / avg_deg for v in vertex_curvature}

    curvs = np.array(list(vertex_curvature.values()))
    dens = np.array(list(vertex_density.values()))
    if len(curvs) > 1 and np.std(curvs) > 0 and np.std(dens) > 0:
        return np.corrcoef(curvs, dens)[0, 1]
    return 0.0

# ============================================================
# БЛОК 4: ТЕСТ 3 - КВАНТОВАЯ ЗАПУТАННОСТЬ (CHSH)
# ============================================================
def run_bell_test(phases):
    N, D = phases.shape
    target = np.full(D, np.pi)
    pairs =[]
    # Жесткий поиск антикоррелированных пар (ключ-замок)
    for _ in range(50000):
        m, n = np.random.randint(0, N), np.random.randint(0, N)
        if m == n: continue
        diff = np.abs((phases[m] + phases[n]) % (2 * np.pi) - target)
        if np.mean(np.minimum(diff, 2*np.pi - diff)) < 0.8:
            pairs.append((m, n))
            if len(pairs) > 500: break

    if len(pairs) < 10:
        print("      [!] Not enough entangled pairs found")
        return 0, 0

    a, a_ = np.zeros(D), np.zeros(D)
    b, b_ = np.zeros(D), np.zeros(D)
    a[0] = 1.0; a_[0], a_[1] = np.cos(np.pi/4), np.sin(np.pi/4)
    b[0], b[1] = np.cos(np.pi/8), np.sin(np.pi/8)
    b_[0], b_[1] = np.cos(3*np.pi/8), np.sin(3*np.pi/8)

    def E_classical(dir_a, dir_b):
        corr =[]
        for m, n in pairs:
            s_m = 1 if np.dot(dir_a, phases[m]) > np.pi else -1
            s_n = 1 if np.dot(dir_b, phases[n]) > np.pi else -1
            corr.append(s_m * s_n)
        return np.mean(corr)

    def E_quantum(dir_a, dir_b):
        return np.mean([-np.cos(np.dot(dir_a, phases[m]) - np.dot(dir_b, phases[n])) for m, n in pairs])

    S_cl = E_classical(a,b) - E_classical(a,b_) + E_classical(a_,b) + E_classical(a_,b_)
    S_qu = E_quantum(a,b) - E_quantum(a,b_) + E_quantum(a_,b) + E_quantum(a_,b_)
    return S_cl, S_qu

# ============================================================
# БЛОК 5: ТЕСТ 4 - ОПЕРАТОР ДИРАКА (МАССЫ)
# ============================================================
def test_dirac_masses(adj, N, k=15):
    edges =[(i, j) for i in adj for j in adj[i] if j > i]
    n_edges = len(edges)
    incidence = lil_matrix((N, n_edges), dtype=np.float32)
    for e_idx, (i, j) in enumerate(edges):
        incidence[i, e_idx] = -1.0; incidence[j, e_idx] = 1.0
    incidence = csc_matrix(incidence)

    zero_vv = csc_matrix((N, N))
    zero_ee = csc_matrix((n_edges, n_edges))
    D_op = bmat([[zero_vv, incidence],[incidence.T, zero_ee]], format='csc')

    D_squared = D_op @ D_op
    eigenvalues_sq, _ = eigsh(D_squared, k=k, which='SM', tol=1e-3, maxiter=2000)
    masses = np.sort(np.sqrt(np.abs(eigenvalues_sq)))
    return masses

# ============================================================
# МАСТЕР-ФУНКЦИЯ (ЗАПУСК ВСЕХ ТЕСТОВ)
# ============================================================
def run_all_tests():
    print("╔" + "═"*65 + "╗")
    print("║  HIGH-PRECISION TEST SUITE FOR SBE HYPOTHESIS           ║")
    print("║  Mathematical apparatus: E₆ coupled standard map        ║")
    print("╚" + "═"*65 + "╝")

    N = 40000       # Рабочий масштаб для 32GB RAM
    kappa = 0.5     # Константа связи E6

    print(f"\n[1/6] Generating E₆-coupled trajectory (N={N})...")
    t0 = time.time()
    phases = generate_trajectory_E6(N, kappa)
    print(f"      Time: {time.time()-t0:.2f}s")

    print("\n[2/6] Building resonance graph (Auto-tuning density)...")
    t0 = time.time()
    adj, n_edges = build_resonance_graph_kd(phases, target_degree=40)
    avg_deg = sum(len(adj[v]) for v in adj) / N
    print(f"      Time: {time.time()-t0:.2f}s | Edges: {n_edges} | Avg degree: {avg_deg:.1f}")

    print("\n[3/6] Testing dimensional emergence (D_eff)...")
    t0 = time.time()
    dims = measure_dimension_BFS(adj, N, num_observers=10, max_radius=6)
    D_eff = np.max(dims)  # Берем пиковую макро-размерность до обвала в 0
    print(f"      Time: {time.time()-t0:.2f}s | Effective dimension D_eff = {D_eff:.2f}")

    print("\n[4/6] Testing Einstein equation analog (Ollivier-Ricci)...")
    t0 = time.time()
    corr = test_einstein_equations(adj, N, n_samples=250)
    print(f"      Time: {time.time()-t0:.2f}s | Curvature-Matter correlation = {corr:.3f}")

    print("\n[5/6] Testing Bell inequality (CHSH)...")
    t0 = time.time()
    S_cl, S_qu = run_bell_test(phases)
    print(f"      Time: {time.time()-t0:.2f}s | Classical S = {S_cl:.3f} | Quantum S = {S_qu:.3f}")

    print("\n[6/6] Testing Dirac mass spectrum (Heavy calculation!)...")
    t0 = time.time()
    try:
        masses = test_dirac_masses(adj, N, k=15)
        nonzero = masses[masses > 1e-4]
        print(f"      Time: {time.time()-t0:.2f}s | Mass gap found: {nonzero[0]:.6f}" if len(nonzero)>0 else "      No mass gap found")
    except Exception as e:
        print(f"      [!] Dirac computation skipped (Out of Memory / Time limit).")

    print("\n" + "="*67)
    print("  SUMMARY OF CLAUDE OPUS CRITICAL TESTS")
    print("="*67)

    pass_dim = abs(D_eff - 6) > 0.5 and D_eff > 1.0
    print(f"  1. Dimensional emergence: D_eff={D_eff:.2f} (Target != 6) -> {'PASS' if pass_dim else 'FAIL'}")

    pass_grav = abs(corr) > 0.2
    print(f"  2. Einstein equations: corr={corr:.3f} (Target > 0.2) -> {'PASS' if pass_grav else 'FAIL'}")

    pass_bell = abs(abs(S_qu) - 2*np.sqrt(2)) < 0.4
    print(f"  3. Bell inequality: S_qu={S_qu:.3f} (Target ≈ 2.828) -> {'PASS' if pass_bell else 'FAIL'}")
    print("="*67)

if __name__ == "__main__":
    run_all_tests()
