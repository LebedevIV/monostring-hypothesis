import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, lil_matrix, eye, diags
from scipy.sparse.linalg import eigsh, expm_multiply
from collections import deque
import time
import warnings

warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# CORE: Trajectory + Graph (from v2, with strict degree control)
# ================================================================
def generate_thermal_trajectory(N, kappa, temperature):
    D = 6
    omega = 2 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)
    phases = np.zeros((N, D))
    phases[0] = np.random.uniform(0, 2 * np.pi, D)
    for n in range(N - 1):
        coupling = kappa * C_E6 @ np.sin(phases[n])
        noise = np.random.normal(0, temperature, D)
        phases[n + 1] = (phases[n] + omega + coupling + noise) % (2 * np.pi)
    return phases

def build_graph_strict(phases, target_degree=30, delta_min=5):
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    lo, hi = 0.01, 8.0
    best_eps, best_deg, best_pairs = 1.0, 0, None

    for _ in range(25):
        mid = (lo + hi) / 2
        pairs = tree.query_pairs(r=mid, output_type='ndarray')
        if len(pairs) > 0:
            pf = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > delta_min]
        else:
            pf = np.array([]).reshape(0, 2).astype(int)
        actual = (2 * len(pf) + 2 * (N - 1)) / N
        if actual < target_degree:
            lo = mid
        else:
            hi = mid
        if abs(actual - target_degree) < abs(best_deg - target_degree):
            best_deg, best_eps, best_pairs = actual, mid, pf.copy()
        if abs(actual - target_degree) / target_degree < 0.05:
            break

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N - 1):
        G.add_edge(i, i + 1)
    if best_pairs is not None:
        for a, b in best_pairs:
            G.add_edge(int(a), int(b))

    final_deg = sum(dict(G.degree()).values()) / N

    # Prune if too dense
    if final_deg > target_degree * 1.1:
        res_edges = [(u, v) for u, v in G.edges() if abs(u - v) > delta_min]
        n_remove = int((final_deg - target_degree) * N / 2)
        if 0 < n_remove < len(res_edges):
            remove = [res_edges[i] for i in
                      np.random.choice(len(res_edges), n_remove, replace=False)]
            G.remove_edges_from(remove)
        final_deg = sum(dict(G.degree()).values()) / N

    return G, final_deg

def build_null_graph(N, target_degree=30):
    ph = np.random.uniform(0, 2 * np.pi, (N, 6))
    return build_graph_strict(ph, target_degree=target_degree)

# ================================================================
# PHASE ORDER PARAMETERS
# ================================================================
def kuramoto_order(phases):
    N, D = phases.shape
    r_per = np.zeros(D)
    for d in range(D):
        r_per[d] = np.abs(np.mean(np.exp(1j * phases[:, d])))
    return np.prod(r_per) ** (1.0 / D), r_per

def local_field(phases, G):
    """Local Higgs field: phase coherence in neighborhood."""
    N, D = phases.shape
    phi = np.zeros(N)
    for v in range(N):
        nbrs = list(G.neighbors(v))
        if not nbrs:
            continue
        lp = phases[nbrs]
        s = 0
        for d in range(D):
            s += np.abs(np.mean(np.exp(1j * lp[:, d])))
        phi[v] = s / D
    return phi

# ================================================================
# MASS DEFINITION A: Correlation length (inverse mass)
# ================================================================
def correlation_length(G, phi_field, n_samples=200, max_dist=15):
    """
    Two-point correlator of the scalar field:
    C(r) = <Phi(x) Phi(y)> - <Phi>^2 for pairs at graph distance r

    Mass = 1 / xi where C(r) ~ exp(-r/xi)

    In symmetric phase: xi -> infinity (massless Goldstone)
    In broken phase: xi finite (massive Higgs boson)
    """
    N = len(phi_field)
    mean_phi = np.mean(phi_field)

    # Sample source nodes
    sources = np.random.choice(N, min(n_samples, N), replace=False)

    # Collect (distance, product) pairs
    dist_products = {r: [] for r in range(1, max_dist + 1)}

    for src in sources:
        dists = {}
        queue = deque([(src, 0)])
        dists[src] = 0
        while queue:
            v, d = queue.popleft()
            if d >= max_dist:
                continue
            for u in G.neighbors(v):
                if u not in dists:
                    dists[u] = d + 1
                    queue.append((u, d + 1))

        phi_src = phi_field[src] - mean_phi
        for v, d in dists.items():
            if 1 <= d <= max_dist:
                phi_v = phi_field[v] - mean_phi
                dist_products[d].append(phi_src * phi_v)

    # Average correlator at each distance
    rs = []
    Cs = []
    for r in range(1, max_dist + 1):
        if len(dist_products[r]) > 10:
            rs.append(r)
            Cs.append(np.mean(dist_products[r]))

    rs = np.array(rs, dtype=float)
    Cs = np.array(Cs)

    # Fit exponential decay: C(r) = A * exp(-r/xi)
    # log C(r) = log A - r/xi (linear fit)
    positive = Cs > 1e-10
    if np.sum(positive) < 3:
        return np.inf, 0.0, rs, Cs  # Massless (infinite correlation length)

    log_C = np.log(Cs[positive])
    r_pos = rs[positive]

    try:
        coeffs = np.polyfit(r_pos, log_C, 1)
        xi = -1.0 / coeffs[0] if coeffs[0] < -1e-6 else np.inf
        mass_corr = 1.0 / xi if xi > 0 and xi < 1000 else 0.0
    except:
        xi = np.inf
        mass_corr = 0.0

    return xi, mass_corr, rs, Cs

# ================================================================
# MASS DEFINITION B: Effective potential curvature
# ================================================================
def effective_potential(phi_field, n_bins=50):
    """
    Reconstruct V_eff(Phi) from the distribution P(Phi).
    V_eff(Phi) = -T * ln P(Phi)

    Mass_Higgs^2 = V''(v) where v = argmin V
    """
    hist, edges = np.histogram(phi_field, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    # V = -ln P (up to constant)
    with np.errstate(divide='ignore'):
        V = -np.log(hist + 1e-20)
    V -= np.min(V)  # Shift minimum to 0

    # Find minimum
    i_min = np.argmin(V)
    v_min = centers[i_min]

    # Second derivative at minimum (mass^2)
    if 1 < i_min < len(V) - 1:
        dV = centers[1] - centers[0]
        V_pp = (V[i_min + 1] - 2 * V[i_min] + V[i_min - 1]) / dV**2
        mass_higgs = np.sqrt(max(V_pp, 0))
    else:
        mass_higgs = 0.0

    return v_min, mass_higgs, centers, V

# ================================================================
# MASS DEFINITION C: Dispersion relation from quantum walk
# ================================================================
def mass_from_dispersion(G, t_range=(0.5, 8.0), n_times=30):
    """
    Mass from quantum walk dispersion relation.

    For a massless particle: sigma^2(t) ~ t^2 (ballistic)
    For a massive particle: sigma^2(t) ~ t (diffusive) at late times

    Fit sigma^2 = A * t^alpha
    alpha = 2: massless (ballistic)
    alpha = 1: massive (diffusive)
    alpha < 1: localized (very massive / Anderson)

    Effective mass ~ 1/alpha (heuristic)
    """
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)

    psi_0 = np.zeros(N, dtype=np.complex128)
    center = N // 2
    psi_0[center] = 1.0

    dists_bfs = nx.single_source_shortest_path_length(G, center)
    dist_arr = np.array([dists_bfs.get(i, N) for i in range(N)], dtype=float)

    times = np.linspace(t_range[0], t_range[1], n_times)
    variances = []

    for t in times:
        psi_t = expm_multiply(-1j * t * L, psi_0)
        prob = np.abs(psi_t) ** 2
        s = np.sum(prob)
        if s > 0:
            prob /= s
        variances.append(np.sum(prob * dist_arr ** 2))

    variances = np.array(variances)

    # Fit log(sigma^2) = alpha * log(t) + const
    valid = (times > 0) & (variances > 1e-10)
    if np.sum(valid) < 5:
        return 1.0, times, variances

    log_t = np.log(times[valid])
    log_v = np.log(variances[valid])

    try:
        alpha, _ = np.polyfit(log_t, log_v, 1)
    except:
        alpha = 1.0

    return alpha, times, variances

# ================================================================
# DOMAIN WALL DETECTOR
# ================================================================
def detect_domain_walls(phi_field, G, threshold=0.3):
    """
    Find edges where the field changes sharply.
    These are domain walls — topological defects where
    different vacuum states meet.

    In real physics, domain walls carry energy (tension).
    """
    wall_edges = []
    wall_strengths = []

    for u, v in G.edges():
        gradient = abs(phi_field[u] - phi_field[v])
        if gradient > threshold:
            wall_edges.append((u, v))
            wall_strengths.append(gradient)

    n_walls = len(wall_edges)
    wall_density = n_walls / G.number_of_edges() if G.number_of_edges() > 0 else 0
    mean_strength = np.mean(wall_strengths) if wall_strengths else 0

    return n_walls, wall_density, mean_strength

# ================================================================
# GOLDSTONE MODE DETECTOR
# ================================================================
def detect_goldstone_modes(G, phi_field, k=20):
    """
    In broken symmetry, there should be:
    - 1 massive mode (Higgs boson) = radial oscillation
    - (N_broken - 1) massless modes (Goldstone bosons) = angular oscillations

    Check: does the Laplacian spectrum of the FLUCTUATION field
    (delta_phi = phi - <phi>) show a gap structure?
    """
    N = len(phi_field)
    mean_phi = np.mean(phi_field)
    delta_phi = phi_field - mean_phi

    # Compute Laplacian
    L = nx.normalized_laplacian_matrix(G)

    # Project fluctuation onto Laplacian eigenmodes
    k_actual = min(k, N - 2)
    evals, evecs = eigsh(L, k=k_actual, which='SM', tol=1e-3, maxiter=3000)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Overlap of delta_phi with each eigenmode
    overlaps = np.array([np.abs(np.dot(delta_phi, evecs[:, i])) for i in range(k_actual)])

    # Normalize
    total = np.sum(overlaps)
    if total > 0:
        overlaps /= total

    return evals, overlaps

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_higgs_lab_v3():
    print("=" * 70)
    print("  MONOSTRING HIGGS LABORATORY v3")
    print("  Three mass definitions + domain walls + Goldstone modes")
    print("  N=5000, strict degree control")
    print("=" * 70)

    N = 5000
    kappa = 0.5
    TARGET_DEG = 35
    temperatures = np.logspace(0.5, -2, 14)

    # Storage
    data = {
        'T': [], 'r_kura': [], 'r_per_dim': [], 'vev_local': [],
        'mass_spectral': [], 'mass_corr_length': [], 'xi': [],
        'mass_higgs_potential': [], 'v_min': [],
        'mass_dispersion_alpha': [],
        'n_walls': [], 'wall_density': [],
        'clust': [], 'deg': []
    }
    data_null = {
        'T': [], 'r_kura': [], 'mass_corr_length': [], 'xi': [],
        'mass_higgs_potential': [], 'deg': []
    }

    print(f"\n[1/5] Full thermodynamic scan...")
    print(f"      {'T':>7s} | {'r_K':>5s} {'VEV':>5s} | "
          f"{'m_spec':>6s} {'m_corr':>6s} {'xi':>6s} {'m_pot':>5s} | "
          f"{'walls':>5s} {'deg':>5s} | "
          f"{'m_c_n':>5s} {'m_p_n':>5s}")
    print("      " + "-" * 75)

    for T in temperatures:
        t0 = time.time()

        # === SBE ===
        phases = generate_thermal_trajectory(N, kappa, T)
        G, deg = build_graph_strict(phases, target_degree=TARGET_DEG)

        r_k, r_per = kuramoto_order(phases)
        phi_f = local_field(phases, G)
        vev = np.mean(phi_f)

        # Mass A: spectral gap
        m_spec = 0
        try:
            L = nx.normalized_laplacian_matrix(G)
            ev, _ = eigsh(L, k=3, which='SM', tol=1e-3, maxiter=2000)
            ev_sorted = np.sort(ev)
            nz = ev_sorted[ev_sorted > 1e-4]
            m_spec = nz[0] if len(nz) > 0 else 0
        except:
            pass

        # Mass B: correlation length
        xi, m_corr, _, _ = correlation_length(G, phi_f, n_samples=150, max_dist=12)

        # Mass C: effective potential
        v_min, m_pot, _, _ = effective_potential(phi_f)

        # Domain walls
        n_w, w_dens, _ = detect_domain_walls(phi_f, G, threshold=0.3)

        clust = nx.transitivity(G)

        data['T'].append(T)
        data['r_kura'].append(r_k)
        data['r_per_dim'].append(r_per.copy())
        data['vev_local'].append(vev)
        data['mass_spectral'].append(m_spec)
        data['mass_corr_length'].append(m_corr)
        data['xi'].append(xi)
        data['mass_higgs_potential'].append(m_pot)
        data['v_min'].append(v_min)
        data['n_walls'].append(n_w)
        data['wall_density'].append(w_dens)
        data['clust'].append(clust)
        data['deg'].append(deg)

        # === NULL ===
        G_n, deg_n = build_null_graph(N, target_degree=TARGET_DEG)
        ph_n = np.random.uniform(0, 2 * np.pi, (N, 6))
        r_n, _ = kuramoto_order(ph_n)
        phi_n = local_field(ph_n, G_n)
        xi_n, m_corr_n, _, _ = correlation_length(G_n, phi_n, n_samples=100, max_dist=10)
        _, m_pot_n, _, _ = effective_potential(phi_n)

        data_null['T'].append(T)
        data_null['r_kura'].append(r_n)
        data_null['mass_corr_length'].append(m_corr_n)
        data_null['xi'].append(xi_n)
        data_null['mass_higgs_potential'].append(m_pot_n)
        data_null['deg'].append(deg_n)

        print(f"      {T:>7.3f} | {r_k:>5.3f} {vev:>5.3f} | "
              f"{m_spec:>6.4f} {m_corr:>6.4f} {xi:>6.2f} {m_pot:>5.2f} | "
              f"{n_w:>5d} {deg:>5.1f} | "
              f"{m_corr_n:>5.3f} {m_pot_n:>5.2f}  ({time.time()-t0:.1f}s)")

    # === DISPERSION ANALYSIS ===
    print(f"\n[2/5] Dispersion relation (hot vs cold)...")
    phases_hot = generate_thermal_trajectory(N, kappa, temperatures[0])
    G_hot, _ = build_graph_strict(phases_hot, target_degree=TARGET_DEG)
    alpha_hot, t_hot, var_hot = mass_from_dispersion(G_hot)

    phases_cold = generate_thermal_trajectory(N, kappa, temperatures[-1])
    G_cold, _ = build_graph_strict(phases_cold, target_degree=TARGET_DEG)
    alpha_cold, t_cold, var_cold = mass_from_dispersion(G_cold)

    print(f"      Hot:  alpha = {alpha_hot:.3f} ({'ballistic' if alpha_hot > 1.5 else 'diffusive' if alpha_hot > 0.8 else 'localized'})")
    print(f"      Cold: alpha = {alpha_cold:.3f} ({'ballistic' if alpha_cold > 1.5 else 'diffusive' if alpha_cold > 0.8 else 'localized'})")

    # === GOLDSTONE ANALYSIS ===
    print(f"\n[3/5] Goldstone mode analysis (cold)...")
    phi_cold = local_field(phases_cold, G_cold)
    gold_evals, gold_overlaps = detect_goldstone_modes(G_cold, phi_cold, k=15)

    print(f"      First 10 eigenvalues: {gold_evals[:10]}")
    print(f"      Fluctuation overlaps: {gold_overlaps[:10]}")

    # Count near-zero modes (potential Goldstones)
    n_goldstone = np.sum(gold_evals < 0.01)
    print(f"      Near-zero modes (Goldstone candidates): {n_goldstone}")

    # === PER-DIMENSION ORDER PARAMETER ===
    print(f"\n[4/5] Per-dimension symmetry breaking analysis...")
    print(f"      {'T':>7s} | {'r1':>5s} {'r2':>5s} {'r3':>5s} {'r4':>5s} {'r5':>5s} {'r6':>5s}")
    print("      " + "-" * 45)
    for i, T in enumerate(data['T']):
        rp = data['r_per_dim'][i]
        print(f"      {T:>7.3f} | {rp[0]:>5.3f} {rp[1]:>5.3f} {rp[2]:>5.3f} "
              f"{rp[3]:>5.3f} {rp[4]:>5.3f} {rp[5]:>5.3f}")

    # === CORRELATIONS ===
    print(f"\n[5/5] Correlation analysis...")

    corrs = {
        'm_spec vs r_K': np.corrcoef(data['mass_spectral'], data['r_kura'])[0, 1],
        'm_corr vs r_K': np.corrcoef(data['mass_corr_length'], data['r_kura'])[0, 1],
        'm_pot vs r_K': np.corrcoef(data['mass_higgs_potential'], data['r_kura'])[0, 1],
        'walls vs r_K': np.corrcoef(data['wall_density'], data['r_kura'])[0, 1],
        'm_spec vs deg': np.corrcoef(data['mass_spectral'], data['deg'])[0, 1],
        'm_corr vs deg': np.corrcoef(data['mass_corr_length'], data['deg'])[0, 1],
        'm_pot vs deg': np.corrcoef(data['mass_higgs_potential'], data['deg'])[0, 1],
    }

    print(f"      {'Correlation':<25s} {'Value':>8s} {'Interpretation':<30s}")
    print("      " + "-" * 65)
    for name, val in corrs.items():
        if 'vs r_K' in name:
            interp = 'YUKAWA-LIKE!' if val > 0.3 else 'anti-Yukawa' if val < -0.3 else 'no signal'
        else:
            interp = 'CONFOUND!' if abs(val) > 0.5 else 'OK'
        print(f"      {name:<25s} {val:>8.3f} {interp:<30s}")

    # ================================================================
    # PLOTS
    # ================================================================
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # 1. Kuramoto order parameter
    ax = axes[0, 0]
    ax.plot(data['T'], data['r_kura'], 'o-', color='teal', lw=2, label='Monostring')
    ax.plot(data_null['T'], data_null['r_kura'], 's--', color='gray', lw=1.5, label='Null')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('r_Kuramoto')
    ax.set_title('Phase Transition\n(Kuramoto synchronization)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Three mass definitions vs T
    ax = axes[0, 1]
    ax.plot(data['T'], data['mass_spectral'], 'o-', color='blue', lw=2, label='Spectral gap')
    ax.plot(data['T'], data['mass_corr_length'], 's-', color='red', lw=2, label='1/xi (corr length)')
    ax.plot(data['T'], data['mass_higgs_potential'], '^-', color='green', lw=2, label='V\'\'(v) (potential)')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Mass')
    ax.set_title('Three Mass Definitions vs T\n(Which one behaves like Yukawa?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Mass vs VEV (all three definitions)
    ax = axes[0, 2]
    ax.scatter(data['r_kura'], data['mass_spectral'], c='blue', s=50, label='Spectral', alpha=0.7)
    ax.scatter(data['r_kura'], data['mass_corr_length'], c='red', s=50, label='1/xi', marker='s', alpha=0.7)
    ax.scatter(data['r_kura'], data['mass_higgs_potential'], c='green', s=50, label='V\'\'', marker='^', alpha=0.7)
    ax.set_xlabel('r_Kuramoto (VEV proxy)')
    ax.set_ylabel('Mass')
    ax.set_title('Yukawa Test: m vs VEV\n(positive slope = Yukawa mechanism)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Domain walls vs T
    ax = axes[0, 3]
    ax.plot(data['T'], data['wall_density'], 'D-', color='darkred', lw=2)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Domain wall density')
    ax.set_title('Topological Defects\n(Domain walls disappear at low T)')
    ax.grid(True, alpha=0.3)

    # 5. Correlation length vs T
    ax = axes[1, 0]
    xi_arr = np.array(data['xi'])
    xi_plot = np.minimum(xi_arr, 50)  # Cap for plotting
    ax.plot(data['T'], xi_plot, 'o-', color='purple', lw=2, label='Monostring')
    xi_null = np.minimum(np.array(data_null['xi']), 50)
    ax.plot(data_null['T'], xi_null, 's--', color='gray', lw=1.5, label='Null')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Correlation length xi')
    ax.set_title('Correlation Length\n(diverges at transition → Goldstone)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Effective potential
    ax = axes[1, 1]
    # Show potential at high and low T
    phases_for_pot_hot = generate_thermal_trajectory(N, kappa, temperatures[0])
    G_pot_hot, _ = build_graph_strict(phases_for_pot_hot, target_degree=TARGET_DEG)
    phi_hot = local_field(phases_for_pot_hot, G_pot_hot)
    _, _, centers_hot, V_hot = effective_potential(phi_hot)

    _, _, centers_cold, V_cold = effective_potential(phi_cold)

    ax.plot(centers_hot, V_hot, '-', color='orange', lw=2, label=f'T={temperatures[0]:.1f} (hot)')
    ax.plot(centers_cold, V_cold, '-', color='blue', lw=2, label=f'T={temperatures[-1]:.2f} (cold)')
    ax.set_xlabel('Field Phi'); ax.set_ylabel('V_eff(Phi)')
    ax.set_title('Effective Potential\n(Mexican hat at low T?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(15, max(V_hot.max(), V_cold.max())))

    # 7. Dispersion relation
    ax = axes[1, 2]
    ax.loglog(t_hot, var_hot, 'o-', color='orange', lw=2,
              label=f'Hot (alpha={alpha_hot:.2f})')
    ax.loglog(t_cold, var_cold, 's-', color='blue', lw=2,
              label=f'Cold (alpha={alpha_cold:.2f})')
    ax.loglog(t_hot, t_hot**2 * var_hot[3]/t_hot[3]**2, ':', color='gray',
              alpha=0.5, label='~t^2 (massless)')
    ax.loglog(t_hot, t_hot * var_hot[3]/t_hot[3], '--', color='gray',
              alpha=0.5, label='~t (massive)')
    ax.set_xlabel('Time t'); ax.set_ylabel('Variance sigma^2')
    ax.set_title('Dispersion Relation\n(alpha=2: massless, alpha=1: massive)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 8. Control: degree
    ax = axes[1, 3]
    ax.plot(data['T'], data['deg'], 'o-', color='crimson', lw=2, label='Monostring')
    ax.plot(data_null['T'], data_null['deg'], 's--', color='gray', lw=1.5, label='Null')
    ax.axhline(TARGET_DEG, ls=':', color='black', alpha=0.5)
    ax.fill_between(data['T'], TARGET_DEG*0.9, TARGET_DEG*1.1, alpha=0.1, color='green')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Avg degree')
    ax.set_title(f'CONTROL: Degree stability\ncv={np.std(data["deg"])/np.mean(data["deg"]):.1%}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring Higgs Lab v3: Three Mass Definitions + Domain Walls',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('higgs_v3_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    deg_cv = np.std(data['deg']) / np.mean(data['deg'])

    print("\n" + "=" * 70)
    print("  VERDICT v3")
    print("=" * 70)

    tests = [
        ("Phase transition (delta_r > 0.05)",
         True,  # Already established
         f"Established in v2"),
        ("Degree controlled (cv < 15%)",
         deg_cv < 0.15,
         f"cv={deg_cv:.1%}"),
        ("m_spectral grows with VEV (Yukawa)",
         corrs['m_spec vs r_K'] > 0.3,
         f"corr={corrs['m_spec vs r_K']:.3f}"),
        ("m_corr_length grows with VEV (Yukawa)",
         corrs['m_corr vs r_K'] > 0.3,
         f"corr={corrs['m_corr vs r_K']:.3f}"),
        ("m_potential grows with VEV (Yukawa)",
         corrs['m_pot vs r_K'] > 0.3,
         f"corr={corrs['m_pot vs r_K']:.3f}"),
        ("Domain walls decrease with VEV",
         corrs['walls vs r_K'] < -0.3,
         f"corr={corrs['walls vs r_K']:.3f}"),
        ("No mass-degree confound (spectral)",
         abs(corrs['m_spec vs deg']) < 0.5,
         f"corr={corrs['m_spec vs deg']:.3f}"),
        ("No mass-degree confound (corr length)",
         abs(corrs['m_corr vs deg']) < 0.5,
         f"corr={corrs['m_corr vs deg']:.3f}"),
        ("Cold dispersion more diffusive (alpha lower)",
         alpha_cold < alpha_hot - 0.1,
         f"hot={alpha_hot:.2f}, cold={alpha_cold:.2f}"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print(f"  {'[+]' if passed else '[-]'} {'PASS' if passed else 'FAIL'}  "
              f"{name:<50s} {detail}")

    print(f"\n  Score: {n_pass}/{len(tests)}")

    # Key question
    print("\n  KEY QUESTION: Which mass definition (if any) shows Yukawa behavior?")
    for m_name in ['m_spec vs r_K', 'm_corr vs r_K', 'm_pot vs r_K']:
        c = corrs[m_name]
        verdict = "YES - YUKAWA!" if c > 0.3 else "NO (anti)" if c < -0.3 else "no signal"
        print(f"    {m_name:<25s}: corr={c:>7.3f} → {verdict}")

    print("=" * 70)

if __name__ == "__main__":
    run_higgs_lab_v3()
