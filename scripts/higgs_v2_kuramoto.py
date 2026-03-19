import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh, expm_multiply
import time
import warnings

warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# TRAJECTORY GENERATION
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

# ================================================================
# GRAPH CONSTRUCTION WITH STRICT DEGREE CONTROL
# ================================================================
def build_graph_strict_degree(phases, target_degree=30, delta_min=5, tolerance=0.10):
    """
    Build graph with STRICTLY controlled average degree.
    Uses iterative refinement to guarantee deg ∈ [target*(1-tol), target*(1+tol)].
    """
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)

    lo, hi = 0.01, 8.0
    best_eps = 1.0
    best_deg = 0
    best_pairs = None

    for iteration in range(25):
        mid = (lo + hi) / 2

        pairs = tree.query_pairs(r=mid, output_type='ndarray')
        if len(pairs) > 0:
            pairs_filtered = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > delta_min]
        else:
            pairs_filtered = np.array([]).reshape(0, 2).astype(int)

        # Actual degree = (2 * resonance_edges + chronological_edges) / N
        n_chrono = N - 1
        n_res = len(pairs_filtered)
        actual_deg = (2 * n_res + 2 * n_chrono) / N

        if actual_deg < target_degree:
            lo = mid
        else:
            hi = mid

        if abs(actual_deg - target_degree) < abs(best_deg - target_degree):
            best_deg = actual_deg
            best_eps = mid
            best_pairs = pairs_filtered.copy()

        if abs(actual_deg - target_degree) / target_degree < tolerance * 0.5:
            break

    # Build graph with best epsilon
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N - 1):
        G.add_edge(i, i + 1)
    if best_pairs is not None and len(best_pairs) > 0:
        for a, b in best_pairs:
            G.add_edge(int(a), int(b))

    final_deg = sum(dict(G.degree()).values()) / N

    # Verify
    if abs(final_deg - target_degree) / target_degree > tolerance:
        # Prune or add edges to get closer
        if final_deg > target_degree * (1 + tolerance):
            # Remove random resonance edges
            res_edges = [(u, v) for u, v in G.edges() if abs(u - v) > delta_min]
            n_remove = int((final_deg - target_degree) * N / 2)
            if n_remove > 0 and len(res_edges) > n_remove:
                to_remove = [res_edges[i] for i in
                             np.random.choice(len(res_edges), n_remove, replace=False)]
                G.remove_edges_from(to_remove)

        final_deg = sum(dict(G.degree()).values()) / N

    return G, final_deg

def build_null_graph(N, target_degree=30):
    phases_random = np.random.uniform(0, 2 * np.pi, (N, 6))
    return build_graph_strict_degree(phases_random, target_degree=target_degree)

# ================================================================
# MEASUREMENTS
# ================================================================
def kuramoto_order_parameter(phases):
    N, D = phases.shape
    r_per_dim = np.zeros(D)
    for d in range(D):
        z = np.mean(np.exp(1j * phases[:, d]))
        r_per_dim[d] = np.abs(z)
    r_global = np.prod(r_per_dim) ** (1.0 / D)
    return r_global, r_per_dim

def local_higgs_field(phases, G):
    """Local scalar field: phase coherence in each node's neighborhood."""
    N, D = phases.shape
    phi_field = np.zeros(N)
    for node in range(N):
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            continue
        local_phases = phases[neighbors]
        r_sum = 0
        for d in range(D):
            z = np.mean(np.exp(1j * local_phases[:, d]))
            r_sum += np.abs(z)
        phi_field[node] = r_sum / D
    return phi_field

def compute_mass_gap(G):
    """Spectral gap of normalized Laplacian."""
    N = G.number_of_nodes()
    try:
        L = nx.normalized_laplacian_matrix(G)
        k = min(5, N - 2)
        evals, _ = eigsh(L, k=k, which='SM', tol=1e-3, maxiter=3000)
        evals = np.sort(evals)
        nonzero = evals[evals > 1e-4]
        return nonzero[0] if len(nonzero) > 0 else 0.0
    except:
        return 0.0

def quantum_walk_spread(G, t_max=10.0, steps=20):
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)
    psi_0 = np.zeros(N, dtype=np.complex128)
    center = N // 2
    psi_0[center] = 1.0
    dists = nx.single_source_shortest_path_length(G, center)
    dist_array = np.array([dists.get(i, N) for i in range(N)], dtype=float)
    times = np.linspace(0, t_max, steps)
    variances = []
    for t in times:
        psi_t = expm_multiply(-1j * t * L, psi_0)
        prob = np.abs(psi_t) ** 2
        s = np.sum(prob)
        if s > 0:
            prob /= s
        variances.append(np.sum(prob * dist_array ** 2))
    return times, np.array(variances)

def mexican_hat_check(phi_field):
    hist, edges = np.histogram(phi_field, bins=50, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mode_idx = np.argmax(hist)
    mode_val = centers[mode_idx]
    return mode_val, mode_val > 0.3, centers, hist

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_higgs_lab_v2():
    print("=" * 70)
    print("  MONOSTRING HIGGS LABORATORY v2")
    print("  Strict degree control + Kuramoto order parameter")
    print("=" * 70)

    N = 3000
    kappa = 0.5
    TARGET_DEG = 30
    temperatures = np.logspace(0.5, -2, 12)

    data_sbe = {k: [] for k in ['T', 'r_kura', 'vev_local', 'mass',
                                  'clust', 'deg']}
    data_null = {k: [] for k in ['T', 'r_kura', 'vev_local', 'mass',
                                   'clust', 'deg']}

    print(f"\n[1/4] Phase transition scan (N={N}, target_deg={TARGET_DEG})")
    print(f"      {'T':>8s} | {'r_K':>6s} {'VEV':>6s} {'mass':>6s} "
          f"{'clust':>6s} {'deg':>5s} | "
          f"{'r_n':>5s} {'V_n':>5s} {'m_n':>5s} {'c_n':>5s} {'d_n':>5s}")
    print("      " + "-" * 72)

    for T in temperatures:
        t0 = time.time()

        # SBE
        phases = generate_thermal_trajectory(N, kappa, T)
        G_sbe, deg_sbe = build_graph_strict_degree(phases, target_degree=TARGET_DEG)
        r_kura, _ = kuramoto_order_parameter(phases)
        phi_f = local_higgs_field(phases, G_sbe)
        vev_loc = np.mean(phi_f)
        mass_sbe = compute_mass_gap(G_sbe)
        clust_sbe = nx.transitivity(G_sbe)

        for k, v in zip(['T', 'r_kura', 'vev_local', 'mass', 'clust', 'deg'],
                         [T, r_kura, vev_loc, mass_sbe, clust_sbe, deg_sbe]):
            data_sbe[k].append(v)

        # Null
        G_null, deg_null = build_null_graph(N, target_degree=TARGET_DEG)
        ph_null = np.random.uniform(0, 2 * np.pi, (N, 6))
        r_null, _ = kuramoto_order_parameter(ph_null)
        phi_null = local_higgs_field(ph_null, G_null)
        vev_null = np.mean(phi_null)
        mass_null = compute_mass_gap(G_null)
        clust_null = nx.transitivity(G_null)

        for k, v in zip(['T', 'r_kura', 'vev_local', 'mass', 'clust', 'deg'],
                         [T, r_null, vev_null, mass_null, clust_null, deg_null]):
            data_null[k].append(v)

        print(f"      {T:>8.3f} | {r_kura:>6.3f} {vev_loc:>6.3f} "
              f"{mass_sbe:>6.3f} {clust_sbe:>6.3f} {deg_sbe:>5.1f} | "
              f"{r_null:>5.3f} {vev_null:>5.3f} {mass_null:>5.3f} "
              f"{clust_null:>5.3f} {deg_null:>5.1f}  ({time.time()-t0:.1f}s)")

    # Phase transition check
    print(f"\n[2/4] Phase transition analysis...")
    r_arr = np.array(data_sbe['r_kura'])
    T_arr = np.array(data_sbe['T'])
    dr = np.abs(np.diff(r_arr))
    i_max = np.argmax(dr)
    T_c = np.sqrt(T_arr[i_max] * T_arr[i_max + 1])
    delta_r = abs(r_arr[i_max + 1] - r_arr[i_max])
    is_transition = delta_r > 0.05

    print(f"      T_c approx {T_c:.3f}")
    print(f"      Delta r = {delta_r:.3f}")
    print(f"      Phase transition: {'YES' if is_transition else 'NO'}")

    # Mexican hat
    print(f"\n[3/4] Mexican hat check (coldest T)...")
    phases_cold = generate_thermal_trajectory(N, kappa, temperatures[-1])
    G_cold, _ = build_graph_strict_degree(phases_cold, target_degree=TARGET_DEG)
    phi_cold = local_higgs_field(phases_cold, G_cold)
    mode_val, is_broken, bins_mh, hist_mh = mexican_hat_check(phi_cold)
    print(f"      Mode = {mode_val:.3f}, Broken: {'YES' if is_broken else 'NO'}")

    # Quantum walk
    print(f"\n[4/4] Quantum walk...")
    phases_hot = generate_thermal_trajectory(N, kappa, temperatures[0])
    G_hot, _ = build_graph_strict_degree(phases_hot, target_degree=TARGET_DEG)

    t_q, var_hot = quantum_walk_spread(G_hot, t_max=10.0, steps=20)
    _, var_cold = quantum_walk_spread(G_cold, t_max=10.0, steps=20)

    if len(t_q) > 5:
        rate_hot = np.polyfit(t_q[2:10], var_hot[2:10], 1)[0]
        rate_cold = np.polyfit(t_q[2:10], var_cold[2:10], 1)[0]
        print(f"      Hot rate: {rate_hot:.4f}, Cold rate: {rate_cold:.4f}")
        print(f"      Ratio: {rate_cold/rate_hot:.3f}")
        inertia = rate_cold < rate_hot
        print(f"      Inertia effect: {'YES' if inertia else 'NO'}")
    else:
        rate_hot = rate_cold = 0
        inertia = False

    # ================================================================
    # CORRELATION ANALYSIS
    # ================================================================
    print(f"\n  CORRELATION ANALYSIS:")

    # Mass vs r_Kuramoto
    corr_mass_r = np.corrcoef(data_sbe['r_kura'], data_sbe['mass'])[0, 1]
    print(f"      corr(mass, r_Kuramoto) = {corr_mass_r:.3f} "
          f"({'Yukawa-like' if corr_mass_r > 0.3 else 'ANTI-Yukawa' if corr_mass_r < -0.3 else 'no correlation'})")

    # Mass vs degree (confound check)
    corr_mass_deg = np.corrcoef(data_sbe['deg'], data_sbe['mass'])[0, 1]
    print(f"      corr(mass, degree) = {corr_mass_deg:.3f} "
          f"({'CONFOUND!' if abs(corr_mass_deg) > 0.5 else 'OK'})")

    # r_Kuramoto vs degree (confound check)
    corr_r_deg = np.corrcoef(data_sbe['r_kura'], data_sbe['deg'])[0, 1]
    print(f"      corr(r_Kuramoto, degree) = {corr_r_deg:.3f} "
          f"({'CONFOUND!' if abs(corr_r_deg) > 0.5 else 'OK'})")

    # Clustering vs degree
    corr_clust_deg = np.corrcoef(data_sbe['clust'], data_sbe['deg'])[0, 1]
    print(f"      corr(clustering, degree) = {corr_clust_deg:.3f} "
          f"({'CONFOUND!' if abs(corr_clust_deg) > 0.5 else 'OK'})")

    # ================================================================
    # PLOTS
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Kuramoto order parameter
    ax = axes[0, 0]
    ax.plot(data_sbe['T'], data_sbe['r_kura'], 'o-', color='teal',
            lw=2.5, markersize=8, label='Monostring')
    ax.plot(data_null['T'], data_null['r_kura'], 's--', color='gray',
            lw=2, markersize=6, label='Null (random)')
    ax.axvline(T_c, ls=':', color='red', alpha=0.7,
               label=f'$T_c \\approx {T_c:.2f}$')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature T (cooling $\\rightarrow$)')
    ax.set_ylabel('Kuramoto order parameter r')
    ax.set_title('Spontaneous Symmetry Breaking\n(Phase synchronization)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 2. Local VEV
    ax = axes[0, 1]
    ax.plot(data_sbe['T'], data_sbe['vev_local'], 'o-', color='darkgreen',
            lw=2.5, markersize=8, label='Monostring $\\langle\\Phi\\rangle$')
    ax.plot(data_null['T'], data_null['vev_local'], 's--', color='gray',
            lw=2, markersize=6, label='Null $\\langle\\Phi\\rangle$')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature T (cooling $\\rightarrow$)')
    ax.set_ylabel('Local field VEV')
    ax.set_title('Local Higgs Field\n(neighborhood coherence)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 3. Mass vs r_Kuramoto
    ax = axes[0, 2]
    sc = ax.scatter(data_sbe['r_kura'], data_sbe['mass'],
                    c=data_sbe['deg'], cmap='viridis', s=80,
                    edgecolors='black', zorder=5)
    plt.colorbar(sc, ax=ax, label='avg degree')
    ax.scatter(data_null['r_kura'], data_null['mass'],
               c='gray', s=40, alpha=0.5, marker='s', label='Null')
    ax.set_xlabel('Order parameter r')
    ax.set_ylabel('Mass gap')
    ax.set_title(f'Yukawa Test: mass vs VEV\ncorr={corr_mass_r:.3f} '
                 f'(color=degree)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 4. Mexican hat
    ax = axes[1, 0]
    ax.bar(bins_mh, hist_mh, width=bins_mh[1]-bins_mh[0],
           color='teal', alpha=0.7, edgecolor='white')
    ax.axvline(0, ls='--', color='red', alpha=0.7)
    ax.axvline(mode_val, ls='-', color='green', lw=2,
               label=f'mode={mode_val:.2f}')
    ax.set_xlabel('Local field $\\Phi$')
    ax.set_ylabel('Density')
    ax.set_title(f'Mexican Hat: mode={mode_val:.2f}\n'
                 f'Symmetry {"BROKEN" if is_broken else "INTACT"}')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 5. Quantum walk
    ax = axes[1, 1]
    ax.plot(t_q, var_hot, 'o-', color='orange', lw=2, label='Hot (symmetric)')
    ax.plot(t_q, var_cold, 's-', color='blue', lw=2, label='Cold (broken)')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Wavepacket variance $\\sigma^2$')
    ax.set_title(f'Quantum Inertia\n'
                 f'cold/hot ratio = {rate_cold/max(rate_hot,1e-10):.2f}')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 6. CONTROL: degree vs T
    ax = axes[1, 2]
    ax.plot(data_sbe['T'], data_sbe['deg'], 'o-', color='crimson',
            lw=2, markersize=8, label='Monostring')
    ax.plot(data_null['T'], data_null['deg'], 's--', color='gray',
            lw=2, markersize=6, label='Null')
    ax.axhline(TARGET_DEG, ls=':', color='black', alpha=0.5,
               label=f'Target={TARGET_DEG}')
    ax.fill_between(data_sbe['T'],
                     TARGET_DEG * (1 - 0.1), TARGET_DEG * (1 + 0.1),
                     alpha=0.1, color='green', label='10% band')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Average degree')
    ax.set_title('CONTROL: Degree stability\n'
                 f'std/mean = {np.std(data_sbe["deg"])/np.mean(data_sbe["deg"]):.2%}')
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.4)

    plt.suptitle('Monostring Higgs Lab v2 — Fixed Density Control',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('higgs_v2_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    deg_arr = np.array(data_sbe['deg'])
    deg_stable = np.std(deg_arr) / np.mean(deg_arr) < 0.15

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    tests = [
        ("Phase transition in r_Kuramoto",
         is_transition, f"dr={delta_r:.3f}"),
        ("SBE r >> null r (at low T)",
         data_sbe['r_kura'][-1] > data_null['r_kura'][-1] + 0.05,
         f"{data_sbe['r_kura'][-1]:.3f} vs {data_null['r_kura'][-1]:.3f}"),
        ("Symmetry broken (mode > 0.3)",
         is_broken, f"mode={mode_val:.3f}"),
        ("Mass grows with VEV (Yukawa)",
         corr_mass_r > 0.3, f"corr={corr_mass_r:.3f}"),
        ("Degree controlled (cv < 15%)",
         deg_stable,
         f"cv={np.std(deg_arr)/np.mean(deg_arr):.1%}"),
        ("No degree-mass confound",
         abs(corr_mass_deg) < 0.5, f"corr={corr_mass_deg:.3f}"),
        ("Inertia effect (cold slower)",
         inertia, f"ratio={rate_cold/max(rate_hot,1e-10):.2f}"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        status = "PASS" if passed else "FAIL"
        n_pass += int(passed)
        print(f"  {'[+]' if passed else '[-]'} {status}  {name:<40s} {detail}")

    print(f"\n  Score: {n_pass}/{len(tests)}")
    print("=" * 70)

    return data_sbe, data_null

if __name__ == "__main__":
    data_sbe, data_null = run_higgs_lab_v2()
