import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
import time
import warnings

warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

def generate_thermal_trajectory(N, kappa, temperature):
    """Generate phases with thermal noise (stochastic quantization-like)."""
    D = 6
    omega = 2 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)
    phases = np.zeros((N, D))
    phases[0] = np.random.uniform(0, 2 * np.pi, D)
    for n in range(N - 1):
        coupling = kappa * C_E6 @ np.sin(phases[n])
        noise = np.random.normal(0, temperature, D)
        phases[n + 1] = (phases[n] + omega + coupling + noise) % (2 * np.pi)
    return phases

def build_graph_fixed_degree(phases, target_degree=30, delta_min=5):
    """Build resonance graph with FIXED average degree (controls for density)."""
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)

    # Binary search for epsilon to achieve target degree
    lo, hi = 0.05, 5.0
    for _ in range(15):
        mid = (lo + hi) / 2
        sample = np.random.choice(N, min(300, N), replace=False)
        degs = [sum(1 for j in tree.query_ball_point(coords[i], mid)
                    if abs(j - i) > delta_min and j != i) for i in sample]
        avg = np.mean(degs)
        if avg < target_degree:
            lo = mid
        else:
            hi = mid
        if abs(avg - target_degree) / max(target_degree, 1) < 0.05:
            break

    eps = (lo + hi) / 2
    pairs = tree.query_pairs(r=eps, output_type='ndarray')
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > delta_min]

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N - 1):
        G.add_edge(i, i + 1)
    for a, b in pairs:
        G.add_edge(a, b)

    actual_deg = sum(dict(G.degree()).values()) / N
    return G, actual_deg

def build_null_graph(N, target_degree=30):
    """Null model: RGG on random T^6 with same average degree."""
    phases_random = np.random.uniform(0, 2 * np.pi, (N, 6))
    return build_graph_fixed_degree(phases_random, target_degree=target_degree)

def kuramoto_order_parameter(phases):
    """
    Correct order parameter for phase synchronization.
    r = 0: disordered (symmetric phase, Higgs VEV = 0)
    r = 1: synchronized (broken symmetry, Higgs VEV = v)
    """
    N, D = phases.shape
    r_per_dim = np.zeros(D)
    psi_per_dim = np.zeros(D)  # Mean phase angle
    for d in range(D):
        z = np.mean(np.exp(1j * phases[:, d]))
        r_per_dim[d] = np.abs(z)
        psi_per_dim[d] = np.angle(z)
    r_global = np.prod(r_per_dim) ** (1.0 / D)
    return r_global, r_per_dim

def higgs_field_on_graph(phases, adj_dict):
    """
    Define a scalar field on the graph via local phase coherence.
    Phi(node) = local Kuramoto order parameter in neighborhood.
    This is a genuine scalar field, not a global statistic.
    """
    N, D = phases.shape
    phi_field = np.zeros(N)

    for node in range(N):
        neighbors = list(adj_dict.get(node, []))
        if len(neighbors) == 0:
            phi_field[node] = 0
            continue
        local_phases = phases[neighbors]
        # Local order parameter
        r_local = 0
        for d in range(D):
            z = np.mean(np.exp(1j * local_phases[:, d]))
            r_local += np.abs(z)
        phi_field[node] = r_local / D

    return phi_field

def mexican_hat_potential(phi_field):
    """
    Check if the field distribution looks like a Mexican hat minimum.
    V(Φ) = -μ²|Φ|² + λ|Φ|⁴
    Minimum at |Φ| = v = μ/√(2λ)

    Test: is the distribution of Φ peaked AWAY from 0?
    """
    hist, bin_edges = np.histogram(phi_field, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the mode (peak of distribution)
    mode_idx = np.argmax(hist)
    mode_value = bin_centers[mode_idx]

    # Is the mode significantly away from 0?
    is_broken = mode_value > 0.3  # Arbitrary but reasonable threshold

    return mode_value, is_broken, bin_centers, hist

def compute_mass_gap(G):
    """Mass gap from Dirac operator (graph Laplacian)."""
    N = G.number_of_nodes()
    edges = [(i, j) for i, j in G.edges() if j > i]
    n_edges = len(edges)

    if n_edges < 10 or N > 10000:
        # Use normalized Laplacian for large graphs
        try:
            L = nx.normalized_laplacian_matrix(G)
            k = min(5, N - 2)
            evals, _ = eigsh(L, k=k, which='SM', tol=1e-3, maxiter=2000)
            evals = np.sort(evals)
            nonzero = evals[evals > 1e-4]
            return nonzero[0] if len(nonzero) > 0 else 0.0
        except:
            return 0.0

    inc = lil_matrix((N, n_edges), dtype=np.float64)
    for e_idx, (i, j) in enumerate(edges):
        inc[i, e_idx] = -1.0
        inc[j, e_idx] = +1.0
    inc = csc_matrix(inc)
    L0 = inc @ inc.T

    try:
        k_eig = min(5, N - 2)
        evals, _ = eigsh(L0, k=k_eig, which='SM', tol=1e-3, maxiter=2000)
        masses = np.sort(np.sqrt(np.abs(evals)))
        nonzero = masses[masses > 1e-4]
        return nonzero[0] if len(nonzero) > 0 else 0.0
    except:
        return 0.0

def quantum_walk_spread(G, t_max=10.0, steps=25):
    """Quantum walk: measure wavepacket variance over time."""
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
        var = np.sum(prob * dist_array ** 2)
        variances.append(var)
    return times, np.array(variances)

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_corrected_higgs_lab():
    print("╔" + "═" * 65 + "╗")
    print("║  CORRECTED HIGGS FIELD LABORATORY                            ║")
    print("║  Fixes: Kuramoto order parameter, fixed graph density,       ║")
    print("║         null model control, local scalar field               ║")
    print("╚" + "═" * 65 + "╝")

    N = 3000
    kappa = 0.5
    target_deg = 30
    temperatures = np.logspace(0.5, -2, 12)

    # Storage
    data_sbe = {'T': [], 'r_kuramoto': [], 'vev_local': [],
                'mass': [], 'clustering': [], 'avg_deg': []}
    data_null = {'T': [], 'r_kuramoto': [], 'vev_local': [],
                 'mass': [], 'clustering': [], 'avg_deg': []}

    print(f"\n[1/4] Phase transition scan (N={N}, target_deg={target_deg})")
    print(f"      {'T':>8s} | {'r_Kura':>7s} {'VEV_loc':>8s} {'mass':>7s} "
          f"{'clust':>7s} {'deg':>5s} | "
          f"{'r_null':>7s} {'VEV_n':>7s} {'m_null':>7s} {'c_null':>7s}")
    print("      " + "─" * 80)

    for T in temperatures:
        t0 = time.time()

        # === SBE MODEL ===
        phases = generate_thermal_trajectory(N, kappa, T)
        G_sbe, deg_sbe = build_graph_fixed_degree(phases, target_degree=target_deg)

        r_kura, _ = kuramoto_order_parameter(phases)
        adj_dict = {n: set(G_sbe.neighbors(n)) for n in G_sbe.nodes()}
        phi_field = higgs_field_on_graph(phases, adj_dict)
        vev_local = np.mean(phi_field)
        mass_sbe = compute_mass_gap(G_sbe)
        clust_sbe = nx.transitivity(G_sbe)

        data_sbe['T'].append(T)
        data_sbe['r_kuramoto'].append(r_kura)
        data_sbe['vev_local'].append(vev_local)
        data_sbe['mass'].append(mass_sbe)
        data_sbe['clustering'].append(clust_sbe)
        data_sbe['avg_deg'].append(deg_sbe)

        # === NULL MODEL ===
        G_null, deg_null = build_null_graph(N, target_degree=target_deg)
        phases_null = np.random.uniform(0, 2 * np.pi, (N, 6))
        r_null, _ = kuramoto_order_parameter(phases_null)
        adj_null = {n: set(G_null.neighbors(n)) for n in G_null.nodes()}
        phi_null = higgs_field_on_graph(phases_null, adj_null)
        vev_null = np.mean(phi_null)
        mass_null = compute_mass_gap(G_null)
        clust_null = nx.transitivity(G_null)

        data_null['T'].append(T)
        data_null['r_kuramoto'].append(r_null)
        data_null['vev_local'].append(vev_null)
        data_null['mass'].append(mass_null)
        data_null['clustering'].append(clust_null)
        data_null['avg_deg'].append(deg_null)

        print(f"      {T:>8.3f} | {r_kura:>7.4f} {vev_local:>8.4f} "
              f"{mass_sbe:>7.4f} {clust_sbe:>7.4f} {deg_sbe:>5.1f} | "
              f"{r_null:>7.4f} {vev_null:>7.4f} {mass_null:>7.4f} "
              f"{clust_null:>7.4f}  ({time.time() - t0:.1f}s)")

    # === PHASE TRANSITION CHECK ===
    print(f"\n[2/4] Checking for phase transition...")
    r_arr = np.array(data_sbe['r_kuramoto'])
    T_arr = np.array(data_sbe['T'])

    # Find steepest change in r_kuramoto
    dr = np.diff(r_arr)
    dT = np.diff(np.log(T_arr))
    slope = dr / dT
    i_max = np.argmax(np.abs(slope))
    T_critical = np.sqrt(T_arr[i_max] * T_arr[i_max + 1])

    print(f"      Steepest change in order parameter at T_c ≈ {T_critical:.3f}")
    print(f"      r_Kuramoto jumps from {r_arr[i_max]:.4f} to {r_arr[i_max + 1]:.4f}")
    is_transition = abs(r_arr[i_max + 1] - r_arr[i_max]) > 0.05
    print(f"      Phase transition detected: {'YES' if is_transition else 'NO'}")

    # === MEXICAN HAT CHECK ===
    print(f"\n[3/4] Mexican hat potential check (coldest temperature)...")
    phases_cold = generate_thermal_trajectory(N, kappa, temperatures[-1])
    G_cold, _ = build_graph_fixed_degree(phases_cold, target_degree=target_deg)
    adj_cold = {n: set(G_cold.neighbors(n)) for n in G_cold.nodes()}
    phi_cold = higgs_field_on_graph(phases_cold, adj_cold)
    mode_val, is_broken, bins_c, hist_c = mexican_hat_potential(phi_cold)
    print(f"      Field distribution mode at Φ = {mode_val:.3f}")
    print(f"      Symmetry broken (mode > 0.3): {'YES' if is_broken else 'NO'}")

    # === QUANTUM WALK COMPARISON ===
    print(f"\n[4/4] Quantum walk comparison (hot vs cold)...")
    phases_hot = generate_thermal_trajectory(N, kappa, temperatures[0])
    G_hot, _ = build_graph_fixed_degree(phases_hot, target_degree=target_deg)

    t_q, var_hot = quantum_walk_spread(G_hot, t_max=10.0, steps=20)
    _, var_cold = quantum_walk_spread(G_cold, t_max=10.0, steps=20)

    # Compare spreading rates
    if len(t_q) > 5:
        rate_hot = np.polyfit(t_q[2:10], var_hot[2:10], 1)[0]
        rate_cold = np.polyfit(t_q[2:10], var_cold[2:10], 1)[0]
        print(f"      Hot vacuum spreading rate: {rate_hot:.4f}")
        print(f"      Cold vacuum spreading rate: {rate_cold:.4f}")
        print(f"      Ratio (cold/hot): {rate_cold / rate_hot:.4f}")
        print(f"      Inertia effect: {'YES (cold slower)' if rate_cold < rate_hot else 'NO'}")

    # ================================================================
    # PLOTS
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Kuramoto order parameter vs T (SBE vs null)
    ax = axes[0, 0]
    ax.plot(data_sbe['T'], data_sbe['r_kuramoto'], 'o-', color='teal',
            lw=2.5, markersize=8, label='SBE (Monostring)')
    ax.plot(data_null['T'], data_null['r_kuramoto'], 's--', color='gray',
            lw=2, markersize=6, label='Null (random)')
    ax.axvline(T_critical, ls=':', color='red', alpha=0.7,
               label=f'T_c ≈ {T_critical:.2f}')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature T (cooling →)')
    ax.set_ylabel('Kuramoto order parameter r')
    ax.set_title('Exp 1: Spontaneous Symmetry Breaking\n'
                 '(Phase synchronization = Higgs VEV)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 2. Local VEV vs T
    ax = axes[0, 1]
    ax.plot(data_sbe['T'], data_sbe['vev_local'], 'o-', color='darkgreen',
            lw=2.5, markersize=8, label='SBE local ⟨Φ⟩')
    ax.plot(data_null['T'], data_null['vev_local'], 's--', color='gray',
            lw=2, markersize=6, label='Null local ⟨Φ⟩')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature T (cooling →)')
    ax.set_ylabel('Local field VEV ⟨Φ⟩')
    ax.set_title('Exp 1b: Local Higgs Field\n'
                 '(neighborhood phase coherence)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 3. Mass vs VEV (Yukawa test)
    ax = axes[0, 2]
    ax.plot(data_sbe['r_kuramoto'], data_sbe['mass'], 'o-', color='indigo',
            lw=2.5, markersize=8, label='SBE')
    ax.plot(data_null['r_kuramoto'], data_null['mass'], 's--', color='gray',
            lw=2, markersize=6, label='Null')
    ax.set_xlabel('Order parameter r (≈ Higgs VEV)')
    ax.set_ylabel('Mass gap Δm')
    ax.set_title('Exp 2: Yukawa Mechanism Test\n'
                 'Does mass grow with VEV? (m = y·⟨Φ⟩)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 4. Mexican hat distribution
    ax = axes[1, 0]
    ax.bar(bins_c, hist_c, width=bins_c[1] - bins_c[0],
           color='teal', alpha=0.7, edgecolor='white')
    ax.axvline(0, ls='--', color='red', alpha=0.7, label='Φ=0 (symmetric)')
    ax.axvline(mode_val, ls='-', color='green', lw=2,
               label=f'Mode={mode_val:.2f}')
    ax.set_xlabel('Local field Φ')
    ax.set_ylabel('Density')
    ax.set_title('Exp 3: Mexican Hat Check\n'
                 f'Symmetry {"BROKEN" if is_broken else "INTACT"} '
                 f'(mode at Φ={mode_val:.2f})')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 5. Quantum walk
    ax = axes[1, 1]
    ax.plot(t_q, var_hot, 'o-', color='orange', lw=2, label='Hot (symmetric)')
    ax.plot(t_q, var_cold, 's-', color='blue', lw=2, label='Cold (broken)')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Wavepacket variance σ²')
    ax.set_title('Exp 4: Quantum Inertia\n'
                 'Does broken symmetry slow propagation?')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    # 6. Control: average degree vs T
    ax = axes[1, 2]
    ax.plot(data_sbe['T'], data_sbe['avg_deg'], 'o-', color='crimson',
            lw=2, label='SBE avg degree')
    ax.plot(data_null['T'], data_null['avg_deg'], 's--', color='gray',
            lw=2, label='Null avg degree')
    ax.axhline(target_deg, ls=':', color='black', alpha=0.5,
               label=f'Target = {target_deg}')
    ax.invert_xaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Average degree')
    ax.set_title('CONTROL: Graph density is FIXED\n'
                 '(rules out trivial density effects)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

    plt.suptitle('Monostring Higgs Lab — Corrected with Null Model & Fixed Density',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('higgs_corrected.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "╔" + "═" * 65 + "╗")
    print("║  VERDICT                                                      ║")
    print("╠" + "═" * 65 + "╣")

    tests = [
        ("Phase transition in r_Kuramoto",
         is_transition,
         f"Δr = {abs(r_arr[i_max + 1] - r_arr[i_max]):.3f}"),
        ("Symmetry broken (Mexican hat mode > 0.3)",
         is_broken,
         f"mode = {mode_val:.3f}"),
        ("SBE r_Kuramoto > null r_Kuramoto (at low T)",
         data_sbe['r_kuramoto'][-1] > data_null['r_kuramoto'][-1] + 0.05,
         f"SBE={data_sbe['r_kuramoto'][-1]:.3f} vs null={data_null['r_kuramoto'][-1]:.3f}"),
        ("Mass grows with VEV (Yukawa: m ∝ ⟨Φ⟩)",
         np.corrcoef(data_sbe['r_kuramoto'], data_sbe['mass'])[0, 1] > 0.3,
         f"corr = {np.corrcoef(data_sbe['r_kuramoto'], data_sbe['mass'])[0, 1]:.3f}"),
        ("Graph density controlled (deg ≈ const)",
         np.std(data_sbe['avg_deg']) / np.mean(data_sbe['avg_deg']) < 0.15,
         f"deg = {np.mean(data_sbe['avg_deg']):.1f} ± {np.std(data_sbe['avg_deg']):.1f}"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        n_pass += int(passed)
        print(f"║  {status}  {name:<45s} {detail:>12s} ║")

    print("╠" + "═" * 65 + "╣")
    print(f"║  Score: {n_pass}/{len(tests)}"
          + " " * (65 - 15 - len(str(n_pass)) - len(str(len(tests)))) + "║")
    print("╚" + "═" * 65 + "╝")

if __name__ == "__main__":
    run_corrected_higgs_lab()
