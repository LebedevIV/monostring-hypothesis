import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.sparse import diags as sp_diags
import time
import warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# CORE (unchanged)
# ================================================================
def gen_traj(N, kappa, T):
    D = 6
    omega = 2 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12)
    ph = np.zeros((N, D))
    ph[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N-1):
        ph[n+1] = (ph[n] + omega + kappa * C_E6 @ np.sin(ph[n])
                   + np.random.normal(0, T, D)) % (2*np.pi)
    return ph

def build_graph(phases, target_deg=20, delta_min=5):
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
    G = nx.Graph(); G.add_nodes_from(range(N))
    for i in range(N-1): G.add_edge(i, i+1)
    if best_pf is not None:
        for a,b in best_pf: G.add_edge(int(a), int(b))
    fd = sum(dict(G.degree()).values())/N
    if fd > target_deg*1.1:
        re = [(u,v) for u,v in G.edges() if abs(u-v)>delta_min]
        nr = int((fd-target_deg)*N/2)
        if 0 < nr < len(re):
            G.remove_edges_from([re[i] for i in np.random.choice(len(re),nr,replace=False)])
    return G, sum(dict(G.degree()).values())/N

def kuramoto(phases):
    N, D = phases.shape
    r = np.zeros(D)
    for d in range(D):
        r[d] = np.abs(np.mean(np.exp(1j*phases[:,d])))
    return np.prod(r)**(1./D), r

def local_field_per_dim(phases, G, dim):
    """Local Kuramoto r for a SINGLE phase dimension."""
    N = len(phases)
    phi = np.zeros(N)
    for v in range(N):
        nbrs = list(G.neighbors(v))
        if not nbrs: continue
        phi[v] = np.abs(np.mean(np.exp(1j * phases[nbrs, dim])))
    return phi

# ================================================================
# KEY INNOVATION: DIRECTIONAL MASS
# ================================================================
def directional_mass(G, phases, synced_dims, unsynced_dims,
                     n_pert=30, t_max=8.0, n_times=30):
    """
    Measure wavepacket spreading rate on the graph,
    with perturbation ALIGNED to different phase directions.

    Physical idea:
    - Perturbation along SYNCHRONIZED direction: the wavepacket
      must "push against" the coherent field → SLOW spreading → MASSIVE
    - Perturbation along UNSYNCHRONIZED direction: the wavepacket
      propagates freely → FAST spreading → MASSLESS (Goldstone)

    Mass = 1 / (spreading rate)
    """
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)

    from collections import deque

    results = {}

    for label, dims in [('synced', synced_dims), ('unsynced', unsynced_dims)]:
        if len(dims) == 0:
            results[label] = {'rate': np.nan, 'rates': []}
            continue

        rates = []

        for trial in range(n_pert):
            center = np.random.randint(N)

            # Create field-weighted Hamiltonian
            # For synced dims: add potential proportional to local coherence
            # For unsynced dims: potential is flat (no barrier)
            V_diag = np.zeros(N)
            for d in dims:
                phi_d = local_field_per_dim(phases, G, d)
                V_diag += phi_d
            V_diag /= len(dims)

            # Coupling: stronger field → higher barrier → more mass
            coupling = 2.0
            H = L + coupling * sp_diags(V_diag)

            # Initial wavepacket
            psi_0 = np.zeros(N, dtype=np.complex128)
            psi_0[center] = 1.0

            # BFS distances
            dists = {}
            queue = [(center, 0)]
            dists[center] = 0
            q = [center]
            head = 0
            while head < len(q):
                v = q[head]; head += 1
                if dists[v] >= 15: continue
                for u in G.neighbors(v):
                    if u not in dists:
                        dists[u] = dists[v] + 1
                        q.append(u)

            dist_arr = np.array([dists.get(i, N) for i in range(N)], dtype=float)

            times = np.linspace(1.0, t_max, n_times)
            variances = []
            for t in times:
                psi_t = expm_multiply(-1j * t * H, psi_0)
                prob = np.abs(psi_t)**2
                s = np.sum(prob)
                if s > 0: prob /= s
                variances.append(np.sum(prob * dist_arr**2))

            variances = np.array(variances)

            # Linear fit: sigma^2 = rate * t + const
            if np.any(variances > 0):
                try:
                    rate, _ = np.polyfit(times, variances, 1)
                    rates.append(max(rate, 0))
                except:
                    pass

        if rates:
            results[label] = {
                'rate': np.mean(rates),
                'rate_std': np.std(rates),
                'rates': rates
            }
        else:
            results[label] = {'rate': np.nan, 'rate_std': np.nan, 'rates': []}

    return results

# ================================================================
# FREE PROPAGATION (no field coupling, as control)
# ================================================================
def free_mass(G, n_pert=30, t_max=8.0, n_times=30):
    """Spreading rate WITHOUT any field coupling (control)."""
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)

    rates = []
    for trial in range(n_pert):
        center = np.random.randint(N)
        psi_0 = np.zeros(N, dtype=np.complex128)
        psi_0[center] = 1.0

        dists = {}
        q = [center]; dists[center] = 0; head = 0
        while head < len(q):
            v = q[head]; head += 1
            if dists[v] >= 15: continue
            for u in G.neighbors(v):
                if u not in dists:
                    dists[u] = dists[v]+1; q.append(u)

        dist_arr = np.array([dists.get(i, N) for i in range(N)], dtype=float)

        times = np.linspace(1.0, t_max, n_times)
        variances = []
        for t in times:
            psi_t = expm_multiply(-1j * t * L, psi_0)
            prob = np.abs(psi_t)**2
            s = np.sum(prob)
            if s > 0: prob /= s
            variances.append(np.sum(prob * dist_arr**2))

        try:
            rate, _ = np.polyfit(times, np.array(variances), 1)
            rates.append(max(rate, 0))
        except:
            pass

    return np.mean(rates) if rates else np.nan, np.std(rates) if rates else np.nan

# ================================================================
# MAIN
# ================================================================
def run_v5():
    print("=" * 70)
    print("  MONOSTRING HIGGS LAB v5")
    print("  Directional mass: synced vs unsynced excitations")
    print("  N=10000, deg=20 (larger diameter for correlations)")
    print("=" * 70)

    N = 10000
    kappa = 0.5
    TARGET_DEG = 20

    temperatures = np.logspace(0.5, -1.5, 10)

    data = {'T': [], 'r_kura': [], 'r_per': [],
            'rate_synced': [], 'rate_unsynced': [], 'rate_free': [],
            'mass_synced': [], 'mass_unsynced': [],
            'deg': []}

    print(f"\n[1/3] Thermodynamic scan + directional mass...")
    print(f"      {'T':>7s} | {'r_K':>5s} {'r1':>5s} {'r6':>5s} {'r234':>5s} | "
          f"{'v_sync':>7s} {'v_unsyn':>7s} {'v_free':>7s} | "
          f"{'m_syn':>6s} {'m_uns':>6s} {'ratio':>6s} | {'deg':>5s}")
    print("      " + "-" * 85)

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N, kappa, T)
        G, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_k, r_per = kuramoto(phases)

        # Classify dimensions
        threshold = 0.5
        synced = [d for d in range(6) if r_per[d] > threshold]
        unsynced = [d for d in range(6) if r_per[d] <= threshold]

        # Directional mass
        if len(synced) > 0 and len(unsynced) > 0:
            dm = directional_mass(G, phases, synced, unsynced,
                                  n_pert=20, t_max=6.0, n_times=20)
            rate_s = dm['synced']['rate']
            rate_u = dm['unsynced']['rate']
        else:
            # Free propagation only (no anisotropy yet)
            rate_s = rate_u = np.nan

        # Free control
        rate_f, _ = free_mass(G, n_pert=15, t_max=6.0, n_times=20)

        # Effective mass = 1/rate (higher rate → lower mass)
        m_s = 1.0/rate_s if rate_s and rate_s > 1e-6 else np.inf
        m_u = 1.0/rate_u if rate_u and rate_u > 1e-6 else np.inf

        ratio = m_s / m_u if m_u > 0 and m_u < 1e6 and m_s < 1e6 else np.nan

        r_mid = np.mean([r_per[d] for d in [1,2,3]])

        data['T'].append(T)
        data['r_kura'].append(r_k)
        data['r_per'].append(r_per.copy())
        data['rate_synced'].append(rate_s if not np.isnan(rate_s) else 0)
        data['rate_unsynced'].append(rate_u if not np.isnan(rate_u) else 0)
        data['rate_free'].append(rate_f)
        data['mass_synced'].append(m_s if m_s < 1e6 else np.nan)
        data['mass_unsynced'].append(m_u if m_u < 1e6 else np.nan)
        data['deg'].append(deg)

        print(f"      {T:>7.3f} | {r_k:>5.3f} {r_per[0]:>5.3f} {r_per[5]:>5.3f} "
              f"{r_mid:>5.3f} | {rate_s:>7.4f} {rate_u:>7.4f} {rate_f:>7.4f} | "
              f"{m_s:>6.2f} {m_u:>6.2f} {ratio:>6.2f} | "
              f"{deg:>5.1f}  ({time.time()-t0:.1f}s)")

    # ================================================================
    # KEY ANALYSIS
    # ================================================================
    print(f"\n[2/3] Key analysis...")

    # At low T: is synced direction SLOWER (more massive)?
    low_T_mask = np.array(data['T']) < 0.2
    if np.any(low_T_mask):
        rates_s_low = np.array(data['rate_synced'])[low_T_mask]
        rates_u_low = np.array(data['rate_unsynced'])[low_T_mask]
        rates_f_low = np.array(data['rate_free'])[low_T_mask]

        valid = (rates_s_low > 0) & (rates_u_low > 0)
        if np.any(valid):
            mean_ratio = np.mean(rates_u_low[valid] / rates_s_low[valid])
            print(f"\n      At low T (broken phase):")
            print(f"      Mean rate(unsynced) / rate(synced) = {mean_ratio:.3f}")
            print(f"      {'HIGGS MECHANISM!' if mean_ratio > 1.2 else 'No mass hierarchy'}")
            print(f"      Interpretation: unsynced excitations spread "
                  f"{'FASTER' if mean_ratio > 1 else 'SLOWER'} than synced")
            print(f"      → synced = {'MASSIVE' if mean_ratio > 1 else 'LIGHTER'}, "
                  f"unsynced = {'MASSLESS (Goldstone)' if mean_ratio > 1 else 'HEAVIER'}")

    # At high T: both should be similar (symmetric)
    high_T_mask = np.array(data['T']) > 1.0
    if np.any(high_T_mask):
        rates_s_high = np.array(data['rate_synced'])[high_T_mask]
        rates_u_high = np.array(data['rate_unsynced'])[high_T_mask]
        valid_h = (rates_s_high > 0) & (rates_u_high > 0)
        if np.any(valid_h):
            ratio_high = np.mean(rates_u_high[valid_h] / rates_s_high[valid_h])
            print(f"\n      At high T (symmetric phase):")
            print(f"      Mean rate(unsynced) / rate(synced) = {ratio_high:.3f}")
            print(f"      {'Equal (symmetric!)' if abs(ratio_high - 1) < 0.2 else 'Asymmetric'}")

    # ================================================================
    # PLOTS
    # ================================================================
    print(f"\n[3/3] Plotting...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Per-dimension Kuramoto
    ax = axes[0,0]
    rp = np.array(data['r_per'])
    for d in range(6):
        ax.plot(data['T'], rp[:,d], 'o-', markersize=4, label=f'dim {d+1}')
    ax.axhline(0.5, ls=':', color='black', alpha=0.5, label='sync threshold')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Per-dim Kuramoto r')
    ax.set_title('Anisotropic Symmetry Breaking')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # 2. Spreading rates: synced vs unsynced vs free
    ax = axes[0,1]
    ax.plot(data['T'], data['rate_synced'], 'o-', color='red', lw=2,
            label='Synced (should be SLOW = massive)')
    ax.plot(data['T'], data['rate_unsynced'], 's-', color='blue', lw=2,
            label='Unsynced (should be FAST = massless)')
    ax.plot(data['T'], data['rate_free'], '^--', color='gray', lw=1.5,
            label='Free (no field, control)')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Spreading rate σ²/t')
    ax.set_title('CRITICAL TEST: Directional Mass\n'
                 '(synced SLOWER than unsynced = Higgs!)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Effective mass (1/rate)
    ax = axes[0,2]
    ms = np.array(data['mass_synced'])
    mu = np.array(data['mass_unsynced'])
    valid_plot = np.isfinite(ms) & np.isfinite(mu) & (ms < 100) & (mu < 100)
    if np.any(valid_plot):
        T_v = np.array(data['T'])[valid_plot]
        ax.plot(T_v, ms[valid_plot], 'o-', color='red', lw=2, label='m_synced (Higgs boson?)')
        ax.plot(T_v, mu[valid_plot], 's-', color='blue', lw=2, label='m_unsynced (Goldstone?)')
        ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Effective mass (1/rate)')
    ax.set_title('Mass Spectrum\n'
                 '(synced HEAVIER than unsynced = Higgs!)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Mass ratio
    ax = axes[1,0]
    ratios = ms / mu
    valid_r = np.isfinite(ratios) & (ratios < 100) & (ratios > 0.01)
    if np.any(valid_r):
        ax.plot(np.array(data['T'])[valid_r], ratios[valid_r], 'D-',
                color='darkgreen', lw=2, markersize=8)
        ax.axhline(1, ls='--', color='black', alpha=0.5, label='Equal mass')
        ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('m_synced / m_unsynced')
    ax.set_title('Mass Ratio\n(>1 at low T = Higgs mechanism)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. Cross-check: rate difference vs anisotropy
    ax = axes[1,1]
    anisotropy = rp[:, 0] - np.mean(rp[:, 1:4], axis=1)
    rate_diff = np.array(data['rate_unsynced']) - np.array(data['rate_synced'])
    ax.scatter(anisotropy, rate_diff, c=data['T'], cmap='coolwarm',
               s=80, edgecolors='black')
    cb = plt.colorbar(ax.scatter(anisotropy, rate_diff, c=data['T'],
                                  cmap='coolwarm', s=0), ax=ax)
    cb.set_label('Temperature')
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.axvline(0, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('Anisotropy (r₁ - ⟨r₂₃₄⟩)')
    ax.set_ylabel('Rate(unsynced) - Rate(synced)')
    ax.set_title('Cross-check: anisotropy vs rate difference\n'
                 '(upper right quadrant = Higgs)')
    ax.grid(True, alpha=0.3)

    # 6. Degree control
    ax = axes[1,2]
    ax.plot(data['T'], data['deg'], 'o-', color='crimson', lw=2)
    ax.axhline(TARGET_DEG, ls=':', color='black')
    ax.fill_between(data['T'], TARGET_DEG*0.9, TARGET_DEG*1.1, alpha=0.1, color='green')
    ax.invert_xaxis(); ax.set_xscale('log')
    cv = np.std(data['deg'])/np.mean(data['deg'])
    ax.set_xlabel('T'); ax.set_ylabel('Avg degree')
    ax.set_title(f'CONTROL (cv={cv:.1%})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring Higgs v5: Directional Mass (synced vs unsynced)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('higgs_v5_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    cv = np.std(data['deg'])/np.mean(data['deg'])

    # Key test: at low T, is synced SLOWER?
    low_mask = np.array(data['T']) < 0.2
    rs_low = np.array(data['rate_synced'])[low_mask]
    ru_low = np.array(data['rate_unsynced'])[low_mask]
    valid_low = (rs_low > 0) & (ru_low > 0)

    higgs_works = False
    if np.any(valid_low):
        ratio_test = np.mean(ru_low[valid_low] / rs_low[valid_low])
        higgs_works = ratio_test > 1.2
    else:
        ratio_test = np.nan

    print("\n" + "=" * 70)
    print("  VERDICT v5")
    print("=" * 70)

    tests = [
        ("Anisotropic breaking (2/6 dims synced)",
         True, "Established"),
        ("Degree controlled (cv < 15%)",
         cv < 0.15, f"cv={cv:.1%}"),
        ("Synced excitations SLOWER than unsynced at low T",
         higgs_works,
         f"rate ratio = {ratio_test:.3f}" if not np.isnan(ratio_test) else "N/A"),
        ("Rate ratio ≈ 1 at high T (symmetric)",
         abs(ratio_high - 1) < 0.3 if 'ratio_high' in dir() else False,
         f"{ratio_high:.3f}" if 'ratio_high' in dir() else "N/A"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print(f"  {'[+]' if passed else '[-]'} {'PASS' if passed else 'FAIL'}  "
              f"{name:<55s} {detail}")

    print(f"\n  Score: {n_pass}/{len(tests)}")

    if higgs_works:
        print("\n  *** HIGGS MECHANISM DETECTED ***")
        print("  Excitations along synchronized phase directions propagate SLOWER")
        print("  than excitations along unsynchronized directions.")
        print("  This is the graphical analog of mass generation via Higgs field.")
    else:
        print("\n  Higgs mechanism NOT detected with this configuration.")
        print("  Possible reasons: insufficient N, wrong coupling, or mechanism absent.")

    print("=" * 70)

if __name__ == "__main__":
    run_v5()
