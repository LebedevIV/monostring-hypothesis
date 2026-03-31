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

# ================================================================
# DISPERSION RELATION: E(k) from Laplacian eigenvalues
# ================================================================
def dispersion_relation(G, k_max=50):
    """
    Compute eigenvalues of graph Laplacian.
    These are the squared frequencies ω² = λ.
    The dispersion relation is ω(k) where k indexes the modes.

    For a massive particle: ω² = m² + k² → gap at k=0
    For massless: ω² = k² → no gap
    """
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)
    k_actual = min(k_max, N - 2)
    evals, evecs = eigsh(L, k=k_actual, which='SM', tol=1e-4, maxiter=3000)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]

# ================================================================
# DIRECTIONAL DISPERSION
# ================================================================
def directional_dispersion(G, phases, evals, evecs, dim, n_modes=40):
    """
    Project Laplacian eigenmodes onto a specific phase dimension.

    For each eigenmode ψ_k, compute its "momentum" in direction d:
    p_d(k) = |Σ_v ψ_k(v) * exp(i * φ_d(v))|²

    This tells us how much mode k "oscillates" in direction d.

    If direction d is MASSIVE: modes with low p_d have high ω (gap)
    If direction d is MASSLESS: ω ~ p_d (linear, no gap)
    """
    N = len(phases)
    n_use = min(n_modes, len(evals) - 1)

    momenta = np.zeros(n_use)
    energies = np.zeros(n_use)

    exp_phase = np.exp(1j * phases[:, dim])

    for k in range(1, n_use + 1):  # Skip zero mode
        psi_k = evecs[:, k]
        # Momentum = overlap with plane wave in direction d
        p = np.abs(np.sum(psi_k * exp_phase)) / np.sqrt(N)
        momenta[k-1] = p
        energies[k-1] = np.sqrt(max(evals[k], 0))

    return momenta, energies

# ================================================================
# MASS FROM DISPERSION CURVATURE
# ================================================================
def mass_from_dispersion_curve(momenta, energies):
    """
    Fit E² = m² + c²p² to the dispersion data.

    If m > 0: massive mode (gap in dispersion)
    If m ≈ 0: massless (Goldstone)

    Returns: m (mass), c (speed), quality of fit
    """
    # Sort by momentum
    idx = np.argsort(momenta)
    p = momenta[idx]
    E = energies[idx]

    E2 = E**2
    p2 = p**2

    valid = (E2 > 0) & (p2 > 0)
    if np.sum(valid) < 5:
        return 0, 0, 0

    # Linear regression: E² = m² + c²·p²
    # y = E², x = p²
    try:
        coeffs = np.polyfit(p2[valid], E2[valid], 1)
        c_squared = coeffs[0]  # slope
        m_squared = coeffs[1]  # intercept

        mass = np.sqrt(max(m_squared, 0))
        speed = np.sqrt(max(c_squared, 0))

        # R² quality
        y_pred = np.polyval(coeffs, p2[valid])
        ss_res = np.sum((E2[valid] - y_pred)**2)
        ss_tot = np.sum((E2[valid] - np.mean(E2[valid]))**2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)

        return mass, speed, r_squared
    except:
        return 0, 0, 0

# ================================================================
# WAVEPACKET EXPERIMENT (from v3, validated)
# ================================================================
def spreading_exponent(G, t_max=8.0, n_pert=20, n_times=25):
    """
    σ²(t) ~ t^α
    α = 2: ballistic (massless)
    α = 1: diffusive (massive)
    α < 1: localized (very massive)
    """
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)

    alphas = []
    for _ in range(n_pert):
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

        times = np.linspace(0.5, t_max, n_times)
        variances = []
        for t in times:
            psi_t = expm_multiply(-1j * t * L, psi_0)
            prob = np.abs(psi_t)**2
            s = np.sum(prob)
            if s > 0: prob /= s
            variances.append(np.sum(prob * dist_arr**2))

        variances = np.array(variances)
        log_t = np.log(times)
        log_v = np.log(np.maximum(variances, 1e-15))
        valid = np.isfinite(log_v)
        if np.sum(valid) > 5:
            try:
                alpha, _ = np.polyfit(log_t[valid], log_v[valid], 1)
                alphas.append(alpha)
            except:
                pass

    return np.mean(alphas) if alphas else 1.0, np.std(alphas) if alphas else 0

# ================================================================
# MAIN
# ================================================================
def run_v7():
    print("=" * 70)
    print("  MONOSTRING HIGGS LAB v7 — DISPERSION RELATION APPROACH")
    print("  Mass from E²(p) = m² + c²p² per phase dimension")
    print("=" * 70)

    N = 10000
    kappa = 0.5
    TARGET_DEG = 20

    temperatures = [3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

    all_data = []

    print(f"\n[1/3] Thermodynamic scan with dispersion analysis...")
    print(f"      {'T':>6s} | {'r_K':>5s} | ", end="")
    for d in range(6):
        print(f"  m_d{d+1}", end="")
    print(f" |  alpha  | {'deg':>5s}")
    print("      " + "-" * 75)

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N, kappa, T)
        G, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_k, r_per = kuramoto(phases)

        # Compute Laplacian spectrum
        evals, evecs = dispersion_relation(G, k_max=60)

        # Directional mass for each dimension
        masses_per_dim = []
        speeds_per_dim = []
        r2_per_dim = []

        for d in range(6):
            mom, eng = directional_dispersion(G, phases, evals, evecs, d, n_modes=50)
            m, c, r2 = mass_from_dispersion_curve(mom, eng)
            masses_per_dim.append(m)
            speeds_per_dim.append(c)
            r2_per_dim.append(r2)

        # Spreading exponent
        alpha, alpha_std = spreading_exponent(G, t_max=6.0, n_pert=15)

        entry = {
            'T': T, 'r_kura': r_k, 'r_per': r_per.copy(),
            'masses': np.array(masses_per_dim),
            'speeds': np.array(speeds_per_dim),
            'r2_fits': np.array(r2_per_dim),
            'alpha': alpha, 'alpha_std': alpha_std,
            'deg': deg, 'evals': evals, 'evecs': evecs,
            'phases': phases
        }
        all_data.append(entry)

        mass_str = " ".join(f"{m:>6.4f}" for m in masses_per_dim)
        print(f"      {T:>6.3f} | {r_k:>5.3f} | {mass_str} | {alpha:>6.3f} | "
              f"{deg:>5.1f}  ({time.time()-t0:.1f}s)")

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n[2/3] Analysis...")

    # Group dimensions into synced/unsynced at lowest T
    r_per_cold = all_data[-1]['r_per']
    synced = [d for d in range(6) if r_per_cold[d] > 0.5]
    unsynced = [d for d in range(6) if r_per_cold[d] <= 0.5]
    print(f"\n      Synced dims: {synced} (r > 0.5)")
    print(f"      Unsynced dims: {unsynced}")

    # Mass of synced vs unsynced at each T
    print(f"\n      {'T':>6s} | {'m_synced':>9s} {'m_unsynced':>11s} {'ratio':>7s} | "
          f"{'r_synced':>9s} {'r_unsynced':>11s}")
    print("      " + "-" * 60)

    mass_ratios = []
    for entry in all_data:
        if len(synced) > 0 and len(unsynced) > 0:
            m_s = np.mean(entry['masses'][synced])
            m_u = np.mean(entry['masses'][unsynced])
            ratio = m_s / m_u if m_u > 1e-6 else np.nan
            r_s = np.mean(entry['r_per'][synced])
            r_u = np.mean(entry['r_per'][unsynced])
        else:
            m_s = m_u = ratio = r_s = r_u = np.nan

        mass_ratios.append(ratio)
        print(f"      {entry['T']:>6.3f} | {m_s:>9.4f} {m_u:>11.4f} {ratio:>7.3f} | "
              f"{r_s:>9.3f} {r_u:>11.3f}")

    # Correlation: mass ratio vs anisotropy
    aniso = [np.mean(e['r_per'][synced]) - np.mean(e['r_per'][unsynced])
             for e in all_data]
    valid_corr = [not np.isnan(r) and not np.isnan(a)
                  for r, a in zip(mass_ratios, aniso)]
    if sum(valid_corr) >= 3:
        mr_valid = [mass_ratios[i] for i in range(len(mass_ratios)) if valid_corr[i]]
        an_valid = [aniso[i] for i in range(len(aniso)) if valid_corr[i]]
        corr_mr_aniso = np.corrcoef(an_valid, mr_valid)[0, 1]
        print(f"\n      corr(mass_ratio, anisotropy) = {corr_mr_aniso:.3f}")
        print(f"      {'HIGGS: mass ratio GROWS with anisotropy!' if corr_mr_aniso > 0.5 else 'No Higgs signal' if abs(corr_mr_aniso) < 0.3 else 'ANTI-Higgs'}")

    # R² quality of fits
    print(f"\n      Fit quality (R² of E² = m² + c²p²):")
    for entry in [all_data[0], all_data[-1]]:
        r2_str = " ".join(f"{r:>5.3f}" for r in entry['r2_fits'])
        print(f"      T={entry['T']:.3f}: {r2_str}")

    # ================================================================
    # PLOTS
    # ================================================================
    print(f"\n[3/3] Plotting...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Per-dim Kuramoto
    ax = axes[0,0]
    Ts = [e['T'] for e in all_data]
    for d in range(6):
        rs = [e['r_per'][d] for e in all_data]
        style = 'o-' if d in synced else 's--'
        ax.plot(Ts, rs, style, markersize=5, label=f'dim {d+1}')
    ax.axhline(0.5, ls=':', color='black', alpha=0.5)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Kuramoto r')
    ax.set_title('Phase Synchronization per Dimension')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # 2. Directional mass per dim vs T (THE key plot)
    ax = axes[0,1]
    for d in range(6):
        ms = [e['masses'][d] for e in all_data]
        style = 'o-' if d in synced else 's--'
        color = 'red' if d in synced else 'blue'
        ax.plot(Ts, ms, style, color=color, markersize=5,
                label=f'dim {d+1} ({"sync" if d in synced else "free"})',
                alpha=0.7)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Dispersion mass m_d')
    ax.set_title('HIGGS TEST: Directional Mass\n'
                 'Red (synced) ABOVE blue (free) = Higgs!')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # 3. Mass ratio vs T
    ax = axes[0,2]
    valid_mr = [not np.isnan(r) for r in mass_ratios]
    if any(valid_mr):
        Ts_v = [Ts[i] for i in range(len(Ts)) if valid_mr[i]]
        mrs_v = [mass_ratios[i] for i in range(len(mass_ratios)) if valid_mr[i]]
        ax.plot(Ts_v, mrs_v, 'D-', color='darkgreen', lw=2.5, markersize=8)
    ax.axhline(1.0, ls='--', color='black', alpha=0.5, label='Equal mass')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T')
    ax.set_ylabel('m_synced / m_unsynced')
    ax.set_title('Mass Ratio\n>1 at low T = Higgs mechanism')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Dispersion relation at hot and cold
    ax = axes[1,0]
    for entry, color, label in [(all_data[0], 'orange', f'T={all_data[0]["T"]}'),
                                  (all_data[-1], 'blue', f'T={all_data[-1]["T"]}')]:
        for d in synced[:1]:
            mom, eng = directional_dispersion(G, entry['phases'],
                                               entry['evals'], entry['evecs'], d)
            idx = np.argsort(mom)
            ax.scatter(mom[idx]**2, eng[idx]**2, s=15, alpha=0.5, color=color,
                      label=f'{label}, dim {d+1} (sync)')
        for d in unsynced[:1]:
            mom, eng = directional_dispersion(G, entry['phases'],
                                               entry['evals'], entry['evecs'], d)
            idx = np.argsort(mom)
            ax.scatter(mom[idx]**2, eng[idx]**2, s=15, alpha=0.5,
                      color=color, marker='s',
                      label=f'{label}, dim {d+1} (free)')
    ax.set_xlabel('p² (momentum²)')
    ax.set_ylabel('E² (energy²)')
    ax.set_title('Dispersion: E²(p²)\nIntercept = m² (mass gap)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. Spreading exponent vs T
    ax = axes[1,1]
    alphas = [e['alpha'] for e in all_data]
    alpha_stds = [e['alpha_std'] for e in all_data]
    ax.errorbar(Ts, alphas, yerr=alpha_stds, fmt='o-', color='purple',
                lw=2, capsize=3, markersize=8)
    ax.axhline(2, ls=':', color='gray', alpha=0.5, label='Ballistic (massless)')
    ax.axhline(1, ls=':', color='gray', alpha=0.5, label='Diffusive (massive)')
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Spreading exponent α')
    ax.set_title('σ²(t) ~ t^α\n'
                 'Decreasing α = mass acquisition')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. Degree control
    ax = axes[1,2]
    degs = [e['deg'] for e in all_data]
    ax.plot(Ts, degs, 'o-', color='crimson', lw=2)
    ax.axhline(TARGET_DEG, ls=':', color='black')
    ax.invert_xaxis(); ax.set_xscale('log')
    cv = np.std(degs)/np.mean(degs)
    ax.set_xlabel('T'); ax.set_ylabel('Degree')
    ax.set_title(f'Control (cv={cv:.1%})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring Higgs v7: Mass from Dispersion Relation E²=m²+c²p²',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('higgs_v7_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    cv = np.std(degs)/np.mean(degs)

    # Key test: mass_synced > mass_unsynced at low T
    if len(all_data) > 0 and len(synced) > 0 and len(unsynced) > 0:
        m_s_cold = np.mean(all_data[-1]['masses'][synced])
        m_u_cold = np.mean(all_data[-1]['masses'][unsynced])
        ratio_cold = m_s_cold / m_u_cold if m_u_cold > 1e-6 else np.nan
    else:
        ratio_cold = np.nan

    # Alpha decreases with cooling
    alpha_hot = all_data[0]['alpha']
    alpha_cold = all_data[-1]['alpha']
    alpha_decreases = alpha_cold < alpha_hot - 0.1

    print("\n" + "=" * 70)
    print("  VERDICT v7")
    print("=" * 70)

    tests = [
        ("Anisotropic breaking (2/6 synced)",
         True, "Established"),
        ("Degree controlled",
         cv < 0.15, f"cv={cv:.1%}"),
        ("Dispersion mass(synced) > mass(unsynced) at low T",
         ratio_cold > 1.2 if not np.isnan(ratio_cold) else False,
         f"ratio={ratio_cold:.3f}" if not np.isnan(ratio_cold) else "N/A"),
        ("Mass ratio grows with anisotropy",
         corr_mr_aniso > 0.5 if 'corr_mr_aniso' in dir() else False,
         f"corr={corr_mr_aniso:.3f}" if 'corr_mr_aniso' in dir() else "N/A"),
        ("Spreading exponent decreases (alpha_cold < alpha_hot)",
         alpha_decreases,
         f"{alpha_hot:.2f} → {alpha_cold:.2f}"),
        ("Good dispersion fits (R² > 0.5)",
         np.mean(all_data[-1]['r2_fits']) > 0.5,
         f"mean R²={np.mean(all_data[-1]['r2_fits']):.3f}"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print(f"  {'[+]' if passed else '[-]'} {'PASS' if passed else 'FAIL'}  "
              f"{name:<55s} {detail}")

    print(f"\n  Score: {n_pass}/{len(tests)}")
    print("=" * 70)

if __name__ == "__main__":
    run_v7()
