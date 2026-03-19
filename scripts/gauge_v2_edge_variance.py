import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, diags as sp_diags
from scipy.sparse.linalg import eigsh
from collections import deque
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
    r = np.zeros(6)
    for d in range(6):
        r[d] = np.abs(np.mean(np.exp(1j*phases[:,d])))
    return r

def torus_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2*np.pi) - np.pi

# ================================================================
# EXPERIMENT 1: PLAQUETTE — FIXED (use F² instead of 1-cos(F))
# ================================================================
def plaquette_action(G, phases):
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    triangles = []
    sampled = np.random.choice(list(G.nodes()), min(2000, G.number_of_nodes()),
                                replace=False)
    for u in sampled:
        for v in adj[u]:
            if v <= u: continue
            for w in adj[u] & adj[v]:
                if w <= v: continue
                triangles.append((u, v, w))

    if not triangles:
        return np.zeros(6), np.zeros(6), 0

    n_tri = len(triangles)
    F = np.zeros((n_tri, 6))

    for idx, (u, v, w) in enumerate(triangles):
        for d in range(6):
            F[idx, d] = (torus_diff(phases[v,d], phases[u,d]) +
                         torus_diff(phases[w,d], phases[v,d]) +
                         torus_diff(phases[u,d], phases[w,d]))

    # Use F² (variance of flux) — much more sensitive than 1-cos(F)
    F_squared = np.mean(F**2, axis=0)

    # Also compute 1-cos for comparison
    S_cos = 1.0 - np.mean(np.cos(F), axis=0)

    return F_squared, S_cos, n_tri, triangles

# ================================================================
# EXPERIMENT 2: EDGE-BASED GAUGE FIELD VARIANCE
# ================================================================
def edge_gauge_variance(G, phases):
    """
    Simpler and more direct: measure variance of phase differences
    along edges for each dimension.

    For synced dim: Δφ ≈ 0 → Var(Δφ) small → massive (condensed)
    For unsynced dim: Δφ random → Var(Δφ) large → massless (fluctuating)
    """
    D = 6
    edge_diffs = {d: [] for d in range(D)}

    for u, v in G.edges():
        for d in range(D):
            dp = torus_diff(phases[v, d], phases[u, d])
            edge_diffs[d].append(dp)

    variance = np.zeros(D)
    mean_abs = np.zeros(D)
    for d in range(D):
        diffs = np.array(edge_diffs[d])
        variance[d] = np.var(diffs)
        mean_abs[d] = np.mean(np.abs(diffs))

    return variance, mean_abs

# ================================================================
# EXPERIMENT 3: GAUGE BOSON SPECTRUM — FIXED
# ================================================================
def gauge_boson_spectrum(G, phases, dim, k=10):
    N = G.number_of_nodes()
    edges = [(u, v) for u, v in G.edges() if u < v]
    n_edges = len(edges)

    if n_edges < 10:
        return np.array([])

    # Incidence matrix: N x n_edges
    inc = lil_matrix((N, n_edges), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        inc[u, idx] = -1.0
        inc[v, idx] = +1.0
    inc = csc_matrix(inc)

    # Gauge weights on edges: cos(Δφ_d)
    gauge_weights = np.zeros(n_edges)
    for idx, (u, v) in enumerate(edges):
        dphi = torus_diff(phases[v, dim], phases[u, dim])
        gauge_weights[idx] = np.cos(dphi)

    # FIX: Diagonal matrix of size n_edges x n_edges
    W_edges = sp_diags(gauge_weights)  # n_edges x n_edges

    # Gauge Laplacian on VERTICES: inc @ W_edges @ inc^T → N x N
    L_gauge = inc @ W_edges @ inc.T

    k_actual = min(k, N - 2)
    try:
        evals, _ = eigsh(L_gauge, k=k_actual, which='SM',
                          tol=1e-3, maxiter=3000)
        return np.sort(evals)
    except:
        return np.array([])

# ================================================================
# EXPERIMENT 4: WILSON LOOPS — FIXED (use actual phase sum)
# ================================================================
def wilson_loops(G, phases, max_size=7, n_samples=300):
    N = G.number_of_nodes()
    results = {s: {d: [] for d in range(6)} for s in range(3, max_size+1)}
    sources = np.random.choice(N, min(n_samples, N), replace=False)

    for src in sources:
        parent = {src: None}
        depth = {src: 0}
        queue = deque([src])

        while queue:
            v = queue.popleft()
            if depth[v] >= max_size // 2:
                continue
            for u in G.neighbors(v):
                if u not in parent:
                    parent[u] = v
                    depth[u] = depth[v] + 1
                    queue.append(u)
                elif u != parent[v] and depth[u] + depth[v] + 1 <= max_size:
                    loop_size = depth[u] + depth[v] + 1
                    if loop_size < 3 or loop_size > max_size:
                        continue

                    # Reconstruct loop
                    path_v = []
                    node = v
                    while node is not None:
                        path_v.append(node)
                        node = parent.get(node)

                    path_u = []
                    node = u
                    while node is not None:
                        path_u.append(node)
                        node = parent.get(node)

                    loop = path_v + path_u[::-1][1:]
                    if len(loop) < 3 or len(loop) > max_size:
                        continue

                    for d in range(6):
                        phase_sum = sum(
                            torus_diff(phases[loop[(i+1)%len(loop)], d],
                                       phases[loop[i], d])
                            for i in range(len(loop))
                        )
                        results[len(loop)][d].append(np.abs(np.exp(1j * phase_sum)))

    W_avg = {}
    for s in range(3, max_size+1):
        W_avg[s] = np.array([np.mean(results[s][d]) if results[s][d] else np.nan
                              for d in range(6)])
    return W_avg

# ================================================================
# EXPERIMENT 5: TOPOLOGICAL CHARGE
# ================================================================
def topological_charge(triangles, phases):
    Q = np.zeros(6)
    for u, v, w in triangles:
        for d in range(6):
            F = (torus_diff(phases[v,d], phases[u,d]) +
                 torus_diff(phases[w,d], phases[v,d]) +
                 torus_diff(phases[u,d], phases[w,d]))
            Q[d] += F
    Q /= (2 * np.pi)
    return Q

# ================================================================
# EXPERIMENT 6: CURVATURE TENSOR
# ================================================================
def curvature_tensor(G, phases):
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    triangles = []
    sampled = np.random.choice(list(G.nodes()), min(1000, G.number_of_nodes()),
                                replace=False)
    for u in sampled:
        for v in adj[u]:
            if v <= u: continue
            for w in adj[u] & adj[v]:
                if w <= v: continue
                triangles.append((u, v, w))

    if not triangles:
        return np.zeros((6, 6))

    F = np.zeros((6, 6))
    for u, v, w in triangles:
        for d1 in range(6):
            dp1_uv = torus_diff(phases[v,d1], phases[u,d1])
            dp1_vw = torus_diff(phases[w,d1], phases[v,d1])
            for d2 in range(d1+1, 6):
                dp2_uv = torus_diff(phases[v,d2], phases[u,d2])
                dp2_vw = torus_diff(phases[w,d2], phases[v,d2])
                val = dp1_uv * dp2_vw - dp2_uv * dp1_vw
                F[d1, d2] += val
                F[d2, d1] -= val

    F /= len(triangles)
    return F

# ================================================================
# MAIN
# ================================================================
def run_gauge_v2():
    print("=" * 70)
    print("  MONOSTRING GAUGE FIELD LAB v2 (FIXED)")
    print("  Fixes: F² metric, gauge Laplacian dimensions, edge variance")
    print("=" * 70)

    N = 8000
    kappa = 0.5
    TARGET_DEG = 25
    temperatures = [3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

    all_data = []

    # ============================================================
    # EXPERIMENT 1 + 2: Plaquette + Edge variance
    # ============================================================
    print(f"\n[1/5] Plaquette flux + Edge gauge variance...")
    print(f"      <F²_d>: flux variance per dim (smaller = more condensed)")
    print(f"      Var(Δφ_d): edge diff variance (smaller = more synced)")

    print(f"\n      {'T':>6s} |", end="")
    for d in range(6):
        print(f" F²_d{d+1}", end="")
    print(f" |", end="")
    for d in range(6):
        print(f" V_d{d+1}", end="")
    print(f" | {'#tri':>6s}")
    print("      " + "-" * 100)

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N, kappa, T)
        G, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_per = kuramoto(phases)

        F_sq, S_cos, n_tri, triangles = plaquette_action(G, phases)
        edge_var, edge_abs = edge_gauge_variance(G, phases)

        entry = {'T': T, 'r_per': r_per.copy(), 'F_sq': F_sq.copy(),
                 'S_cos': S_cos.copy(), 'edge_var': edge_var.copy(),
                 'edge_abs': edge_abs.copy(), 'n_tri': n_tri,
                 'triangles': triangles, 'phases': phases, 'G': G, 'deg': deg}
        all_data.append(entry)

        Fstr = " ".join(f"{f:>5.3f}" for f in F_sq)
        Vstr = " ".join(f"{v:>5.3f}" for v in edge_var)
        print(f"      {T:>6.3f} | {Fstr} | {Vstr} | "
              f"{n_tri:>6d}  ({time.time()-t0:.1f}s)")

    # Classify
    r_cold = all_data[-1]['r_per']
    synced = [d for d in range(6) if r_cold[d] > 0.5]
    unsynced = [d for d in range(6) if r_cold[d] <= 0.5]
    print(f"\n      Synced: {synced}, Unsynced: {unsynced}")

    # ============================================================
    # ANALYSIS: Flux ratio and Edge variance ratio
    # ============================================================
    print(f"\n      GAUGE CONDENSATION ANALYSIS:")
    print(f"      {'T':>6s} | {'<F²>_s':>8s} {'<F²>_u':>8s} {'ratio':>7s} | "
          f"{'Var_s':>7s} {'Var_u':>7s} {'ratio':>7s}")
    print("      " + "-" * 60)

    flux_ratios = []
    var_ratios = []

    for entry in all_data:
        if len(synced) > 0 and len(unsynced) > 0:
            Fs = np.mean(entry['F_sq'][synced])
            Fu = np.mean(entry['F_sq'][unsynced])
            fr = Fu / Fs if Fs > 1e-10 else np.nan

            Vs = np.mean(entry['edge_var'][synced])
            Vu = np.mean(entry['edge_var'][unsynced])
            vr = Vu / Vs if Vs > 1e-10 else np.nan
        else:
            Fs = Fu = fr = Vs = Vu = vr = np.nan

        flux_ratios.append(fr)
        var_ratios.append(vr)

        print(f"      {entry['T']:>6.3f} | {Fs:>8.5f} {Fu:>8.5f} {fr:>7.2f} | "
              f"{Vs:>7.5f} {Vu:>7.5f} {vr:>7.2f}")

    # Correlations
    anisos = [np.mean(e['r_per'][synced]) - np.mean(e['r_per'][unsynced])
              if len(synced)>0 and len(unsynced)>0 else 0 for e in all_data]

    valid_f = [not np.isnan(r) for r in flux_ratios]
    valid_v = [not np.isnan(r) for r in var_ratios]

    if sum(valid_f) >= 3:
        fv = [flux_ratios[i] for i in range(len(flux_ratios)) if valid_f[i]]
        av = [anisos[i] for i in range(len(anisos)) if valid_f[i]]
        corr_flux = np.corrcoef(av, fv)[0, 1]
        print(f"\n      corr(flux_ratio, anisotropy) = {corr_flux:.3f}")
    else:
        corr_flux = 0

    if sum(valid_v) >= 3:
        vv = [var_ratios[i] for i in range(len(var_ratios)) if valid_v[i]]
        av2 = [anisos[i] for i in range(len(anisos)) if valid_v[i]]
        corr_var = np.corrcoef(av2, vv)[0, 1]
        print(f"      corr(var_ratio, anisotropy) = {corr_var:.3f}")
    else:
        corr_var = 0

    # ============================================================
    # EXPERIMENT 3: GAUGE BOSON SPECTRUM (FIXED)
    # ============================================================
    print(f"\n[2/5] Gauge boson spectrum (fixed dimensions)...")

    entry_cold = all_data[-1]
    gauge_gaps = np.zeros(6)

    print(f"      {'Dim':>5s} {'Type':>5s} | First 5 eigenvalues")
    print("      " + "-" * 50)

    for d in range(6):
        evals = gauge_boson_spectrum(entry_cold['G'], entry_cold['phases'], d, k=8)
        if len(evals) > 0:
            nonzero = evals[np.abs(evals) > 1e-4]
            gauge_gaps[d] = np.min(np.abs(nonzero)) if len(nonzero) > 0 else 0
            ev_str = " ".join(f"{e:>8.4f}" for e in evals[:5])
            label = "SYNC" if d in synced else "FREE"
            print(f"      d{d+1:>3d} {label:>5s} | {ev_str}")

    if len(synced) > 0 and len(unsynced) > 0:
        gs = np.mean(gauge_gaps[synced])
        gu = np.mean(gauge_gaps[unsynced])
        print(f"\n      Gauge gap synced: {gs:.4f}")
        print(f"      Gauge gap unsynced: {gu:.4f}")
        if gu > 1e-6:
            print(f"      Ratio sync/unsync: {gs/gu:.3f}")

    # ============================================================
    # EXPERIMENT 4: WILSON LOOPS
    # ============================================================
    print(f"\n[3/5] Wilson loops...")

    for label, idx in [("HOT", 0), ("COLD", -1)]:
        entry = all_data[idx]
        W_avg = wilson_loops(entry['G'], entry['phases'], max_size=6)

        print(f"\n      {label} (T={entry['T']:.3f}):")
        for s in sorted(W_avg.keys()):
            W_s = np.mean(W_avg[s][synced]) if len(synced) > 0 else np.nan
            W_u = np.mean(W_avg[s][unsynced]) if len(unsynced) > 0 else np.nan
            print(f"      Size {s}: W_sync={W_s:.4f}, W_unsync={W_u:.4f}, "
                  f"diff={W_s-W_u:.5f}")

    # ============================================================
    # EXPERIMENT 5: TOPOLOGICAL CHARGE
    # ============================================================
    print(f"\n[4/5] Topological charge...")

    for label, idx in [("HOT", 0), ("COLD", -1)]:
        entry = all_data[idx]
        if entry['triangles']:
            Q = topological_charge(entry['triangles'], entry['phases'])
            print(f"      {label}: Q = [{' '.join(f'{q:>7.2f}' for q in Q)}]")

    # ============================================================
    # EXPERIMENT 6: CURVATURE TENSOR
    # ============================================================
    print(f"\n[5/5] Curvature tensor F_{{d1,d2}}...")

    F_cold = curvature_tensor(all_data[-1]['G'], all_data[-1]['phases'])
    print(f"\n      COLD (T={all_data[-1]['T']:.3f}):")
    for d1 in range(6):
        row = " ".join(f"{F_cold[d1,d2]:>7.4f}" for d2 in range(6))
        print(f"      [{row}]")

    if len(synced) >= 2 and len(unsynced) >= 2:
        F_ss = np.sqrt(np.sum(F_cold[np.ix_(synced, synced)]**2))
        F_uu = np.sqrt(np.sum(F_cold[np.ix_(unsynced, unsynced)]**2))
        F_su = np.sqrt(np.sum(F_cold[np.ix_(synced, unsynced)]**2))
        print(f"\n      |F_sync-sync|  = {F_ss:.4f}")
        print(f"      |F_unsync-unsync| = {F_uu:.4f}")
        print(f"      |F_sync-unsync|  = {F_su:.4f}")

    # ================================================================
    # PLOTS
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    Ts = [e['T'] for e in all_data]

    # 1. Flux variance per dim
    ax = axes[0, 0]
    for d in range(6):
        vals = [e['F_sq'][d] for e in all_data]
        c = 'red' if d in synced else 'blue'
        s = 'o-' if d in synced else 's--'
        ax.plot(Ts, vals, s, color=c, markersize=5,
                label=f'd{d+1} ({"S" if d in synced else "F"})', alpha=0.7)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel(r'$\langle F^2_d \rangle$')
    ax.set_title('Plaquette Flux Variance\n'
                 'Low = condensed (massive), High = fluctuating (massless)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # 2. Edge variance per dim
    ax = axes[0, 1]
    for d in range(6):
        vals = [e['edge_var'][d] for e in all_data]
        c = 'red' if d in synced else 'blue'
        s = 'o-' if d in synced else 's--'
        ax.plot(Ts, vals, s, color=c, markersize=5, alpha=0.7)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel(r'$\mathrm{Var}(\Delta\phi_d)$')
    ax.set_title('Edge Phase Difference Variance\nLow = synced, High = random')
    ax.grid(True, alpha=0.3)

    # 3. Flux ratio (unsync/sync)
    ax = axes[0, 2]
    valid_fr = [(Ts[i], flux_ratios[i]) for i in range(len(Ts))
                if not np.isnan(flux_ratios[i])]
    if valid_fr:
        t_fr, r_fr = zip(*valid_fr)
        ax.plot(t_fr, r_fr, 'D-', color='darkgreen', lw=2.5, markersize=8)
    ax.axhline(1, ls='--', color='black', alpha=0.5)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T')
    ax.set_ylabel(r'$\langle F^2 \rangle_{unsync} / \langle F^2 \rangle_{sync}$')
    ax.set_title('GAUGE HIGGS TEST\n>1 = gauge mass hierarchy')
    ax.grid(True, alpha=0.3)

    # 4. Gauge gap per dimension
    ax = axes[1, 0]
    colors = ['red' if d in synced else 'blue' for d in range(6)]
    ax.bar(range(6), gauge_gaps, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'd{d+1}\n{"S" if d in synced else "F"}' for d in range(6)])
    ax.set_ylabel('Gauge spectral gap')
    ax.set_title('Gauge Boson Mass per Dimension')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Curvature tensor
    ax = axes[1, 1]
    im = ax.imshow(np.abs(F_cold), cmap='YlOrRd', aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([f'd{d+1}' for d in range(6)])
    ax.set_yticklabels([f'd{d+1}' for d in range(6)])
    ax.set_title('Curvature Tensor |F_{d1,d2}|')

    # 6. Degree control
    ax = axes[1, 2]
    degs = [e['deg'] for e in all_data]
    ax.plot(Ts, degs, 'o-', color='crimson', lw=2)
    ax.axhline(TARGET_DEG, ls=':', color='black')
    cv = np.std(degs)/np.mean(degs)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Degree')
    ax.set_title(f'Control (cv={cv:.1%})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring Gauge Field Lab v2\n'
                 'Phase differences as gauge connections',
                 fontsize=14, fontweight='bold')
    for ax_row in axes:
        for ax in ax_row:
            # Fix log scale with zero values
            if ax.get_xscale() == 'log':
                xlim = ax.get_xlim()
                if xlim[0] <= 0:
                    ax.set_xlim(left=max(xlim[0], 1e-3))
            if ax.get_yscale() == 'log':
                ylim = ax.get_ylim()
                if ylim[0] <= 0:
                    ax.set_ylim(bottom=max(ylim[0], 1e-6))

    plt.tight_layout()
    plt.savefig('gauge_v2_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    cv = np.std(degs)/np.mean(degs)

    # Flux ratio at low T
    fr_cold = flux_ratios[-1] if not np.isnan(flux_ratios[-1]) else 0
    fr_hot = flux_ratios[0] if not np.isnan(flux_ratios[0]) else 1

    # Var ratio at low T
    vr_cold = var_ratios[-1] if not np.isnan(var_ratios[-1]) else 0

    print("\n" + "=" * 70)
    print("  GAUGE FIELD VERDICT v2")
    print("=" * 70)

    tests = [
        ("Flux ratio > 1 at low T (gauge condensation)",
         fr_cold > 1.05,
         f"ratio = {fr_cold:.3f}"),
        ("Flux ratio ≈ 1 at high T (symmetric)",
         abs(fr_hot - 1) < 0.2 or np.isnan(flux_ratios[0]),
         f"ratio = {fr_hot:.3f}"),
        ("Flux ratio correlates with anisotropy",
         corr_flux > 0.5,
         f"corr = {corr_flux:.3f}"),
        ("Edge var ratio > 1 at low T",
         vr_cold > 1.05,
         f"ratio = {vr_cold:.3f}"),
        ("Edge var ratio correlates with anisotropy",
         corr_var > 0.5,
         f"corr = {corr_var:.3f}"),
        ("Gauge gap differs sync vs unsync",
         abs(gs - gu) / max(gu, 1e-6) > 0.05 if 'gs' in dir() else False,
         f"{gs:.4f} vs {gu:.4f}" if 'gs' in dir() else "N/A"),
        ("Degree controlled",
         cv < 0.15,
         f"cv = {cv:.1%}"),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        print(f"  {'[+]' if passed else '[-]'} {'PASS' if passed else 'FAIL'}  "
              f"{name:<50s} {detail}")

    print(f"\n  Score: {n_pass}/{len(tests)}")

    if n_pass >= 5:
        print("\n  *** GAUGE HIGGS MECHANISM CONFIRMED ***")
    elif n_pass >= 3:
        print("\n  * Partial gauge Higgs signal *")
    else:
        print("\n  No gauge Higgs mechanism detected.")

    print("=" * 70)

if __name__ == "__main__":
    run_gauge_v2()
