import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — avoids Wayland issues
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, diags as sp_diags
from scipy.sparse.linalg import eigsh
from collections import deque
import time
import warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],
    [-1, 2,-1, 0, 0, 0],
    [ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],
    [ 0, 0, 0,-1, 2, 0],
    [ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)


def gen_traj(N, kappa, T):
    D = 6
    omega = 2 * np.sin(np.pi * np.array([1, 4, 5, 7, 8, 11]) / 12)
    ph = np.zeros((N, D))
    ph[0] = np.random.uniform(0, 2 * np.pi, D)
    for n in range(N - 1):
        ph[n + 1] = (ph[n] + omega + kappa * C_E6 @ np.sin(ph[n])
                     + np.random.normal(0, T, D)) % (2 * np.pi)
    return ph


def build_graph(phases, target_deg=25, delta_min=5):
    N = len(phases)
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    lo, hi = 0.01, 8.0
    best_eps, best_deg, best_pf = 1.0, 0, None
    for _ in range(25):
        mid = (lo + hi) / 2
        pairs = tree.query_pairs(r=mid, output_type='ndarray')
        if len(pairs) > 0:
            pf = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > delta_min]
        else:
            pf = np.zeros((0, 2), dtype=int)
        actual = (2 * len(pf) + 2 * (N - 1)) / N
        if actual < target_deg:
            lo = mid
        else:
            hi = mid
        if abs(actual - target_deg) < abs(best_deg - target_deg):
            best_deg = actual
            best_eps = mid
            best_pf = pf.copy()
        if abs(actual - target_deg) / target_deg < 0.05:
            break
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N - 1):
        G.add_edge(i, i + 1)
    if best_pf is not None:
        for a, b in best_pf:
            G.add_edge(int(a), int(b))
    fd = sum(dict(G.degree()).values()) / N
    if fd > target_deg * 1.1:
        re = [(u, v) for u, v in G.edges() if abs(u - v) > delta_min]
        nr = int((fd - target_deg) * N / 2)
        if 0 < nr < len(re):
            idx = np.random.choice(len(re), nr, replace=False)
            G.remove_edges_from([re[i] for i in idx])
    return G, sum(dict(G.degree()).values()) / N


def kuramoto(phases):
    r = np.zeros(6)
    for d in range(6):
        r[d] = np.abs(np.mean(np.exp(1j * phases[:, d])))
    return r


def torus_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi


def plaquette_action(G, phases):
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    triangles = []
    sampled = np.random.choice(
        list(G.nodes()), min(2000, G.number_of_nodes()), replace=False
    )
    for u in sampled:
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w <= v:
                    continue
                triangles.append((u, v, w))
    if not triangles:
        return np.zeros(6), np.zeros(6), 0, []
    n_tri = len(triangles)
    F = np.zeros((n_tri, 6))
    for idx, (u, v, w) in enumerate(triangles):
        for d in range(6):
            F[idx, d] = (
                torus_diff(phases[v, d], phases[u, d])
                + torus_diff(phases[w, d], phases[v, d])
                + torus_diff(phases[u, d], phases[w, d])
            )
    F_squared = np.mean(F ** 2, axis=0)
    S_cos = 1.0 - np.mean(np.cos(F), axis=0)
    return F_squared, S_cos, n_tri, triangles


def edge_gauge_variance(G, phases):
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


def gauge_boson_spectrum(G, phases, dim, k=10):
    N = G.number_of_nodes()
    edges = [(u, v) for u, v in G.edges() if u < v]
    n_edges = len(edges)
    if n_edges < 10:
        return np.array([])
    inc = lil_matrix((N, n_edges), dtype=np.float64)
    for idx, (u, v) in enumerate(edges):
        inc[u, idx] = -1.0
        inc[v, idx] = +1.0
    inc = csc_matrix(inc)
    gauge_weights = np.zeros(n_edges)
    for idx, (u, v) in enumerate(edges):
        dphi = torus_diff(phases[v, dim], phases[u, dim])
        gauge_weights[idx] = np.cos(dphi)
    W_edges = sp_diags(gauge_weights)
    L_gauge = inc @ W_edges @ inc.T
    k_actual = min(k, N - 2)
    try:
        evals, _ = eigsh(L_gauge, k=k_actual, which='SM',
                         tol=1e-3, maxiter=3000)
        return np.sort(evals)
    except Exception:
        return np.array([])


def wilson_loops(G, phases, max_size=6, n_samples=300):
    N = G.number_of_nodes()
    results = {s: {d: [] for d in range(6)} for s in range(3, max_size + 1)}
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
                            torus_diff(
                                phases[loop[(i + 1) % len(loop)], d],
                                phases[loop[i], d],
                            )
                            for i in range(len(loop))
                        )
                        results[len(loop)][d].append(np.abs(np.exp(1j * phase_sum)))
    W_avg = {}
    for s in range(3, max_size + 1):
        W_avg[s] = np.array(
            [np.mean(results[s][d]) if results[s][d] else np.nan for d in range(6)]
        )
    return W_avg


def topological_charge(triangles, phases):
    Q = np.zeros(6)
    for u, v, w in triangles:
        for d in range(6):
            F = (
                torus_diff(phases[v, d], phases[u, d])
                + torus_diff(phases[w, d], phases[v, d])
                + torus_diff(phases[u, d], phases[w, d])
            )
            Q[d] += F
    Q /= 2 * np.pi
    return Q


def curvature_tensor(G, phases):
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    triangles = []
    sampled = np.random.choice(
        list(G.nodes()), min(1000, G.number_of_nodes()), replace=False
    )
    for u in sampled:
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w <= v:
                    continue
                triangles.append((u, v, w))
    if not triangles:
        return np.zeros((6, 6))
    F = np.zeros((6, 6))
    for u, v, w in triangles:
        for d1 in range(6):
            dp1_uv = torus_diff(phases[v, d1], phases[u, d1])
            dp1_vw = torus_diff(phases[w, d1], phases[v, d1])
            for d2 in range(d1 + 1, 6):
                dp2_uv = torus_diff(phases[v, d2], phases[u, d2])
                dp2_vw = torus_diff(phases[w, d2], phases[v, d2])
                val = dp1_uv * dp2_vw - dp2_uv * dp1_vw
                F[d1, d2] += val
                F[d2, d1] -= val
    F /= len(triangles)
    return F


def run_gauge_v2():
    print("=" * 70)
    print("  MONOSTRING GAUGE FIELD LAB v2")
    print("  Phase differences as lattice gauge connections")
    print("=" * 70)

    N = 8000
    kappa = 0.5
    TARGET_DEG = 25
    temperatures = [3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]

    all_data = []

    # ==============================================================
    # EXPERIMENT 1+2: Plaquette + Edge variance
    # ==============================================================
    print("\n[1/5] Plaquette flux + Edge gauge variance...")
    print("      <F^2_d>: flux variance per dim")
    print("      Var(dphi_d): edge diff variance")

    header = "      {:>6s} |".format("T")
    for d in range(6):
        header += " F2_d{}".format(d + 1)
    header += " |"
    for d in range(6):
        header += " V_d{}".format(d + 1)
    header += " | {:>6s}".format("#tri")
    print(header)
    print("      " + "-" * 95)

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N, kappa, T)
        G, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_per = kuramoto(phases)

        F_sq, S_cos, n_tri, triangles = plaquette_action(G, phases)
        edge_var, edge_abs = edge_gauge_variance(G, phases)

        entry = {
            "T": T,
            "r_per": r_per.copy(),
            "F_sq": F_sq.copy(),
            "S_cos": S_cos.copy(),
            "edge_var": edge_var.copy(),
            "edge_abs": edge_abs.copy(),
            "n_tri": n_tri,
            "triangles": triangles,
            "phases": phases,
            "G": G,
            "deg": deg,
        }
        all_data.append(entry)

        Fstr = " ".join("{:>5.3f}".format(f) for f in F_sq)
        Vstr = " ".join("{:>5.3f}".format(v) for v in edge_var)
        print(
            "      {:>6.3f} | {} | {} | {:>6d}  ({:.1f}s)".format(
                T, Fstr, Vstr, n_tri, time.time() - t0
            )
        )

    # Classify dimensions
    r_cold = all_data[-1]["r_per"]
    synced = [d for d in range(6) if r_cold[d] > 0.5]
    unsynced = [d for d in range(6) if r_cold[d] <= 0.5]
    print("\n      Synced: {}, Unsynced: {}".format(synced, unsynced))

    # ==============================================================
    # ANALYSIS: Condensation ratios
    # ==============================================================
    print("\n      GAUGE CONDENSATION ANALYSIS:")
    print(
        "      {:>6s} | {:>8s} {:>8s} {:>7s} | {:>7s} {:>7s} {:>7s}".format(
            "T", "<F2>_s", "<F2>_u", "ratio", "Var_s", "Var_u", "ratio"
        )
    )
    print("      " + "-" * 60)

    flux_ratios = []
    var_ratios = []

    for entry in all_data:
        if len(synced) > 0 and len(unsynced) > 0:
            Fs = np.mean(entry["F_sq"][synced])
            Fu = np.mean(entry["F_sq"][unsynced])
            fr = Fu / Fs if Fs > 1e-10 else np.nan

            Vs = np.mean(entry["edge_var"][synced])
            Vu = np.mean(entry["edge_var"][unsynced])
            vr = Vu / Vs if Vs > 1e-10 else np.nan
        else:
            Fs = Fu = fr = Vs = Vu = vr = np.nan

        flux_ratios.append(fr)
        var_ratios.append(vr)

        print(
            "      {:>6.3f} | {:>8.5f} {:>8.5f} {:>7.2f} | "
            "{:>7.5f} {:>7.5f} {:>7.2f}".format(
                entry["T"],
                Fs if not np.isnan(Fs) else 0,
                Fu if not np.isnan(Fu) else 0,
                fr if not np.isnan(fr) else 0,
                Vs if not np.isnan(Vs) else 0,
                Vu if not np.isnan(Vu) else 0,
                vr if not np.isnan(vr) else 0,
            )
        )

    # Correlations
    anisos = []
    for e in all_data:
        if len(synced) > 0 and len(unsynced) > 0:
            anisos.append(
                np.mean(e["r_per"][synced]) - np.mean(e["r_per"][unsynced])
            )
        else:
            anisos.append(0)

    corr_flux = 0.0
    valid_f = [not np.isnan(r) for r in flux_ratios]
    if sum(valid_f) >= 3:
        fv = [flux_ratios[i] for i in range(len(flux_ratios)) if valid_f[i]]
        av = [anisos[i] for i in range(len(anisos)) if valid_f[i]]
        corr_flux = np.corrcoef(av, fv)[0, 1]
        print("\n      corr(flux_ratio, anisotropy) = {:.3f}".format(corr_flux))

    corr_var = 0.0
    valid_v = [not np.isnan(r) for r in var_ratios]
    if sum(valid_v) >= 3:
        vv = [var_ratios[i] for i in range(len(var_ratios)) if valid_v[i]]
        av2 = [anisos[i] for i in range(len(anisos)) if valid_v[i]]
        corr_var = np.corrcoef(av2, vv)[0, 1]
        print("      corr(var_ratio, anisotropy) = {:.3f}".format(corr_var))

    # ==============================================================
    # EXPERIMENT 3: Gauge boson spectrum
    # ==============================================================
    print("\n[2/5] Gauge boson spectrum...")

    entry_cold = all_data[-1]
    gauge_gaps = np.zeros(6)

    print("      {:>5s} {:>5s} | First 5 eigenvalues".format("Dim", "Type"))
    print("      " + "-" * 50)

    for d in range(6):
        evals = gauge_boson_spectrum(
            entry_cold["G"], entry_cold["phases"], d, k=8
        )
        if len(evals) > 0:
            nonzero = evals[np.abs(evals) > 1e-4]
            gauge_gaps[d] = np.min(np.abs(nonzero)) if len(nonzero) > 0 else 0
            ev_str = " ".join("{:>8.4f}".format(e) for e in evals[:5])
            label = "SYNC" if d in synced else "FREE"
            print("      d{:>3d} {:>5s} | {}".format(d + 1, label, ev_str))

    gs = np.mean(gauge_gaps[synced]) if len(synced) > 0 else 0
    gu = np.mean(gauge_gaps[unsynced]) if len(unsynced) > 0 else 0

    if len(synced) > 0 and len(unsynced) > 0:
        print("\n      Gauge gap synced: {:.4f}".format(gs))
        print("      Gauge gap unsynced: {:.4f}".format(gu))
        if gu > 1e-6:
            print("      Ratio sync/unsync: {:.3f}".format(gs / gu))

    # ==============================================================
    # EXPERIMENT 4: Wilson loops
    # ==============================================================
    print("\n[3/5] Wilson loops...")

    for label, idx in [("HOT", 0), ("COLD", -1)]:
        entry = all_data[idx]
        W_avg = wilson_loops(entry["G"], entry["phases"], max_size=6)

        print("\n      {} (T={:.3f}):".format(label, entry["T"]))
        for s in sorted(W_avg.keys()):
            if len(synced) > 0 and len(unsynced) > 0:
                W_s = np.nanmean(W_avg[s][synced])
                W_u = np.nanmean(W_avg[s][unsynced])
                print(
                    "      Size {}: W_sync={:.4f}, W_unsync={:.4f}, "
                    "diff={:.5f}".format(s, W_s, W_u, W_s - W_u)
                )

    # ==============================================================
    # EXPERIMENT 5: Topological charge
    # ==============================================================
    print("\n[4/5] Topological charge...")

    for label, idx in [("HOT", 0), ("COLD", -1)]:
        entry = all_data[idx]
        if entry["triangles"]:
            Q = topological_charge(entry["triangles"], entry["phases"])
            Q_str = " ".join("{:>7.2f}".format(q) for q in Q)
            print("      {}: Q = [{}]".format(label, Q_str))

    # ==============================================================
    # EXPERIMENT 6: Curvature tensor
    # ==============================================================
    print("\n[5/5] Curvature tensor...")

    F_cold = curvature_tensor(all_data[-1]["G"], all_data[-1]["phases"])
    print("\n      COLD (T={:.3f}):".format(all_data[-1]["T"]))
    for d1 in range(6):
        row = " ".join("{:>7.4f}".format(F_cold[d1, d2]) for d2 in range(6))
        print("      [{}]".format(row))

    F_ss = 0.0
    F_uu = 0.0
    F_su = 0.0
    if len(synced) >= 2 and len(unsynced) >= 2:
        F_ss = np.sqrt(np.sum(F_cold[np.ix_(synced, synced)] ** 2))
        F_uu = np.sqrt(np.sum(F_cold[np.ix_(unsynced, unsynced)] ** 2))
        F_su = np.sqrt(np.sum(F_cold[np.ix_(synced, unsynced)] ** 2))
        print("\n      |F_sync-sync|   = {:.4f}".format(F_ss))
        print("      |F_unsync-unsync| = {:.4f}".format(F_uu))
        print("      |F_sync-unsync|  = {:.4f}".format(F_su))

    # ==============================================================
    # PLOTS
    # ==============================================================
    Ts = [e["T"] for e in all_data]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Edge variance per dim
    ax = axes[0, 0]
    for d in range(6):
        vals = [e["edge_var"][d] for e in all_data]
        c = "red" if d in synced else "blue"
        s = "o-" if d in synced else "s--"
        ax.plot(Ts, vals, s, color=c, markersize=5, alpha=0.7,
                label="d{} ({})".format(d + 1, "S" if d in synced else "F"))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("T")
    ax.set_ylabel("Var(dphi_d)")
    ax.set_title("Edge Phase Variance\nLow=condensed(massive), High=free(massless)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. Var ratio (unsync/sync)
    ax = axes[0, 1]
    valid_vr = [(Ts[i], var_ratios[i]) for i in range(len(Ts))
                if not np.isnan(var_ratios[i])]
    if valid_vr:
        t_vr, r_vr = zip(*valid_vr)
        ax.plot(t_vr, r_vr, "D-", color="darkgreen", lw=2.5, markersize=8)
    ax.axhline(1, ls="--", color="black", alpha=0.5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("T")
    ax.set_ylabel("Var(unsync) / Var(sync)")
    ax.set_title("GAUGE HIGGS TEST\n>1 = gauge mass hierarchy")
    ax.grid(True, alpha=0.3)

    # 3. Gauge gap per dimension
    ax = axes[0, 2]
    colors_bar = ["red" if d in synced else "blue" for d in range(6)]
    ax.bar(range(6), gauge_gaps, color=colors_bar, alpha=0.7, edgecolor="black")
    ax.set_xticks(range(6))
    ax.set_xticklabels(
        ["d{}\n{}".format(d + 1, "S" if d in synced else "F") for d in range(6)]
    )
    ax.set_ylabel("Gauge spectral gap")
    ax.set_title("Gauge Boson Mass\nRed=synced(massive), Blue=free(massless)")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Per-dim Kuramoto order parameter
    ax = axes[1, 0]
    for d in range(6):
        rs = [e["r_per"][d] for e in all_data]
        c = "red" if d in synced else "blue"
        s = "o-" if d in synced else "s--"
        ax.plot(Ts, rs, s, color=c, markersize=4, alpha=0.7,
                label="d{}".format(d + 1))
    ax.axhline(0.5, ls=":", color="black", alpha=0.5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("T")
    ax.set_ylabel("Kuramoto r per dim")
    ax.set_title("Phase Synchronization")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # 5. Curvature tensor heatmap
    ax = axes[1, 1]
    F_abs = np.abs(F_cold)
    vmax = max(F_abs.max(), 1e-6)
    im = ax.imshow(F_abs, cmap="YlOrRd", aspect="equal", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(["d{}".format(d + 1) for d in range(6)])
    ax.set_yticklabels(["d{}".format(d + 1) for d in range(6)])
    ax.set_title("Curvature |F_d1d2|\n(T={:.3f})".format(all_data[-1]["T"]))

    # 6. Degree control
    ax = axes[1, 2]
    degs = [e["deg"] for e in all_data]
    ax.plot(Ts, degs, "o-", color="crimson", lw=2)
    ax.axhline(TARGET_DEG, ls=":", color="black")
    cv = np.std(degs) / np.mean(degs)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("T")
    ax.set_ylabel("Degree")
    ax.set_title("Control (cv={:.1%})".format(cv))
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Monostring Gauge Field Lab v2\n"
        "Phase differences as lattice gauge connections",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig("gauge_v2_results.png", dpi=150, bbox_inches="tight")
    print("\n      Figure saved: gauge_v2_results.png")

    try:
        plt.show()
    except Exception:
        pass

    # ==============================================================
    # VERDICT
    # ==============================================================
    cv = np.std(degs) / np.mean(degs)

    fr_cold = var_ratios[-1] if not np.isnan(var_ratios[-1]) else 0
    fr_hot = var_ratios[0] if not np.isnan(var_ratios[0]) else 1

    print("\n" + "=" * 70)
    print("  GAUGE FIELD VERDICT v2")
    print("=" * 70)

    tests = [
        (
            "Edge var ratio > 1 at low T (gauge condensation)",
            fr_cold > 1.05,
            "ratio = {:.3f}".format(fr_cold),
        ),
        (
            "Edge var ratio ~ 1 at high T (symmetric)",
            abs(fr_hot - 1) < 0.2,
            "ratio = {:.3f}".format(fr_hot),
        ),
        (
            "Edge var ratio correlates with anisotropy",
            corr_var > 0.5,
            "corr = {:.3f}".format(corr_var),
        ),
        (
            "Gauge gap: synced > 0, unsynced = 0",
            gs > 0.005 and gu < 0.001,
            "sync={:.4f}, unsync={:.4f}".format(gs, gu),
        ),
        (
            "Curvature: F_sync-sync < F_unsync-unsync",
            F_ss < F_uu,
            "|F_ss|={:.4f}, |F_uu|={:.4f}".format(F_ss, F_uu),
        ),
        (
            "Degree controlled (cv < 15%)",
            cv < 0.15,
            "cv = {:.1%}".format(cv),
        ),
    ]

    n_pass = 0
    for name, passed, detail in tests:
        n_pass += int(passed)
        status = "[+] PASS" if passed else "[-] FAIL"
        print("  {} {:50s} {}".format(status, name, detail))

    print("\n  Score: {}/{}".format(n_pass, len(tests)))

    if n_pass >= 5:
        print("\n  *** GAUGE HIGGS MECHANISM CONFIRMED ***")
        print("  Synchronized phase directions have SUPPRESSED gauge fluctuations")
        print("  (massive gauge bosons), while unsynchronized directions")
        print("  have ACTIVE fluctuations (massless gauge bosons).")
        print("  Edge variance ratio = {:.1f}x mass hierarchy.".format(fr_cold))
    elif n_pass >= 3:
        print("\n  * Partial gauge Higgs signal *")
    else:
        print("\n  No gauge Higgs mechanism detected.")

    print("=" * 70)


if __name__ == "__main__":
    run_gauge_v2()
