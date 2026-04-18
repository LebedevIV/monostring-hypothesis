"""
Part VIII — Step 5: One Monostring -> Multiple Transport Classes
================================================================

Question:
Can one monostring graph support several distinct transport sectors
("field-like" mode groups) with different propagation properties?

Core idea:
- Build a spatial graph from orbit points on the torus
  using a TORUS-RESPECTING embedding: phi -> (cos phi, sin phi)
- Compute normalized graph Laplacian
- Split spectrum into three bands:
    low-lambda  = IR / long-wavelength sector
    mid-lambda  = intermediate sector
    high-lambda = UV / short-wavelength sector
- Launch a source-localized excitation filtered into each band
- Evolve by continuous-time quantum walk / local graph Hamiltonian
- Measure:
    1) arrival time vs graph geodesic distance
    2) attenuation vs distance
    3) geodesic concentration (does the signal concentrate
       near the shortest path corridor?)

Null hypothesis H0:
- low/mid/high bands have no statistically meaningful transport differences
- E6 behaves like Random

Interesting result H1:
- bands differ in speed / attenuation / geodesic focusing
- possibly E6 differs from Random

Important honesty:
- We are NOT claiming Standard Model gauge fields
- We are NOT deriving photon/electron masses
- We are only testing whether one graph supports multiple
  distinct transport classes
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from scipy import stats
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. ALGEBRAS
# ============================================================

ALGEBRAS = {
    "E6": {
        "rank": 6, "h": 12,
        "exponents": [1, 4, 5, 7, 8, 11],
        "color": "#2196F3",
    },
    "Random": {
        "rank": 6, "h": None,
        "exponents": None,
        "color": "#9E9E9E",
    },
    # Optional:
    # "A6": {
    #     "rank": 6, "h": 7,
    #     "exponents": [1, 2, 3, 4, 5, 6],
    #     "color": "#9C27B0",
    # },
    # "E8": {
    #     "rank": 8, "h": 30,
    #     "exponents": [1, 7, 11, 13, 17, 19, 23, 29],
    #     "color": "#4CAF50",
    # },
}


def get_omega(name, seed=42):
    alg = ALGEBRAS[name]
    if alg["exponents"] is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.5, 2.0, alg["rank"])
    m = np.array(alg["exponents"], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg["h"])


# ============================================================
# 2. ORBIT GENERATION
# ============================================================

def generate_orbit(omega, T=900, kappa=0.05, warmup=500, seed=42):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))

    for _ in range(warmup):
        phi = (phi + omega + kappa * np.sin(phi)) % (2*np.pi)

    traj = np.zeros((T, len(omega)))
    for t in range(T):
        phi = (phi + omega + kappa * np.sin(phi)) % (2*np.pi)
        traj[t] = phi

    return traj


# ============================================================
# 3. TORUS EMBEDDING
# ============================================================

def embed_torus(orbit):
    """
    Map angles phi_i to Euclidean embedding:
      phi_i -> (cos phi_i, sin phi_i)

    This respects torus periodicity and avoids false boundary splits.
    """
    return np.concatenate([np.cos(orbit), np.sin(orbit)], axis=1)


# ============================================================
# 4. BUILD WEIGHTED k-NN GRAPH
# ============================================================

def build_knn_graph(X, k=10):
    """
    X: embedded orbit points in Euclidean space.
    Graph edges:
      - cost   = Euclidean distance in embedding (for shortest paths)
      - weight = Gaussian similarity exp(-d^2 / 2σ^2) (for Laplacian)
    """
    n = len(X)
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k+1)

    sigma = np.median(dists[:, 1:])

    edge_best = {}

    for i in range(n):
        for m in range(1, k+1):
            j = int(idxs[i, m])
            if i == j:
                continue
            d = float(dists[i, m])
            a, b = (i, j) if i < j else (j, i)

            if (a, b) not in edge_best or d < edge_best[(a, b)]["cost"]:
                edge_best[(a, b)] = {
                    "cost": d,
                    "weight": np.exp(-d**2 / (2*sigma**2))
                }

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (a, b), data in edge_best.items():
        G.add_edge(a, b, cost=data["cost"], weight=data["weight"])

    return G, sigma


def extract_giant_component(G):
    comps = list(nx.connected_components(G))
    giant = max(comps, key=len)
    Gg = G.subgraph(giant).copy()
    mapping = {old: i for i, old in enumerate(Gg.nodes())}
    Gg = nx.relabel_nodes(Gg, mapping)
    return Gg, len(comps), len(giant)


# ============================================================
# 5. NORMALIZED LAPLACIAN
# ============================================================

def graph_to_matrices(G):
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    C = np.full((n, n), np.inf, dtype=float)

    for i, j, data in G.edges(data=True):
        w = data["weight"]
        c = data["cost"]
        A[i, j] = A[j, i] = w
        C[i, j] = C[j, i] = c

    deg = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    Dhalf = np.diag(dinv_sqrt)
    Lnorm = np.eye(n) - Dhalf @ A @ Dhalf

    # Numerical cleanup
    Lnorm = 0.5 * (Lnorm + Lnorm.T)

    return A, C, Lnorm, deg


def spectral_decomposition(Lnorm):
    vals, vecs = np.linalg.eigh(Lnorm)
    vals = np.maximum(vals, 0.0)
    return vals, vecs


# ============================================================
# 6. SPECTRAL BANDS
# ============================================================

def make_bands(vals, frac=0.20):
    nz = np.where(vals > 1e-10)[0]
    m = len(nz)
    q = max(8, int(frac * m))

    low = nz[:q]
    mid_center = m // 2
    mid = nz[max(0, mid_center - q//2): min(m, mid_center + q//2)]
    high = nz[-q:]

    return {
        "lowλ": low,
        "midλ": mid,
        "highλ": high,
    }


# ============================================================
# 7. SOURCE + TARGETS
# ============================================================

def choose_source(G, n_candidates=20):
    degs = np.array([G.degree(i) for i in G.nodes()])
    cand = np.argsort(degs)[-min(n_candidates, len(degs)):]

    best_node = None
    best_mean = np.inf

    for c in cand:
        lengths = nx.single_source_dijkstra_path_length(G, c, weight="cost")
        mean_len = np.mean(list(lengths.values()))
        if mean_len < best_mean:
            best_mean = mean_len
            best_node = c

    return int(best_node), float(best_mean)


def select_targets(G, source, n_targets=24):
    lengths = nx.single_source_dijkstra_path_length(G, source, weight="cost")
    items = sorted([(node, d) for node, d in lengths.items() if node != source],
                   key=lambda x: x[1])

    nodes = np.array([x[0] for x in items], dtype=int)
    dvals = np.array([x[1] for x in items], dtype=float)

    if len(nodes) < n_targets:
        return nodes, dvals

    qmin = np.quantile(dvals, 0.15)
    qmax = np.quantile(dvals, 0.90)

    mask = (dvals >= qmin) & (dvals <= qmax)
    nodes2 = nodes[mask]
    dvals2 = dvals[mask]

    if len(nodes2) < n_targets:
        nodes2, dvals2 = nodes, dvals

    qs = np.linspace(0.0, 1.0, n_targets)
    picked_nodes = []
    picked_dists = []

    for q in qs:
        target_d = np.quantile(dvals2, q)
        idx = np.argmin(np.abs(dvals2 - target_d))
        picked_nodes.append(int(nodes2[idx]))
        picked_dists.append(float(dvals2[idx]))

    # unique preserve order
    seen = set()
    out_nodes, out_dists = [], []
    for n, d in zip(picked_nodes, picked_dists):
        if n not in seen:
            seen.add(n)
            out_nodes.append(n)
            out_dists.append(d)

    return np.array(out_nodes, dtype=int), np.array(out_dists, dtype=float)


# ============================================================
# 8. BAND-FILTERED INITIAL STATE
# ============================================================

def make_band_state(vecs, band_idx, source):
    n = vecs.shape[0]
    delta = np.zeros(n, dtype=float)
    delta[source] = 1.0

    coeff = vecs.T @ delta
    coeff_band = np.zeros_like(coeff)
    coeff_band[band_idx] = coeff[band_idx]

    psi0 = vecs @ coeff_band
    norm = np.linalg.norm(psi0)
    if norm > 0:
        psi0 /= norm

    coeff0 = vecs.T @ psi0
    return psi0.astype(complex), coeff0.astype(complex)


# ============================================================
# 9. LOCAL UNITARY EVOLUTION ON GRAPH
# ============================================================

def evolve_targets(vals, vecs, coeff0, targets, times):
    """
    psi(t) = U exp(-i Lambda t) U^T psi(0)
    Numerically computed spectrally, but physically generated
    by local graph Hamiltonian H = L_norm.
    """
    phases = np.exp(-1j * vals[:, None] * times[None, :])  # (n_modes, n_times)
    B = coeff0[:, None] * phases
    U_targets = vecs[targets, :]                           # (n_targets, n_modes)
    amps = U_targets @ B                                   # (n_targets, n_times)
    probs = np.abs(amps)**2
    return probs


def evolve_all_nodes(vals, vecs, coeff0, t):
    phase = np.exp(-1j * vals * t)
    psi = vecs @ (coeff0 * phase)
    prob = np.abs(psi)**2
    return prob


# ============================================================
# 10. TRANSPORT METRICS
# ============================================================

def first_arrival_times(probs, times, frac=0.50):
    """
    First time reaching frac * peak probability at each target.
    """
    n_targets = probs.shape[0]
    t_arr = np.zeros(n_targets)
    p_peak = np.zeros(n_targets)

    for i in range(n_targets):
        p = probs[i]
        pmax = p.max()
        p_peak[i] = pmax
        if pmax <= 1e-14:
            t_arr[i] = np.nan
            continue

        threshold = frac * pmax
        idx = np.argmax(p >= threshold)
        t_arr[i] = times[idx]

    return t_arr, p_peak


def fit_transport(distances, t_arr, p_peak):
    mask = np.isfinite(distances) & np.isfinite(t_arr) & np.isfinite(p_peak)
    mask &= (p_peak > 1e-14)

    d = distances[mask]
    t = t_arr[mask]
    p = p_peak[mask]

    out = {
        "n": len(d),
        "speed": np.nan,
        "intercept": np.nan,
        "r2_time": np.nan,
        "p_time": np.nan,
        "att_len": np.nan,
        "att_slope": np.nan,
        "r2_att": np.nan,
        "d": d,
        "t": t,
        "p": p,
    }

    if len(d) >= 5:
        sl, ic, r, pv, _ = stats.linregress(d, t)
        out["intercept"] = ic
        out["r2_time"] = r**2
        out["p_time"] = pv
        if sl > 0:
            out["speed"] = 1.0 / sl

        # attenuation: log peak prob ~ a - d / ell
        logp = np.log(p + 1e-15)
        sl2, ic2, r2, pv2, _ = stats.linregress(d, logp)
        out["att_slope"] = sl2
        out["r2_att"] = r2**2
        if sl2 < 0:
            out["att_len"] = -1.0 / sl2
        else:
            out["att_len"] = np.inf

    return out


# ============================================================
# 11. GEODESIC FOCUSING
# ============================================================

def corridor_nodes(G, path, radius=1):
    S = set(path)
    frontier = set(path)
    for _ in range(radius):
        new_frontier = set()
        for v in frontier:
            new_frontier.update(G.neighbors(v))
        S.update(new_frontier)
        frontier = new_frontier
    return np.array(sorted(S), dtype=int)


def geodesic_focus_score(G, vals, vecs, coeff0, source, target, t_hit,
                         n_random=100, radius=1, seed=42):
    path = nx.shortest_path(G, source, target, weight="cost")
    corridor = corridor_nodes(G, path, radius=radius)

    P = evolve_all_nodes(vals, vecs, coeff0, t_hit)
    p_geo = P[corridor].sum()

    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    m = len(corridor)

    baseline = []
    for _ in range(n_random):
        sample = rng.choice(n, size=m, replace=False)
        baseline.append(P[sample].sum())
    baseline = np.array(baseline)

    mu = baseline.mean()
    sd = baseline.std(ddof=1) + 1e-12
    z = (p_geo - mu) / sd

    return {
        "path_len_nodes": len(path),
        "corridor_size": m,
        "p_geo": p_geo,
        "p_rand_mean": mu,
        "p_rand_std": sd,
        "z_geo": z,
    }


# ============================================================
# 12. MAIN ANALYSIS FOR ONE ALGEBRA
# ============================================================

def run_one_algebra(name, T=900, k_nn=10, tmax=120.0, n_times=320, seed=42):
    color = ALGEBRAS[name]["color"]
    omega = get_omega(name, seed=seed)

    print("\n" + "="*60)
    print(f"ALGEBRA: {name}")
    print(f"omega = {np.round(omega, 4).tolist()}")

    # Orbit and torus embedding
    orbit = generate_orbit(omega, T=T, kappa=0.05, warmup=500, seed=seed)
    X = embed_torus(orbit)

    # Graph
    G0, sigma = build_knn_graph(X, k=k_nn)
    G, n_comp, giant_size = extract_giant_component(G0)

    print(f"Orbit points: {len(orbit)}")
    print(f"k-NN graph: nodes={G0.number_of_nodes()}, edges={G0.number_of_edges()}, sigma={sigma:.4f}")
    print(f"Connected components before trimming: {n_comp}")
    print(f"Giant component size: {giant_size} ({100*giant_size/len(orbit):.1f}% of nodes)")

    # Matrices
    A, C, Lnorm, deg = graph_to_matrices(G)
    vals, vecs = spectral_decomposition(Lnorm)
    bands = make_bands(vals, frac=0.20)

    print(f"Spectrum: λ_min={vals[0]:.3e}, λ_max={vals[-1]:.4f}")
    for bname, idx in bands.items():
        print(f"  {bname:<6}: {len(idx):3d} modes, λ in [{vals[idx[0]]:.4f}, {vals[idx[-1]]:.4f}]")

    # Source + targets
    source, mean_d = choose_source(G)
    targets, dist_targets = select_targets(G, source, n_targets=24)
    print(f"Source node: {source}, mean shortest-path distance={mean_d:.4f}")
    print(f"Targets selected: {len(targets)}")

    # Time grid
    times = np.linspace(0.0, tmax, n_times)

    band_results = {}

    for bname, idx in bands.items():
        psi0, coeff0 = make_band_state(vecs, idx, source)
        probs = evolve_targets(vals, vecs, coeff0, targets, times)
        t_arr, p_peak = first_arrival_times(probs, times, frac=0.50)
        tr = fit_transport(dist_targets, t_arr, p_peak)

        # far target for geodesic-focusing test
        far_i = np.argmax(dist_targets)
        far_target = int(targets[far_i])
        t_hit = float(t_arr[far_i]) if np.isfinite(t_arr[far_i]) else float(times[-1])

        focus = geodesic_focus_score(
            G, vals, vecs, coeff0,
            source=source,
            target=far_target,
            t_hit=t_hit,
            n_random=100,
            radius=1,
            seed=seed
        )

        lam_center = float(np.median(vals[idx]))
        m_proxy = float(np.sqrt(max(lam_center, 0.0)))

        print(f"\n  [{bname}]")
        print(f"    λ_center ~ {lam_center:.4f},  mass-proxy sqrt(λ) ~ {m_proxy:.4f}")
        print(f"    speed          = {tr['speed']:.4f}")
        print(f"    R²(time fit)   = {tr['r2_time']:.4f}")
        print(f"    attenuation ℓ  = {tr['att_len']:.4f}")
        print(f"    geodesic z     = {focus['z_geo']:.4f}")
        print(f"    p_geo          = {focus['p_geo']:.4f}  vs random {focus['p_rand_mean']:.4f} ± {focus['p_rand_std']:.4f}")

        band_results[bname] = {
            "idx": idx,
            "psi0": psi0,
            "coeff0": coeff0,
            "times": times,
            "probs": probs,
            "t_arr": t_arr,
            "p_peak": p_peak,
            "transport": tr,
            "focus": focus,
            "λ_center": lam_center,
            "m_proxy": m_proxy,
        }

    return {
        "name": name,
        "color": color,
        "omega": omega,
        "orbit": orbit,
        "X": X,
        "G": G,
        "A": A,
        "Lnorm": Lnorm,
        "vals": vals,
        "vecs": vecs,
        "bands": bands,
        "source": source,
        "targets": targets,
        "dist_targets": dist_targets,
        "band_results": band_results,
    }


# ============================================================
# 13. FIGURE
# ============================================================

def make_figure(all_results, fname="part8_step5_multifield_transport.png"):
    algs = list(all_results.keys())
    n_algs = len(algs)

    fig = plt.figure(figsize=(22, 7 * n_algs))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Part VIII — Step 5: One Monostring -> Multiple Transport Classes\n"
        "Band-limited transport on the corrected torus graph",
        fontsize=14, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(n_algs, 3, figure=fig, hspace=0.38, wspace=0.28)

    band_colors = {
        "lowλ": "#1565C0",
        "midλ": "#FB8C00",
        "highλ": "#C62828",
    }

    for row, alg in enumerate(algs):
        res = all_results[alg]
        vals = res["vals"]
        dist_targets = res["dist_targets"]

        # --- Spectrum with band coloring
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor("#f0f4f8")
        n = np.arange(len(vals))
        ax.plot(n, vals, ".", color="black", alpha=0.35, ms=3)

        for bname, idx in res["bands"].items():
            ax.plot(idx, vals[idx], ".", color=band_colors[bname], ms=5, label=bname)

        ax.set_title(f"{alg}: normalized Laplacian spectrum", fontsize=11, fontweight="bold")
        ax.set_xlabel("mode index")
        ax.set_ylabel("λ")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- arrival time vs distance
        ax = fig.add_subplot(gs[row, 1])
        ax.set_facecolor("#f0f4f8")

        for bname in ["lowλ", "midλ", "highλ"]:
            br = res["band_results"][bname]
            tr = br["transport"]
            d = tr["d"]
            t = tr["t"]
            c = band_colors[bname]

            ax.scatter(d, t, s=45, alpha=0.75, color=c, label=f"{bname}")
            if len(d) >= 5 and np.isfinite(tr["speed"]):
                sl = 1.0 / tr["speed"]
                ic = tr["intercept"]
                xfit = np.linspace(d.min(), d.max(), 100)
                ax.plot(xfit, sl*xfit + ic, "-", color=c, lw=2)

        ax.set_title(f"{alg}: arrival time vs geodesic distance", fontsize=11, fontweight="bold")
        ax.set_xlabel("graph geodesic distance from source")
        ax.set_ylabel("first-arrival time")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- summary bars
        ax = fig.add_subplot(gs[row, 2])
        ax.set_facecolor("#f0f4f8")

        labels = ["lowλ", "midλ", "highλ"]
        xs = np.arange(3)

        speeds = [res["band_results"][b]["transport"]["speed"] for b in labels]
        zscores = [res["band_results"][b]["focus"]["z_geo"] for b in labels]

        ax2 = ax.twinx()
        ax.bar(xs - 0.18, speeds, width=0.36,
               color=[band_colors[b] for b in labels], alpha=0.75,
               edgecolor="k", label="speed")
        ax2.bar(xs + 0.18, zscores, width=0.36,
                color=[band_colors[b] for b in labels], alpha=0.35,
                edgecolor="k", hatch="//", label="geodesic z")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("speed", color="black")
        ax2.set_ylabel("geodesic focus z", color="black")
        ax.axhline(0, color="black", lw=0.8)
        ax2.axhline(2.0, color="green", ls="--", lw=1.5, alpha=0.7)
        ax.set_title(f"{alg}: speed vs geodesic focusing", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved: {fname}")


# ============================================================
# 14. TEXT SUMMARY
# ============================================================

def final_summary(all_results):
    print("\n" + "="*78)
    print("FINAL SUMMARY — STEP 5")
    print("="*78)
    print(f"\n{'Alg':<10} {'Band':<8} {'λ_center':>10} {'m~sqrt(λ)':>12} {'speed':>10} {'R²_t':>8} {'att_len':>10} {'z_geo':>8}")
    print("-"*92)

    for alg in all_results:
        res = all_results[alg]
        for bname in ["lowλ", "midλ", "highλ"]:
            br = res["band_results"][bname]
            tr = br["transport"]
            z = br["focus"]["z_geo"]
            print(f"{alg:<10} {bname:<8} {br['λ_center']:>10.4f} {br['m_proxy']:>12.4f} "
                  f"{tr['speed']:>10.4f} {tr['r2_time']:>8.4f} {tr['att_len']:>10.4f} {z:>8.3f}")

    print("\nInterpretation guide:")
    print("  higher speed      -> faster transport sector")
    print("  high R²_t         -> cleaner ballistic relation t_arr ~ d")
    print("  larger att_len    -> weaker attenuation with distance")
    print("  z_geo > 2         -> signal concentrated near shortest-path corridor")
    print("\nHonest verdict logic:")
    print("  If low/mid/high are similar -> no evidence for multiple field classes")
    print("  If one band is fastest + weakly attenuated + z_geo>2 -> photon-like transport candidate")
    print("  If high-λ is slow / strongly attenuated -> localized or massive-like sector")
    print("  If E6 ≈ Random in all metrics -> algebra not special here")


# ============================================================
# 15. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Part VIII Step 5: One Monostring -> Multiple Transport Classes\n")

    all_results = {}
    for alg in ["E6", "Random"]:
        all_results[alg] = run_one_algebra(
            alg,
            T=900,
            k_nn=10,
            tmax=120.0,
            n_times=320,
            seed=42
        )

    make_figure(all_results)
    final_summary(all_results)
