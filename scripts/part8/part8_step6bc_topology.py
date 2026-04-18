"""
Part VIII — Step 6bc: Why A6 > E6, and Topological Invariants
==============================================================

Two experiments:

EXP 6b: WHY IS A6 STRONGER THAN E6?
  A6 z_geo=6.60 > E6 z_geo=5.82 > E8 z_geo=3.68 > Random z_geo=0.04

  Hypotheses:
  H_freq:   A6 has simpler frequency ratios → more resonances
            → denser clustering → stronger geodesic alignment
  H_ipr:    A6 eigenmodes are less localized (smaller IPR)
            → more spatially coherent → better geodesic overlap
  H_geom:   A6 orbit geometry (D_corr) differs from E6
            → different graph structure → different transport

  Method:
    Compare for E6, A6, E8:
    (a) Frequency rationality: min gcd-like measure
    (b) Eigenmode IPR distribution
    (c) Fiedler value (algebraic connectivity)
    (d) Graph diameter and mean path length
    (e) Orbit correlation dimension D_corr

EXP 6c: TOPOLOGICAL INVARIANT FOR highλ MODES
  Inspired by "geodesic Chern number" idea.

  What we CAN compute honestly:
  (1) SPECTRAL FLOW along a path
      Parameterize paths γ in the graph.
      For each edge along geodesic path:
      compute how much highλ probability flows
      across that edge vs random edges.
      = "geodesic current"

  (2) WINDING NUMBER of eigenvector phases
      For each highλ eigenmode k:
      walk along the geodesic path source→target
      track phase of vecs[j,k] as j moves along path
      count windings = topological winding number W_k
      If W_k ≠ 0 → mode is topologically non-trivial

  (3) PARTICIPATION RATIO vs GRAPH CYCLES
      Count independent cycles (= graph first Betti number β₁)
      Test: does β₁ correlate with z_geo across algebras?
      If yes: topology (cycles) drives geodesic focusing

Null hypothesis for 6c:
  H0: winding numbers W_k = 0 for all k (trivial)
  H0: β₁ does not correlate with z_geo
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from scipy import stats
from scipy.spatial import cKDTree
from fractions import Fraction
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 0. ALGEBRAS
# ============================================================

ALGEBRAS = {
    "E6": {
        "rank": 6, "h": 12,
        "exponents": [1, 4, 5, 7, 8, 11],
        "color": "#2196F3",
    },
    "A6": {
        "rank": 6, "h": 7,
        "exponents": [1, 2, 3, 4, 5, 6],
        "color": "#9C27B0",
    },
    "E8": {
        "rank": 8, "h": 30,
        "exponents": [1, 7, 11, 13, 17, 19, 23, 29],
        "color": "#4CAF50",
    },
    "Random": {
        "rank": 6, "h": None,
        "exponents": None,
        "color": "#9E9E9E",
    },
}

BAND_COLORS = {
    "lowλ":  "#1565C0",
    "midλ":  "#FB8C00",
    "highλ": "#C62828",
}
BAND_NAMES = ["lowλ", "midλ", "highλ"]


def get_omega(name, seed=42):
    alg = ALGEBRAS[name]
    if alg["exponents"] is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.5, 2.0, alg["rank"])
    m = np.array(alg["exponents"], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg["h"])


# ============================================================
# 1. ORBIT + GRAPH (shared)
# ============================================================

def generate_orbit(omega, T=900, kappa=0.05,
                   warmup=500, seed=42):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega + kappa * np.sin(phi)) % (2*np.pi)
    traj = np.zeros((T, len(omega)))
    for t in range(T):
        phi = (phi + omega + kappa * np.sin(phi)) % (2*np.pi)
        traj[t] = phi
    return traj


def embed_torus(orbit):
    return np.concatenate(
        [np.cos(orbit), np.sin(orbit)], axis=1)


def build_graph(X, k=10):
    n    = len(X)
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
            if (a, b) not in edge_best \
               or d < edge_best[(a, b)]["cost"]:
                edge_best[(a, b)] = {
                    "cost":   d,
                    "weight": float(
                        np.exp(-d**2 / (2*sigma**2)))
                }
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (a, b), data in edge_best.items():
        G.add_edge(a, b,
                   cost=data["cost"],
                   weight=data["weight"])
    comps   = list(nx.connected_components(G))
    giant   = max(comps, key=len)
    Gg      = G.subgraph(giant).copy()
    mapping = {old: i for i, old
               in enumerate(sorted(Gg.nodes()))}
    return nx.relabel_nodes(Gg, mapping), sigma


def graph_spectrum(G):
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        A[i, j] = A[j, i] = d["weight"]
    deg  = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0/np.sqrt(deg), 0.0)
    Ln   = np.eye(n) - np.diag(dinv) @ A @ np.diag(dinv)
    Ln   = 0.5*(Ln + Ln.T)
    vals, vecs = np.linalg.eigh(Ln)
    vals = np.maximum(vals, 0.0)
    return vals, vecs


def make_bands(vals, frac=0.20):
    nz = np.where(vals > 1e-10)[0]
    m  = len(nz)
    q  = max(8, int(frac * m))
    mc = m // 2
    return {
        "lowλ":  nz[:q],
        "midλ":  nz[max(0, mc-q//2): min(m, mc+q//2)],
        "highλ": nz[-q:],
    }


def band_state(vecs, idx, source):
    n   = vecs.shape[0]
    d   = np.zeros(n); d[source] = 1.0
    c   = vecs.T @ d
    cb  = np.zeros_like(c); cb[idx] = c[idx]
    psi = vecs @ cb
    nm  = np.linalg.norm(psi)
    if nm > 0:
        psi /= nm
    return (vecs.T @ psi).astype(complex)


def prob_vec(vals, vecs, c0, t):
    return np.abs(
        vecs @ (c0 * np.exp(-1j * vals * t)))**2


def geo_corridor(G, src, tgt, radius=1):
    path  = nx.shortest_path(G, src, tgt, weight="cost")
    S     = set(path)
    front = set(path)
    for _ in range(radius):
        nxt = set()
        for v in front:
            nxt.update(G.neighbors(v))
        S.update(nxt); front = nxt
    return list(path), np.array(sorted(S), dtype=int)


def z_geo_measure(G, vals, vecs, idx, src, tgt,
                  t=50.0, n_rand=80, seed=0):
    rng      = np.random.default_rng(seed)
    n        = G.number_of_nodes()
    _, corr  = geo_corridor(G, src, tgt)
    m        = len(corr)
    c0       = band_state(vecs, idx, src)
    P        = prob_vec(vals, vecs, c0, t)
    p_geo    = P[corr].sum()
    rand_p   = [P[rng.choice(n, m, False)].sum()
                for _ in range(n_rand)]
    mu, sd   = np.mean(rand_p), np.std(rand_p) + 1e-12
    return float((p_geo - mu) / sd)


# ============================================================
# 2. EXP 6b: WHY A6 > E6?
# ============================================================

# ── 2a. Frequency rationality
def freq_rationality(omega):
    """
    Measure how 'rational' the frequency ratios are.
    For each pair (i,j): approximate omega_i/omega_j
    as a fraction with bounded denominator.
    Smaller denominator → more rational → more resonances.

    Returns: mean denominator (lower = more rational)
    """
    n     = len(omega)
    denoms = []
    for i in range(n):
        for j in range(i+1, n):
            ratio = omega[i] / omega[j]
            frac  = Fraction(ratio).limit_denominator(100)
            denoms.append(frac.denominator)
    return float(np.mean(denoms)), float(np.median(denoms))


# ── 2b. Orbit correlation dimension D_corr
def correlation_dimension(orbit, n_pairs=2000,
                           r_bins=20, seed=42):
    """
    Estimate D_corr from correlation integral C(r).
    D_corr = d(log C)/d(log r) in scaling region.
    """
    rng  = np.random.default_rng(seed)
    T    = len(orbit)
    idx  = rng.choice(T, size=min(n_pairs, T),
                      replace=False)
    sub  = orbit[idx]

    # Pairwise distances (subsample)
    n    = len(sub)
    dists = []
    step = max(1, n // 100)
    for i in range(0, n-1, step):
        d = np.linalg.norm(sub[i] - sub[i+1:], axis=1)
        dists.extend(d.tolist())
    dists = np.array(dists)
    if len(dists) == 0:
        return np.nan

    r_vals = np.logspace(
        np.log10(np.percentile(dists, 5)),
        np.log10(np.percentile(dists, 95)),
        r_bins)
    C_r = np.array([(dists < r).mean() for r in r_vals])
    pos = C_r > 0
    if pos.sum() < 4:
        return np.nan

    sl, _, _, _, _ = stats.linregress(
        np.log(r_vals[pos]), np.log(C_r[pos]))
    return float(sl)


# ── 2c. Graph topology measures
def graph_topology(G):
    n    = G.number_of_nodes()
    e    = G.number_of_edges()

    # Fiedler value = 2nd smallest Laplacian eigenvalue
    # Proxy: use networkx algebraic_connectivity
    try:
        fiedler = nx.algebraic_connectivity(
            G, weight="weight", method="lanczos")
    except Exception:
        fiedler = np.nan

    # First Betti number β₁ = e - n + 1
    # (number of independent cycles)
    beta1 = e - n + 1

    # Mean shortest path (subsample for speed)
    rng   = np.random.default_rng(42)
    nodes = list(G.nodes())
    sample_size = min(50, len(nodes))
    sample = rng.choice(nodes, size=sample_size,
                        replace=False)
    path_lengths = []
    for s in sample:
        lens = nx.single_source_dijkstra_path_length(
            G, int(s), weight="cost")
        path_lengths.extend(lens.values())
    mean_path = float(np.mean(path_lengths))

    # Diameter (approximate: max over sample)
    diam_approx = 0.0
    for s in sample[:20]:
        lens = nx.single_source_dijkstra_path_length(
            G, int(s), weight="cost")
        diam_approx = max(diam_approx, max(lens.values()))

    return {
        "fiedler":   float(fiedler),
        "beta1":     int(beta1),
        "mean_path": mean_path,
        "diam":      diam_approx,
        "n":         n,
        "e":         e,
    }


# ── 2d. IPR comparison
def ipr_per_band(vecs, bands):
    n = vecs.shape[0]
    result = {}
    for bname in BAND_NAMES:
        idx = bands[bname]
        ipr = np.sum(vecs[:, idx]**4, axis=0)
        result[bname] = {
            "mean":  float(ipr.mean()),
            "std":   float(ipr.std()),
            "ratio": float(ipr.mean() * n),
        }
    return result


# ── 2e. Multi-pair z_geo for 6b
def multi_pair_zgeo(G, vals, vecs, bands,
                    bname="highλ", n_pairs=20,
                    t=50.0, seed=42):
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    z_vals = []
    for pi in range(n_pairs):
        src = int(rng.choice(nodes))
        tgt = int(rng.choice(nodes))
        if src == tgt or not nx.has_path(G, src, tgt):
            continue
        idx = bands[bname]
        z   = z_geo_measure(G, vals, vecs, idx,
                            src, tgt, t=t,
                            n_rand=60, seed=pi)
        z_vals.append(z)
    return np.array(z_vals)


# ============================================================
# 3. EXP 6c: TOPOLOGICAL INVARIANTS
# ============================================================

# ── 3a. Geodesic current
def geodesic_current(G, vals, vecs, bands,
                     source, target,
                     t=50.0):
    """
    For each band: compute the probability current
    flowing along each edge of the geodesic path.

    Current on edge (u,v):
    J(u,v) = Im[ psi*(u) * H(u,v) * psi(v) ]
    where H(u,v) = -weight(u,v) (off-diagonal of -L)

    Compare geodesic path edges vs random edges.
    """
    path, _ = geo_corridor(G, source, target, radius=0)

    results = {}
    for bname in BAND_NAMES:
        idx = bands[bname]
        c0  = band_state(vecs, idx, source)
        psi = vecs @ (c0 * np.exp(-1j * vals * t))

        # Currents on geodesic edges
        geo_currents = []
        for u, v in zip(path[:-1], path[1:]):
            if G.has_edge(u, v):
                w = G[u][v]["weight"]
                J = 2.0 * w * np.imag(
                    np.conj(psi[u]) * psi[v])
                geo_currents.append(abs(J))

        # Currents on random edges (baseline)
        rng  = np.random.default_rng(42)
        all_edges = list(G.edges())
        n_sample  = max(len(geo_currents) * 10, 50)
        rand_edges = [all_edges[i] for i in
                      rng.choice(len(all_edges),
                                 n_sample, replace=False)]
        rand_currents = []
        for u, v in rand_edges:
            w = G[u][v]["weight"]
            J = 2.0 * w * np.imag(
                np.conj(psi[u]) * psi[v])
            rand_currents.append(abs(J))

        J_geo  = np.mean(geo_currents)  if geo_currents  else 0.0
        J_rand = np.mean(rand_currents) if rand_currents else 1e-10
        ratio  = J_geo / (J_rand + 1e-12)

        results[bname] = {
            "J_geo":  J_geo,
            "J_rand": J_rand,
            "ratio":  ratio,
            "n_geo":  len(geo_currents),
        }

    return results


# ── 3b. Winding number of eigenvector phases
def winding_numbers(vecs, bands, G, source, target):
    """
    For each eigenmode k in highλ band:
    walk along geodesic path source → target.
    Track complex phase θ(j) = arg(vecs[j, k]).
    Winding number W_k = total phase winding / (2π).

    W_k ≠ 0 → topologically non-trivial mode.

    Returns: array of winding numbers for highλ modes.
    """
    path, _ = geo_corridor(G, source, target, radius=0)

    if len(path) < 3:
        return np.array([0.0])

    idx_high = bands["highλ"]
    windings = []

    for k in idx_high:
        v = vecs[:, k]
        # Complex phase along path
        phases = np.angle(v[path])
        # Phase differences
        dphi = np.diff(phases)
        # Unwrap to handle 2π jumps
        dphi_unwrapped = np.arctan2(
            np.sin(dphi), np.cos(dphi))
        total_winding = dphi_unwrapped.sum() / (2*np.pi)
        windings.append(total_winding)

    return np.array(windings)


# ── 3c. Betti number vs z_geo
def betti_vs_zgeo(results_6b):
    """
    Collect (β₁, z_geo_mean) pairs across algebras.
    Test Spearman correlation.
    """
    betti  = []
    z_geo  = []
    names  = []

    for name, res in results_6b.items():
        if "topo" in res and "z_high" in res:
            betti.append(res["topo"]["beta1"])
            z_geo.append(res["z_high"].mean())
            names.append(name)

    if len(betti) < 3:
        return None, None, names, betti, z_geo

    rho, p = stats.spearmanr(betti, z_geo)
    return float(rho), float(p), names, betti, z_geo


# ============================================================
# 4. MAIN
# ============================================================

def run_step6bc(seed=42, T=900, k=10, n_pairs=20):
    print("=" * 64)
    print("Part VIII Step 6bc: Why A6>E6? + Topological Invariants")
    print("=" * 64)

    alg_names = ["E6", "A6", "E8", "Random"]
    results   = {}

    for name in alg_names:
        print(f"\n{'='*56}")
        print(f"ALGEBRA: {name}")

        omega = get_omega(name, seed=seed)
        orbit = generate_orbit(omega, T=T,
                               kappa=0.05, seed=seed)
        X     = embed_torus(orbit)
        G, _  = build_graph(X, k=k)
        n_nodes = G.number_of_nodes()
        vals, vecs = graph_spectrum(G)
        bands = make_bands(vals)

        # ── 6b analyses
        mean_denom, med_denom = freq_rationality(omega)
        D_corr = correlation_dimension(orbit, seed=seed)
        topo   = graph_topology(G)
        ipr    = ipr_per_band(vecs, bands)
        z_high = multi_pair_zgeo(
            G, vals, vecs, bands,
            bname="highλ", n_pairs=n_pairs,
            t=50.0, seed=seed)

        print(f"\n  omega = {np.round(omega,4).tolist()}")
        print(f"  Freq rationality: "
              f"mean_denom={mean_denom:.2f}, "
              f"med_denom={med_denom:.2f}")
        print(f"  D_corr = {D_corr:.3f}")
        print(f"  Fiedler = {topo['fiedler']:.5f}")
        print(f"  β₁ (cycles) = {topo['beta1']}")
        print(f"  Mean path = {topo['mean_path']:.3f}")

        print(f"\n  IPR/ref by band:")
        for b in BAND_NAMES:
            r = ipr[b]["ratio"]
            print(f"    {b}: IPR/ref = {r:.3f}")

        print(f"\n  z_geo(highλ): "
              f"mean={z_high.mean():.3f} ± "
              f"{z_high.std()/max(1,np.sqrt(len(z_high))):.3f}")

        # ── Choose source/target for 6c
        degs   = np.array([G.degree(i) for i in G.nodes()])
        source = int(np.argmax(degs))
        geo_d  = {}
        for j, v in nx.single_source_dijkstra_path_length(
                G, source, weight="cost").items():
            geo_d[j] = v
        fin    = [(j, d) for j, d in geo_d.items()]
        target = max(fin, key=lambda x: x[1])[0]

        # ── 6c: geodesic current
        curr = geodesic_current(
            G, vals, vecs, bands,
            source=source, target=target, t=50.0)
        print(f"\n  Geodesic current ratio J_geo/J_rand:")
        for b in BAND_NAMES:
            r = curr[b]["ratio"]
            print(f"    {b}: {r:.4f}  "
                  f"(geo={curr[b]['J_geo']:.4e}, "
                  f"rand={curr[b]['J_rand']:.4e})")

        # ── 6c: winding numbers
        W = winding_numbers(vecs, bands, G, source, target)
        n_nonzero = np.sum(np.abs(W) > 0.1)
        W_mean    = float(np.abs(W).mean())
        W_max     = float(np.abs(W).max())
        print(f"\n  Winding numbers (highλ modes):")
        print(f"    |W| mean = {W_mean:.4f}")
        print(f"    |W| max  = {W_max:.4f}")
        print(f"    Non-trivial (|W|>0.1): "
              f"{n_nonzero} / {len(W)}")

        results[name] = {
            "omega":       omega,
            "G":           G,
            "vals":        vals,
            "vecs":        vecs,
            "bands":       bands,
            "source":      source,
            "target":      target,
            "freq_denom":  (mean_denom, med_denom),
            "D_corr":      D_corr,
            "topo":        topo,
            "ipr":         ipr,
            "z_high":      z_high,
            "current":     curr,
            "winding":     W,
            "color":       ALGEBRAS[name]["color"],
        }

    # ── β₁ vs z_geo correlation
    rho_b, p_b, nm_b, beta1_b, zgeo_b = \
        betti_vs_zgeo(results)
    print(f"\n[Betti β₁ vs z_geo correlation]")
    if rho_b is not None:
        for nm, b1, z in zip(nm_b, beta1_b, zgeo_b):
            print(f"  {nm}: β₁={b1}, z_geo={z:.3f}")
        print(f"  Spearman ρ={rho_b:.4f}, p={p_b:.4f}")
        if p_b < 0.05:
            print(f"  → β₁ correlates with z_geo ✓")
        else:
            print(f"  → No significant correlation ✗")

    results["_betti_test"] = {
        "rho": rho_b, "p": p_b,
        "names": nm_b, "beta1": beta1_b,
        "zgeo": zgeo_b,
    }

    return results


# ============================================================
# 5. FIGURE
# ============================================================

def make_figure(results,
                fname="part8_step6bc_topology.png"):

    alg_names = ["E6", "A6", "E8", "Random"]

    fig = plt.figure(figsize=(28, 28))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Part VIII — Step 6bc: Why A6 > E6? "
        "Topological Invariants\n"
        "Frequency rationality  |  Graph topology  |  "
        "Geodesic current  |  Winding numbers  |  "
        "Betti β₁",
        fontsize=14, fontweight="bold", y=1.001)

    gs = gridspec.GridSpec(5, 4, figure=fig,
                           hspace=0.52, wspace=0.35)

    alg_colors = [ALGEBRAS[nm]["color"]
                  for nm in alg_names]

    # ── Row 0: z_geo vs key properties scatter
    props = [
        ("freq_denom",  lambda r: r["freq_denom"][0],
         "Mean freq denominator\n(lower = more rational)"),
        ("D_corr",      lambda r: r["D_corr"],
         "Correlation dimension D_corr"),
        ("fiedler",     lambda r: r["topo"]["fiedler"],
         "Fiedler value\n(algebraic connectivity)"),
        ("beta1",       lambda r: r["topo"]["beta1"],
         "First Betti number β₁\n(number of graph cycles)"),
    ]

    for col, (pname, pfunc, plabel) in enumerate(props):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#f0f4f8")

        for nm, clr in zip(alg_names, alg_colors):
            if nm not in results:
                continue
            x_val = pfunc(results[nm])
            y_val = results[nm]["z_high"].mean()
            ax.scatter(x_val, y_val,
                       s=220, c=clr, zorder=5,
                       edgecolors="k", lw=2)
            ax.annotate(
                nm, xy=(x_val, y_val),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=10, fontweight="bold",
                color=clr)

        ax.axhline(2.0, color="green", ls="--",
                   lw=1.5, alpha=0.7)
        ax.set_xlabel(plabel, fontsize=9)
        ax.set_ylabel("z_geo(highλ)", fontsize=9)
        ax.set_title(f"z_geo vs {pname}",
                     fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # ── Row 1: IPR per band (all algebras)
    for col, bname in enumerate(BAND_NAMES):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("#f0f4f8")

        xs    = np.arange(len(alg_names))
        iprs  = [results[nm]["ipr"][bname]["ratio"]
                 if nm in results else 0.0
                 for nm in alg_names]
        bars  = ax.bar(xs, iprs,
                       color=alg_colors, alpha=0.75,
                       edgecolor="k", lw=1.2)

        ax.axhline(1.0, color="black", ls="--",
                   lw=1.5, label="extended (1×)")
        ax.set_xticks(xs)
        ax.set_xticklabels(alg_names, fontsize=10)
        ax.set_ylabel("IPR / (1/N)")
        ax.set_title(f"IPR ratio: {bname}\n"
                     f"1=extended, higher=localized",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # IPR summary across bands
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor("#f0f4f8")
    x_ipr = np.arange(len(BAND_NAMES))
    width = 0.2
    for ci, (nm, clr) in enumerate(
            zip(alg_names, alg_colors)):
        if nm not in results:
            continue
        vals_ipr = [results[nm]["ipr"][b]["ratio"]
                    for b in BAND_NAMES]
        ax.plot(x_ipr, vals_ipr,
                "o-", color=clr, lw=2.0,
                ms=9, label=nm, alpha=0.85)
    ax.set_xticks(x_ipr)
    ax.set_xticklabels(BAND_NAMES, fontsize=10)
    ax.set_ylabel("IPR / (1/N)")
    ax.set_title("IPR gradient by algebra\n"
                 "monotonic = structured hierarchy",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Geodesic current ratios
    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_facecolor("#f0f4f8")
    x_curr = np.arange(len(BAND_NAMES))
    for nm, clr in zip(alg_names, alg_colors):
        if nm not in results:
            continue
        ratios = [results[nm]["current"][b]["ratio"]
                  for b in BAND_NAMES]
        ax.plot(x_curr, ratios,
                "o-", color=clr, lw=2.2,
                ms=10, label=nm, alpha=0.85)
    ax.axhline(1.0, color="black", ls="--",
               lw=1.5, alpha=0.6,
               label="J_geo = J_rand")
    ax.set_xticks(x_curr)
    ax.set_xticklabels(BAND_NAMES, fontsize=11)
    ax.set_ylabel("J_geo / J_rand")
    ax.set_title("Geodesic current ratio\n"
                 ">1: more current along geodesic",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Winding numbers
    ax = fig.add_subplot(gs[2, 2:4])
    ax.set_facecolor("#f0f4f8")
    for nm, clr in zip(alg_names, alg_colors):
        if nm not in results:
            continue
        W   = np.abs(results[nm]["winding"])
        ax.hist(W, bins=20, color=clr,
                alpha=0.55, edgecolor="k", lw=0.4,
                label=f"{nm} (mean={W.mean():.3f})",
                density=True)
    ax.axvline(0.1, color="red", ls="--",
               lw=1.8, alpha=0.8,
               label="|W|=0.1 threshold")
    ax.set_xlabel("|Winding number|", fontsize=11)
    ax.set_ylabel("density")
    ax.set_title("Winding numbers of highλ eigenmodes\n"
                 "|W|>0.1: topologically non-trivial",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Betti β₁ vs z_geo
    ax = fig.add_subplot(gs[3, 0:2])
    ax.set_facecolor("#f0f4f8")
    bt = results.get("_betti_test", {})
    if bt.get("names"):
        for nm, b1, z, clr in zip(
                bt["names"], bt["beta1"],
                bt["zgeo"], alg_colors):
            ax.scatter(b1, z, s=250, c=clr,
                       zorder=5, edgecolors="k", lw=2)
            ax.annotate(nm, xy=(b1, z),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=12, fontweight="bold",
                        color=clr)
        # Regression line
        if bt.get("rho") is not None:
            b_arr = np.array(bt["beta1"])
            z_arr = np.array(bt["zgeo"])
            if len(b_arr) >= 2:
                sl, ic, _, _, _ = \
                    stats.linregress(b_arr, z_arr)
                x_fit = np.linspace(
                    b_arr.min(), b_arr.max(), 50)
                ax.plot(x_fit, sl*x_fit + ic,
                        "k--", lw=2, alpha=0.6)
            rho = bt["rho"]; pv = bt["p"]
            ax.set_title(
                f"Betti β₁ vs z_geo(highλ)\n"
                f"ρ={rho:.3f}, p={pv:.4f}  "
                f"{'correlated ✓' if pv < 0.05 else 'ns'}",
                fontsize=9, fontweight="bold")
    ax.axhline(2.0, color="green", ls="--",
               lw=1.5, alpha=0.7)
    ax.set_xlabel("First Betti number β₁ (graph cycles)",
                  fontsize=11)
    ax.set_ylabel("z_geo(highλ)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Row 3: Summary table
    ax = fig.add_subplot(gs[3, 2:4])
    ax.set_facecolor("#e8eaf6")
    ax.axis("off")

    rows_txt = [
        ["Algebra", "z_geo", "β₁", "Fiedler",
         "D_corr", "mean_denom"],
    ]
    for nm in alg_names:
        if nm not in results:
            continue
        r = results[nm]
        rows_txt.append([
            nm,
            f"{r['z_high'].mean():.2f}",
            f"{r['topo']['beta1']}",
            f"{r['topo']['fiedler']:.5f}",
            f"{r['D_corr']:.3f}",
            f"{r['freq_denom'][0]:.1f}",
        ])

    table = ax.table(
        cellText=rows_txt[1:],
        colLabels=rows_txt[0],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.1, 1.0, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#3F51B5")
            cell.set_text_props(color="white",
                                fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#E8EAF6")
        cell.set_edgecolor("#BDBDBD")

    ax.set_title("Summary table",
                 fontsize=11, fontweight="bold",
                 pad=20)

    # ── Row 4: z_geo violin (all algebras)
    ax = fig.add_subplot(gs[4, 0:3])
    ax.set_facecolor("#f0f4f8")
    data_viol = [results[nm]["z_high"]
                 for nm in alg_names if nm in results]
    pos_v     = np.arange(1, len(alg_names)+1)
    vp = ax.violinplot(data_viol, positions=pos_v,
                       showmedians=True,
                       showextrema=True)
    for pc, clr in zip(vp["bodies"], alg_colors):
        pc.set_facecolor(clr); pc.set_alpha(0.65)
    ax.axhline(2.0, color="green", ls="--",
               lw=2.0, label="z=2 threshold")
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    ax.set_xticks(pos_v)
    ax.set_xticklabels(alg_names, fontsize=13)
    ax.set_ylabel("z_geo(highλ)", fontsize=12)
    ax.set_title("z_geo(highλ): full distribution\n"
                 "Is Coxeter vs Random the key split?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Final interpretation panel
    ax = fig.add_subplot(gs[4, 3])
    ax.set_facecolor("#e8f5e9")
    ax.axis("off")
    txt = [
        "KEY FINDINGS 6bc",
        "=" * 20,
        "",
        "WHY A6 > E6?",
        "→ More rational freqs",
        "  (lower denominator)",
        "→ Different D_corr?",
        "→ Different β₁?",
        "",
        "TOPOLOGICAL TESTS:",
        "J_geo/J_rand:",
        "  >1 → geodesic current",
        "",
        "|W| winding number:",
        "  >0.1 → non-trivial",
        "",
        "β₁ vs z_geo:",
        "  ρ>0 → cycles drive",
        "  focusing",
        "",
        "HONEST LIMIT:",
        "These are graph-theoretic",
        "invariants, not QFT",
        "topological invariants.",
    ]
    ax.text(0.05, 0.97, "\n".join(txt),
            transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            va="top", color="#1b5e20")

    plt.savefig(fname, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {fname}")


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

def final_summary(results):
    alg_names = ["E6", "A6", "E8", "Random"]

    print("\n" + "="*72)
    print("FINAL SUMMARY — Step 6bc")
    print("="*72)

    print(f"\n{'Alg':<8} {'z_geo':>8} {'β₁':>8} "
          f"{'Fiedler':>10} {'D_corr':>8} "
          f"{'denom':>8} {'J_ratio_hi':>12} "
          f"{'|W|_mean':>10}")
    print("-"*82)

    for nm in alg_names:
        if nm not in results:
            continue
        r   = results[nm]
        z   = r["z_high"].mean()
        b1  = r["topo"]["beta1"]
        fi  = r["topo"]["fiedler"]
        dc  = r["D_corr"]
        dn  = r["freq_denom"][0]
        jr  = r["current"]["highλ"]["ratio"]
        wm  = float(np.abs(r["winding"]).mean())
        print(f"{nm:<8} {z:>8.3f} {b1:>8d} "
              f"{fi:>10.5f} {dc:>8.3f} "
              f"{dn:>8.2f} {jr:>12.4f} {wm:>10.4f}")

    # Key comparisons
    print("\n[Key comparisons]")
    if "E6" in results and "A6" in results:
        z_e6 = results["E6"]["z_high"]
        z_a6 = results["A6"]["z_high"]
        _, p = stats.mannwhitneyu(
            z_a6, z_e6, alternative="greater")
        print(f"  A6 vs E6 (z_high): "
              f"A6={z_a6.mean():.3f}, E6={z_e6.mean():.3f}, "
              f"p(A6>E6)={p:.4f}")

    if "E6" in results and "Random" in results:
        z_e6  = results["E6"]["z_high"]
        z_rnd = results["Random"]["z_high"]
        _, p  = stats.mannwhitneyu(
            z_e6, z_rnd, alternative="greater")
        print(f"  E6 vs Random: "
              f"E6={z_e6.mean():.3f}, "
              f"Rnd={z_rnd.mean():.3f}, "
              f"p(E6>Rnd)={p:.4e}")

    # Betti test
    bt = results.get("_betti_test", {})
    if bt.get("rho") is not None:
        print(f"\n[Betti β₁ correlation]")
        print(f"  ρ(β₁, z_geo) = {bt['rho']:.4f}, "
              f"p = {bt['p']:.4f}")

    # Winding numbers
    print(f"\n[Winding numbers |W| > 0.1]")
    for nm in alg_names:
        if nm not in results:
            continue
        W     = np.abs(results[nm]["winding"])
        frac  = (W > 0.1).mean()
        print(f"  {nm}: {frac:.2%} non-trivial modes")

    # Why A6 > E6? Summary
    print("\n[Hypothesis evaluation: Why A6 > E6?]")
    if "E6" in results and "A6" in results:
        a6, e6 = results["A6"], results["E6"]
        tests = [
            ("Freq rationality",
             a6["freq_denom"][0] < e6["freq_denom"][0],
             f"A6 denom={a6['freq_denom'][0]:.1f} "
             f"< E6 denom={e6['freq_denom'][0]:.1f}"),
            ("More graph cycles (β₁)",
             a6["topo"]["beta1"] > e6["topo"]["beta1"],
             f"A6 β₁={a6['topo']['beta1']} "
             f"vs E6 β₁={e6['topo']['beta1']}"),
            ("Higher connectivity",
             a6["topo"]["fiedler"] > e6["topo"]["fiedler"],
             f"A6 Fiedler={a6['topo']['fiedler']:.4f} "
             f"vs E6={e6['topo']['fiedler']:.4f}"),
            ("Higher geodesic current",
             a6["current"]["highλ"]["ratio"] >
             e6["current"]["highλ"]["ratio"],
             f"A6 ratio={a6['current']['highλ']['ratio']:.3f} "
             f"vs E6={e6['current']['highλ']['ratio']:.3f}"),
        ]
        for hyp, confirmed, detail in tests:
            mark = "✓" if confirmed else "✗"
            print(f"  H({hyp}): {mark}  {detail}")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Part VIII Step 6bc\n")
    print("Estimated runtime: 10-20 minutes\n")

    results = run_step6bc(
        seed=42, T=900, k=10, n_pairs=20)
    make_figure(results)
    final_summary(results)
