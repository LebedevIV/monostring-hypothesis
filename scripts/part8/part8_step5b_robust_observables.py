"""
Part VIII — Step 5b: Transport Classes with Robust Observables
==============================================================

Three honest observables (no first-arrival threshold bias):

  1. SPREADING LAW  <d(t)> and <d²(t)>
     ballistic:  <d> ~ t        (field-like propagation)
     diffusive:  <d²> ~ t       (random-walk-like)
     localized:  saturates       (Anderson-like)

  2. TIME-AVERAGED GEODESIC OVERLAP  Z_geo(t)
     z-score of probability mass in geodesic corridor,
     integrated over time.
     z > 2: signal concentrates near shortest paths
     z ~ 0: random spatial distribution

  3. INVERSE PARTICIPATION RATIO (IPR)
     IPR = sum_i |psi_i|^4
     IPR ~ 1/N : extended (field-like)
     IPR ~ 1   : fully localized (particle-like)

Null hypotheses:
  H0_spread:  all bands show the same spreading law exponent
  H0_geo:     all bands have z_geo ~ 0 (no geodesic focusing)
  H0_ipr:     all bands have IPR ~ 1/N (equally extended)

If H0 rejected for some bands but not others:
  -> evidence for multiple distinct transport classes
  -> "multiple fields from one graph"
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from scipy import stats
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
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
# 3. TORUS EMBEDDING  phi_i -> (cos phi_i, sin phi_i)
# ============================================================

def embed_torus(orbit):
    return np.concatenate([np.cos(orbit), np.sin(orbit)], axis=1)


# ============================================================
# 4. GRAPH
# ============================================================

def build_knn_graph(X, k=10):
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
            if (a, b) not in edge_best or d < edge_best[(a,b)]["cost"]:
                edge_best[(a, b)] = {
                    "cost":   d,
                    "weight": float(np.exp(-d**2 / (2*sigma**2)))
                }

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (a, b), data in edge_best.items():
        G.add_edge(a, b, cost=data["cost"], weight=data["weight"])
    return G, sigma


def extract_giant(G):
    comps  = list(nx.connected_components(G))
    giant  = max(comps, key=len)
    Gg     = G.subgraph(giant).copy()
    mapping = {old: i for i, old in enumerate(sorted(Gg.nodes()))}
    Gg     = nx.relabel_nodes(Gg, mapping)
    return Gg


def graph_to_matrices(G):
    n   = G.number_of_nodes()
    A   = np.zeros((n, n))
    C   = np.full((n, n), np.inf)
    for i, j, data in G.edges(data=True):
        A[i,j] = A[j,i] = data["weight"]
        C[i,j] = C[j,i] = data["cost"]
    deg  = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0/np.sqrt(deg), 0.0)
    D12  = np.diag(dinv)
    Ln   = np.eye(n) - D12 @ A @ D12
    Ln   = 0.5*(Ln + Ln.T)
    return A, C, Ln, deg


def spectral_decomp(Ln):
    vals, vecs = np.linalg.eigh(Ln)
    vals = np.maximum(vals, 0.0)
    return vals, vecs


def make_bands(vals, frac=0.20):
    nz = np.where(vals > 1e-10)[0]
    m  = len(nz)
    q  = max(8, int(frac * m))
    low  = nz[:q]
    mc   = m // 2
    mid  = nz[max(0, mc - q//2): min(m, mc + q//2)]
    high = nz[-q:]
    return {"lowλ": low, "midλ": mid, "highλ": high}


# ============================================================
# 5. SOURCE + GEODESIC DISTANCES
# ============================================================

def choose_source(G, n_cand=20):
    degs  = np.array([G.degree(i) for i in G.nodes()])
    cands = np.argsort(degs)[-min(n_cand, len(degs)):]
    best_node, best_mean = None, np.inf
    for c in cands:
        lens = nx.single_source_dijkstra_path_length(G, int(c), weight="cost")
        mu   = np.mean(list(lens.values()))
        if mu < best_mean:
            best_mean, best_node = mu, int(c)
    return best_node, best_mean


def all_geodesics_from(G, source):
    """Returns array d[j] = cost-geodesic from source to j."""
    n = G.number_of_nodes()
    lens = nx.single_source_dijkstra_path_length(G, source, weight="cost")
    d = np.full(n, np.nan)
    for j, v in lens.items():
        d[j] = v
    return d


def geodesic_corridor(G, source, target, radius=1):
    path = nx.shortest_path(G, source, target, weight="cost")
    S    = set(path)
    front = set(path)
    for _ in range(radius):
        nxt = set()
        for v in front:
            nxt.update(G.neighbors(v))
        S.update(nxt); front = nxt
    return np.array(sorted(S), dtype=int)


# ============================================================
# 6. BAND STATE
# ============================================================

def make_band_state(vecs, band_idx, source):
    n   = vecs.shape[0]
    d   = np.zeros(n); d[source] = 1.0
    c   = vecs.T @ d
    cb  = np.zeros_like(c); cb[band_idx] = c[band_idx]
    psi = vecs @ cb
    nm  = np.linalg.norm(psi)
    if nm > 0:
        psi /= nm
    return psi.astype(complex), (vecs.T @ psi).astype(complex)


# ============================================================
# 7. SPECTRAL EVOLUTION  psi(t) = sum_k coeff_k exp(-i lam_k t) v_k
# ============================================================

def prob_at_time(vals, vecs, coeff0, t):
    """Full probability vector P_j(t) for all nodes j."""
    amp = vecs @ (coeff0 * np.exp(-1j * vals * t))
    return np.abs(amp)**2


# ============================================================
# 8. OBSERVABLE 1 — SPREADING  <d(t)>  <d²(t)>
# ============================================================

def compute_spreading(vals, vecs, coeff0, geo_dist, times):
    """
    <d(t)>  = sum_j d(s,j) * P_j(t)
    <d²(t)> = sum_j d(s,j)^2 * P_j(t)

    Uses only nodes with finite geodesic distance.
    """
    mask  = np.isfinite(geo_dist)
    d_fin = geo_dist[mask]

    mean_d  = np.zeros(len(times))
    mean_d2 = np.zeros(len(times))

    for ti, t in enumerate(times):
        P       = prob_at_time(vals, vecs, coeff0, t)
        Pm      = P[mask]
        norm    = Pm.sum()
        if norm > 0:
            Pm /= norm
        mean_d[ti]  = (d_fin   * Pm).sum()
        mean_d2[ti] = (d_fin**2 * Pm).sum()

    return mean_d, mean_d2


def fit_spreading(times, mean_d, mean_d2, t_fit_start_frac=0.10):
    """
    Fit <d> ~ A * t^alpha  and  <d²> ~ B * t^beta
    in log-log space over the second half of the time range.

    alpha ~ 1  : ballistic
    alpha ~ 0.5: diffusive (beta ~ 1)
    alpha ~ 0  : localized (saturation)
    """
    ns    = max(4, int(t_fit_start_frac * len(times)))
    t_fit = times[ns:]
    d_fit = mean_d[ns:]
    d2_fit = mean_d2[ns:]

    results = {}

    for name, y in [("<d>", d_fit), ("<d²>", d2_fit)]:
        pos   = (t_fit > 0) & (y > 0)
        if pos.sum() < 4:
            results[name] = {"alpha": np.nan, "r2": np.nan}
            continue
        lt = np.log(t_fit[pos])
        ly = np.log(y[pos])
        sl, ic, r, pv, _ = stats.linregress(lt, ly)
        results[name] = {
            "alpha": sl,
            "r2":    r**2,
            "A":     np.exp(ic),
            "pval":  pv,
        }

    # Classify
    a = results.get("<d>", {}).get("alpha", np.nan)
    if np.isnan(a):
        cls = "unknown"
    elif a > 0.75:
        cls = "ballistic"
    elif a > 0.35:
        cls = "diffusive"
    else:
        cls = "localized"
    results["class"] = cls

    return results


# ============================================================
# 9. OBSERVABLE 2 — TIME-AVERAGED GEODESIC OVERLAP
# ============================================================

def time_avg_geodesic_overlap(vals, vecs, coeff0, G,
                               source, target,
                               times, n_rand=120,
                               corridor_radius=1, seed=42):
    """
    For each time snapshot compute z-score of
    probability mass inside the geodesic corridor.
    Return mean z-score over time.

    Uses corridor_nodes from shortest-path source->target.
    """
    rng      = np.random.default_rng(seed)
    n        = G.number_of_nodes()
    corridor = geodesic_corridor(G, source, target, corridor_radius)
    m        = len(corridor)

    z_series = np.zeros(len(times))

    for ti, t in enumerate(times):
        P      = prob_at_time(vals, vecs, coeff0, t)
        p_geo  = P[corridor].sum()

        # baseline: random sets of size m
        rand_p = []
        for _ in range(n_rand):
            samp = rng.choice(n, size=m, replace=False)
            rand_p.append(P[samp].sum())
        rand_p = np.array(rand_p)
        mu, sd = rand_p.mean(), rand_p.std(ddof=1) + 1e-12
        z_series[ti] = (p_geo - mu) / sd

    return float(z_series.mean()), z_series


# ============================================================
# 10. OBSERVABLE 3 — IPR
# ============================================================

def compute_ipr_per_mode(vecs, band_idx):
    """
    IPR_k = sum_j |v_{jk}|^4  for each eigenvector k in band.

    IPR ~ 1/N : plane-wave-like (extended)
    IPR ~ 1   : fully localized (delta-function)
    """
    n        = vecs.shape[0]
    ipr_vals = np.sum(vecs[:, band_idx]**4, axis=0)
    return ipr_vals, float(ipr_vals.mean()), float(ipr_vals.std())


def compute_ipr_of_state(psi):
    """
    IPR of the probability distribution |psi_j|^2.
    """
    p = np.abs(psi)**2
    p = p / (p.sum() + 1e-14)
    return float(np.sum(p**2))


# ============================================================
# 11. MAIN ANALYSIS
# ============================================================

BAND_COLORS = {
    "lowλ":  "#1565C0",
    "midλ":  "#FB8C00",
    "highλ": "#C62828",
}
BAND_NAMES = ["lowλ", "midλ", "highλ"]


def run_one(name, T=900, k_nn=10,
            tmax=200.0, n_times=400, seed=42):

    color = ALGEBRAS[name]["color"]
    omega = get_omega(name, seed=seed)

    print("\n" + "="*62)
    print(f"ALGEBRA: {name}")
    print(f"omega = {np.round(omega,4).tolist()}")

    orbit = generate_orbit(omega, T=T, kappa=0.05,
                           warmup=500, seed=seed)
    X     = embed_torus(orbit)
    G0, sigma = build_knn_graph(X, k=k_nn)
    G     = extract_giant(G0)

    n = G.number_of_nodes()
    print(f"Graph: {n} nodes, {G.number_of_edges()} edges, σ={sigma:.4f}")

    A, C, Ln, deg = graph_to_matrices(G)
    vals, vecs    = spectral_decomp(Ln)
    bands         = make_bands(vals, frac=0.20)

    print(f"Spectrum: λ_min={vals[0]:.2e}, λ_max={vals[-1]:.4f}")
    for bname, idx in bands.items():
        lmin = vals[idx[0]]; lmax = vals[idx[-1]]
        print(f"  {bname}: {len(idx)} modes, "
              f"λ∈[{lmin:.4f},{lmax:.4f}]")

    source, _ = choose_source(G)
    geo_dist  = all_geodesics_from(G, source)

    # Far target for corridor tests
    finite_mask = np.where(np.isfinite(geo_dist))[0]
    far_target  = int(finite_mask[np.argmax(geo_dist[finite_mask])])
    print(f"Source: {source},  Far target: {far_target}, "
          f"d_geo={geo_dist[far_target]:.4f}")

    times = np.linspace(0.0, tmax, n_times)

    band_res = {}

    for bname in BAND_NAMES:
        idx = bands[bname]
        psi0, c0 = make_band_state(vecs, idx, source)

        # ── Observable 1: Spreading
        mean_d, mean_d2 = compute_spreading(
            vals, vecs, c0, geo_dist, times)
        spr = fit_spreading(times, mean_d, mean_d2)

        # ── Observable 2: Time-averaged geodesic overlap
        z_avg, z_series = time_avg_geodesic_overlap(
            vals, vecs, c0, G,
            source=source, target=far_target,
            times=times, n_rand=120,
            corridor_radius=1, seed=seed)

        # ── Observable 3: IPR of eigenmodes
        ipr_modes, ipr_mean, ipr_std = compute_ipr_per_mode(vecs, idx)

        # IPR of initial state
        ipr_state = compute_ipr_of_state(psi0)

        # Extended reference IPR = 1/n
        ipr_ref = 1.0 / n

        lam_c = float(np.median(vals[idx]))

        print(f"\n  [{bname}]  λ_center={lam_c:.4f}  "
              f"sqrt(λ)={np.sqrt(lam_c):.4f}")
        print(f"    Spreading class    : {spr['class']}")
        a_d  = spr.get('<d>',  {}).get('alpha', np.nan)
        r2_d = spr.get('<d>',  {}).get('r2',    np.nan)
        a_d2 = spr.get('<d²>', {}).get('alpha', np.nan)
        print(f"    <d>  exponent α    : {a_d:.4f}  "
              f"(R²={r2_d:.4f})")
        print(f"    <d²> exponent β    : {a_d2:.4f}")
        print(f"    Time-avg z_geo     : {z_avg:.4f}")
        print(f"    IPR(modes) mean    : {ipr_mean:.6f}  "
              f"(ref 1/N={ipr_ref:.6f})")
        print(f"    IPR(modes)/ref     : {ipr_mean/ipr_ref:.3f}")
        print(f"    IPR(state)         : {ipr_state:.6f}")

        band_res[bname] = {
            "idx":        idx,
            "psi0":       psi0,
            "c0":         c0,
            "mean_d":     mean_d,
            "mean_d2":    mean_d2,
            "spreading":  spr,
            "z_avg":      z_avg,
            "z_series":   z_series,
            "ipr_modes":  ipr_modes,
            "ipr_mean":   ipr_mean,
            "ipr_std":    ipr_std,
            "ipr_state":  ipr_state,
            "ipr_ref":    ipr_ref,
            "lambda_c":   lam_c,
        }

    return {
        "name":      name,
        "color":     color,
        "omega":     omega,
        "G":         G,
        "vals":      vals,
        "vecs":      vecs,
        "bands":     bands,
        "times":     times,
        "source":    source,
        "far_target":far_target,
        "geo_dist":  geo_dist,
        "band_res":  band_res,
        "n":         n,
    }


# ============================================================
# 12. STATISTICAL TESTS
# ============================================================

def band_pairwise_tests(band_res, alg_name):
    """
    Test whether bands are statistically distinct
    on IPR and z_geo.
    """
    print(f"\n  [{alg_name}] Pairwise band tests:")

    # IPR comparison: Welch t-test on per-mode IPR
    pairs = [("lowλ","highλ"), ("lowλ","midλ"), ("midλ","highλ")]
    for b1, b2 in pairs:
        x1 = band_res[b1]["ipr_modes"]
        x2 = band_res[b2]["ipr_modes"]
        t, p = stats.ttest_ind(x1, x2, equal_var=False)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else
              ("*" if p < 0.05 else "ns"))
        print(f"    IPR({b1}) vs IPR({b2}): "
              f"t={t:+.3f}, p={p:.3e}  {sig}")

    # z_geo comparison (scalar per band, just print)
    z_vals = {b: band_res[b]["z_avg"] for b in BAND_NAMES}
    print(f"    z_geo: " +
          "  ".join(f"{b}={z_vals[b]:.3f}" for b in BAND_NAMES))
    print(f"    Spreading class: " +
          "  ".join(f"{b}={band_res[b]['spreading']['class']}"
                    for b in BAND_NAMES))


# ============================================================
# 13. FIGURE
# ============================================================

def make_figure(all_results,
                fname="part8_step5b_robust_observables.png"):

    algs   = list(all_results.keys())
    n_algs = len(algs)

    # 4 rows per algebra:
    # row 0: spreading <d(t)>
    # row 1: spreading <d²(t)>
    # row 2: z_geo(t) time series
    # row 3: IPR per mode (all bands)
    n_rows = 4 * n_algs
    n_cols = len(BAND_NAMES) + 1          # 3 bands + summary

    fig = plt.figure(figsize=(6*(n_cols), 4.5*n_rows))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Part VIII — Step 5b: Transport Classes with Robust Observables\n"
        "Spreading law  |  Time-averaged geodesic overlap  |  IPR",
        fontsize=15, fontweight="bold", y=1.001)

    gs = gridspec.GridSpec(n_rows, n_cols,
                           figure=fig,
                           hspace=0.52, wspace=0.32)

    for alg_row, alg in enumerate(algs):
        res   = all_results[alg]
        times = res["times"]
        n_pts = res["n"]
        color = res["color"]
        row0  = alg_row * 4

        # ── shared title strip
        for c in range(n_cols):
            ax = fig.add_subplot(gs[row0, c])
            if c == 0:
                ax.set_facecolor("#e8eaf6")
                ax.text(0.5, 0.5,
                        f"{alg}  (n={n_pts} nodes)",
                        ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="#1a237e",
                        transform=ax.transAxes)
            ax.axis("off")

        # Per-band rows
        for bi, bname in enumerate(BAND_NAMES):
            br  = res["band_res"][bname]
            clr = BAND_COLORS[bname]
            spr = br["spreading"]

            # ── Row 1: <d(t)>
            ax = fig.add_subplot(gs[row0+1, bi])
            ax.set_facecolor("#f0f4f8")
            md = br["mean_d"]
            ax.plot(times, md, color=clr, lw=2.0, alpha=0.85)

            # power-law overlay
            a  = spr.get("<d>",  {}).get("alpha", np.nan)
            A_ = spr.get("<d>",  {}).get("A",     np.nan)
            if np.isfinite(a) and np.isfinite(A_):
                t_ov = times[times > 0]
                ax.plot(t_ov, A_ * t_ov**a, "k--",
                        lw=1.8, alpha=0.6,
                        label=f"α={a:.3f}")
                ax.legend(fontsize=9)

            ax.set_title(f"{alg} | {bname}\n"
                         f"<d(t)>  class={spr['class']}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("time t"); ax.set_ylabel("<d(t)>")
            ax.grid(True, alpha=0.3)

            # ── Row 2: <d²(t)>
            ax = fig.add_subplot(gs[row0+2, bi])
            ax.set_facecolor("#f0f4f8")
            md2 = br["mean_d2"]
            ax.plot(times, md2, color=clr, lw=2.0, alpha=0.85)

            b_  = spr.get("<d²>", {}).get("alpha", np.nan)
            B_  = spr.get("<d²>", {}).get("A",     np.nan)
            if np.isfinite(b_) and np.isfinite(B_):
                t_ov = times[times > 0]
                ax.plot(t_ov, B_ * t_ov**b_, "k--",
                        lw=1.8, alpha=0.6,
                        label=f"β={b_:.3f}")
                ax.legend(fontsize=9)

            ax.set_title(f"{alg} | {bname}\n<d²(t)>",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("time t"); ax.set_ylabel("<d²(t)>")
            ax.grid(True, alpha=0.3)

            # ── Row 3: z_geo(t)
            ax = fig.add_subplot(gs[row0+3, bi])
            ax.set_facecolor("#f0f4f8")
            zs = br["z_series"]
            ax.plot(times, zs, color=clr, lw=1.5, alpha=0.7)
            ax.axhline(br["z_avg"], color=clr, lw=2.5,
                       ls="-", label=f"mean z={br['z_avg']:.3f}")
            ax.axhline(2.0,  color="green",  ls="--",
                       lw=1.5, alpha=0.7, label="z=2")
            ax.axhline(0.0,  color="black",  ls=":",
                       lw=1.0, alpha=0.5)
            ax.axhline(-2.0, color="red",    ls="--",
                       lw=1.5, alpha=0.7, label="z=-2")
            ax.set_title(f"{alg} | {bname}\n"
                         f"Geodesic overlap z(t)",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("time t"); ax.set_ylabel("z-score")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # ── Summary column (col 3): IPR + bar summary
        ax_ipr = fig.add_subplot(gs[row0+1:row0+3, n_cols-1])
        ax_ipr.set_facecolor("#f0f4f8")

        for bname in BAND_NAMES:
            br  = res["band_res"][bname]
            ipr = br["ipr_modes"]
            clr = BAND_COLORS[bname]
            ipr_ref = br["ipr_ref"]
            ax_ipr.hist(ipr / ipr_ref, bins=30, color=clr,
                        alpha=0.55, edgecolor="k", lw=0.4,
                        label=f"{bname} mean={br['ipr_mean']/ipr_ref:.2f}×")

        ax_ipr.axvline(1.0, color="black", ls="--",
                       lw=2, label="extended ref=1×(1/N)")
        ax_ipr.set_xlabel("IPR / (1/N)  [1=extended, N=localized]",
                          fontsize=9)
        ax_ipr.set_ylabel("mode count")
        ax_ipr.set_title(f"{alg}: IPR distribution\nper eigenvector",
                         fontsize=10, fontweight="bold")
        ax_ipr.legend(fontsize=8)
        ax_ipr.grid(True, alpha=0.3)

        # Bar summary of z_avg and alpha_d
        ax_bar = fig.add_subplot(gs[row0+3, n_cols-1])
        ax_bar.set_facecolor("#f0f4f8")

        xs     = np.arange(len(BAND_NAMES))
        z_avgs = [res["band_res"][b]["z_avg"]
                  for b in BAND_NAMES]
        alphas = [res["band_res"][b]["spreading"]
                  .get("<d>", {}).get("alpha", np.nan)
                  for b in BAND_NAMES]
        clrs   = [BAND_COLORS[b] for b in BAND_NAMES]

        ax2    = ax_bar.twinx()

        bars1 = ax_bar.bar(xs - 0.2, z_avgs, width=0.38,
                           color=clrs, alpha=0.8,
                           edgecolor="k", lw=1.0,
                           label="z_geo")
        bars2 = ax2.bar(xs + 0.2, alphas, width=0.38,
                        color=clrs, alpha=0.35,
                        edgecolor="k", lw=1.0,
                        hatch="//", label="α_spread")

        ax_bar.axhline(2.0,  color="green", ls="--",
                       lw=1.5, alpha=0.7)
        ax_bar.axhline(-2.0, color="red",   ls="--",
                       lw=1.5, alpha=0.7)
        ax2.axhline(1.0,  color="blue",   ls=":",
                    lw=1.5, alpha=0.7,
                    label="α=1 (ballistic)")
        ax2.axhline(0.5,  color="orange", ls=":",
                    lw=1.5, alpha=0.7,
                    label="α=0.5 (diffusive)")

        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels(BAND_NAMES, fontsize=10)
        ax_bar.set_ylabel("z_geo (bars solid)", fontsize=9)
        ax2.set_ylabel("α spreading (bars hatch)", fontsize=9)
        ax_bar.set_title(f"{alg}: summary", fontsize=10,
                         fontweight="bold")
        ax_bar.grid(True, alpha=0.2, axis="y")

        # Combine legends
        h1, l1 = ax_bar.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_bar.legend(h1+h2, l1+l2, fontsize=8, loc="upper right")

    plt.savefig(fname, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {fname}")


# ============================================================
# 14. SUMMARY TABLE
# ============================================================

def final_summary(all_results):
    print("\n" + "="*90)
    print("FINAL SUMMARY — STEP 5b")
    print("="*90)
    hdr = (f"{'Alg':<10} {'Band':<8} "
           f"{'λ_c':>8} {'class':>12} "
           f"{'α<d>':>8} {'β<d²>':>8} "
           f"{'z_avg':>8} "
           f"{'IPR/ref':>9}")
    print(hdr)
    print("-"*90)

    for alg, res in all_results.items():
        for bname in BAND_NAMES:
            br  = res["band_res"][bname]
            spr = br["spreading"]
            a   = spr.get("<d>",  {}).get("alpha", np.nan)
            b   = spr.get("<d²>", {}).get("alpha", np.nan)
            cls = spr["class"]
            z   = br["z_avg"]
            ipr_ratio = br["ipr_mean"] / br["ipr_ref"]
            lc  = br["lambda_c"]
            print(f"{alg:<10} {bname:<8} "
                  f"{lc:>8.4f} {cls:>12} "
                  f"{a:>8.4f} {b:>8.4f} "
                  f"{z:>8.4f} "
                  f"{ipr_ratio:>9.3f}")
        print()

    print("-"*90)
    print("\nH0 rejection guide:")
    print("  Spreading:  bands differ in class (ballistic/diffusive/localized)?")
    print("  z_geo > 2:  band concentrates along geodesic corridor?")
    print("  IPR/ref:    1× = extended (field-like),  >>1 = localized (particle-like)")
    print("\nE6 vs Random comparison:")

    for bname in BAND_NAMES:
        e6z  = all_results["E6"]["band_res"][bname]["z_avg"]
        rnz  = all_results["Random"]["band_res"][bname]["z_avg"]
        e6c  = all_results["E6"]["band_res"][bname]["spreading"]["class"]
        rnc  = all_results["Random"]["band_res"][bname]["spreading"]["class"]
        e6ip = (all_results["E6"]["band_res"][bname]["ipr_mean"]
                / all_results["E6"]["band_res"][bname]["ipr_ref"])
        rnip = (all_results["Random"]["band_res"][bname]["ipr_mean"]
                / all_results["Random"]["band_res"][bname]["ipr_ref"])
        print(f"  {bname}: z_geo  E6={e6z:.3f}  Random={rnz:.3f}  "
              f"| class  E6={e6c:<10}  Random={rnc:<10}  "
              f"| IPR/ref  E6={e6ip:.3f}  Random={rnip:.3f}")


# ============================================================
# 15. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Part VIII Step 5b: Transport Classes — Robust Observables\n")

    all_results = {}
    for alg in ["E6", "Random"]:
        res = run_one(alg, T=900, k_nn=10,
                      tmax=200.0, n_times=400, seed=42)
        band_pairwise_tests(res["band_res"], alg)
        all_results[alg] = res

    make_figure(all_results)
    final_summary(all_results)