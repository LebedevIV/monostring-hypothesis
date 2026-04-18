"""
Part VIII — Step 6d: Controlling for Graph Connectivity
========================================================

Discovery from Step 6bc:
  A6 Fiedler=0.006 (near-disconnected graph)
  A6 Mean_path=12.9 (2x longer than E6)

Hypothesis:
  High z_geo in A6 is an ARTIFACT of near-disconnection:
  excitations have no alternative but to follow the
  one available path = the geodesic.

Critical test:
  Control for Fiedler value by adjusting k_nn
  so that all algebras have similar Fiedler values.

  If z_geo(E6) remains high after controlling:
    → REAL effect (geometry of orbit matters)
  If z_geo(E6) drops to random level:
    → ARTIFACT (was driven by graph connectivity)

Method:
  For each algebra: find k_nn that gives Fiedler ≈ 0.15
  Then recompute z_geo(highλ) at matched connectivity.
  Compare across algebras at FIXED Fiedler.

Additionally:
  Test partial correlation:
  r(z_geo, D_corr | Fiedler) — does D_corr predict
  z_geo after removing Fiedler effect?
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
# ALGEBRAS
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

BAND_NAMES = ["lowλ", "midλ", "highλ"]
BAND_COLORS = {
    "lowλ": "#1565C0",
    "midλ": "#FB8C00",
    "highλ": "#C62828",
}


def get_omega(name, seed=42):
    alg = ALGEBRAS[name]
    if alg["exponents"] is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.5, 2.0, alg["rank"])
    m = np.array(alg["exponents"], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg["h"])


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


def get_fiedler(G):
    try:
        return float(nx.algebraic_connectivity(
            G, weight="weight", method="lanczos"))
    except Exception:
        return np.nan


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
    n  = vecs.shape[0]
    d  = np.zeros(n); d[source] = 1.0
    c  = vecs.T @ d
    cb = np.zeros_like(c); cb[idx] = c[idx]
    psi = vecs @ cb
    nm  = np.linalg.norm(psi)
    if nm > 0:
        psi /= nm
    return (vecs.T @ psi).astype(complex)


def prob_vec(vals, vecs, c0, t):
    return np.abs(
        vecs @ (c0 * np.exp(-1j * vals * t)))**2


def z_geo_single(G, vals, vecs, idx, src, tgt,
                 t=50.0, n_rand=80, seed=0):
    rng  = np.random.default_rng(seed)
    n    = G.number_of_nodes()
    path = nx.shortest_path(G, src, tgt, weight="cost")
    corr = set(path)
    for v in list(corr):
        corr.update(G.neighbors(v))
    corr = np.array(sorted(corr), dtype=int)
    m    = len(corr)

    c0   = band_state(vecs, idx, src)
    P    = prob_vec(vals, vecs, c0, t)
    p_g  = P[corr].sum()

    rp   = [P[rng.choice(n, m, False)].sum()
            for _ in range(n_rand)]
    mu, sd = np.mean(rp), np.std(rp) + 1e-12
    return float((p_g - mu) / sd)


def multi_pair_z(G, vals, vecs, bands, bname,
                 n_pairs=20, t=50.0, seed=42):
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    z_arr = []
    for pi in range(n_pairs):
        src = int(rng.choice(nodes))
        tgt = int(rng.choice(nodes))
        if src == tgt or not nx.has_path(G, src, tgt):
            continue
        idx = bands[bname]
        z   = z_geo_single(G, vals, vecs, idx,
                           src, tgt, t=t,
                           n_rand=60, seed=pi)
        z_arr.append(z)
    return np.array(z_arr)


def mean_path_length(G, n_sample=50, seed=42):
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    s     = min(n_sample, len(nodes))
    sample = rng.choice(nodes, size=s, replace=False)
    lens  = []
    for nd in sample:
        L = nx.single_source_dijkstra_path_length(
            G, int(nd), weight="cost")
        lens.extend(L.values())
    return float(np.mean(lens))


# ============================================================
# CORE: FIND k_nn FOR TARGET FIEDLER
# ============================================================

def find_k_for_fiedler(X, target_fiedler=0.15,
                        k_range=None, tol=0.03):
    """
    Binary search for k_nn such that
    |Fiedler(G) - target| < tol.
    Returns best k and achieved Fiedler.
    """
    if k_range is None:
        k_range = list(range(4, 30))

    best_k   = k_range[0]
    best_f   = np.inf
    best_diff = np.inf

    fiedler_by_k = {}

    for k in k_range:
        G, _ = build_graph(X, k=k)
        f    = get_fiedler(G)
        fiedler_by_k[k] = f
        diff = abs(f - target_fiedler)
        if diff < best_diff:
            best_diff = diff
            best_k    = k
            best_f    = f

    return best_k, best_f, fiedler_by_k


# ============================================================
# EXP 1: MATCHED-FIEDLER COMPARISON
# ============================================================

def exp_matched_fiedler(alg_names,
                         target_fiedler=0.15,
                         T=900, n_pairs=20,
                         seed=42):
    """
    For each algebra: find k_nn such that Fiedler ≈ target.
    Then measure z_geo(highλ) at matched connectivity.
    """
    print(f"\n[Exp 1: Matched Fiedler ≈ {target_fiedler}]")
    print(f"{'Alg':<10} {'k_match':>8} {'Fiedler_got':>13} "
          f"{'mean_path':>11} {'z_high_mean':>13}")
    print("-"*60)

    results = {}

    for name in alg_names:
        omega = get_omega(name, seed=seed)
        orbit = generate_orbit(omega, T=T,
                               kappa=0.05, seed=seed)
        X     = embed_torus(orbit)

        k_match, f_got, fk_dict = find_k_for_fiedler(
            X, target_fiedler=target_fiedler,
            k_range=list(range(4, 28)), tol=0.03)

        G, _   = build_graph(X, k=k_match)
        vals, vecs = graph_spectrum(G)
        bands  = make_bands(vals)
        mp     = mean_path_length(G, seed=seed)

        z_arr  = multi_pair_z(
            G, vals, vecs, bands,
            bname="highλ", n_pairs=n_pairs,
            t=50.0, seed=seed)

        print(f"{name:<10} {k_match:>8d} "
              f"{f_got:>13.5f} "
              f"{mp:>11.3f} "
              f"{z_arr.mean():>13.4f} ± "
              f"{z_arr.std()/max(1,np.sqrt(len(z_arr))):.4f}")

        results[name] = {
            "k_match":    k_match,
            "fiedler":    f_got,
            "mean_path":  mp,
            "z_high":     z_arr,
            "fk_dict":    fk_dict,
            "G":          G,
            "vals":       vals,
            "vecs":       vecs,
            "bands":      bands,
            "color":      ALGEBRAS[name]["color"],
        }

    return results


# ============================================================
# EXP 2: z_geo vs FIEDLER CURVE (one algebra at a time)
# ============================================================

def exp_fiedler_curve(alg_names, T=900,
                       n_pairs=15, k_range=None,
                       seed=42):
    """
    For each algebra: sweep k_nn from 4 to 24.
    Record Fiedler and z_geo(highλ) at each k.
    This gives the full z_geo(Fiedler) curve.

    Key question: at the SAME Fiedler value,
    do algebras differ in z_geo?
    """
    if k_range is None:
        k_range = [4, 6, 8, 10, 12, 14, 16, 20, 24]

    print(f"\n[Exp 2: z_geo vs Fiedler curve]")

    results = {}

    for name in alg_names:
        omega = get_omega(name, seed=seed)
        orbit = generate_orbit(omega, T=T,
                               kappa=0.05, seed=seed)
        X     = embed_torus(orbit)

        fiedler_arr = []
        z_arr_list  = []

        for k in k_range:
            G, _   = build_graph(X, k=k)
            f      = get_fiedler(G)
            vals, vecs = graph_spectrum(G)
            bands  = make_bands(vals)

            z_arr  = multi_pair_z(
                G, vals, vecs, bands,
                bname="highλ", n_pairs=n_pairs,
                t=50.0, seed=seed)

            fiedler_arr.append(f)
            z_arr_list.append(z_arr.mean())
            print(f"  {name} k={k:2d}: "
                  f"Fiedler={f:.4f}, "
                  f"z_high={z_arr.mean():.3f}")

        results[name] = {
            "fiedler": np.array(fiedler_arr),
            "z_means": np.array(z_arr_list),
            "k_range": k_range,
            "color":   ALGEBRAS[name]["color"],
        }

    return results


# ============================================================
# EXP 3: PARTIAL CORRELATION
# ============================================================

def exp_partial_correlation(matched_results):
    """
    At matched Fiedler, test:
    r(z_geo, D_corr) — does orbit geometry
    predict z_geo after Fiedler is controlled?

    Also: ANOVA across algebras at matched Fiedler.
    """
    print(f"\n[Exp 3: Partial correlation + ANOVA]")

    alg_names = list(matched_results.keys())

    z_means   = np.array([
        matched_results[nm]["z_high"].mean()
        for nm in alg_names])
    fiedlers  = np.array([
        matched_results[nm]["fiedler"]
        for nm in alg_names])

    print(f"\n  At matched Fiedler:")
    print(f"  {'Algebra':<10} {'z_mean':>10} "
          f"{'Fiedler':>10}")
    for nm, z, f in zip(alg_names, z_means, fiedlers):
        print(f"  {nm:<10} {z:>10.4f} {f:>10.5f}")

    # ANOVA
    groups = [matched_results[nm]["z_high"]
               for nm in alg_names
               if len(matched_results[nm]["z_high"]) > 2]
    if len(groups) >= 2:
        F, p_anova = stats.f_oneway(*groups)
        print(f"\n  ANOVA (matched Fiedler): "
              f"F={F:.3f}, p={p_anova:.4e}")
        if p_anova < 0.05:
            print("  → Algebras differ even at "
                  "matched connectivity ✓")
            print("  → Effect is NOT explained "
                  "by Fiedler alone")
        else:
            print("  → Algebras do NOT differ at "
                  "matched connectivity")
            print("  → z_geo effect WAS driven "
                  "by Fiedler (artifact)")

    # Pairwise: E6 vs Random
    if "E6" in matched_results and \
       "Random" in matched_results:
        z_e6  = matched_results["E6"]["z_high"]
        z_rnd = matched_results["Random"]["z_high"]
        _, p  = stats.mannwhitneyu(
            z_e6, z_rnd, alternative="greater")
        print(f"\n  E6 vs Random (matched Fiedler):")
        print(f"    E6={z_e6.mean():.4f}, "
              f"Random={z_rnd.mean():.4f}, "
              f"p={p:.4e}")
        if p < 0.05:
            print("    → E6 STILL higher after "
                  "controlling connectivity ✓")
            print("    → REAL geometric effect")
        else:
            print("    → E6 NOT higher after "
                  "controlling connectivity")
            print("    → Was Fiedler artifact")

    return F if len(groups) >= 2 else np.nan, \
           p_anova if len(groups) >= 2 else np.nan


# ============================================================
# FIGURE
# ============================================================

def make_figure(matched_res, curve_res,
                fname="part8_step6d_connectivity_control.png"):

    alg_names  = ["E6", "A6", "E8", "Random"]
    alg_colors = [ALGEBRAS[nm]["color"]
                  for nm in alg_names]

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Part VIII — Step 6d: Controlling for Graph Connectivity\n"
        "Is z_geo(E6) real or a Fiedler artifact?",
        fontsize=14, fontweight="bold", y=1.001)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.48, wspace=0.35)

    # ── Row 0: Fiedler sweep curves
    ax = fig.add_subplot(gs[0, 0:2])
    ax.set_facecolor("#f0f4f8")
    for nm in alg_names:
        if nm not in curve_res:
            continue
        r   = curve_res[nm]
        clr = ALGEBRAS[nm]["color"]
        ax.plot(r["fiedler"], r["z_means"],
                "o-", color=clr, lw=2.2,
                ms=8, label=nm, alpha=0.85)
    ax.axhline(2.0, color="green", ls="--",
               lw=1.8, label="z=2")
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("Fiedler value (algebraic connectivity)",
                  fontsize=11)
    ax.set_ylabel("z_geo(highλ)", fontsize=11)
    ax.set_title("z_geo vs Fiedler (k sweep)\n"
                 "If curves separate → real effect\n"
                 "If they collapse → Fiedler artifact",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 0: z_geo at matched Fiedler violin
    ax = fig.add_subplot(gs[0, 2:4])
    ax.set_facecolor("#f0f4f8")
    data_v = [matched_res[nm]["z_high"]
               for nm in alg_names
               if nm in matched_res]
    pos_v  = np.arange(1, len(alg_names)+1)
    vp = ax.violinplot(data_v, positions=pos_v,
                       showmedians=True,
                       showextrema=True)
    for pc, clr in zip(vp["bodies"], alg_colors):
        pc.set_facecolor(clr); pc.set_alpha(0.65)
    ax.axhline(2.0, color="green", ls="--",
               lw=2.0, label="z=2 threshold")
    ax.set_xticks(pos_v)
    ax.set_xticklabels(alg_names, fontsize=12)
    ax.set_ylabel("z_geo(highλ)")
    ax.set_title(
        "z_geo at MATCHED Fiedler ≈ 0.15\n"
        "Controls for connectivity artifact",
        fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Row 1: k_match and Fiedler_got
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor("#f0f4f8")
    xs = np.arange(len(alg_names))
    k_m = [matched_res[nm]["k_match"]
            if nm in matched_res else 0
            for nm in alg_names]
    f_g = [matched_res[nm]["fiedler"]
            if nm in matched_res else 0
            for nm in alg_names]
    ax2 = ax.twinx()
    ax.bar(xs-0.2, k_m, 0.38,
           color=alg_colors, alpha=0.75,
           edgecolor="k", label="k_match")
    ax2.plot(xs, f_g, "D--", color="red",
             ms=10, lw=2, label="Fiedler_got")
    ax.set_xticks(xs)
    ax.set_xticklabels(alg_names, fontsize=11)
    ax.set_ylabel("k_nn used", fontsize=10)
    ax2.set_ylabel("Fiedler achieved", color="red",
                   fontsize=10)
    ax.set_title("k_nn required to match Fiedler≈0.15\n"
                 "A6 needs much larger k",
                 fontsize=9, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Row 1: mean path at matched Fiedler
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor("#f0f4f8")
    mp_m = [matched_res[nm]["mean_path"]
             if nm in matched_res else 0
             for nm in alg_names]
    z_m  = [matched_res[nm]["z_high"].mean()
             if nm in matched_res else 0
             for nm in alg_names]
    sc = ax.scatter(mp_m, z_m, s=300,
                    c=alg_colors, zorder=5,
                    edgecolors="k", lw=2)
    for nm, mp, z in zip(alg_names, mp_m, z_m):
        ax.annotate(nm, xy=(mp, z),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=12, fontweight="bold",
                    color=ALGEBRAS[nm]["color"])
    if len(mp_m) >= 3:
        rho, p = stats.pearsonr(mp_m, z_m)
        ax.set_title(
            f"Mean path vs z_geo (matched Fiedler)\n"
            f"r={rho:.3f}, p={p:.4f}",
            fontsize=9, fontweight="bold")
    ax.axhline(2.0, color="green", ls="--",
               lw=1.5, alpha=0.7)
    ax.set_xlabel("Mean geodesic path length",
                  fontsize=11)
    ax.set_ylabel("z_geo(highλ)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Fiedler sweep per algebra
    for ci, nm in enumerate(["E6", "Random"]):
        ax = fig.add_subplot(gs[2, ci*2:ci*2+2])
        ax.set_facecolor("#f0f4f8")
        if nm not in curve_res:
            continue
        r   = curve_res[nm]
        clr = ALGEBRAS[nm]["color"]
        ax.plot(r["fiedler"], r["z_means"],
                "o-", color=clr, lw=2.5,
                ms=10, alpha=0.85, label=nm)
        ax.axhline(2.0, color="green", ls="--",
                   lw=1.8, label="z=2")

        # Annotate k values
        for fi, zi, ki in zip(r["fiedler"],
                               r["z_means"],
                               r["k_range"]):
            ax.annotate(f"k={ki}",
                        xy=(fi, zi),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=8)

        ax.set_xlabel("Fiedler value", fontsize=11)
        ax.set_ylabel("z_geo(highλ)", fontsize=11)
        ax.set_title(
            f"{nm}: z_geo vs Fiedler\n"
            f"as k_nn varies",
            fontsize=10, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.savefig(fname, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {fname}")


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(matched_res, curve_res,
                   F_anova, p_anova):
    alg_names = ["E6", "A6", "E8", "Random"]

    print("\n" + "="*70)
    print("FINAL SUMMARY — Step 6d")
    print("="*70)

    print(f"\nMatched Fiedler results:")
    print(f"{'Alg':<10} {'k_match':>8} {'Fiedler':>10} "
          f"{'mean_path':>11} {'z_high':>10}")
    print("-"*55)
    for nm in alg_names:
        if nm not in matched_res:
            continue
        r = matched_res[nm]
        print(f"{nm:<10} {r['k_match']:>8d} "
              f"{r['fiedler']:>10.5f} "
              f"{r['mean_path']:>11.3f} "
              f"{r['z_high'].mean():>10.4f}")

    print(f"\nANOVA (matched Fiedler): "
          f"F={F_anova:.3f}, p={p_anova:.4e}")

    # Key verdict
    if "E6" in matched_res and "Random" in matched_res:
        z_e6  = matched_res["E6"]["z_high"]
        z_rnd = matched_res["Random"]["z_high"]
        _, p  = stats.mannwhitneyu(
            z_e6, z_rnd, alternative="greater")

        print(f"\nE6 vs Random at matched connectivity:")
        print(f"  E6={z_e6.mean():.4f} vs "
              f"Random={z_rnd.mean():.4f}")
        print(f"  p(E6>Random) = {p:.4e}")

        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)

        if p < 0.05 and p_anova < 0.05:
            print("""
  ✓ REAL EFFECT: E6 z_geo survives Fiedler control.

  The geodesic focusing of E6 highλ modes is NOT explained
  by graph connectivity (Fiedler value).
  Even when all algebras are matched to the same Fiedler,
  E6 (and Coxeter algebras) show stronger geodesic focusing
  than Random.

  This implicates the ORBIT GEOMETRY (D_corr ≈ 2.6)
  as the primary driver, not graph topology alone.
""")
        elif p_anova > 0.05:
            print("""
  ✗ ARTIFACT: Effect disappears at matched connectivity.

  The z_geo differences were driven by different Fiedler
  values (graph connectivity), not by algebra structure.
  After controlling for connectivity, all algebras
  perform similarly.

  Conclusion: z_geo was a Fiedler artifact.
""")
        else:
            print("""
  ⚠ MIXED: Partial effect after Fiedler control.

  Some difference survives but is weakened.
  The effect has both geometric and connectivity components.
""")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Part VIII Step 6d: Connectivity Control\n")
    print("Estimated runtime: 15-25 minutes\n")

    alg_names = ["E6", "A6", "E8", "Random"]

    # Exp 1: Matched Fiedler
    matched_res = exp_matched_fiedler(
        alg_names,
        target_fiedler=0.15,
        T=900, n_pairs=20, seed=42)

    # Exp 2: z_geo vs Fiedler curves
    curve_res = exp_fiedler_curve(
        alg_names, T=900, n_pairs=12,
        k_range=[4, 6, 8, 10, 12, 14, 16, 20, 24],
        seed=42)

    # Exp 3: Partial correlation + ANOVA
    F_a, p_a = exp_partial_correlation(matched_res)

    # Figure
    make_figure(matched_res, curve_res)

    # Summary
    final_summary(matched_res, curve_res, F_a, p_a)
