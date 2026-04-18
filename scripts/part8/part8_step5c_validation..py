"""
Part VIII — Step 5c: Validation of E6 Geodesic Focusing
========================================================

Step 5b showed:
  E6 highλ: z_geo = 2.374 (above threshold)
  Random highλ: z_geo = 1.115

This could be:
  (A) real E6-specific structure
  (B) artifact of one specific source-target pair

Step 5c tests:
  1. MULTI-PAIR VALIDATION
     Repeat z_geo measurement for N_pairs random (source, target) pairs
     Test: is mean z_geo(highλ, E6) > mean z_geo(highλ, Random)?
     Stat: Mann-Whitney U test

  2. IPR GRADIENT TEST
     Is the monotonic IPR increase (low < mid < high)
     specific to E6 or appears in Random too?
     Stat: Spearman correlation IPR vs lambda_center

  3. SPREADING HIERARCHY TEST
     Is α(highλ) > α(lowλ) significant in E6
     but not in Random?
     Method: bootstrap over orbit seeds

  4. ROBUSTNESS: vary k_nn in [6, 8, 10, 12, 14]
     Does z_geo(highλ, E6) stay above threshold?
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
# 1. SETUP — reuse from Step 5b
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

BAND_NAMES  = ["lowλ", "midλ", "highλ"]
BAND_COLORS = {"lowλ": "#1565C0", "midλ": "#FB8C00", "highλ": "#C62828"}


def get_omega(name, seed=42):
    alg = ALGEBRAS[name]
    if alg["exponents"] is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.5, 2.0, alg["rank"])
    m = np.array(alg["exponents"], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg["h"])


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


def embed_torus(orbit):
    return np.concatenate([np.cos(orbit), np.sin(orbit)], axis=1)


def build_graph(X, k=10):
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k+1)
    sigma = np.median(dists[:, 1:])
    edge_best = {}
    n = len(X)
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
    comps   = list(nx.connected_components(G))
    giant   = max(comps, key=len)
    Gg      = G.subgraph(giant).copy()
    mapping = {old: i for i, old in enumerate(sorted(Gg.nodes()))}
    return nx.relabel_nodes(Gg, mapping), sigma


def graph_matrices(G):
    n   = G.number_of_nodes()
    A   = np.zeros((n, n))
    for i, j, d in G.edges(data=True):
        A[i,j] = A[j,i] = d["weight"]
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
        "midλ":  nz[max(0, mc - q//2): min(m, mc + q//2)],
        "highλ": nz[-q:],
    }


def band_state(vecs, idx, source):
    n   = vecs.shape[0]
    d   = np.zeros(n); d[source] = 1.0
    c   = vecs.T @ d
    cb  = np.zeros_like(c); cb[idx] = c[idx]
    psi = vecs @ cb
    nm  = np.linalg.norm(psi)
    if nm > 0: psi /= nm
    return (vecs.T @ psi).astype(complex)


def prob_t(vals, vecs, c0, t):
    return np.abs(vecs @ (c0 * np.exp(-1j * vals * t)))**2


def corridor_nodes(G, source, target, radius=1):
    path  = nx.shortest_path(G, source, target, weight="cost")
    S     = set(path)
    front = set(path)
    for _ in range(radius):
        nxt = set()
        for v in front: nxt.update(G.neighbors(v))
        S.update(nxt); front = nxt
    return np.array(sorted(S), dtype=int)


def z_geo_single(vals, vecs, c0, G, source, target,
                 t_sample=50.0, n_rand=100, seed=0):
    """
    z-score of geodesic overlap at one time point t_sample.
    Fast version for multi-pair averaging.
    """
    rng      = np.random.default_rng(seed)
    n        = G.number_of_nodes()
    corridor = corridor_nodes(G, source, target, radius=1)
    m        = len(corridor)

    P       = prob_t(vals, vecs, c0, t_sample)
    p_geo   = P[corridor].sum()

    rand_p  = []
    for _ in range(n_rand):
        samp = rng.choice(n, size=m, replace=False)
        rand_p.append(P[samp].sum())
    rand_p = np.array(rand_p)
    mu, sd = rand_p.mean(), rand_p.std(ddof=1) + 1e-12
    return float((p_geo - mu) / sd)


def ipr_of_modes(vecs, idx):
    return np.sum(vecs[:, idx]**4, axis=0)


def spreading_alpha(vals, vecs, c0, G, source,
                    tmax=200.0, n_times=200):
    """
    Compute <d(t)> spreading exponent α from log-log fit.
    """
    lens    = nx.single_source_dijkstra_path_length(
                  G, source, weight="cost")
    n       = G.number_of_nodes()
    geo     = np.full(n, np.nan)
    for j, v in lens.items(): geo[j] = v
    mask    = np.isfinite(geo)
    d_fin   = geo[mask]

    times   = np.linspace(1.0, tmax, n_times)
    mean_d  = np.zeros(n_times)
    for ti, t in enumerate(times):
        P       = prob_t(vals, vecs, c0, t)
        Pm      = P[mask]; s = Pm.sum()
        if s > 0: Pm /= s
        mean_d[ti] = (d_fin * Pm).sum()

    pos = mean_d > 0
    if pos.sum() < 4:
        return np.nan, np.nan
    sl, ic, r, pv, _ = stats.linregress(
        np.log(times[pos]), np.log(mean_d[pos]))
    return float(sl), float(r**2)


# ============================================================
# 2. TEST 1: MULTI-PAIR z_geo
# ============================================================

def test_multi_pair_zgeo(name, n_pairs=30,
                          t_sample=50.0,
                          T=900, k=10,
                          seed_orbit=42,
                          seed_pairs=7):
    """
    Sample n_pairs random (source, target) pairs.
    For each pair and each band compute z_geo.
    Return dict: band -> array of z_geo values.
    """
    omega = get_omega(name, seed=seed_orbit)
    orbit = generate_orbit(omega, T=T, seed=seed_orbit)
    X     = embed_torus(orbit)
    G, _  = build_graph(X, k=k)
    n     = G.number_of_nodes()
    vals, vecs = graph_matrices(G)
    bands = make_bands(vals)

    rng   = np.random.default_rng(seed_pairs)
    nodes = list(G.nodes())

    z_results = {b: [] for b in BAND_NAMES}

    for pair_i in range(n_pairs):
        src = int(rng.choice(nodes))
        tgt = int(rng.choice(nodes))
        if src == tgt:
            continue

        # Check connectivity
        if not nx.has_path(G, src, tgt):
            continue

        for bname in BAND_NAMES:
            idx = bands[bname]
            c0  = band_state(vecs, idx, src)
            z   = z_geo_single(vals, vecs, c0, G,
                               src, tgt,
                               t_sample=t_sample,
                               n_rand=80,
                               seed=pair_i)
            z_results[bname].append(z)

    return {b: np.array(v) for b, v in z_results.items()}


# ============================================================
# 3. TEST 2: IPR GRADIENT
# ============================================================

def test_ipr_gradient(name, T=900, k=10, seed=42):
    """
    Test whether IPR increases monotonically with lambda_center.
    Returns Spearman rho for (lambda_center vs mean_IPR) per band.
    """
    omega = get_omega(name, seed=seed)
    orbit = generate_orbit(omega, T=T, seed=seed)
    X     = embed_torus(orbit)
    G, _  = build_graph(X, k=k)
    vals, vecs = graph_matrices(G)
    bands = make_bands(vals)

    lam_c  = []
    ipr_m  = []

    for bname in BAND_NAMES:
        idx   = bands[bname]
        ipr   = ipr_of_modes(vecs, idx)
        lam_c.append(float(np.median(vals[idx])))
        ipr_m.append(float(ipr.mean()))

    rho, p = stats.spearmanr(lam_c, ipr_m)
    return float(rho), float(p), lam_c, ipr_m


# ============================================================
# 4. TEST 3: SPREADING HIERARCHY BOOTSTRAP
# ============================================================

def test_spreading_bootstrap(name, n_boot=30,
                              T=900, k=10,
                              tmax=150.0, n_times=150,
                              base_seed=42):
    """
    Bootstrap over orbit seeds.
    At each seed: compute α(highλ) - α(lowλ).
    Return distribution of this contrast.
    """
    contrasts = []

    for b in range(n_boot):
        seed = base_seed + b * 17
        omega = get_omega(name, seed=base_seed)
        orbit = generate_orbit(omega, T=T, seed=seed)
        X     = embed_torus(orbit)
        G, _  = build_graph(X, k=k)
        n     = G.number_of_nodes()
        vals, vecs = graph_matrices(G)
        bands = make_bands(vals)

        src_degs = np.array([G.degree(i) for i in G.nodes()])
        src      = int(np.argmax(src_degs))

        a_low = a_high = np.nan
        for bname in ["lowλ", "highλ"]:
            idx = bands[bname]
            c0  = band_state(vecs, idx, src)
            a, r2 = spreading_alpha(vals, vecs, c0, G, src,
                                    tmax=tmax, n_times=n_times)
            if bname == "lowλ":  a_low  = a
            if bname == "highλ": a_high = a

        if np.isfinite(a_high) and np.isfinite(a_low):
            contrasts.append(a_high - a_low)

    return np.array(contrasts)


# ============================================================
# 5. TEST 4: ROBUSTNESS vs k_nn
# ============================================================

def test_knn_robustness(name, k_vals=None,
                         T=900, n_pairs=15,
                         t_sample=50.0,
                         seed=42):
    """
    For each k in k_vals: compute mean z_geo(highλ).
    Test: does E6 highλ stay above Random highλ?
    """
    if k_vals is None:
        k_vals = [6, 8, 10, 12, 14]

    omega = get_omega(name, seed=seed)
    orbit = generate_orbit(omega, T=T, seed=seed)
    X     = embed_torus(orbit)

    rng   = np.random.default_rng(seed + 1)
    z_by_k = {}

    for k in k_vals:
        G, _  = build_graph(X, k=k)
        nodes = list(G.nodes())
        n     = G.number_of_nodes()
        vals, vecs = graph_matrices(G)
        bands = make_bands(vals)

        z_high = []
        for pi in range(n_pairs):
            src = int(rng.choice(nodes))
            tgt = int(rng.choice(nodes))
            if src == tgt or not nx.has_path(G, src, tgt):
                continue
            idx = bands["highλ"]
            c0  = band_state(vecs, idx, src)
            z   = z_geo_single(vals, vecs, c0, G,
                               src, tgt,
                               t_sample=t_sample,
                               n_rand=60,
                               seed=pi)
            z_high.append(z)

        z_by_k[k] = np.array(z_high)

    return z_by_k


# ============================================================
# 6. MAIN
# ============================================================

def run_all_tests(n_pairs=30, n_boot=20, seed=42):
    results = {}

    for name in ["E6", "Random"]:
        print(f"\n{'='*62}")
        print(f"ALGEBRA: {name}")

        # Test 1: Multi-pair z_geo
        print(f"\n  Test 1: Multi-pair z_geo (n_pairs={n_pairs})...")
        z_mp = test_multi_pair_zgeo(
            name, n_pairs=n_pairs,
            t_sample=50.0, T=900, k=10,
            seed_orbit=seed, seed_pairs=seed+1)

        for bname in BAND_NAMES:
            arr = z_mp[bname]
            mu  = arr.mean() if len(arr) > 0 else np.nan
            se  = arr.std(ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else np.nan
            print(f"    z_geo({bname}): "
                  f"mean={mu:.4f} ± {se:.4f}  "
                  f"(n={len(arr)})")

        # Test 2: IPR gradient
        print(f"\n  Test 2: IPR gradient...")
        rho, p, lam_c, ipr_m = test_ipr_gradient(
            name, T=900, k=10, seed=seed)
        print(f"    Spearman(λ_center, mean_IPR): "
              f"ρ={rho:.4f}, p={p:.4f}")
        print(f"    λ_centers: {[f'{x:.4f}' for x in lam_c]}")
        print(f"    mean IPRs: {[f'{x:.6f}' for x in ipr_m]}")
        if rho > 0.9 and p < 0.1:
            print(f"    → MONOTONIC IPR gradient ✓")
        else:
            print(f"    → No clear monotonic gradient")

        # Test 3: Bootstrap spreading hierarchy
        print(f"\n  Test 3: Bootstrap spreading α(high)-α(low) "
              f"(n_boot={n_boot})...")
        contrasts = test_spreading_bootstrap(
            name, n_boot=n_boot, T=900, k=10,
            tmax=150.0, n_times=120, base_seed=seed)
        if len(contrasts) > 1:
            mu_c  = contrasts.mean()
            se_c  = contrasts.std(ddof=1) / np.sqrt(len(contrasts))
            t_c   = mu_c / (contrasts.std(ddof=1) + 1e-12) * np.sqrt(len(contrasts))
            _, pv = stats.ttest_1samp(contrasts, 0.0)
            print(f"    Δα = α(high)-α(low): "
                  f"mean={mu_c:.4f} ± {se_c:.4f}  "
                  f"t={t_c:.3f}  p={pv:.4f}")
            if pv < 0.05 and mu_c > 0:
                print(f"    → highλ spreads faster than lowλ ✓")
            else:
                print(f"    → No significant hierarchy ✗")
        else:
            print(f"    → Insufficient bootstrap samples")

        # Test 4: k_nn robustness
        print(f"\n  Test 4: k_nn robustness for highλ z_geo...")
        k_vals = [6, 8, 10, 12, 14]
        z_by_k = test_knn_robustness(
            name, k_vals=k_vals, T=900,
            n_pairs=12, t_sample=50.0, seed=seed)
        for k, arr in z_by_k.items():
            mu  = arr.mean() if len(arr) > 0 else np.nan
            se  = arr.std(ddof=1)/np.sqrt(len(arr)) if len(arr) > 1 else np.nan
            print(f"    k={k:>2}: z_geo(highλ) = {mu:.4f} ± {se:.4f}")

        results[name] = {
            "z_multipair": z_mp,
            "ipr_gradient": (rho, p, lam_c, ipr_m),
            "boot_contrast": contrasts,
            "knn_robust": z_by_k,
        }

    return results


# ============================================================
# 7. STATISTICAL COMPARISON E6 vs RANDOM
# ============================================================

def compare_e6_vs_random(results):
    print("\n" + "="*70)
    print("E6 vs RANDOM: Statistical comparison")
    print("="*70)

    # Test 1: Mann-Whitney on z_geo per band
    print("\nTest 1 — Mann-Whitney U: z_geo(band, E6) vs z_geo(band, Random)")
    print(f"{'Band':<10} {'E6 mean':>10} {'Rand mean':>11} "
          f"{'U stat':>10} {'p-value':>10} {'verdict':>15}")
    print("-"*70)
    for bname in BAND_NAMES:
        x_e6  = results["E6"]["z_multipair"][bname]
        x_rnd = results["Random"]["z_multipair"][bname]
        if len(x_e6) < 3 or len(x_rnd) < 3:
            print(f"{bname:<10} {'—':>10} {'—':>11} {'—':>10} {'—':>10}")
            continue
        U, p  = stats.mannwhitneyu(x_e6, x_rnd, alternative="greater")
        mu_e6 = x_e6.mean()
        mu_rn = x_rnd.mean()
        sig   = "E6 > Rnd ***" if p < 0.001 else \
                "E6 > Rnd **"  if p < 0.01  else \
                "E6 > Rnd *"   if p < 0.05  else "ns"
        print(f"{bname:<10} {mu_e6:>10.4f} {mu_rn:>11.4f} "
              f"{U:>10.1f} {p:>10.4e} {sig:>15}")

    # Test 2: IPR gradient comparison
    print("\nTest 2 — IPR gradient (Spearman ρ):")
    for name in ["E6", "Random"]:
        rho, p, _, _ = results[name]["ipr_gradient"]
        print(f"  {name}: ρ={rho:.4f}, p={p:.4f}  "
              f"{'monotonic ✓' if rho > 0.9 else 'not monotonic ✗'}")

    # Test 3: Bootstrap Δα
    print("\nTest 3 — Spreading hierarchy Δα = α(high) - α(low):")
    for name in ["E6", "Random"]:
        c = results[name]["boot_contrast"]
        if len(c) > 1:
            _, pv = stats.ttest_1samp(c, 0.0)
            print(f"  {name}: Δα={c.mean():.4f} ± {c.std():.4f}  "
                  f"p={pv:.4f}  "
                  f"{'sig pos ✓' if pv < 0.05 and c.mean() > 0 else 'ns'}")

    # Test 4: k_nn robustness summary
    print("\nTest 4 — k_nn robustness of z_geo(highλ):")
    k_vals = [6, 8, 10, 12, 14]
    print(f"{'k':<5} {'E6 z_high':>12} {'Rnd z_high':>12} {'E6>Rnd?':>10}")
    print("-"*42)
    for k in k_vals:
        ze = results["E6"]["knn_robust"].get(k, np.array([]))
        zr = results["Random"]["knn_robust"].get(k, np.array([]))
        me = ze.mean() if len(ze) > 0 else np.nan
        mr = zr.mean() if len(zr) > 0 else np.nan
        ans = "yes" if me > mr else "no"
        print(f"{k:<5} {me:>12.4f} {mr:>12.4f} {ans:>10}")


# ============================================================
# 8. FIGURE
# ============================================================

def make_figure(results, fname="part8_step5c_validation.png"):
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        "Part VIII — Step 5c: Validation of E6 Geodesic Focusing\n"
        "Multi-pair z_geo  |  IPR gradient  |  "
        "Bootstrap Δα  |  k_nn robustness",
        fontsize=14, fontweight="bold", y=1.001)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.48, wspace=0.36)

    alg_colors = {"E6": "#2196F3", "Random": "#9E9E9E"}

    # ── Row 0: Multi-pair z_geo violin per band ────────
    for bi, bname in enumerate(BAND_NAMES):
        ax = fig.add_subplot(gs[0, bi])
        ax.set_facecolor("#f0f4f8")

        pos  = [1, 2]
        data = [results["E6"]["z_multipair"][bname],
                results["Random"]["z_multipair"][bname]]
        vp   = ax.violinplot(data, positions=pos,
                             showmedians=True,
                             showextrema=True)
        for pc, clr in zip(vp["bodies"],
                           ["#2196F3", "#9E9E9E"]):
            pc.set_facecolor(clr)
            pc.set_alpha(0.6)

        ax.axhline(2.0, color="green", ls="--",
                   lw=1.8, alpha=0.8, label="z=2")
        ax.axhline(0.0, color="black", ls=":",
                   lw=1.0, alpha=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["E6", "Random"], fontsize=11)
        ax.set_ylabel("z_geo", fontsize=10)
        ax.set_title(f"z_geo: {bname}\n(multi-pair)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # ── Row 0 col 3: k_nn robustness ──────────────────
    ax = fig.add_subplot(gs[0, 3])
    ax.set_facecolor("#f0f4f8")
    k_vals = [6, 8, 10, 12, 14]
    for name in ["E6", "Random"]:
        mus = []
        ses = []
        for k in k_vals:
            arr = results[name]["knn_robust"].get(k, np.array([]))
            mus.append(arr.mean() if len(arr) > 0 else np.nan)
            ses.append(arr.std(ddof=1)/np.sqrt(max(len(arr),1)))
        mus = np.array(mus)
        ses = np.array(ses)
        ax.errorbar(k_vals, mus, yerr=ses,
                    fmt="o-", lw=2.2, ms=8,
                    color=alg_colors[name],
                    label=name, capsize=4)
    ax.axhline(2.0, color="green", ls="--",
               lw=1.8, alpha=0.8, label="z=2 threshold")
    ax.axhline(0.0, color="black", ls=":", lw=1.0, alpha=0.5)
    ax.set_xlabel("k_nn", fontsize=10)
    ax.set_ylabel("z_geo(highλ)", fontsize=10)
    ax.set_title("k_nn robustness\n(highλ band)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Row 1: IPR gradient ────────────────────────────
    for ci, name in enumerate(["E6", "Random"]):
        ax = fig.add_subplot(gs[1, ci*2:ci*2+2])
        ax.set_facecolor("#f0f4f8")
        rho, p, lam_c, ipr_m = results[name]["ipr_gradient"]

        clrs = [BAND_COLORS[b] for b in BAND_NAMES]
        for xi, (lc, im, bn, clr) in enumerate(
                zip(lam_c, ipr_m, BAND_NAMES, clrs)):
            ax.scatter(lc, im, s=250, color=clr,
                       zorder=5, edgecolors="k", lw=2,
                       label=bn)
            ax.annotate(f" {bn}", xy=(lc, im),
                        fontsize=10, va="center")

        ax.plot(sorted(lam_c),
                [ipr_m[lam_c.index(x)] for x in sorted(lam_c)],
                "k--", lw=1.5, alpha=0.5)

        ax.set_xlabel("λ_center", fontsize=11)
        ax.set_ylabel("mean IPR per mode", fontsize=11)
        ax.set_title(
            f"{name}: IPR gradient\n"
            f"Spearman ρ={rho:.3f}, p={p:.4f}  "
            f"{'monotonic ✓' if rho > 0.9 else '✗'}",
            fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Row 2: Bootstrap Δα distributions ─────────────
    for ci, name in enumerate(["E6", "Random"]):
        ax = fig.add_subplot(gs[2, ci*2:ci*2+2])
        ax.set_facecolor("#f0f4f8")
        c   = results[name]["boot_contrast"]
        clr = alg_colors[name]

        if len(c) > 1:
            ax.hist(c, bins=15, color=clr, alpha=0.7,
                    edgecolor="k", lw=0.8)
            ax.axvline(c.mean(), color="red", lw=2.5,
                       label=f"mean={c.mean():.4f}")
            ax.axvline(0.0, color="black", ls="--",
                       lw=1.8, alpha=0.7, label="zero")
            _, pv = stats.ttest_1samp(c, 0.0)
            ax.set_title(
                f"{name}: Bootstrap Δα = α(high)-α(low)\n"
                f"mean={c.mean():.4f}  p={pv:.4f}  "
                f"{'sig ✓' if pv < 0.05 and c.mean() > 0 else 'ns'}",
                fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "insufficient samples",
                    ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{name}: Bootstrap Δα",
                         fontsize=10, fontweight="bold")

        ax.set_xlabel("Δα = α(highλ) − α(lowλ)", fontsize=11)
        ax.set_ylabel("count", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.savefig(fname, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {fname}")


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Part VIII Step 5c: Validation of E6 Geodesic Focusing\n")
    print("This may take 5-15 minutes depending on your machine.\n")

    results = run_all_tests(n_pairs=30, n_boot=20, seed=42)
    compare_e6_vs_random(results)
    make_figure(results)

    # Final honest verdict
    print("\n" + "="*70)
    print("HONEST VERDICT")
    print("="*70)

    z_e6_high  = results["E6"]["z_multipair"]["highλ"]
    z_rnd_high = results["Random"]["z_multipair"]["highλ"]

    if len(z_e6_high) > 2 and len(z_rnd_high) > 2:
        _, p_mw = stats.mannwhitneyu(
            z_e6_high, z_rnd_high, alternative="greater")
        above_e6  = (z_e6_high > 2.0).mean()
        above_rnd = (z_rnd_high > 2.0).mean()
        print(f"\n  highλ z_geo: E6 mean={z_e6_high.mean():.4f} "
              f"vs Random mean={z_rnd_high.mean():.4f}")
        print(f"  Fraction above z=2: E6={above_e6:.2%}, "
              f"Random={above_rnd:.2%}")
        print(f"  Mann-Whitney p (E6>Random): {p_mw:.4f}")

        if p_mw < 0.05 and above_e6 > above_rnd + 0.1:
            print("""
  RESULT: E6-SPECIFIC GEODESIC FOCUSING IN highλ BAND ✓

  The high-frequency Laplacian modes of the E6 orbit graph
  show significantly stronger concentration along graph geodesics
  compared to a random-frequency null model.

  Physical interpretation:
    The high-λ sector of the monostring state graph
    "knows about" shortest paths in a way that
    random-frequency strings do not.
    This is a weak but reproducible signature of
    E6-specific transport structure.

  What this is NOT:
    - Not Standard Model gauge fields
    - Not photon propagation at speed c
    - Not Lorentz-invariant field theory
""")
        elif p_mw < 0.10:
            print("""
  RESULT: MARGINAL E6 SIGNAL (p < 0.10)
  Requires larger n_pairs for confirmation.
""")
        else:
            print("""
  RESULT: NO STATISTICALLY SIGNIFICANT E6 ADVANTAGE
  The geodesic focusing in highλ is not specific to E6.
  H0 not rejected.
""")
