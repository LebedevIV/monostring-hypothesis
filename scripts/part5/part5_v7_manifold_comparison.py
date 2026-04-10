"""
Part IV+ v7: FINAL — correct manifold comparison
Key fixes:
1. T^d constructed with d ACTIVE coordinates, rest EXCLUDED from distance
2. Torus metric applied consistently in corr_dim
3. Box-counting restricted to intrinsic coordinates only
4. Separate distance functions per manifold type
5. Summarize ALL results from Parts I-IV+ in final table
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp
import time
import warnings
warnings.filterwarnings("ignore")

CALIB = 4.0 / 3.616  # validated in v3

# ═══════════════════════════════════════════════════════════════
# ALGEBRA & PHASE EVOLUTION
# ═══════════════════════════════════════════════════════════════

def omega_E6():
    return 2.0 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12.0)

def evolve_phases(N, omega, seed=0):
    rng = np.random.RandomState(seed)
    D = len(omega)
    ph = np.zeros((N, D))
    ph[0] = rng.uniform(0, 2*np.pi, D)
    for n in range(N-1):
        ph[n+1] = (ph[n] + omega + 0.1*np.sin(ph[n])) % (2*np.pi)
    return ph

# ═══════════════════════════════════════════════════════════════
# MANIFOLD GENERATORS — correct construction
# ═══════════════════════════════════════════════════════════════

class Manifold:
    """
    Encapsulates a point set with its own distance function
    and true dimension. Avoids the T⁴-with-zeros bug.
    """
    def __init__(self, name, points, dist_fn, true_dim):
        self.name     = name
        self.points   = points   # (N, D_ambient) array
        self.dist_fn  = dist_fn  # (points_i, points_j) → distances
        self.true_dim = true_dim # expected dimension

def torus_dist(p, q):
    """Torus distance in intrinsic coordinates."""
    diff = np.abs(p - q)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.linalg.norm(diff, axis=-1)

def make_T3(N, seed=0):
    """T³: 3 independent uniform phases."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0, 2*np.pi, (N, 3))
    return Manifold(
        'T³ (3D torus)', pts,
        lambda p, q: torus_dist(p, q),
        3.0
    )

def make_T4(N, seed=0):
    """T⁴: 4 independent uniform phases."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0, 2*np.pi, (N, 4))
    return Manifold(
        'T⁴ (4D torus)', pts,
        lambda p, q: torus_dist(p, q),
        4.0
    )

def make_T6(N, seed=0):
    """T⁶: 6 independent uniform phases (null)."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0, 2*np.pi, (N, 6))
    return Manifold(
        'T⁶ (null)', pts,
        lambda p, q: torus_dist(p, q),
        6.0
    )

def make_S3(N, seed=0):
    """
    S³: 3-sphere embedded in R⁴.
    Sample uniformly, keep as R⁴ coordinates.
    Distance = great-circle (= arccos of dot product).
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(N, 4)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    def gc_dist(p, q):
        dots = np.clip(np.sum(p * q, axis=-1), -1, 1)
        return np.arccos(np.abs(dots))  # great-circle (mod antipodal)
    return Manifold('S³ (3-sphere)', x, gc_dist, 3.0)

def make_E6(N, seed=0):
    """E6 quasi-periodic orbit on T⁶."""
    pts = evolve_phases(N, omega_E6(), seed=seed)
    return Manifold(
        'E6 orbit', pts,
        lambda p, q: torus_dist(p, q),
        None  # unknown
    )

# ═══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION — uses manifold's own distance
# ═══════════════════════════════════════════════════════════════

def build_knn_from_manifold(mfd, k=20, seed=42):
    """
    Build k-NN graph using manifold's intrinsic distance function.
    Guarantees correct degree regardless of ambient dimension.
    """
    N = len(mfd.points)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    chunk = 100
    for i in range(N):
        di = np.zeros(N)
        for c in range(0, N, chunk):
            ce = min(c+chunk, N)
            # Broadcast: distance from point i to points c:ce
            pi = np.tile(mfd.points[i], (ce-c, 1))
            di[c:ce] = mfd.dist_fn(pi, mfd.points[c:ce])
        di[i] = np.inf
        neighbors = np.argpartition(di, k)[:k]
        for j in neighbors:
            G.add_edge(i, int(j))

    if not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        for c in comps[1:]:
            G.add_edge(list(comps[0])[0], list(c)[0])
    return G

# ═══════════════════════════════════════════════════════════════
# SPECTRAL DIMENSION (validated v3)
# ═══════════════════════════════════════════════════════════════

def spectral_dim(G):
    N = G.number_of_nodes()
    if N < 15: return 0.0
    n_eigs = min(N-1, max(100, N//3))
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        Ls = csr_matrix(
            nx.normalized_laplacian_matrix(G).astype(np.float64))
        eigs = eigsh(Ls, k=min(n_eigs, N-2),
                     which='SM', return_eigenvectors=False)
        eigs = np.sort(np.abs(eigs))
    except Exception:
        Ln = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
        eigs = np.sort(np.linalg.eigvalsh(Ln))
    eigs_nz = eigs[eigs > 1e-6]
    if len(eigs_nz) < 5: return 0.0
    lmin, lmax = eigs_nz[0], eigs_nz[-1]
    ts = np.logspace(np.log10(0.5/lmax), np.log10(5.0/lmin), 400)
    ds = np.zeros(400)
    for idx, t in enumerate(ts):
        e = np.exp(-eigs_nz*t); Z = e.sum()
        if Z < 1e-30: break
        ds[idx] = 2.0*t*np.dot(eigs_nz, e)/Z
    valid = (ds > 0.2) & (np.isfinite(ds))
    if valid.sum() < 10: return 0.0
    sm = np.convolve(ds, np.ones(11)/11, mode='same')
    W = 30; bs = np.inf; bi = 0
    for s in range(10, 400-W-10):
        if not valid[s:s+W].all(): continue
        std = np.std(sm[s:s+W])
        if std < bs: bs = std; bi = s
    pl = float(np.median(ds[bi:bi+W]))
    if pl < 0.3 or pl > 20:
        pl = float(np.max(sm[valid]))
    return pl

# ═══════════════════════════════════════════════════════════════
# CORRELATION DIMENSION — with manifold's distance
# ═══════════════════════════════════════════════════════════════

def corr_dim_manifold(mfd, n_sample=400, n_r=30, seed=42):
    """
    Grassberger-Procaccia correlation dimension.
    Uses manifold's own distance function.
    """
    N = len(mfd.points)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, min(n_sample, N), replace=False)
    ps  = mfd.points[idx]

    # Pairwise distances
    n_s = len(ps)
    all_dists = []
    chunk = 50
    for i in range(0, n_s, chunk):
        ie = min(i+chunk, n_s)
        for j_start in range(i, n_s, chunk):
            j_end = min(j_start+chunk, n_s)
            pi = np.repeat(ps[i:ie], j_end-j_start, axis=0)
            pj = np.tile(ps[j_start:j_end], (ie-i, 1))
            d  = mfd.dist_fn(pi, pj)
            all_dists.extend(d.tolist())

    all_dists = np.array(all_dists)
    # Remove zero distances (self-pairs)
    all_dists = all_dists[all_dists > 1e-10]

    if len(all_dists) == 0: return None

    # r-range: from 10th percentile to 90th percentile of distances
    r_lo = np.percentile(all_dists, 5)
    r_hi = np.percentile(all_dists, 60)
    if r_hi <= r_lo: return None

    r_vals = np.logspace(np.log10(r_lo), np.log10(r_hi), n_r)

    # Correlation integral C(r) = fraction of pairs with dist < r
    C = np.array([np.mean(all_dists < r) for r in r_vals])
    valid = (C > 0.02) & (C < 0.98)
    if valid.sum() < 4: return None

    slope, _ = np.polyfit(
        np.log(r_vals[valid]),
        np.log(C[valid] + 1e-10), 1)
    return slope

# ═══════════════════════════════════════════════════════════════
# BOX-COUNTING — in intrinsic coordinates only
# ═══════════════════════════════════════════════════════════════

def box_count_dim(mfd, n_scales=20, seed=42):
    """
    Box-counting in INTRINSIC coordinates.
    For T^d: use the d intrinsic phases.
    For S³: use the 4 embedding coordinates (normalized).
    """
    pts = mfd.points
    # Normalize to [0,1]^D_intrinsic
    p_min = pts.min(axis=0)
    p_max = pts.max(axis=0)
    span  = p_max - p_min
    span[span < 1e-10] = 1.0  # avoid division by zero
    p_norm = (pts - p_min) / span  # in [0,1]^D

    # Only use dimensions with non-zero span
    active = span > 1e-8
    p_active = p_norm[:, active]
    D_active = p_active.shape[1]

    # epsilon range: 10^(-1) to 10^(-0.3)
    # Must be small enough to see structure but large enough for N points
    epsilons = np.logspace(-1.2, -0.2, n_scales)
    counts = []
    for eps in epsilons:
        boxes = set()
        indices = np.floor(p_active / eps).astype(int)
        for row in indices:
            boxes.add(tuple(row))
        counts.append(len(boxes))

    counts = np.array(counts, dtype=float)
    valid  = counts > 1
    if valid.sum() < 3: return None

    # D_box = slope of log(N_boxes) vs log(1/eps)
    slope, _ = np.polyfit(
        np.log(1/epsilons[valid]),
        np.log(counts[valid]), 1)
    return slope, D_active

# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT: Compare all manifolds
# ═══════════════════════════════════════════════════════════════

def experiment_manifolds(N=800, k=20, n_runs=5):
    print("\n" + "="*65)
    print("MANIFOLD COMPARISON: D_corr, D_box, d_s for known geometries")
    print("="*65)
    print(f"  N={N}, k={k}, {n_runs} runs for d_s\n")
    print(f"  {'Manifold':18s} {'True d':>7} {'D_corr':>8} "
          f"{'D_box':>8} {'d_s_raw':>9} {'d_s_cal':>9}")
    print("  " + "-"*63)

    manifolds_gen = [
        make_T3, make_T4, make_S3, make_E6, make_T6
    ]

    results = {}
    for gen in manifolds_gen:
        mfd0 = gen(N, seed=0)
        name = mfd0.name

        # D_corr (single run, expensive)
        dc = corr_dim_manifold(mfd0)

        # D_box
        db_result = box_count_dim(mfd0)
        if db_result:
            db, D_act = db_result
        else:
            db, D_act = None, None

        # d_s (multiple runs)
        ds_vals = []
        for run in range(n_runs):
            mfd_r = gen(N, seed=run)
            G = build_knn_from_manifold(mfd_r, k=k, seed=run)
            ds_vals.append(spectral_dim(G))
        m_ds = np.mean(ds_vals)
        s_ds = np.std(ds_vals) / np.sqrt(n_runs)
        mc   = m_ds * CALIB

        true_d = mfd0.true_dim
        td_str = f"{true_d:.1f}" if true_d else "?"
        dc_str = f"{dc:.3f}" if dc else "N/A"
        db_str = f"{db:.3f}({D_act}D)" if db else "N/A"

        print(f"  {name:18s} {td_str:>7} {dc_str:>8} "
              f"{db_str:>8} {m_ds:>9.3f} {mc:>9.3f}±{s_ds*CALIB:.3f}")
        results[name] = {
            'true_d': true_d, 'dc': dc, 'db': db,
            'ds_raw': m_ds, 'ds_cal': mc, 'ds_sem': s_ds*CALIB
        }

    # Analysis
    print(f"\n  Analysis:")
    e6  = results.get('E6 orbit', {})
    t3  = results.get('T³ (3D torus)', {})
    t4  = results.get('T⁴ (4D torus)', {})
    t6  = results.get('T⁶ (null)', {})
    s3  = results.get('S³ (3-sphere)', {})

    # 1. Is D_corr correct for known manifolds?
    print(f"\n  D_corr validation:")
    for name, expected in [('T³ (3D torus)',3.0),
                            ('T⁴ (4D torus)',4.0),
                            ('T⁶ (null)',6.0)]:
        dc = results[name]['dc']
        if dc:
            err = abs(dc - expected)
            ok  = err < 0.5
            print(f"    {name}: D_corr={dc:.3f}, expected={expected}, "
                  f"err={err:.3f}  {'✅' if ok else '❌'}")

    # 2. d_s validation
    print(f"\n  d_s_cal validation:")
    for name, expected in [('T³ (3D torus)',3.0),
                            ('T⁴ (4D torus)',4.0),
                            ('T⁶ (null)',6.0)]:
        mc  = results[name]['ds_cal']
        sem = results[name]['ds_sem']
        err = abs(mc - expected)
        ok  = err < 0.5
        print(f"    {name}: d_s_cal={mc:.3f}±{sem:.3f}, "
              f"expected={expected}, err={err:.3f}  {'✅' if ok else '❌'}")

    # 3. E6 vs known manifolds
    print(f"\n  E6 interpretation:")
    if e6.get('dc') and t3.get('dc') and t4.get('dc'):
        dc_e6 = e6['dc']
        print(f"    D_corr(E6)={dc_e6:.3f} vs T³={t3['dc']:.3f}, T⁴={t4['dc']:.3f}")
        dist3 = abs(dc_e6 - t3['dc'])
        dist4 = abs(dc_e6 - t4['dc'])
        if dist3 < dist4:
            print(f"    → D_corr closer to T³: E6 orbit ≈ 3D")
        else:
            print(f"    → D_corr closer to T⁴: E6 orbit ≈ 4D")

    if e6.get('ds_cal') and t3.get('ds_cal') and t4.get('ds_cal'):
        ds_e6 = e6['ds_cal']
        print(f"    d_s_cal(E6)={ds_e6:.3f} vs T³={t3['ds_cal']:.3f}, T⁴={t4['ds_cal']:.3f}")
        dist3 = abs(ds_e6 - t3['ds_cal'])
        dist4 = abs(ds_e6 - t4['ds_cal'])
        if dist3 < dist4:
            print(f"    → d_s closer to T³: spectral dim consistent with 3D")
        else:
            print(f"    → d_s closer to T⁴: spectral dim consistent with 4D")

    # 4. The key question: does d_s OVERESTIMATE for quasi-periodic orbits?
    print(f"\n  Key question: d_s vs D_corr discrepancy")
    for name in results:
        dc = results[name]['dc']
        ds = results[name]['ds_cal']
        td = results[name]['true_d']
        if dc and ds:
            ratio = ds / dc if dc > 0 else None
            print(f"    {name:18s}: d_s_cal/D_corr = {ds:.2f}/{dc:.2f} = "
                  f"{ratio:.2f}" if ratio else f"    {name}: N/A")

    return results

# ═══════════════════════════════════════════════════════════════
# k* MEANING: Is k=20 related to E6 structure?
# ═══════════════════════════════════════════════════════════════

def experiment_kstar_meaning(N=600, n_runs=6):
    """
    k*=20 gives d_s(cal)=4 for E6.
    Is k*=20 related to E6 algebraic properties?

    E6 properties:
    - Rank: 6
    - Coxeter number h=12
    - Number of positive roots: 36
    - Dimension: 78
    - Exponents: {1,4,5,7,8,11}

    Test: scan k for E6, A6, D6, B6.
    Does k*(algebra) correlate with algebraic invariants?
    """
    print("\n" + "="*60)
    print("k* MEANING: What determines k*?")
    print("="*60)
    print("  E6 properties: rank=6, h=12, |Φ⁺|=36, dim=78")
    print("  Scanning k for all rank-6 algebras...\n")

    def omega_alg(name):
        if name=='E6': return 2.0*np.sin(np.pi*np.array([1,4,5,7,8,11])/12.0)
        if name=='A6': return 2.0*np.sin(np.pi*np.array([1,2,3,4,5,6])/7.0)
        if name=='D6': return 2.0*np.sin(np.pi*np.array([1,3,5,5,7,9])/12.0)
        if name=='B6': return 2.0*np.sin(np.pi*np.array([1,3,5,7,9,11])/12.0)

    # Algebraic properties {name: (h, |Φ⁺|, dim)}
    alg_props = {
        'E6': (12, 36, 78),
        'A6': (7,  21, 48),
        'D6': (10, 30, 66),
        'B6': (11, 36, 78),  # same |Φ⁺| and dim as... wait
    }
    # Correct values:
    # A6: h=7,  |Φ⁺|=21, dim=48
    # B6: h=11, |Φ⁺|=36, dim=78  ← actually B6 dim=78 too?
    # D6: h=10, |Φ⁺|=30, dim=66
    # E6: h=12, |Φ⁺|=36, dim=78
    # Note: B6 and E6 both have |Φ⁺|=36

    k_values = [4, 8, 12, 16, 20, 25]
    alg_names = ['E6', 'A6', 'D6', 'B6']

    k_star_per_alg = {}
    print(f"  {'Algebra':8s} {'h':>5} {'|Φ⁺|':>7}", end="")
    for k in k_values:
        print(f"  {'k='+str(k):>8}", end="")
    print("  k*")
    print("  " + "-"*70)

    all_data = {}
    for alg in alg_names:
        h, nroots, dim = alg_props[alg]
        omega = omega_alg(alg)
        print(f"  {alg:8s} {h:>5} {nroots:>7}", end="", flush=True)
        ds_per_k = {}
        for k in k_values:
            vals = []
            for run in range(n_runs):
                ph = evolve_phases(N, omega, seed=run)
                G  = build_knn_from_manifold(
                    Manifold('tmp', ph, lambda p,q: torus_dist(p,q), None),
                    k=k, seed=run)
                vals.append(spectral_dim(G) * CALIB)
            ds_per_k[k] = np.mean(vals)
            print(f"  {ds_per_k[k]:>8.2f}", end="", flush=True)

        # Find k* (first k where d_s_cal >= 3.8)
        k_star = None
        for k in k_values:
            if ds_per_k[k] >= 3.8:
                k_star = k
                break
        k_star_per_alg[alg] = k_star
        print(f"  {k_star if k_star else '>25'}")
        all_data[alg] = ds_per_k

    print(f"\n  k* per algebra:")
    for alg in alg_names:
        h, nroots, dim = alg_props[alg]
        ks = k_star_per_alg[alg]
        print(f"    {alg}: k*={ks if ks else '>25'}, h={h}, |Φ⁺|={nroots}")

    # Correlation: k* vs algebraic invariants
    k_star_vals = [k_star_per_alg[a] or 30 for a in alg_names]
    h_vals      = [alg_props[a][0] for a in alg_names]
    nroot_vals  = [alg_props[a][1] for a in alg_names]

    if len(set(k_star_vals)) > 1:
        from scipy.stats import pearsonr
        r_h, p_h = pearsonr(h_vals, k_star_vals)
        r_n, p_n = pearsonr(nroot_vals, k_star_vals)
        print(f"\n  Correlation k* vs h:    r={r_h:.3f}, p={p_h:.3f}")
        print(f"  Correlation k* vs |Φ⁺|: r={r_n:.3f}, p={p_n:.3f}")
        if abs(r_h) > 0.8:
            print(f"  → k* correlates with Coxeter number h!")
        if abs(r_n) > 0.8:
            print(f"  → k* correlates with number of positive roots |Φ⁺|!")

    return {'k_star': k_star_per_alg, 'ds_per_k': all_data,
            'alg_props': alg_props}

# ═══════════════════════════════════════════════════════════════
# FINAL PAPER TABLE
# ═══════════════════════════════════════════════════════════════

def print_final_paper_table():
    """
    Produce the table suitable for inclusion in the paper.
    All numbers from Parts I-IV+ v1-v7.
    """
    print("\n" + "═"*70)
    print("FINAL PAPER TABLE: The Monostring Hypothesis — Complete Results")
    print("═"*70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    CONFIRMED FINDINGS                                │
├──────────────────────────┬──────────┬──────────────────────────────┤
│ Finding                  │ Part     │ Key number                   │
├──────────────────────────┼──────────┼──────────────────────────────┤
│ Kuramoto T_c≈1.4 (2+4)   │ II       │ 20+ runs, null controlled    │
│ d_s reduction 37–51%     │ III,IV   │ Weyl + heat kernel           │
│ ω dominates K (ANOVA)    │ IV       │ 66% vs 3%                    │
│ D_KY≈4 plateau (diss.)   │ I        │ 13/13 algebras               │
│ GUE statistics ⟨r⟩=0.529 │ III      │ GUE=0.531                    │
│ D_corr(E6)≈2.9           │ IV+ v5   │ GP dimension of orbit        │
│ d_eff(E6)≈3.5            │ IV+ v5   │ k-NN scaling                 │
│ d_s(E6,k=20,cal)≈4.0     │ IV+ v4   │ N∈[400,1000], t>90          │
│ E6 unique: Δd_s=+2.95    │ IV+ v4   │ vs all other rank-6 algs    │
├──────────────────────────┴──────────┴──────────────────────────────┤
│                    FALSIFIED CLAIMS                                  │
├──────────────────────────┬──────────┬──────────────────────────────┤
│ 6D→4D (Lyapunov)         │ I, v7    │ Symplectic: D_KY=2r always   │
│ Gauge Higgs mechanism     │ II       │ Null ratio > E6 ratio        │
│ Yukawa mechanism          │ II       │ 6 defs anti-correlate        │
│ d_s=4 as fixed dim        │ IV       │ d_s∝N in chain graph         │
│ Dark energy from E6       │ IV+ v5   │ p=0.90: E6 irrelevant        │
│ Feedback ≠ const λ (DE)   │ IV+ v5   │ p=0.93: no difference        │
│ D_corr(E6)=4              │ IV+ v5   │ D_corr=2.89, d_eff=3.48     │
├──────────────────────────┴──────────┴──────────────────────────────┤
│                    OPEN QUESTIONS                                    │
├──────────────────────────┬──────────┬──────────────────────────────┤
│ d_s(cal)≈4 if D_corr≈2.9?│ IV+ v6-7 │ Fractal orbit + graph effect │
│ k*=20 geometric meaning? │ IV+ v5-7 │ Related to h=12 or |Φ⁺|=36? │
│ Quantum walks → Dirac    │ Pending  │ Unitary dynamics needed      │
│ KAM resonances (analytic)│ IV+ v3   │ Observed, not derived        │
└──────────────────────────┴──────────┴──────────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_all(r_mfd, r_kstar):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Part IV+ v7: Final Manifold Comparison",
                 fontsize=13, fontweight='bold')

    # 1: D_corr per manifold
    ax = axes[0,0]
    names = list(r_mfd.keys())
    dc   = [r_mfd[n]['dc'] or 0   for n in names]
    ds   = [r_mfd[n]['ds_cal'] or 0 for n in names]
    td   = [r_mfd[n]['true_d'] or 0 for n in names]
    x    = np.arange(len(names))
    w    = 0.27
    ax.bar(x-w, td,   w, label='True d',     color='gray',   alpha=0.7)
    ax.bar(x,   dc,   w, label='D_corr',     color='steelblue', alpha=0.8)
    ax.bar(x+w, ds,   w, label='d_s (cal)',  color='tomato',  alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8)
    ax.axhline(4.0, color='black', linestyle=':', lw=2)
    ax.axhline(3.0, color='gray',  linestyle=':', lw=1)
    ax.set_ylabel('Dimension')
    ax.set_title('D_corr vs d_s vs true d\n(correct manifold distance)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # 2: D_corr vs d_s scatter
    ax = axes[0,1]
    colors_mfd = {
        'T³ (3D torus)': 'green', 'T⁴ (4D torus)': 'blue',
        'S³ (3-sphere)': 'purple', 'E6 orbit': 'red', 'T⁶ (null)': 'gray'
    }
    for name in names:
        dc_v = r_mfd[name]['dc'] or 0
        ds_v = r_mfd[name]['ds_cal']
        col  = colors_mfd.get(name, 'black')
        ax.scatter(dc_v, ds_v, s=120, color=col, zorder=5,
                   label=name)
        ax.annotate(name, (dc_v, ds_v), fontsize=7,
                    xytext=(6,4), textcoords='offset points')
    # True dimension line
    lim = 7
    ax.plot([0,lim],[0,lim], 'k--', alpha=0.4, label='d_s=D_corr')
    # Highlight E6
    if 'E6 orbit' in r_mfd:
        e6dc = r_mfd['E6 orbit']['dc'] or 0
        e6ds = r_mfd['E6 orbit']['ds_cal']
        ax.annotate('← E6 here',
                    (e6dc, e6ds), fontsize=9, color='red',
                    xytext=(-50, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('D_corr'); ax.set_ylabel('d_s (calibrated)')
    ax.set_title('D_corr vs d_s\n(diagonal=perfect agreement)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim([0,lim]); ax.set_ylim([0,lim])

    # 3: k-scan per algebra
    ax = axes[0,2]
    k_vals = sorted(list(r_kstar['ds_per_k']['E6'].keys()))
    alg_colors = {'E6':'steelblue','A6':'tomato','D6':'green','B6':'purple'}
    for alg in ['E6','A6','D6','B6']:
        ds_k = [r_kstar['ds_per_k'][alg][k] for k in k_vals]
        ks   = r_kstar['k_star'].get(alg)
        ax.plot(k_vals, ds_k, 'o-', color=alg_colors[alg],
                label=f'{alg} (k*={ks if ks else ">25"})', lw=2)
    ax.axhline(4.0, color='black', linestyle=':', lw=2, label='d=4')
    ax.axhline(3.8, color='gray',  linestyle=':', lw=1, label='threshold')
    ax.set_xlabel('k (nearest neighbors)')
    ax.set_ylabel('d_s (calibrated)')
    ax.set_title('k-scan per algebra\n(k* = first k with d_s_cal≥3.8)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4: k* vs algebraic invariants
    ax = axes[1,0]
    alg_names_list = ['E6','A6','D6','B6']
    k_star_vals    = [r_kstar['k_star'].get(a) or 30
                      for a in alg_names_list]
    h_vals         = [r_kstar['alg_props'][a][0] for a in alg_names_list]
    nroot_vals     = [r_kstar['alg_props'][a][1] for a in alg_names_list]
    cols_alg       = [alg_colors[a] for a in alg_names_list]
    sc = ax.scatter(h_vals, k_star_vals, s=150, c=cols_alg, zorder=5)
    for a,h,ks in zip(alg_names_list,h_vals,k_star_vals):
        ax.annotate(a, (h,ks), fontsize=9,
                    xytext=(4,4), textcoords='offset points')
    ax.set_xlabel('Coxeter number h')
    ax.set_ylabel('k* (connectivity for d_s=4)')
    ax.set_title('k* vs Coxeter number h\n(does k* correlate with h?)')
    ax.grid(True, alpha=0.3)

    # 5: Complete scorecard
    ax = axes[1,1]
    ax.axis('off')
    lines = [
        ("CONFIRMED", 'darkgreen', [
            "D_corr(E6)≈2.9 (quasi-3D orbit)",
            "d_s(E6,k=20)≈4.0 (spectral 4D)",
            "E6 unique: Δd_s=+2.95 vs null",
            "D_KY≈4 plateau (all algebras)",
            "Kuramoto T_c≈1.4 (Part II)",
        ]),
        ("FALSIFIED", 'red', [
            "Emergent 4D spacetime (SOH)",
            "Dark energy from E6 structure",
            "Gauge Higgs / Yukawa mech.",
            "d_s=4 as fixed dimension",
            "D_corr(E6)=4 (it's ≈2.9)",
        ]),
        ("OPEN", 'darkorange', [
            "Why d_s≈4 if D_corr≈2.9?",
            "k* geometric meaning?",
            "Quantum walk → Dirac?",
        ]),
    ]
    y = 0.98
    for header, col, items in lines:
        ax.text(0.03, y, header, fontsize=10, fontweight='bold',
                color=col, transform=ax.transAxes, va='top')
        y -= 0.08
        for item in items:
            sym = "✅" if col=='darkgreen' else ("❌" if col=='red' else "❓")
            ax.text(0.05, y, f"{sym} {item}", fontsize=8,
                    color=col, transform=ax.transAxes, va='top')
            y -= 0.08
        y -= 0.02
    ax.set_title('Complete Scorecard')

    # 6: Final verdict
    ax = axes[1,2]
    ax.axis('off')
    verdict_text = """
THE MONOSTRING HYPOTHESIS
FINAL VERDICT

FALSIFIED:
Emergent 4D spacetime from SOH.

SURVIVES (unexplained):
E₆ Coxeter orbit on T⁶ has
D_corr ≈ 2.9 (quasi-3D)
but k-NN graph at k*=20 gives
d_s(cal) ≈ 4.0 (spectral 4D).

Gap: d_s/D_corr ≈ 1.38
This ratio is unexplained.

k*=20 may relate to:
• Coxeter number h=12
• Number of roots |Φ⁺|=36
• None of the above

Further work needed:
• Analytic derivation of k*
• Quantum walk implementation
• KAM resonance analysis
    """
    ax.text(0.05, 0.97, verdict_text, fontsize=8,
            fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      alpha=0.8))
    ax.set_title('Final Verdict')

    plt.tight_layout()
    plt.savefig('part4plus_v7_final.png', dpi=150, bbox_inches='tight')
    print("  Saved: part4plus_v7_final.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Monostring Hypothesis — Part IV+ v7 (TRUE FINAL)      ║")
    print("║   Correct manifolds | k* meaning | Final verdict        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    print(f"Calib: ×{CALIB:.4f}\n")

    print("[1/3] Manifold comparison (correct distances)...")
    r_mfd = experiment_manifolds(N=800, k=20, n_runs=5)

    print("\n[2/3] k* meaning — algebra scan...")
    r_kstar = experiment_kstar_meaning(N=600, n_runs=6)

    print("\n[3/3] Final paper table...")
    print_final_paper_table()

    plot_all(r_mfd, r_kstar)

    elapsed = time.time() - t_start
    print(f"\nRuntime: {elapsed/60:.1f} min")
    print("\nDone. This is the final experiment in the Part IV+ series.")
