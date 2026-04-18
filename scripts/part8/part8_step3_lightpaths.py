"""
Part VIII — Step 3: Shortest Paths as Light Rays
=================================================
Does the geodesic in the monostring phase graph
correspond to null propagation in emergent 3D space?

Method:
1. Generate E6 orbit (T=5000 points)
2. Build k-NN graph
3. Sample random pairs (A, B) on orbit
4. Measure: graph distance d_G(A,B)
            3D PCA distance d_3(A,B)
            T⁶ distance d_6(A,B)
5. Test: d_G ∝ d_3 (not d_6)?
   → Light propagates in 3D, not 6D
6. Measure "speed": c_eff = d_3 / d_G
   → Is c_eff constant? (Lorentz-like)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# 1. ORBIT GENERATION (same convention as Parts V-VI)
# ══════════════════════════════════════════════════════════════════

def get_e6_omega():
    m = np.array([1, 4, 5, 7, 8, 11], dtype=float)
    h = 12.0
    return 2.0 * np.sin(np.pi * m / h)


def get_random_omega(seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 2.0, 6)


def generate_orbit(omega, T=5000, kappa=0.05,
                   warmup=500, seed=42):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)
    traj = np.zeros((T, len(omega)))
    for t in range(T):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)
        traj[t] = phi
    return traj


# ══════════════════════════════════════════════════════════════════
# 2. BUILD PHASE GRAPH
# ══════════════════════════════════════════════════════════════════

def build_phase_graph(orbit, k=8, include_sequential=True):
    """
    Build k-NN graph of orbit points.

    Each node = one monostring state (time step).
    Edges:
      - k nearest neighbours in T⁶ (spatial proximity)
      - sequential edges t→t+1 (temporal flow)

    Edge weight = Euclidean distance in T⁶
    (on torus: min(|Δφ|, 2π-|Δφ|) per dimension)
    """
    T, rank = orbit.shape

    # Torus distance
    def torus_dist(a, b):
        d = np.abs(a - b)
        d = np.minimum(d, 2*np.pi - d)
        return np.sqrt(np.sum(d**2, axis=-1))

    # k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1,
                             algorithm='ball_tree',
                             metric='euclidean')
    nbrs.fit(orbit)
    distances, indices = nbrs.kneighbors(orbit)

    G = nx.Graph()
    G.add_nodes_from(range(T))

    # k-NN edges (spatial)
    for i in range(T):
        for j_idx in range(1, k+1):
            j   = indices[i, j_idx]
            w   = distances[i, j_idx]
            G.add_edge(i, j, weight=w,
                       edge_type='spatial')

    # Sequential edges (temporal flow)
    if include_sequential:
        for t in range(T-1):
            w = torus_dist(orbit[t], orbit[t+1])
            if not G.has_edge(t, t+1):
                G.add_edge(t, t+1, weight=w,
                           edge_type='temporal')
            # else keep min weight

    return G


# ══════════════════════════════════════════════════════════════════
# 3. PCA PROJECTION TO 3D
# ══════════════════════════════════════════════════════════════════

def project_to_3d(orbit):
    """
    Project T⁶ orbit to 3D using PCA.
    Returns: (T, 3) array and explained variance.
    """
    pca = PCA(n_components=3)
    proj = pca.fit_transform(orbit)
    return proj, pca.explained_variance_ratio_


# ══════════════════════════════════════════════════════════════════
# 4. SAMPLE PAIRS AND MEASURE DISTANCES
# ══════════════════════════════════════════════════════════════════

def sample_and_measure(orbit, proj_3d, G,
                        n_pairs=300, seed=42):
    """
    Sample random pairs (A, B) and measure:
    - d_graph: shortest path length in graph
    - d_3d:    Euclidean distance in 3D PCA space
    - d_6d:    Euclidean distance in T⁶
    - d_time:  |t_A - t_B| (step index difference)

    Returns dict of arrays.
    """
    T    = orbit.shape[0]
    rng  = np.random.default_rng(seed)

    # Sample pairs with varied separations
    # Use both close and far pairs
    pairs = []

    # Random pairs
    for _ in range(n_pairs):
        a = rng.integers(0, T)
        b = rng.integers(0, T)
        if a != b:
            pairs.append((int(a), int(b)))

    # Remove duplicates
    pairs = list(set(pairs))[:n_pairs]

    d_graph = []
    d_3d    = []
    d_6d    = []
    d_time  = []
    valid   = []

    print(f"  Computing {len(pairs)} shortest paths...")

    # Precompute shortest paths from sample of sources
    # (full APSP too slow for T=5000)
    sources = list(set([p[0] for p in pairs]))

    # Use Dijkstra from each source
    path_lengths = {}
    for s in sources:
        lengths = nx.single_source_dijkstra_path_length(
            G, s, weight='weight')
        path_lengths[s] = lengths

    for a, b in pairs:
        if a in path_lengths and b in path_lengths[a]:
            d_g = path_lengths[a][b]
        else:
            continue

        # 3D distance
        d3 = np.linalg.norm(proj_3d[a] - proj_3d[b])

        # 6D torus distance
        diff = np.abs(orbit[a] - orbit[b])
        diff = np.minimum(diff, 2*np.pi - diff)
        d6   = np.sqrt(np.sum(diff**2))

        # Time distance
        dt = abs(a - b)

        d_graph.append(d_g)
        d_3d.append(d3)
        d_6d.append(d6)
        d_time.append(dt)
        valid.append((a, b))

    print(f"  Valid pairs: {len(d_graph)}")

    return {
        'd_graph': np.array(d_graph),
        'd_3d':    np.array(d_3d),
        'd_6d':    np.array(d_6d),
        'd_time':  np.array(d_time),
        'pairs':   valid,
    }


# ══════════════════════════════════════════════════════════════════
# 5. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyse_correlations(data, label=""):
    """
    Test which distance best predicts graph distance.

    Key question: does d_graph ∝ d_3d (not d_6d)?
    → Light propagates in effective 3D, not T⁶
    """
    dg = data['d_graph']
    d3 = data['d_3d']
    d6 = data['d_6d']
    dt = data['d_time']

    print(f"\n  [{label}] Correlation analysis:")
    print(f"  {'Predictor':>12}  {'r':>8}  {'R²':>8}  "
          f"{'p':>10}  interpretation")
    print(f"  {'-'*65}")

    results = {}

    for name, x in [('d_3d', d3),
                     ('d_6d', d6),
                     ('d_time', dt)]:
        r, p = stats.pearsonr(x, dg)
        r2   = r**2

        if r2 > 0.9:
            interp = "STRONG predictor"
        elif r2 > 0.7:
            interp = "moderate predictor"
        elif r2 > 0.3:
            interp = "weak predictor"
        else:
            interp = "NO correlation"

        print(f"  {name:>12}  {r:>8.4f}  {r2:>8.4f}  "
              f"{p:>10.2e}  {interp}")

        results[name] = {'r': r, 'r2': r2, 'p': p}

    # Key comparison: is d_3d better than d_6d?
    r2_3d = results['d_3d']['r2']
    r2_6d = results['d_6d']['r2']

    print(f"\n  Key result: R²(d_3d) = {r2_3d:.4f} "
          f"vs R²(d_6d) = {r2_6d:.4f}")

    if r2_3d > r2_6d + 0.05:
        print(f"  → 3D distance is BETTER predictor: "
              f"light lives in 3D ✓")
    elif r2_6d > r2_3d + 0.05:
        print(f"  → 6D distance is BETTER predictor: "
              f"no dimensional reduction ✗")
    else:
        print(f"  → 3D and 6D equally predictive "
              f"(inconclusive)")

    # Effective speed: c_eff = d_3d / d_graph
    c_eff = d3 / (dg + 1e-10)
    print(f"\n  Effective speed c_eff = d_3d/d_graph:")
    print(f"  mean  = {c_eff.mean():.4f}")
    print(f"  std   = {c_eff.std():.4f}")
    print(f"  CV    = {c_eff.std()/c_eff.mean():.4f}")

    if c_eff.std() / c_eff.mean() < 0.1:
        print(f"  → c_eff ≈ CONSTANT: Lorentz-like ✓")
    else:
        print(f"  → c_eff variable: no Lorentz invariance ✗")

    results['c_eff'] = c_eff
    return results


# ══════════════════════════════════════════════════════════════════
# 6. CAUSAL STRUCTURE TEST
# ══════════════════════════════════════════════════════════════════

def causal_structure_test(data, c_eff_mean, label=""):
    """
    Test for light-cone structure.

    For each pair (A, B):
    - "Spacelike" if d_3d/d_graph > c_eff_mean
    - "Timelike" if d_3d/d_graph < c_eff_mean
    - "Lightlike" if d_3d/d_graph ≈ c_eff_mean

    If causal structure exists:
    → Spacelike pairs should be unreachable faster than c
    → Timelike pairs should cluster
    """
    dg   = data['d_graph']
    d3   = data['d_3d']
    dt   = data['d_time']

    c_ratio = d3 / (dg + 1e-10)

    # Classify
    tol  = 0.2 * c_eff_mean
    spacelike = c_ratio > c_eff_mean + tol
    timelike  = c_ratio < c_eff_mean - tol
    lightlike = ~spacelike & ~timelike

    n_space = spacelike.sum()
    n_time  = timelike.sum()
    n_light = lightlike.sum()
    n_total = len(dg)

    print(f"\n  [{label}] Causal structure:")
    print(f"  Spacelike:  {n_space:4d} "
          f"({100*n_space/n_total:.1f}%)")
    print(f"  Timelike:   {n_time:4d} "
          f"({100*n_time/n_total:.1f}%)")
    print(f"  Light-like: {n_light:4d} "
          f"({100*n_light/n_total:.1f}%)")

    # Spearman test: does temporal separation
    # predict causal class?
    rho, p = stats.spearmanr(dt, c_ratio)
    print(f"\n  Spearman(Δt, c_ratio): "
          f"ρ={rho:.4f}, p={p:.4e}")

    if p < 0.01 and rho < -0.2:
        print(f"  → Larger Δt → smaller c_ratio "
              f"(temporal pairs are 'timelike') ✓")
    else:
        print(f"  → No causal ordering by Δt ✗")

    return {
        'spacelike': spacelike,
        'timelike':  timelike,
        'lightlike': lightlike,
        'c_ratio':   c_ratio,
        'rho':       rho,
        'p_rho':     p,
    }


# ══════════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_step3(T=3000, k_nn=8, n_pairs=300, seed=42):
    print("=" * 62)
    print("PART VIII Step 3: Shortest Paths as Light Rays")
    print("=" * 62)

    all_out = {}

    for alg_name, omega_fn in [
        ('E6',     get_e6_omega),
        ('Random', lambda: get_random_omega(seed))
    ]:
        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}")

        omega = omega_fn()
        print(f"  omega = {np.round(omega, 4).tolist()}")

        # Generate orbit
        print(f"  Generating orbit T={T}...")
        orbit = generate_orbit(omega, T=T,
                                kappa=0.05,
                                warmup=500,
                                seed=seed)

        # PCA to 3D
        proj_3d, var_ratio = project_to_3d(orbit)
        print(f"  3D explained variance: "
              f"{100*var_ratio.sum():.1f}%")

        # Build graph
        print(f"  Building k={k_nn} NN graph...")
        G = build_phase_graph(orbit, k=k_nn,
                               include_sequential=True)
        print(f"  Graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

        # Sample pairs
        print(f"  Sampling {n_pairs} pairs...")
        data = sample_and_measure(
            orbit, proj_3d, G,
            n_pairs=n_pairs, seed=seed)

        # Correlation analysis
        corr = analyse_correlations(data, label=alg_name)

        # Causal structure
        c_eff_mean = corr['c_eff'].mean()
        causal = causal_structure_test(
            data, c_eff_mean, label=alg_name)

        all_out[alg_name] = {
            'orbit':    orbit,
            'proj_3d':  proj_3d,
            'G':        G,
            'data':     data,
            'corr':     corr,
            'causal':   causal,
            'omega':    omega,
            'var_ratio': var_ratio,
        }

    return all_out


# ══════════════════════════════════════════════════════════════════
# 8. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(all_out):
    alg_list = ['E6', 'Random']
    colors   = {'E6': '#2196F3', 'Random': '#9E9E9E'}

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Part VIII — Step 3: Shortest Paths as Light Rays\n"
        "Does the graph geodesic = null geodesic in 3D?",
        fontsize=14, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.50, wspace=0.38)

    for col, alg_name in enumerate(alg_list):
        out  = all_out[alg_name]
        data = out['data']
        corr = out['corr']
        causal = out['causal']
        c    = colors[alg_name]

        dg = data['d_graph']
        d3 = data['d_3d']
        d6 = data['d_6d']
        dt = data['d_time']

        # ── Row 0 col 0-1: d_graph vs d_3d ───────────
        ax = fig.add_subplot(gs[0, col*2])
        ax.set_facecolor('#f0f4f8')

        r2_3d = corr['d_3d']['r2']
        ax.scatter(d3, dg, alpha=0.4, s=25,
                   color=c, edgecolors='none')

        # Fit line
        sl, ic, _, _, _ = stats.linregress(d3, dg)
        x_fit = np.linspace(d3.min(), d3.max(), 100)
        ax.plot(x_fit, sl*x_fit + ic, 'r-',
                lw=2.5, label=f'fit R²={r2_3d:.3f}')

        ax.set_xlabel('3D distance d₃(A,B)',
                      fontsize=10)
        ax.set_ylabel('Graph distance d_G(A,B)',
                      fontsize=10)
        ax.set_title(f'{alg_name}: d_G vs d_3D\n'
                     f'R²={r2_3d:.4f}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Row 0 col 1-2: d_graph vs d_6d ───────────
        ax = fig.add_subplot(gs[0, col*2+1])
        ax.set_facecolor('#f0f4f8')

        r2_6d = corr['d_6d']['r2']
        ax.scatter(d6, dg, alpha=0.4, s=25,
                   color='#FF9800', edgecolors='none')

        sl6, ic6, _, _, _ = stats.linregress(d6, dg)
        x6_fit = np.linspace(d6.min(), d6.max(), 100)
        ax.plot(x6_fit, sl6*x6_fit + ic6, 'r-',
                lw=2.5, label=f'fit R²={r2_6d:.3f}')

        ax.set_xlabel('6D distance d₆(A,B)',
                      fontsize=10)
        ax.set_ylabel('Graph distance d_G(A,B)',
                      fontsize=10)
        ax.set_title(f'{alg_name}: d_G vs d_6D\n'
                     f'R²={r2_6d:.4f}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Row 1: c_eff distributions ────────────────────
    for col, alg_name in enumerate(alg_list):
        out  = all_out[alg_name]
        corr = out['corr']
        c    = colors[alg_name]

        ax = fig.add_subplot(gs[1, col*2:col*2+2])
        ax.set_facecolor('#f0f4f8')

        c_eff = corr['c_eff']

        ax.hist(c_eff, bins=40, color=c,
                alpha=0.7, edgecolor='k', lw=0.5)
        ax.axvline(c_eff.mean(), color='red',
                   lw=2.5, ls='-',
                   label=f'mean={c_eff.mean():.4f}')
        ax.axvline(c_eff.mean() + c_eff.std(),
                   color='red', lw=1.5, ls='--',
                   label=f'±σ={c_eff.std():.4f}')
        ax.axvline(c_eff.mean() - c_eff.std(),
                   color='red', lw=1.5, ls='--')

        cv = c_eff.std() / c_eff.mean()
        ax.set_xlabel('c_eff = d_3d / d_graph',
                      fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(
            f'{alg_name}: Distribution of effective speed\n'
            f'CV = {cv:.4f}  '
            f'{"≈ CONSTANT (Lorentz-like)" if cv<0.15 else "variable (no Lorentz)"}',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Row 2: 3D orbit with causal coloring ──────────
    for col, alg_name in enumerate(alg_list):
        out    = all_out[alg_name]
        proj   = out['proj_3d']
        causal = out['causal']
        data   = out['data']

        ax = fig.add_subplot(gs[2, col*2],
                             projection='3d')
        ax.set_facecolor('#f0f4f8')

        # Plot orbit (subsample)
        step = max(1, len(proj)//500)
        ax.plot(proj[::step, 0],
                proj[::step, 1],
                proj[::step, 2],
                'k-', alpha=0.15, lw=0.5)

        # Color pairs by causal type
        pairs = data['pairs']
        c_ratio = causal['c_ratio']

        n_show = min(50, len(pairs))
        for idx in range(n_show):
            a, b = pairs[idx]
            cr   = c_ratio[idx]

            if causal['spacelike'][idx]:
                clr = 'blue'
                lw  = 1.5
            elif causal['timelike'][idx]:
                clr = 'red'
                lw  = 1.5
            else:
                clr = 'green'
                lw  = 2.0

            ax.plot([proj[a,0], proj[b,0]],
                    [proj[a,1], proj[b,1]],
                    [proj[a,2], proj[b,2]],
                    '-', color=clr,
                    alpha=0.5, lw=lw)

        ax.set_title(
            f'{alg_name}: 3D orbit\n'
            'Blue=spacelike, Red=timelike, '
            'Green=lightlike',
            fontsize=9, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)

    # ── Row 2: Causal pie charts ───────────────────────
    for col, alg_name in enumerate(alg_list):
        out    = all_out[alg_name]
        causal = out['causal']
        c      = colors[alg_name]

        ax = fig.add_subplot(gs[2, col*2+1])
        ax.set_facecolor('#f0f4f8')

        n_space = causal['spacelike'].sum()
        n_time  = causal['timelike'].sum()
        n_light = causal['lightlike'].sum()
        total   = n_space + n_time + n_light

        sizes  = [n_space, n_time, n_light]
        labels = [f'Spacelike\n{n_space}',
                  f'Timelike\n{n_time}',
                  f'Lightlike\n{n_light}']
        clrs   = ['#2196F3', '#F44336', '#4CAF50']

        ax.pie(sizes, labels=labels,
               colors=clrs, autopct='%1.1f%%',
               startangle=90,
               textprops={'fontsize': 10})

        rho = causal['rho']
        p   = causal['p_rho']
        ax.set_title(
            f'{alg_name}: Causal classification\n'
            f'Spearman(Δt,c_ratio): ρ={rho:.3f}, '
            f'p={p:.3e}',
            fontsize=9, fontweight='bold')

    plt.savefig('part8_step3_lightpaths.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: part8_step3_lightpaths.png")


# ══════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VIII Step 3: Shortest Paths as Light Rays\n")

    all_out = run_step3(T=3000, k_nn=8,
                         n_pairs=300, seed=42)
    make_figure(all_out)

    # Final summary
    print("\n" + "="*62)
    print("FINAL SUMMARY: Step 3")
    print("="*62)

    print(f"\n{'Algebra':<10} {'R²(d_3d)':>10} "
          f"{'R²(d_6d)':>10} {'CV(c_eff)':>10} "
          f"{'Verdict'}")
    print("-"*65)

    for alg_name in ['E6', 'Random']:
        out  = all_out[alg_name]
        corr = out['corr']

        r2_3d = corr['d_3d']['r2']
        r2_6d = corr['d_6d']['r2']
        cv    = (corr['c_eff'].std() /
                 corr['c_eff'].mean())

        if r2_3d > r2_6d + 0.05 and cv < 0.15:
            verdict = "LIGHT in 3D ✓"
        elif r2_3d > r2_6d + 0.05:
            verdict = "3D preferred, c variable"
        elif cv < 0.15:
            verdict = "c≈const but no 3D preference"
        else:
            verdict = "No structure ✗"

        print(f"{alg_name:<10} {r2_3d:>10.4f} "
              f"{r2_6d:>10.4f} {cv:>10.4f} "
              f"{verdict}")

    print("\nPhysical interpretation:")
    print("  R²(d_3d) >> R²(d_6d): graph distance")
    print("  measured in emergent 3D, not T⁶")
    print("  CV(c_eff) < 0.15: effective 'speed of light'")
    print("  is approximately constant = Lorentz-like")
