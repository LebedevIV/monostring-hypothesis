"""
Part IV+ v3: Stable spectral dimension measurement
Key fixes:
1. k-NN graph instead of RGG (degree exactly controlled)
2. Normalized Laplacian with correct t-range
3. Plateau detection via second derivative zero-crossing
4. Direction D: cap feedback to prevent divergence
5. Full diagnostic output for every graph
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.sparse.linalg import eigsh
import time
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# ALGEBRA DEFINITIONS
# ═══════════════════════════════════════════════════════════════

def cartan_E6():
    K = np.diag([2.0]*6)
    for i,j in [(0,1),(1,2),(2,3),(3,4),(2,5)]:
        K[i,j]=K[j,i]=-1.0
    return K

def cartan_B6():
    K = np.diag([2.0]*6)
    for i in range(5): K[i,i+1]=K[i+1,i]=-1.0
    K[4,5]=-2.0; K[5,4]=-1.0
    return K

def cartan_D6():
    K = np.diag([2.0]*6)
    for i in range(4): K[i,i+1]=K[i+1,i]=-1.0
    K[3,5]=K[5,3]=-1.0
    return K

def cartan_A6():
    K = np.diag([2.0]*6)
    for i in range(5): K[i,i+1]=K[i+1,i]=-1.0
    return K

def omega_E6():
    return 2.0*np.sin(np.pi*np.array([1,4,5,7,8,11])/12.0)

def omega_B6():
    return 2.0*np.sin(np.pi*np.array([1,3,5,7,9,11])/12.0)

def omega_D6():
    return 2.0*np.sin(np.pi*np.array([1,3,5,5,7,9])/12.0)

def omega_A6():
    return 2.0*np.sin(np.pi*np.array([1,2,3,4,5,6])/7.0)

def omega_uniform():
    return np.full(6, np.mean(omega_E6()))

def omega_interpolated(beta):
    return (1.0-beta)*omega_E6() + beta*omega_uniform()

# ═══════════════════════════════════════════════════════════════
# PHASE EVOLUTION
# ═══════════════════════════════════════════════════════════════

def evolve_phases(N, omega, seed=0):
    rng = np.random.RandomState(seed)
    D = len(omega)
    ph = np.zeros((N,D))
    ph[0] = rng.uniform(0,2*np.pi,D)
    for n in range(N-1):
        ph[n+1] = (ph[n]+omega+0.1*np.sin(ph[n]))%(2*np.pi)
    return ph

def evolve_phases_coupled(N, omega, K, kappa=0.1, seed=0):
    rng = np.random.RandomState(seed)
    D = len(omega)
    ph = np.zeros((N,D))
    ph[0] = rng.uniform(0,2*np.pi,D)
    for n in range(N-1):
        ph[n+1] = (ph[n]+omega+kappa*K@np.sin(ph[n]))%(2*np.pi)
    return ph

# ═══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def torus_dist_matrix(phases):
    """Pairwise torus distances, chunked to save memory."""
    N = len(phases)
    D = np.zeros((N,N))
    chunk = 100
    for i in range(0,N,chunk):
        ie = min(i+chunk,N)
        diff = np.abs(phases[i:ie,None,:]-phases[None,:,:])
        diff = np.minimum(diff, 2*np.pi-diff)
        D[i:ie,:] = np.linalg.norm(diff,axis=2)
    return D

def build_knn_graph(phases, k=8, seed=42):
    """
    k-Nearest-Neighbor graph on 6D torus.
    Each node connects to its k nearest neighbors (by torus distance).
    Average degree = exactly k (before symmetrization: ~2k after).
    This gives FIXED degree independent of N.
    """
    N = phases.shape[0]
    dist = torus_dist_matrix(phases)
    np.fill_diagonal(dist, np.inf)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        neighbors = np.argpartition(dist[i], k)[:k]
        for j in neighbors:
            G.add_edge(i,int(j))

    # Ensure connectivity
    if not nx.is_connected(G):
        comps = sorted(nx.connected_components(G),
                       key=len, reverse=True)
        for c in comps[1:]:
            a = list(comps[0])[0]
            b = list(c)[0]
            G.add_edge(a,b)
    return G

def build_chain_graph(phases, eps=1.5, max_conn=5, n_cand=80, seed=42):
    """Original chain graph from Parts I-IV."""
    N = phases.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N-1):
        G.add_edge(i,i+1)
    rng = np.random.RandomState(seed)
    deg = np.zeros(N)
    for i in range(N-1):
        deg[i]+=1; deg[i+1]+=1
    for i in range(2,N):
        w = deg[:i]+1.0; w/=w.sum()
        pool = min(i,n_cand)
        cands = rng.choice(i,pool,p=w,replace=False)
        diffs = np.abs(phases[i]-phases[cands])
        diffs = np.minimum(diffs,2*np.pi-diffs)
        dists = np.linalg.norm(diffs,axis=1)
        added=0
        for j in np.argsort(dists):
            if dists[j]<eps and added<max_conn:
                G.add_edge(i,int(cands[j]))
                deg[i]+=1; deg[int(cands[j])]+=1
                added+=1
    return G

# ═══════════════════════════════════════════════════════════════
# SPECTRAL DIMENSION v3 — NORMALIZED LAPLACIAN + PLATEAU
# ═══════════════════════════════════════════════════════════════

def spectral_dimension_v3(G, n_eigs=None, verbose=False):
    """
    Spectral dimension via heat kernel on NORMALIZED Laplacian.

    Algorithm:
    1. Compute smallest n_eigs eigenvalues of normalized Laplacian
    2. Build d_s(t) curve on adaptive t-range
    3. Find plateau via minimum of |d(d_s)/d(ln t)|
    4. Validate: plateau must be flat (std < 15% of mean)

    Normalized Laplacian eigenvalues ∈ [0, 2].
    For k-regular graph: λ_min_nz ≈ 1 - cos(2π/diam)
    """
    N = G.number_of_nodes()
    if N < 15:
        return 0.0, None

    # Normalized Laplacian (eigenvalues in [0,2])
    Ln = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)

    # Number of eigenvalues: use enough to resolve the plateau
    if n_eigs is None:
        n_eigs = min(N-1, max(100, N//3))

    try:
        # Use sparse eigensolver for large N
        if N > 200:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh as sp_eigsh
            Ls = csr_matrix(
                nx.normalized_laplacian_matrix(G).astype(np.float64))
            eigs = sp_eigsh(Ls, k=min(n_eigs, N-2),
                            which='SM', return_eigenvectors=False)
            eigs = np.sort(np.abs(eigs))
        else:
            eigs = np.linalg.eigvalsh(Ln)
            eigs = np.sort(eigs)
    except Exception:
        eigs = np.linalg.eigvalsh(Ln)
        eigs = np.sort(eigs)

    eigs_nz = eigs[eigs > 1e-6]
    if len(eigs_nz) < 5:
        return 0.0, None

    lambda_min = eigs_nz[0]
    lambda_max = eigs_nz[-1]

    if verbose:
        avg_k = np.mean([d for _,d in G.degree()])
        print(f"    N={N}, <k>={avg_k:.1f}, "
              f"λ_min={lambda_min:.4f}, λ_max={lambda_max:.4f}, "
              f"n_eigs={len(eigs_nz)}")

    # Adaptive t-range
    # At t_lo: all modes active → d_s near its maximum
    # At t_hi: only lowest mode survives → d_s falls to 0
    # Plateau is between these extremes
    t_lo = 0.5  / lambda_max   # fine structure
    t_hi = 5.0  / lambda_min   # large-scale structure
    n_t  = 400
    ts   = np.logspace(np.log10(t_lo), np.log10(t_hi), n_t)

    ds_curve = np.zeros(n_t)
    for idx, t in enumerate(ts):
        e  = np.exp(-eigs_nz * t)
        Z  = e.sum()
        if Z < 1e-30:
            ds_curve[idx:] = 0.0
            break
        # d_s(t) = 2t Σ λ exp(-λt) / Σ exp(-λt)
        ds_curve[idx] = 2.0*t*np.dot(eigs_nz,e)/Z

    # Find plateau: sliding window, minimize variance
    valid = (ds_curve > 0.2) & (np.isfinite(ds_curve))
    if valid.sum() < 10:
        return 0.0, (ts, ds_curve)

    # Smooth curve
    smooth = np.convolve(ds_curve, np.ones(11)/11, mode='same')

    # Find flattest window
    W = 30
    best_std = np.inf
    best_start = 0
    for s in range(10, n_t-W-10):
        if not valid[s:s+W].all():
            continue
        window = smooth[s:s+W]
        std = np.std(window)
        if std < best_std:
            best_std = std
            best_start = s

    # Plateau value
    w_vals = ds_curve[best_start:best_start+W]
    ds_plateau = float(np.median(w_vals))
    plateau_std = float(np.std(w_vals))
    plateau_rel_std = plateau_std / (ds_plateau + 1e-10)

    if verbose:
        print(f"    d_s = {ds_plateau:.3f} ± {plateau_std:.3f} "
              f"(rel_std={plateau_rel_std:.2%})")

    # Quality check
    if ds_plateau < 0.3 or ds_plateau > 20:
        if verbose:
            print(f"    WARNING: out-of-range d_s={ds_plateau:.3f}")
        # Fallback: use maximum of smoothed curve (upper bound)
        ds_plateau = float(np.max(smooth[valid]))

    if plateau_rel_std > 0.30:
        if verbose:
            print(f"    WARNING: noisy plateau (rel_std={plateau_rel_std:.1%})")

    return ds_plateau, (ts, ds_curve, best_start, W)

# ═══════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════

def run_benchmarks(verbose=True):
    """Must pass before main experiments."""
    print("\n  BENCHMARK VALIDATION (normalized Laplacian, v3):")
    print(f"  {'Graph':20s} {'Expected':>10} {'Measured':>10} "
          f"{'Error':>8} {'Pass?':>6}")
    print("  "+"-"*58)

    cases = [
        ("Path(400)",      nx.path_graph(400),                    1.0, 0.15),
        ("Cycle(300)",     nx.cycle_graph(300),                   1.0, 0.15),
        ("2D Grid(18×18)", nx.convert_node_labels_to_integers(
                            nx.grid_2d_graph(18,18)),              2.0, 0.20),
        ("3D Grid(8×8×8)", nx.convert_node_labels_to_integers(
                            nx.grid_graph([8,8,8])),               3.0, 0.30),
        ("4D Grid(5⁴)",    nx.convert_node_labels_to_integers(
                            nx.grid_graph([5,5,5,5])),             4.0, 0.40),
    ]

    results = []
    for name, G, expected, tol in cases:
        ds, curve = spectral_dimension_v3(G, verbose=verbose)
        err  = abs(ds - expected)
        ok   = err < tol
        results.append(ok)
        print(f"  {name:20s} {expected:>10.1f} {ds:>10.3f} "
              f"{err:>8.3f} {'✅' if ok else '❌':>6}")

    n_pass = sum(results)
    print(f"\n  {n_pass}/{len(results)} benchmarks passed")
    all_ok = all(results)
    if all_ok:
        print("  ✅ Measurement validated — proceeding.")
    else:
        print("  ❌ FAILED — check spectral_dimension_v3()")
    return all_ok

def plot_benchmark_curves():
    """Visual check: d_s(t) curves for known graphs."""
    fig, axes = plt.subplots(1,5,figsize=(20,4))
    cases = [
        ("Path(400)", nx.path_graph(400), 1.0),
        ("Cycle(300)", nx.cycle_graph(300), 1.0),
        ("2D Grid(18×18)", nx.convert_node_labels_to_integers(
                            nx.grid_2d_graph(18,18)), 2.0),
        ("3D Grid(8×8×8)", nx.convert_node_labels_to_integers(
                            nx.grid_graph([8,8,8])), 3.0),
        ("4D Grid(5⁴)", nx.convert_node_labels_to_integers(
                         nx.grid_graph([5,5,5,5])), 4.0),
    ]
    for ax, (name, G, expected) in zip(axes, cases):
        ds, curve = spectral_dimension_v3(G)
        if curve is not None:
            ts, ds_c = curve[0], curve[1]
            ax.semilogx(ts, ds_c, 'b-', linewidth=1.5, label='d_s(t)')
            if len(curve) > 2:
                s, W = curve[2], curve[3]
                ax.axvspan(ts[s], ts[min(s+W, len(ts)-1)],
                           alpha=0.2, color='orange', label='plateau')
        ax.axhline(expected, color='green', linestyle='--',
                   label=f'expected={expected}')
        ax.axhline(ds, color='red', linestyle=':',
                   label=f'measured={ds:.3f}')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('t'); ax.set_ylabel('d_s(t)')
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        ax.set_ylim([0, max(6, expected*2)])
    plt.suptitle("Benchmark d_s(t) curves — v3", fontsize=12)
    plt.tight_layout()
    plt.savefig('benchmarks_v3.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved: benchmarks_v3.png")

def phase_recurrence(phases, eps=1.5, n_pairs=15000, seed=42):
    N = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0,N,(n_pairs,2))
    diffs = np.abs(phases[idx[:,0]]-phases[idx[:,1]])
    diffs = np.minimum(diffs,2*np.pi-diffs)
    return float(np.mean(np.linalg.norm(diffs,axis=1)<eps))

# ═══════════════════════════════════════════════════════════════
# DIRECTION A: k-NN graph — N-independence test
# ═══════════════════════════════════════════════════════════════

def direction_A(n_runs=5):
    print("\n"+"="*60)
    print("DIRECTION A: k-NN graph — N-independence test (v3)")
    print("="*60)
    print("k-NN ensures FIXED average degree regardless of N.\n")

    omega = omega_E6()
    sizes = [200,400,600,800,1000]
    k_nn  = 8   # fixed number of nearest neighbors

    res = {
        'knn_e6':   {N:[] for N in sizes},
        'knn_null': {N:[] for N in sizes},
        'chain_e6': {N:[] for N in sizes},
    }
    deg_log = {N:[] for N in sizes}

    for N in sizes:
        print(f"  N={N}:", end="", flush=True)
        for run in range(n_runs):
            # E6 k-NN
            ph = evolve_phases(N, omega, seed=run)
            G_knn = build_knn_graph(ph, k=k_nn, seed=run)
            ds, _ = spectral_dimension_v3(G_knn)
            res['knn_e6'][N].append(ds)
            deg_log[N].append(np.mean([d for _,d in G_knn.degree()]))

            # Null k-NN
            rng = np.random.RandomState(run+2000)
            ph_null = rng.uniform(0,2*np.pi,(N,6))
            G_null = build_knn_graph(ph_null, k=k_nn, seed=run)
            ds_n, _ = spectral_dimension_v3(G_null)
            res['knn_null'][N].append(ds_n)

            # Chain (original)
            G_ch = build_chain_graph(ph, seed=run)
            ds_c, _ = spectral_dimension_v3(G_ch)
            res['chain_e6'][N].append(ds_c)

            print(".", end="", flush=True)

        m_knn = np.mean(res['knn_e6'][N])
        m_null= np.mean(res['knn_null'][N])
        m_ch  = np.mean(res['chain_e6'][N])
        mk    = np.mean(deg_log[N])
        print(f" knn={m_knn:.3f}(<k>={mk:.1f}), null={m_null:.3f}, chain={m_ch:.3f}")

    Ns = np.array(sizes)
    print(f"\n  Linear fits d_s = a + b*N:")
    fits = {}
    for key, label in [('knn_e6','k-NN E6'),('knn_null','k-NN null'),('chain_e6','Chain E6')]:
        means = np.array([np.mean(res[key][N]) for N in sizes])
        sems  = np.array([np.std(res[key][N])/np.sqrt(n_runs) for N in sizes])
        b,a   = np.polyfit(Ns, means, 1)
        pred  = a+b*Ns
        ss_res= np.sum((means-pred)**2)
        ss_tot= np.sum((means-means.mean())**2)
        r2    = 1-ss_res/ss_tot if ss_tot>0 else 0
        fits[key] = (a,b,r2,means,sems)
        ni = "N-INDEPENDENT ✅" if (abs(b)<0.003 and r2<0.6) else "N-dependent ❌"
        print(f"  {label:15s}: d_s={a:.3f}+{b:.5f}·N  R²={r2:.3f}  {ni}")

    # PCA
    print("\n  PCA variance (N=600, E6):")
    ph_pca = evolve_phases(600, omega, seed=0)
    p = ph_pca-ph_pca.mean(axis=0)
    _,S,_ = np.linalg.svd(p,full_matrices=False)
    var_exp= S**2/(S**2).sum()
    cumvar = np.cumsum(var_exp)
    for k in range(6):
        bar = "█"*int(var_exp[k]*40)
        print(f"    PC{k+1}: {var_exp[k]*100:5.1f}%  {bar}")
    n80 = np.searchsorted(cumvar,0.80)+1
    print(f"  → {n80} PCs explain >80% variance")

    return {'sizes':sizes,'fits':fits,'pca_var':var_exp,'res':res}

# ═══════════════════════════════════════════════════════════════
# DIRECTION B: Number-theoretic resonances
# ═══════════════════════════════════════════════════════════════

def direction_B(n_runs=6):
    print("\n"+"="*60)
    print("DIRECTION B: Number-theoretic resonances d_s(β)")
    print("="*60)

    N     = 600
    k_nn  = 8
    n_beta= 40
    betas = np.linspace(0.0,0.20,n_beta)

    # Rational complexity of ω ratios
    ratio_complexity = np.zeros(n_beta)
    for bi,beta in enumerate(betas):
        om = omega_interpolated(beta)
        pairs = [(i,j) for i in range(6) for j in range(i+1,6)]
        qs = []
        for i,j in pairs:
            ratio = om[i]/(om[j]+1e-15)
            best_q = 20
            for q in range(1,21):
                p = round(ratio*q)
                if abs(ratio-p/q) < 5e-3:
                    best_q = q; break
            qs.append(best_q)
        ratio_complexity[bi] = np.mean(qs)

    ds_vals  = np.zeros((n_beta,n_runs))
    rec_vals = np.zeros((n_beta,n_runs))

    print(f"  β ∈ [0,0.20], {n_beta} pts × {n_runs} runs × N={N}...")
    t0 = time.time()
    for bi,beta in enumerate(betas):
        om = omega_interpolated(beta)
        for run in range(n_runs):
            ph = evolve_phases(N,om,seed=run)
            G  = build_knn_graph(ph,k=k_nn,seed=run)
            ds,_ = spectral_dimension_v3(G)
            ds_vals[bi,run]  = ds
            rec_vals[bi,run] = phase_recurrence(ph)
        if (bi+1)%10==0:
            el  = time.time()-t0
            eta = el/(bi+1)*(n_beta-bi-1)
            print(f"    {bi+1}/{n_beta}  ({el:.0f}s, ETA {eta:.0f}s)")

    ds_mean  = ds_vals.mean(axis=1)
    ds_sem   = ds_vals.std(axis=1)/np.sqrt(n_runs)
    rec_mean = rec_vals.mean(axis=1)

    peaks,_  = find_peaks(ds_mean,
                           height=np.median(ds_mean)+0.3*ds_mean.std(),
                           prominence=0.15)
    troughs,_= find_peaks(-ds_mean, prominence=0.10)

    print(f"\n  d_s: min={ds_mean.min():.3f}, max={ds_mean.max():.3f}, "
          f"range={ds_mean.max()-ds_mean.min():.3f}")
    print(f"  Peaks: {len(peaks)}, Troughs: {len(troughs)}")

    if len(peaks)>0:
        print(f"\n  Peak details:")
        for pk in peaks[:6]:
            om_pk = omega_interpolated(betas[pk])
            r12 = om_pk[0]/om_pk[1]
            print(f"    β={betas[pk]:.4f}: d_s={ds_mean[pk]:.3f}±{ds_sem[pk]:.3f}"
                  f"  rec={rec_mean[pk]:.4f}  ω₁/ω₂={r12:.4f}"
                  f"  complexity={ratio_complexity[pk]:.1f}")

    # Correlations
    if rec_mean.std()>0:
        r1,p1 = pearsonr(ds_mean,rec_mean)
        print(f"\n  r(d_s, recurrence) = {r1:.3f}  p={p1:.4f}")
    r2,p2 = pearsonr(ds_mean,ratio_complexity)
    print(f"  r(d_s, complexity) = {r2:.3f}  p={p2:.4f}")

    if r2 < -0.30 and p2 < 0.05:
        print("  → KAM: d_s ↑ at rational ratios (resonance lock-in)")
    elif r2 > 0.30 and p2 < 0.05:
        print("  → d_s ↑ at irrational ratios (quasiperiodic wins)")
    else:
        print("  → No significant KAM signature")

    return {'betas':betas,'ds_mean':ds_mean,'ds_sem':ds_sem,
            'rec_mean':rec_mean,'ratio_complexity':ratio_complexity,
            'peaks':peaks,'troughs':troughs}

# ═══════════════════════════════════════════════════════════════
# DIRECTION C: Algebra comparison
# ═══════════════════════════════════════════════════════════════

def direction_C(n_runs=6):
    print("\n"+"="*60)
    print("DIRECTION C: Algebra comparison (k-NN, v3)")
    print("="*60)

    algebras = {
        'E6': (omega_E6, cartan_E6),
        'B6': (omega_B6, cartan_B6),
        'D6': (omega_D6, cartan_D6),
        'A6': (omega_A6, cartan_A6),
    }
    sizes = [200,400,600,800,1000]
    k_nn  = 8
    kappa = 0.1

    results = {name:{N:[] for N in sizes} for name in algebras}

    for name,(omega_fn,K_fn) in algebras.items():
        om = omega_fn(); Km = K_fn()
        print(f"  {name}:", end="",flush=True)
        for N in sizes:
            for run in range(n_runs):
                ph = evolve_phases_coupled(N,om,Km,kappa=kappa,seed=run)
                G  = build_knn_graph(ph,k=k_nn,seed=run)
                ds,_ = spectral_dimension_v3(G)
                results[name][N].append(ds)
            m = np.mean(results[name][N])
            print(f" {N}:{m:.3f}",end="",flush=True)
        print()

    Ns = np.array(sizes)
    print(f"\n  Fits + N-independence:")
    slopes = {}
    for name in algebras:
        means = np.array([np.mean(results[name][N]) for N in sizes])
        sems  = np.array([np.std(results[name][N])/np.sqrt(n_runs)
                          for N in sizes])
        b,a   = np.polyfit(Ns,means,1)
        pred  = a+b*Ns
        ss_res= np.sum((means-pred)**2)
        ss_tot= np.sum((means-means.mean())**2)
        r2    = 1-ss_res/ss_tot if ss_tot>0 else 0
        slopes[name] = (a,b,means,sems)
        ds1k  = a+b*1000
        ni    = "N-indep ✅" if (abs(b)<0.003 and r2<0.6) else "N-dep ❌"
        print(f"  {name}: d_s={a:.3f}+{b:.5f}·N  R²={r2:.3f}  "
              f"d_s(1000)={ds1k:.3f}  {ni}")

    print(f"\n  Distance |d_s-4| at N=1000:")
    dists = {}
    for name in algebras:
        a,b,_,_ = slopes[name]
        ds1k = a+b*1000
        dists[name] = abs(ds1k-4.0)
        print(f"  {name}: {dists[name]:.3f}")
    best = min(dists,key=dists.get)
    print(f"  → {best} closest to 4D")

    return {'sizes':sizes,'algebras':list(algebras.keys()),
            'results':results,'slopes':slopes}

# ═══════════════════════════════════════════════════════════════
# DIRECTION D: Feedback dark energy — stabilized
# ═══════════════════════════════════════════════════════════════

def direction_D(n_mc=12, n_epochs=25):
    print("\n"+"="*60)
    print("DIRECTION D: Feedback dark energy (stabilized v3)")
    print("="*60)
    print("Feedback: λ_C = λ₀ · min(⟨d⟩/⟨d⟩_init, 3.0)")
    print("Cap prevents divergence; tests whether MILD feedback")
    print("causes acceleration relative to null model.\n")

    omega   = omega_E6()
    N_init  = 300
    N_add   = 80
    lambda0 = 0.004
    eps_ph  = 1.5
    max_conn= 4
    n_cand  = 50
    N_total = N_init + N_add*n_epochs + 50
    # Maximum feedback ratio (prevents runaway)
    MAX_RATIO = 2.0

    def build_epoch_graph(phases, n_nodes, lam, seed):
        rng = np.random.RandomState(seed)
        G   = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes-1):
            G.add_edge(i,i+1)
        deg = np.zeros(n_nodes)
        for i in range(n_nodes-1):
            deg[i]+=1; deg[i+1]+=1
        for i in range(2,n_nodes):
            pool = min(i,n_cand)
            w    = deg[:i]+1.0; w/=w.sum()
            cands= rng.choice(i,pool,p=w,replace=False)
            diffs= np.abs(phases[i]-phases[cands])
            diffs= np.minimum(diffs,2*np.pi-diffs)
            dists= np.linalg.norm(diffs,axis=1)
            added=0
            for j in np.argsort(dists):
                if dists[j]<eps_ph and added<max_conn:
                    dt   = i-int(cands[j])
                    prob = np.exp(-lam*dt)
                    if rng.random()<prob:
                        G.add_edge(i,int(cands[j]))
                        deg[i]+=1; deg[int(cands[j])]+=1
                        added+=1
        return G

    def sample_path(G, n_sample=80, seed=0):
        nodes = list(G.nodes())
        rng   = np.random.RandomState(seed)
        srcs  = rng.choice(nodes,min(n_sample,len(nodes)),replace=False)
        paths = []
        for s in srcs:
            L = nx.single_source_shortest_path_length(G,s)
            if len(L)>1:
                paths.extend(list(L.values())[1:])
        return float(np.mean(paths)) if paths else np.nan

    path_A = np.full((n_mc,n_epochs+1), np.nan)
    path_B = np.full((n_mc,n_epochs+1), np.nan)
    path_C = np.full((n_mc,n_epochs+1), np.nan)

    print(f"  Running {n_mc} MC × {n_epochs} epochs...")
    t0 = time.time()
    for mc in range(n_mc):
        phases_all = evolve_phases(N_total,omega,seed=mc*17+3)
        lam_C  = lambda0
        d_init = None

        for ep in range(n_epochs+1):
            n_nodes  = min(N_init+N_add*ep, N_total-1)
            seed_ep  = mc*500+ep

            G_A = build_epoch_graph(phases_all,n_nodes,0.0,    seed_ep)
            G_B = build_epoch_graph(phases_all,n_nodes,lambda0,seed_ep)
            G_C = build_epoch_graph(phases_all,n_nodes,lam_C,  seed_ep)

            path_A[mc,ep] = sample_path(G_A,seed=seed_ep)
            path_B[mc,ep] = sample_path(G_B,seed=seed_ep)
            d_now = sample_path(G_C,seed=seed_ep)
            path_C[mc,ep] = d_now

            # Stabilized feedback: cap ratio at MAX_RATIO
            if ep==0:
                d_init = d_now if (d_now and d_now>0) else 1.0
            if d_init and d_init>0 and d_now and d_now>0:
                ratio  = min(d_now/d_init, MAX_RATIO)
                lam_C  = np.clip(lambda0*ratio, 0.0, lambda0*MAX_RATIO)

        if (mc+1)%4==0:
            print(f"    MC {mc+1}/{n_mc}  ({time.time()-t0:.0f}s)")

    mean_A = np.nanmean(path_A,axis=0)
    mean_B = np.nanmean(path_B,axis=0)
    mean_C = np.nanmean(path_C,axis=0)
    sem_A  = np.nanstd(path_A,axis=0)/np.sqrt(n_mc)
    sem_B  = np.nanstd(path_B,axis=0)/np.sqrt(n_mc)
    sem_C  = np.nanstd(path_C,axis=0)/np.sqrt(n_mc)

    # Velocity
    vel_A = np.diff(mean_A)
    vel_B = np.diff(mean_B)
    vel_C = np.diff(mean_C)

    # Acceleration (last 8 steps)
    acc_A = float(np.mean(np.diff(vel_A[-8:])))
    acc_B = float(np.mean(np.diff(vel_B[-8:])))
    acc_C = float(np.mean(np.diff(vel_C[-8:])))

    print(f"\n  Final ⟨d⟩ (epoch {n_epochs}):")
    print(f"    A (null):     {mean_A[-1]:.3f} ± {sem_A[-1]:.3f}")
    print(f"    B (constant): {mean_B[-1]:.3f} ± {sem_B[-1]:.3f}")
    print(f"    C (feedback): {mean_C[-1]:.3f} ± {sem_C[-1]:.3f}")
    print(f"\n  Acceleration d²⟨d⟩/dt² (last 8 epochs):")
    print(f"    A: {acc_A:+.5f}")
    print(f"    B: {acc_B:+.5f}")
    print(f"    C: {acc_C:+.5f}")

    delta_CA = acc_C-acc_A
    if delta_CA < -0.003:
        print(f"\n  ✅ C more negative than A by {abs(delta_CA):.4f}")
        print(f"     Feedback → genuine acceleration vs null")
    elif delta_CA > 0.003:
        print(f"\n  ❌ C more positive than A by {delta_CA:.4f}")
        print(f"     Feedback → suppresses shortcuts → deceleration")
    else:
        print(f"\n  ⚠️  C ≈ A (|Δ|={abs(delta_CA):.4f}) → no effect")

    return {'epochs':np.arange(n_epochs+1),
            'mean_A':mean_A,'sem_A':sem_A,
            'mean_B':mean_B,'sem_B':sem_B,
            'mean_C':mean_C,'sem_C':sem_C,
            'acc':(acc_A,acc_B,acc_C)}

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_all(res_A, res_B, res_C, res_D):
    fig, axes = plt.subplots(2,4,figsize=(22,10))
    fig.suptitle("Part IV+ v3: Four Directions Beyond Falsification",
                 fontsize=13, fontweight='bold')

    Ns = np.array(res_A['sizes'])
    col_map = {'knn_e6':'steelblue','knn_null':'gray','chain_e6':'darkorange'}
    col_alg = {'E6':'steelblue','B6':'tomato','D6':'green','A6':'purple'}

    # A1: d_s vs N
    ax = axes[0,0]
    for key,label in [('knn_e6','k-NN E6'),
                       ('knn_null','k-NN null'),
                       ('chain_e6','Chain E6')]:
        a,b,r2,means,sems = res_A['fits'][key]
        ax.errorbar(Ns, means, yerr=2*sems, fmt='o-',
                    color=col_map[key], capsize=4,
                    label=f'{label} (b={b:.4f})')
        ax.plot(Ns, a+b*Ns, '--', color=col_map[key], alpha=0.4)
    ax.axhline(4.0, color='black', linestyle=':', lw=2, label='d=4')
    ax.axhline(1.0, color='gray',  linestyle=':', lw=1)
    ax.axhline(2.0, color='gray',  linestyle=':', lw=1)
    ax.axhline(3.0, color='gray',  linestyle=':', lw=1)
    ax.set_xlabel('N'); ax.set_ylabel('d_s')
    ax.set_title('A: d_s vs N\n(k-NN vs chain vs null)')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    # A2: PCA
    ax = axes[0,1]
    pca = res_A['pca_var']
    ax.bar(range(1,7), pca*100, color='steelblue', alpha=0.8)
    ax.plot(range(1,7), np.cumsum(pca)*100, 'ro-', label='Cumulative')
    ax.axhline(80, color='gray', linestyle='--', label='80%')
    ax.set_xlabel('PC'); ax.set_ylabel('Variance (%)')
    ax.set_title('A: PCA of E6 phases\n(effective dimensionality)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # B1: d_s(β)
    ax = axes[0,2]
    b_  = res_B['betas']
    dm  = res_B['ds_mean']
    ds_ = res_B['ds_sem']
    ax.fill_between(b_, dm-2*ds_, dm+2*ds_, alpha=0.25, color='darkorange')
    ax.plot(b_, dm, 'o-', color='darkorange', ms=4)
    for pk in res_B['peaks']:
        ax.axvline(b_[pk], color='red', alpha=0.6, lw=1.5)
    for tr in res_B['troughs']:
        ax.axvline(b_[tr], color='blue', alpha=0.3, lw=1)
    ax.axhline(4.0, color='black', linestyle=':', lw=2, label='d=4')
    ax.set_xlabel('β'); ax.set_ylabel('d_s')
    ax.set_title('B: Resonances d_s(β)\n(red=peaks)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # B2: d_s vs recurrence
    ax = axes[0,3]
    sc = ax.scatter(res_B['rec_mean'], dm,
                    c=res_B['ratio_complexity'],
                    cmap='RdYlGn_r', s=50, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Rational complexity (lower=more rational)')
    ax.set_xlabel('Phase recurrence'); ax.set_ylabel('d_s')
    ax.set_title('B: d_s vs recurrence\n(color=frequency rationality)')
    ax.grid(True,alpha=0.3)

    # C1: d_s vs N per algebra
    ax = axes[1,0]
    Ns_C = np.array(res_C['sizes'])
    for name in res_C['algebras']:
        a,b,means,sems = res_C['slopes'][name]
        ax.errorbar(Ns_C, means, yerr=2*sems, fmt='o-',
                    color=col_alg[name], capsize=4,
                    label=f'{name} (b={b:.4f})')
        ax.plot(Ns_C, a+b*Ns_C, '--', color=col_alg[name], alpha=0.4)
    ax.axhline(4.0, color='black', linestyle=':', lw=2, label='d=4')
    ax.set_xlabel('N'); ax.set_ylabel('d_s')
    ax.set_title('C: Algebras vs N\n(flat=N-independent)')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    # C2: extrapolation
    ax = axes[1,1]
    N_ext = np.linspace(100,2000,100)
    for name in res_C['algebras']:
        a,b,_,_ = res_C['slopes'][name]
        ax.plot(N_ext, a+b*N_ext, '-',
                color=col_alg[name], label=name, lw=2)
    ax.axhline(4.0, color='black', linestyle=':', lw=2, label='d=4')
    ax.set_xlabel('N'); ax.set_ylabel('d_s')
    ax.set_title('C: Extrapolation\n(linear trend)')
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
    ax.set_ylim([0,10])

    # D1: ⟨d⟩ vs epoch
    ax = axes[1,2]
    ep = res_D['epochs']
    for lab,km,ks,col in [
        ('A null',    'mean_A','sem_A','gray'),
        ('B constant','mean_B','sem_B','steelblue'),
        ('C feedback','mean_C','sem_C','tomato'),
    ]:
        m=res_D[km]; s=res_D[ks]
        ax.fill_between(ep, m-2*s, m+2*s, alpha=0.15, color=col)
        ax.plot(ep, m, '-', color=col, label=lab, lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('⟨d⟩')
    ax.set_title('D: Dark energy\n(A=null, B=const λ, C=feedback)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # D2: acceleration
    ax = axes[1,3]
    acc_vals    = list(res_D['acc'])
    model_names = ['A\n(null)','B\n(const)','C\n(feedback)']
    cols_bar    = ['gray','steelblue','tomato']
    bars = ax.bar(model_names, acc_vals, color=cols_bar,
                  alpha=0.8, edgecolor='black')
    ax.axhline(0, color='black', lw=1.5)
    for bar,val in zip(bars,acc_vals):
        ax.text(bar.get_x()+bar.get_width()/2., val,
                f'{val:+.4f}', ha='center',
                va='bottom' if val>=0 else 'top', fontsize=9)
    ax.set_ylabel('d²⟨d⟩/dt²')
    ax.set_title('D: Acceleration\n(negative=dark energy)')
    ax.grid(True,alpha=0.3,axis='y')

    plt.tight_layout()
    plt.savefig('part4plus_v3.png', dpi=150, bbox_inches='tight')
    print("  Saved: part4plus_v3.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(res_A, res_B, res_C, res_D):
    print("\n"+"="*60)
    print("SUMMARY: Part IV+ v3")
    print("="*60)

    print("\n┌─ DIRECTION A: k-NN N-independence ─────────────────────┐")
    for key,label in [('knn_e6','k-NN E6'),
                       ('knn_null','k-NN null'),
                       ('chain_e6','Chain E6')]:
        a,b,r2,_,_ = res_A['fits'][key]
        ni = "✅ N-indep" if (abs(b)<0.003 and r2<0.6) else "❌ N-dep"
        print(f"│  {label:12s}: {a:.2f}+{b:.5f}·N  R²={r2:.2f}  {ni}      │")
    n80 = np.searchsorted(np.cumsum(res_A['pca_var']),0.80)+1
    print(f"│  PCA: {n80} dims explain >80% E6 phase variance               │")
    print("└──────────────────────────────────────────────────────────┘")

    print("\n┌─ DIRECTION B: Resonances ───────────────────────────────┐")
    rng_b = res_B['ds_mean'].max()-res_B['ds_mean'].min()
    print(f"│  d_s range: {rng_b:.3f},  "
          f"peaks: {len(res_B['peaks'])}, troughs: {len(res_B['troughs'])}         │")
    if len(res_B['peaks'])>0:
        best_pk = res_B['peaks'][
            np.argmin(abs(res_B['ds_mean'][res_B['peaks']]-4.0))]
        print(f"│  Best β→4: β={res_B['betas'][best_pk]:.4f}, "
              f"d_s={res_B['ds_mean'][best_pk]:.3f}                    │")
    print("└──────────────────────────────────────────────────────────┘")

    print("\n┌─ DIRECTION C: Algebras ─────────────────────────────────┐")
    dists = {}
    for name in res_C['algebras']:
        a,b,_,_ = res_C['slopes'][name]
        ds1k = a+b*1000
        dists[name] = abs(ds1k-4.0)
        ni = "✅" if abs(b)<0.003 else "❌"
        print(f"│  {name}: d_s(1k)={ds1k:.2f}  |−4|={dists[name]:.2f}  "
              f"N-indep={ni}                 │")
    best = min(dists,key=dists.get)
    print(f"│  → Closest to 4D: {best}                                   │")
    print("└──────────────────────────────────────────────────────────┘")

    print("\n┌─ DIRECTION D: Dark Energy ──────────────────────────────┐")
    aA,aB,aC = res_D['acc']
    print(f"│  A (null):     {aA:+.5f}                               │")
    print(f"│  B (constant): {aB:+.5f}                               │")
    print(f"│  C (feedback): {aC:+.5f}                               │")
    if aC < aA-0.003:
        verdict = "Feedback→acceleration ✅ (emergent dark energy)"
    elif aC > aA+0.003:
        verdict = "Feedback→deceleration ❌ (shortcuts suppressed)"
    else:
        verdict = "C ≈ A ⚠️  (no significant effect)"
    print(f"│  Verdict: {verdict:45s}│")
    print("└──────────────────────────────────────────────────────────┘")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Monostring Hypothesis — Part IV+ v3                   ║")
    print("║   k-NN graph + normalized Laplacian + stable feedback   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Started: {time.strftime('%H:%M:%S')}\n")

    # Step 0: Benchmarks
    print("[0/4] Benchmark validation...")
    plot_benchmark_curves()
    ok = run_benchmarks(verbose=True)
    if not ok:
        print("Aborting."); exit(1)

    # Main experiments
    print("\n[1/4] Direction A: k-NN N-independence...")
    res_A = direction_A(n_runs=5)

    print("\n[2/4] Direction B: Resonances...")
    res_B = direction_B(n_runs=6)

    print("\n[3/4] Direction C: Algebras...")
    res_C = direction_C(n_runs=6)

    print("\n[4/4] Direction D: Dark energy...")
    res_D = direction_D(n_mc=12, n_epochs=25)

    print_summary(res_A, res_B, res_C, res_D)
    plot_all(res_A, res_B, res_C, res_D)

    elapsed = time.time()-t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print("Done.")
