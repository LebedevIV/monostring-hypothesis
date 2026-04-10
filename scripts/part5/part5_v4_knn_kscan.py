"""
Part IV+ v4: Targeted fixes
1. k-NN scan: find k* such that d_s(E6, k*) = 4.0
2. Chain graph diagnostic: why does it fail at N>600?
3. Direction D: correct interpretation of acceleration
4. Benchmark correction factor for 4D underestimate
5. Statistical test: is E6 d_s significantly > null?
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from scipy.signal import find_peaks
import time
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# ALGEBRA DEFINITIONS (same as v3)
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

def build_knn_graph(phases, k=8, seed=42):
    """k-NN graph on 6D torus. Degree ≈ 2k after symmetrization."""
    N = phases.shape[0]
    # Chunked torus distance
    chunk = 150
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        dists_i = np.zeros(N)
        for c in range(0,N,chunk):
            ce = min(c+chunk,N)
            diff = np.abs(phases[i]-phases[c:ce])
            diff = np.minimum(diff,2*np.pi-diff)
            dists_i[c:ce] = np.linalg.norm(diff,axis=1)
        dists_i[i] = np.inf
        neighbors = np.argpartition(dists_i,k)[:k]
        for j in neighbors:
            G.add_edge(i,int(j))
    # Ensure connectivity
    if not nx.is_connected(G):
        comps = sorted(nx.connected_components(G),key=len,reverse=True)
        for c in comps[1:]:
            G.add_edge(list(comps[0])[0], list(c)[0])
    return G

def build_chain_graph(phases, eps=1.5, max_conn=5, n_cand=80, seed=42):
    """Original chain graph. Returns graph + diagnostics."""
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
# SPECTRAL DIMENSION v3 (validated in previous run)
# ═══════════════════════════════════════════════════════════════

def spectral_dimension_v3(G, n_eigs=None, verbose=False):
    N = G.number_of_nodes()
    if N < 15:
        return 0.0, None

    Ln = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)

    if n_eigs is None:
        n_eigs = min(N-1, max(100, N//3))

    try:
        if N > 200:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh as sp_eigsh
            Ls = csr_matrix(
                nx.normalized_laplacian_matrix(G).astype(np.float64))
            eigs = sp_eigsh(Ls, k=min(n_eigs,N-2),
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
              f"λ_min={lambda_min:.5f}, λ_max={lambda_max:.4f}, "
              f"n_eigs={len(eigs_nz)}, connected={nx.is_connected(G)}")

    t_lo = 0.5/lambda_max
    t_hi = 5.0/lambda_min
    n_t  = 400
    ts   = np.logspace(np.log10(t_lo),np.log10(t_hi),n_t)

    ds_curve = np.zeros(n_t)
    for idx,t in enumerate(ts):
        e = np.exp(-eigs_nz*t)
        Z = e.sum()
        if Z < 1e-30:
            break
        ds_curve[idx] = 2.0*t*np.dot(eigs_nz,e)/Z

    valid = (ds_curve > 0.2) & (np.isfinite(ds_curve))
    if valid.sum() < 10:
        return 0.0, (ts,ds_curve)

    smooth = np.convolve(ds_curve,np.ones(11)/11,mode='same')
    W = 30
    best_std = np.inf
    best_start = 0
    for s in range(10,n_t-W-10):
        if not valid[s:s+W].all():
            continue
        std = np.std(smooth[s:s+W])
        if std < best_std:
            best_std = std
            best_start = s

    w_vals = ds_curve[best_start:best_start+W]
    ds_plateau = float(np.median(w_vals))
    plateau_std = float(np.std(w_vals))

    if verbose:
        print(f"    d_s = {ds_plateau:.3f} ± {plateau_std:.3f}")

    if ds_plateau < 0.3 or ds_plateau > 20:
        ds_plateau = float(np.max(smooth[valid]))

    return ds_plateau, (ts,ds_curve,best_start,W)

# Calibration factor from benchmarks:
# 4D grid measured 3.616 vs expected 4.0 → factor = 4.0/3.616 = 1.106
CALIB_FACTOR_4D = 4.0 / 3.616

def spectral_dimension_calibrated(G, n_eigs=None, verbose=False):
    """
    Apply calibration factor derived from 4D grid benchmark.
    Raw d_s is multiplied by CALIB_FACTOR_4D only for
    reporting purposes — raw value also returned.
    """
    ds_raw, curve = spectral_dimension_v3(G, n_eigs=n_eigs, verbose=verbose)
    ds_cal = ds_raw * CALIB_FACTOR_4D
    return ds_raw, ds_cal, curve

def phase_recurrence(phases, eps=1.5, n_pairs=15000, seed=42):
    N = len(phases)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0,N,(n_pairs,2))
    diffs = np.abs(phases[idx[:,0]]-phases[idx[:,1]])
    diffs = np.minimum(diffs,2*np.pi-diffs)
    return float(np.mean(np.linalg.norm(diffs,axis=1)<eps))

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: k-SCAN — find k* where d_s(E6) = 4.0
# ═══════════════════════════════════════════════════════════════

def experiment_k_scan(n_runs=6):
    """
    Scan k (number of nearest neighbors) for E6 and null model.
    Find k* where d_s(E6, k*) ≈ 4.0.
    Key question: is k* the same for null, or E6-specific?
    """
    print("\n"+"="*60)
    print("EXPERIMENT 1: k-scan — find k* for d_s(E6)=4")
    print("="*60)

    omega = omega_E6()
    N = 600
    k_values = [4, 6, 8, 10, 12, 16, 20, 25]

    ds_e6   = {k:[] for k in k_values}
    ds_null = {k:[] for k in k_values}

    print(f"  N={N}, {n_runs} runs per k")
    for k in k_values:
        print(f"  k={k:2d}:", end="", flush=True)
        for run in range(n_runs):
            # E6
            ph = evolve_phases(N, omega, seed=run)
            G_e6 = build_knn_graph(ph, k=k, seed=run)
            ds,_ = spectral_dimension_v3(G_e6)
            ds_e6[k].append(ds)
            # Null
            rng = np.random.RandomState(run+5000)
            ph_n = rng.uniform(0,2*np.pi,(N,6))
            G_null = build_knn_graph(ph_n, k=k, seed=run)
            ds_n,_ = spectral_dimension_v3(G_null)
            ds_null[k].append(ds_n)
            print(".", end="", flush=True)

        m_e6  = np.mean(ds_e6[k])
        s_e6  = np.std(ds_e6[k])/np.sqrt(n_runs)
        m_null= np.mean(ds_null[k])
        s_null= np.std(ds_null[k])/np.sqrt(n_runs)
        # t-test
        t_stat, p_val = ttest_ind(ds_e6[k], ds_null[k])
        sig = "***" if p_val<0.001 else ("**" if p_val<0.01 else
              ("*" if p_val<0.05 else "ns"))
        print(f" E6={m_e6:.3f}±{s_e6:.3f}  "
              f"null={m_null:.3f}±{s_null:.3f}  "
              f"Δ={m_e6-m_null:+.3f}  {sig}")

    # Find k*: where E6 first reaches d_s ≥ 3.8
    means_e6 = [np.mean(ds_e6[k]) for k in k_values]
    print(f"\n  Calibrated d_s (×{CALIB_FACTOR_4D:.3f}):")
    k_star = None
    for k, m in zip(k_values, means_e6):
        mc = m * CALIB_FACTOR_4D
        print(f"    k={k:2d}: raw={m:.3f}, calibrated={mc:.3f}", end="")
        if mc >= 3.9 and k_star is None:
            k_star = k
            print(f"  ← k* (d_s≈4)")
        else:
            print()

    if k_star:
        print(f"\n  k* = {k_star}: E6 achieves d_s≈4 at this connectivity")
        # Is null also 4 at k*?
        m_null_kstar = np.mean(ds_null[k_star]) * CALIB_FACTOR_4D
        print(f"  Null at k*: d_s = {m_null_kstar:.3f}")
        if m_null_kstar < 3.5:
            print(f"  ✅ E6 unique: null stays below 3.5 at k={k_star}")
        else:
            print(f"  ❌ Null also reaches ≈4 at k={k_star} — not E6-specific")
    else:
        print(f"\n  ❌ E6 never reaches calibrated d_s≈4 in k ∈ {k_values}")

    return {'k_values': k_values,
            'ds_e6':   {k: (np.mean(ds_e6[k]),
                             np.std(ds_e6[k])/np.sqrt(n_runs))
                        for k in k_values},
            'ds_null': {k: (np.mean(ds_null[k]),
                             np.std(ds_null[k])/np.sqrt(n_runs))
                        for k in k_values},
            'k_star':  k_star}

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Chain graph diagnostic
# ═══════════════════════════════════════════════════════════════

def experiment_chain_diagnostic(n_runs=5):
    """
    Why does chain graph fail at N>600?
    Measure: connectivity, avg degree, λ_min, d_s vs N.
    Plot d_s(t) curves for N=400,600,800,1000.
    """
    print("\n"+"="*60)
    print("EXPERIMENT 2: Chain graph diagnostic")
    print("="*60)

    omega = omega_E6()
    sizes = [400, 600, 800, 1000]

    print(f"  {'N':>6} {'<k>':>6} {'connected':>12} {'λ_min':>10} "
          f"{'d_s_raw':>10} {'d_s_cal':>10}")
    print("  "+"-"*60)

    results = {}
    fig, axes = plt.subplots(1,4,figsize=(20,4))
    fig.suptitle("Chain graph d_s(t) curves: why does it fail?",
                 fontsize=11)

    for ni,(N,ax) in enumerate(zip(sizes,axes)):
        ds_runs = []
        for run in range(n_runs):
            ph = evolve_phases(N, omega, seed=run)
            G  = build_chain_graph(ph, seed=run)
            avg_k = np.mean([d for _,d in G.degree()])
            conn  = nx.is_connected(G)
            ds_raw, ds_cal, curve = spectral_dimension_calibrated(
                G, verbose=False)
            ds_runs.append(ds_raw)

            # Plot first run
            if run == 0:
                Ln = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
                eigs = np.linalg.eigvalsh(Ln)
                eigs_nz = np.sort(eigs[eigs>1e-6])
                lmin = eigs_nz[0] if len(eigs_nz)>0 else 0
                lmax = eigs_nz[-1] if len(eigs_nz)>0 else 2

                if curve is not None:
                    ts, ds_curve = curve[0], curve[1]
                    ax.semilogx(ts, ds_curve, 'b-', lw=1.5,
                                label=f'd_s(t)')
                    if len(curve)>2:
                        s,W = curve[2],curve[3]
                        ax.axvspan(ts[s],ts[min(s+W,len(ts)-1)],
                                   alpha=0.2,color='orange',
                                   label='plateau')
                ax.axhline(ds_raw,color='red',linestyle='--',
                           label=f'raw={ds_raw:.2f}')
                ax.axhline(ds_raw*CALIB_FACTOR_4D,
                           color='green',linestyle=':',
                           label=f'cal={ds_raw*CALIB_FACTOR_4D:.2f}')
                ax.axhline(4.0,color='black',linestyle=':',
                           alpha=0.5,label='d=4')
                ax.set_title(f'N={N}\n<k>={avg_k:.1f}, '
                             f'conn={conn}, λ_min={lmin:.4f}')
                ax.set_xlabel('t'); ax.set_ylabel('d_s(t)')
                ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
                ax.set_ylim([0,8])

        m  = np.mean(ds_runs)
        mc = m*CALIB_FACTOR_4D
        se = np.std(ds_runs)/np.sqrt(n_runs)

        # Get diagnostics from last run
        ph = evolve_phases(N,omega,seed=0)
        G  = build_chain_graph(ph,seed=0)
        avg_k = np.mean([d for _,d in G.degree()])
        conn  = nx.is_connected(G)
        Ln    = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
        eigs  = np.sort(np.linalg.eigvalsh(Ln))
        lmin  = eigs[eigs>1e-6][0] if (eigs>1e-6).any() else 0

        print(f"  {N:>6} {avg_k:>6.2f} {str(conn):>12} "
              f"{lmin:>10.5f} {m:>10.3f} {mc:>10.3f}")
        results[N] = (m, mc, se, avg_k, conn, lmin)

    plt.tight_layout()
    plt.savefig('chain_diagnostic.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("  Saved: chain_diagnostic.png")

    # Diagnosis
    print("\n  Diagnosis:")
    for N,(m,mc,se,avg_k,conn,lmin) in results.items():
        issues = []
        if not conn: issues.append("DISCONNECTED")
        if lmin < 1e-4: issues.append(f"λ_min≈0 ({lmin:.2e})")
        if avg_k < 3: issues.append(f"sparse (<k>={avg_k:.1f})")
        if mc < 1.0: issues.append("d_s collapsed")
        status = ", ".join(issues) if issues else "OK"
        print(f"    N={N}: {status}")

    return results

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: E6 uniqueness — statistical test
# ═══════════════════════════════════════════════════════════════

def experiment_e6_uniqueness(n_runs=10):
    """
    Rigorous test: is E6 d_s significantly different from
    other algebras AND from null?
    Uses k=8 k-NN, N=800, 10 runs each.
    Reports: mean, SEM, 95% CI, t-test vs null.
    """
    print("\n"+"="*60)
    print("EXPERIMENT 3: E6 uniqueness (rigorous statistical test)")
    print("="*60)

    algebras = {
        'E6':   (omega_E6,   cartan_E6,  0.1),
        'B6':   (omega_B6,   cartan_B6,  0.1),
        'D6':   (omega_D6,   cartan_D6,  0.1),
        'A6':   (omega_A6,   cartan_A6,  0.1),
        'Null': (None,       None,       0.0),
    }
    N   = 800
    k   = 8

    raw_data = {name:[] for name in algebras}

    print(f"  N={N}, k={k}, {n_runs} runs each\n")
    for name,(omega_fn,K_fn,kappa) in algebras.items():
        print(f"  {name}:", end="",flush=True)
        for run in range(n_runs):
            if name == 'Null':
                rng = np.random.RandomState(run+9000)
                ph = rng.uniform(0,2*np.pi,(N,6))
            elif kappa > 0:
                ph = evolve_phases_coupled(N,omega_fn(),K_fn(),
                                           kappa=kappa,seed=run)
            else:
                ph = evolve_phases(N,omega_fn(),seed=run)
            G = build_knn_graph(ph,k=k,seed=run)
            ds,_ = spectral_dimension_v3(G)
            raw_data[name].append(ds)
            print(".",end="",flush=True)
        print()

    # Report
    print(f"\n  {'Algebra':8s} {'mean':>8} {'SEM':>8} "
          f"{'95% CI':>18} {'cal':>8} {'vs null':>12}")
    print("  "+"-"*68)
    null_data = np.array(raw_data['Null'])
    for name in algebras:
        d = np.array(raw_data[name])
        m  = d.mean()
        se = d.std()/np.sqrt(n_runs)
        ci_lo = m - 1.96*se
        ci_hi = m + 1.96*se
        mc = m * CALIB_FACTOR_4D
        if name != 'Null':
            t,p = ttest_ind(d, null_data)
            sig = "***" if p<0.001 else ("**" if p<0.01 else
                  ("*" if p<0.05 else "ns"))
            vs = f"t={t:.2f} {sig}"
        else:
            vs = "(reference)"
        in_4 = "✅" if (ci_lo*CALIB_FACTOR_4D <= 4.0 <=
                        ci_hi*CALIB_FACTOR_4D) else "❌"
        print(f"  {name:8s} {m:8.3f} {se:8.3f} "
              f"[{ci_lo:.3f},{ci_hi:.3f}] {mc:8.3f} {vs:>16}  "
              f"d_s_cal∋4? {in_4}")

    # Pairwise: E6 vs others
    print(f"\n  Pairwise E6 vs others:")
    e6_data = np.array(raw_data['E6'])
    for name in ['B6','D6','A6']:
        d = np.array(raw_data[name])
        t,p = ttest_ind(e6_data, d)
        sig = "***" if p<0.001 else ("**" if p<0.01 else
              ("*" if p<0.05 else "ns"))
        print(f"    E6 vs {name}: Δ={e6_data.mean()-d.mean():+.3f}  "
              f"t={t:.2f}  p={p:.4f}  {sig}")

    return raw_data

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Direction D — correct interpretation
# ═══════════════════════════════════════════════════════════════

def experiment_dark_energy(n_mc=15, n_epochs=30):
    """
    Correct analysis of dark energy models.
    Key fix: measure VELOCITY (d⟨d⟩/dt) not just second derivative.
    Dark energy = sustained positive velocity (expansion) that
    ACCELERATES, i.e., d²⟨d⟩/dt² > 0 AND increasing.

    Also: compare to ΛCDM — does ⟨d⟩(t) fit a(t)∝exp(H·t)?
    """
    print("\n"+"="*60)
    print("EXPERIMENT 4: Dark energy — correct analysis")
    print("="*60)
    print("Dark energy requires: ⟨d⟩ growing AND d²⟨d⟩/dt² > 0")
    print("(not just 'more negative than null')\n")

    omega   = omega_E6()
    N_init  = 300
    N_add   = 60
    lambda0 = 0.004
    eps_ph  = 1.5
    max_conn= 4
    n_cand  = 50
    MAX_RATIO = 2.0
    N_total = N_init + N_add*n_epochs + 50

    def build_graph(phases, n_nodes, lam, seed):
        rng = np.random.RandomState(seed)
        G   = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes-1): G.add_edge(i,i+1)
        deg = np.zeros(n_nodes)
        for i in range(n_nodes-1): deg[i]+=1; deg[i+1]+=1
        for i in range(2,n_nodes):
            pool=min(i,n_cand)
            w=deg[:i]+1.0; w/=w.sum()
            cands=rng.choice(i,pool,p=w,replace=False)
            diffs=np.abs(phases[i]-phases[cands])
            diffs=np.minimum(diffs,2*np.pi-diffs)
            dists=np.linalg.norm(diffs,axis=1)
            added=0
            for j in np.argsort(dists):
                if dists[j]<eps_ph and added<max_conn:
                    prob=np.exp(-lam*(i-int(cands[j])))
                    if rng.random()<prob:
                        G.add_edge(i,int(cands[j]))
                        deg[i]+=1; deg[int(cands[j])]+=1
                        added+=1
        return G

    def sample_path(G, n_sample=80, seed=0):
        nodes=list(G.nodes())
        rng=np.random.RandomState(seed)
        srcs=rng.choice(nodes,min(n_sample,len(nodes)),replace=False)
        paths=[]
        for s in srcs:
            L=nx.single_source_shortest_path_length(G,s)
            if len(L)>1: paths.extend(list(L.values())[1:])
        return float(np.mean(paths)) if paths else np.nan

    path_A = np.full((n_mc,n_epochs+1),np.nan)
    path_B = np.full((n_mc,n_epochs+1),np.nan)
    path_C = np.full((n_mc,n_epochs+1),np.nan)

    print(f"  {n_mc} MC × {n_epochs} epochs...")
    t0=time.time()
    for mc in range(n_mc):
        phases_all = evolve_phases(N_total,omega,seed=mc*17+3)
        lam_C=lambda0; d_init=None
        for ep in range(n_epochs+1):
            n_nodes=min(N_init+N_add*ep,N_total-1)
            se=mc*600+ep
            G_A=build_graph(phases_all,n_nodes,0.0,se)
            G_B=build_graph(phases_all,n_nodes,lambda0,se)
            G_C=build_graph(phases_all,n_nodes,lam_C,se)
            path_A[mc,ep]=sample_path(G_A,seed=se)
            path_B[mc,ep]=sample_path(G_B,seed=se)
            d_now=sample_path(G_C,seed=se)
            path_C[mc,ep]=d_now
            if ep==0: d_init=d_now if (d_now and d_now>0) else 1.0
            if d_init and d_init>0 and d_now and d_now>0:
                ratio=min(d_now/d_init,MAX_RATIO)
                lam_C=np.clip(lambda0*ratio,0,lambda0*MAX_RATIO)
        if (mc+1)%5==0:
            print(f"    MC {mc+1}/{n_mc} ({time.time()-t0:.0f}s)")

    mean_A=np.nanmean(path_A,axis=0)
    mean_B=np.nanmean(path_B,axis=0)
    mean_C=np.nanmean(path_C,axis=0)
    sem_A=np.nanstd(path_A,axis=0)/np.sqrt(n_mc)
    sem_B=np.nanstd(path_B,axis=0)/np.sqrt(n_mc)
    sem_C=np.nanstd(path_C,axis=0)/np.sqrt(n_mc)

    ep_arr = np.arange(n_epochs+1)

    # Velocity (first difference)
    vel_A = np.diff(mean_A)
    vel_B = np.diff(mean_B)
    vel_C = np.diff(mean_C)

    # Acceleration (second difference) — last 10 epochs
    acc_A_arr = np.diff(vel_A[-10:])
    acc_B_arr = np.diff(vel_B[-10:])
    acc_C_arr = np.diff(vel_C[-10:])
    acc_A = float(np.mean(acc_A_arr))
    acc_B = float(np.mean(acc_B_arr))
    acc_C = float(np.mean(acc_C_arr))

    # Fit ⟨d⟩(t) to exponential: a(t) = a0·exp(H·t)
    # Dark energy → H > 0 and growing
    def fit_exp_growth(y, x):
        """Fit log(y) = log(a0) + H·x via linear regression."""
        logy = np.log(np.maximum(y, 1e-6))
        H,loga0 = np.polyfit(x, logy, 1)
        return H, np.exp(loga0)

    ep_fit = ep_arr[-15:]  # Last 15 epochs
    H_A, a0_A = fit_exp_growth(mean_A[-15:], ep_fit)
    H_B, a0_B = fit_exp_growth(mean_B[-15:], ep_fit)
    H_C, a0_C = fit_exp_growth(mean_C[-15:], ep_fit)

    print(f"\n  Results:")
    print(f"  {'Model':12s} {'⟨d⟩_final':>12} {'vel_final':>12} "
          f"{'acc(last10)':>14} {'H(exp fit)':>12}")
    print("  "+"-"*64)
    for label,m,v,acc,H in [
        ("A (null)",    mean_A[-1], vel_A[-1], acc_A, H_A),
        ("B (const λ)", mean_B[-1], vel_B[-1], acc_B, H_B),
        ("C (feedback)",mean_C[-1], vel_C[-1], acc_C, H_C),
    ]:
        print(f"  {label:12s} {m:12.3f} {v:12.4f} {acc:14.5f} {H:12.5f}")

    print(f"\n  Dark energy requires acc > 0 AND H > 0:")
    for label,acc,H,vel_last in [
        ("A", acc_A, H_A, vel_A[-1]),
        ("B", acc_B, H_B, vel_B[-1]),
        ("C", acc_C, H_C, vel_C[-1]),
    ]:
        is_de = (acc > 0) and (H > 0) and (vel_last > 0)
        print(f"  Model {label}: acc={acc:+.5f}, H={H:+.5f}, "
              f"v_final={vel_last:+.4f}  → "
              f"{'DARK ENERGY ✅' if is_de else 'NOT dark energy ❌'}")

    return {'epochs': ep_arr,
            'mean_A':mean_A,'sem_A':sem_A,
            'mean_B':mean_B,'sem_B':sem_B,
            'mean_C':mean_C,'sem_C':sem_C,
            'vel': (vel_A,vel_B,vel_C),
            'acc': (acc_A,acc_B,acc_C),
            'H':   (H_A,H_B,H_C)}

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_all(res_kscan, res_chain, res_uniq, res_de):
    fig, axes = plt.subplots(2,4,figsize=(24,10))
    fig.suptitle("Part IV+ v4: Four Targeted Experiments",
                 fontsize=13,fontweight='bold')

    # 1: k-scan E6 vs null
    ax = axes[0,0]
    k_vals = res_kscan['k_values']
    m_e6   = [res_kscan['ds_e6'][k][0]   for k in k_vals]
    s_e6   = [res_kscan['ds_e6'][k][1]   for k in k_vals]
    m_null = [res_kscan['ds_null'][k][0]  for k in k_vals]
    s_null = [res_kscan['ds_null'][k][1]  for k in k_vals]
    # Calibrated
    mc_e6  = [m*CALIB_FACTOR_4D for m in m_e6]
    mc_null= [m*CALIB_FACTOR_4D for m in m_null]
    ax.errorbar(k_vals, mc_e6,
                yerr=[2*s*CALIB_FACTOR_4D for s in s_e6],
                fmt='bo-', capsize=4, label='E6 (calibrated)')
    ax.errorbar(k_vals, mc_null,
                yerr=[2*s*CALIB_FACTOR_4D for s in s_null],
                fmt='r^--', capsize=4, label='Null (calibrated)')
    ax.axhline(4.0,color='black',linestyle=':',lw=2,label='d=4')
    if res_kscan['k_star']:
        ax.axvline(res_kscan['k_star'],color='green',
                   linestyle='--',lw=2,
                   label=f'k*={res_kscan["k_star"]}')
    ax.set_xlabel('k (nearest neighbors)')
    ax.set_ylabel('d_s (calibrated)')
    ax.set_title('1: k-scan\nFind k* where d_s(E6)=4')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # 2: k-scan difference E6-null
    ax = axes[0,1]
    diff   = [e-n for e,n in zip(mc_e6,mc_null)]
    colors = ['green' if d>0 else 'red' for d in diff]
    ax.bar(k_vals, diff, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(0,color='black',lw=1)
    ax.set_xlabel('k')
    ax.set_ylabel('Δd_s (E6 − null, calibrated)')
    ax.set_title('1: E6 advantage over null\n(green=E6 higher)')
    ax.grid(True,alpha=0.3,axis='y')

    # 3: E6 uniqueness — box plot
    ax = axes[0,2]
    alg_names = list(res_uniq.keys())
    data_cal  = [np.array(res_uniq[n])*CALIB_FACTOR_4D
                 for n in alg_names]
    bp = ax.boxplot(data_cal, labels=alg_names,
                    patch_artist=True, notch=True)
    colors_box=['steelblue','tomato','green','purple','gray']
    for patch,col in zip(bp['boxes'],colors_box):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.axhline(4.0,color='black',linestyle=':',lw=2,label='d=4')
    ax.set_ylabel('d_s (calibrated)')
    ax.set_title('3: Algebra uniqueness\n(notched box = 95% CI of median)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # 4: Chain diagnostic (skip — already saved separately)
    ax = axes[0,3]
    N_vals = list(res_chain.keys())
    ds_ch  = [res_chain[N][0]*CALIB_FACTOR_4D for N in N_vals]
    lmins  = [res_chain[N][5] for N in N_vals]
    ax2    = ax.twinx()
    ax.plot(N_vals, ds_ch, 'bo-', label='d_s (calibrated)', lw=2)
    ax2.plot(N_vals, lmins, 'r^--', label='λ_min', lw=2)
    ax.axhline(4.0,color='black',linestyle=':',lw=2)
    ax.axhline(1.0,color='gray',linestyle=':',lw=1)
    ax.set_xlabel('N'); ax.set_ylabel('d_s', color='blue')
    ax2.set_ylabel('λ_min', color='red')
    ax.set_title('2: Chain graph vs N\n(why does it fail?)')
    lines1,labs1 = ax.get_legend_handles_labels()
    lines2,labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7)
    ax.grid(True,alpha=0.3)

    # 5: Dark energy — ⟨d⟩ vs epoch
    ax = axes[1,0]
    ep = res_de['epochs']
    for lab,km,ks,col in [
        ('A null',    'mean_A','sem_A','gray'),
        ('B const',   'mean_B','sem_B','steelblue'),
        ('C feedback','mean_C','sem_C','tomato'),
    ]:
        m=res_de[km]; s=res_de[ks]
        ax.fill_between(ep,m-2*s,m+2*s,alpha=0.15,color=col)
        ax.plot(ep,m,'-',color=col,label=lab,lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('⟨d⟩')
    ax.set_title('4: ⟨d⟩ vs epoch')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # 6: Velocity
    ax = axes[1,1]
    vel_A,vel_B,vel_C = res_de['vel']
    ep2 = ep[1:]
    ax.plot(ep2,vel_A,'gray',lw=2,label='A null')
    ax.plot(ep2,vel_B,'steelblue',lw=2,label='B const')
    ax.plot(ep2,vel_C,'tomato',lw=2,label='C feedback')
    ax.axhline(0,color='black',lw=1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('d⟨d⟩/dt (velocity)')
    ax.set_title('4: Expansion velocity\n(positive = expanding)')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    # 7: Acceleration
    ax = axes[1,2]
    acc_A,acc_B,acc_C = res_de['acc']
    H_A,H_B,H_C = res_de['H']
    model_names=['A\n(null)','B\n(const)','C\n(feedback)']
    acc_vals=[acc_A,acc_B,acc_C]
    cols_bar=['gray','steelblue','tomato']
    bars=ax.bar(model_names,acc_vals,color=cols_bar,
                alpha=0.8,edgecolor='black')
    ax.axhline(0,color='black',lw=1.5)
    for bar,val in zip(bars,acc_vals):
        ax.text(bar.get_x()+bar.get_width()/2.,val,
                f'{val:+.4f}',ha='center',
                va='bottom' if val>=0 else 'top',fontsize=9)
    ax.set_ylabel('d²⟨d⟩/dt²')
    ax.set_title('4: Acceleration\n(positive = dark energy)')
    ax.grid(True,alpha=0.3,axis='y')

    # 8: Exponential Hubble fit
    ax = axes[1,3]
    H_vals = [H_A,H_B,H_C]
    bars2 = ax.bar(model_names, H_vals, color=cols_bar,
                   alpha=0.8, edgecolor='black')
    ax.axhline(0,color='black',lw=1.5)
    for bar,val in zip(bars2,H_vals):
        ax.text(bar.get_x()+bar.get_width()/2.,val,
                f'{val:+.4f}',ha='center',
                va='bottom' if val>=0 else 'top',fontsize=9)
    ax.set_ylabel('H (exponential growth rate)')
    ax.set_title('4: Hubble parameter\n(positive = exp expansion)')
    ax.grid(True,alpha=0.3,axis='y')

    plt.tight_layout()
    plt.savefig('part4plus_v4.png',dpi=150,bbox_inches='tight')
    print("  Saved: part4plus_v4.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(res_kscan, res_chain, res_uniq, res_de):
    print("\n"+"="*60)
    print("SUMMARY: Part IV+ v4")
    print(f"  Calibration factor: ×{CALIB_FACTOR_4D:.3f} "
          f"(from 4D grid benchmark)")
    print("="*60)

    print("\n┌─ EXP 1: k-scan ─────────────────────────────────────────┐")
    k_star = res_kscan['k_star']
    if k_star:
        m_e6  = res_kscan['ds_e6'][k_star][0]*CALIB_FACTOR_4D
        m_null= res_kscan['ds_null'][k_star][0]*CALIB_FACTOR_4D
        print(f"│  k* = {k_star}: E6 d_s(cal)={m_e6:.3f}, "
              f"null={m_null:.3f}                  │")
        if m_null < 3.5:
            print(f"│  ✅ E6 reaches d_s≈4 at k={k_star}; null stays below 3.5  │")
        else:
            print(f"│  ❌ Null also ≈4 at k={k_star} — geometry, not E6         │")
    else:
        print(f"│  ❌ E6 never reaches calibrated d_s=4                    │")
    print("└──────────────────────────────────────────────────────────┘")

    print("\n┌─ EXP 2: Chain diagnostic ──────────────────────────────┐")
    for N,v in res_chain.items():
        m,mc,se,avg_k,conn,lmin = v
        print(f"│  N={N:4d}: d_s_cal={mc:.2f}, <k>={avg_k:.1f}, "
              f"λ_min={lmin:.4f}, conn={conn}     │")
    print("└──────────────────────────────────────────────────────────┘")

    print("\n┌─ EXP 3: E6 uniqueness ─────────────────────────────────┐")
    for name,data in res_uniq.items():
        d = np.array(data)
        m  = d.mean()*CALIB_FACTOR_4D
        se = d.std()/np.sqrt(len(d))*CALIB_FACTOR_4D
        ci_lo = m-1.96*se; ci_hi = m+1.96*se
        in4 = "✅" if ci_lo<=4.0<=ci_hi else "❌"
        print(f"│  {name:6s}: {m:.3f}±{se:.3f}  CI=[{ci_lo:.2f},{ci_hi:.2f}]  "
              f"contains 4.0? {in4}               │")
    print("└──────────────────────────────────────────────────────────┘")

    print("\n┌─ EXP 4: Dark energy ───────────────────────────────────┐")
    acc_A,acc_B,acc_C = res_de['acc']
    H_A,H_B,H_C      = res_de['H']
    vel_A,vel_B,vel_C = res_de['vel']
    for lab,acc,H,v_last in [
        ("A",acc_A,H_A,vel_A[-1]),
        ("B",acc_B,H_B,vel_B[-1]),
        ("C",acc_C,H_C,vel_C[-1]),
    ]:
        is_de = acc>0 and H>0 and v_last>0
        print(f"│  {lab}: acc={acc:+.4f}, H={H:+.4f}, v={v_last:+.4f}  "
              f"{'DARK ENERGY ✅' if is_de else 'no dark energy ❌':20s}  │")
    print("└──────────────────────────────────────────────────────────┘")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Monostring Hypothesis — Part IV+ v4                   ║")
    print("║   k-scan | chain diagnostic | E6 uniqueness | DE        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Started: {time.strftime('%H:%M:%S')}")
    print(f"Calibration factor: ×{CALIB_FACTOR_4D:.4f} "
          f"(4D grid: 3.616→4.0)\n")

    print("[1/4] k-scan: find k* for d_s(E6)=4...")
    res_kscan = experiment_k_scan(n_runs=6)

    print("\n[2/4] Chain graph diagnostic...")
    res_chain = experiment_chain_diagnostic(n_runs=5)

    print("\n[3/4] E6 uniqueness (rigorous)...")
    res_uniq = experiment_e6_uniqueness(n_runs=10)

    print("\n[4/4] Dark energy (corrected)...")
    res_de = experiment_dark_energy(n_mc=15, n_epochs=30)

    print_summary(res_kscan, res_chain, res_uniq, res_de)
    plot_all(res_kscan, res_chain, res_uniq, res_de)

    elapsed = time.time()-t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")
    print("Done.")
