import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, eye, diags, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog
from collections import deque
import time, random, warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],
    [-1, 2,-1, 0, 0, 0],
    [ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],
    [ 0, 0, 0,-1, 2, 0],
    [ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# TRAJECTORY
# ================================================================
def generate_trajectory(N, kappa, omega=None):
    D = 6
    if omega is None:
        omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    phases = np.zeros((N, D))
    phases[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N - 1):
        coupling = kappa * C_E6 @ np.sin(phases[n])
        phases[n+1] = (phases[n] + omega + coupling) % (2 * np.pi)
    return phases

def estimate_correlation_dimension(phases, max_pairs=100000):
    N = len(phases)
    n_pairs = min(max_pairs, N*(N-1)//2)
    idx_a = np.random.randint(0, N, n_pairs)
    idx_b = np.random.randint(0, N, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    diff = np.abs(phases[idx_a] - phases[idx_b])
    diff = np.minimum(diff, 2*np.pi - diff)
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_vals = np.logspace(np.log10(np.percentile(dists,1)),
                         np.log10(np.percentile(dists,90)), 60)
    C_r = np.array([np.mean(dists < r) for r in r_vals])
    valid = (C_r > 0.05) & (C_r < 0.40)
    if np.sum(valid) < 5:
        valid = (C_r > 0.01) & (C_r < 0.5)
    if np.sum(valid) < 5:
        return np.nan
    return np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)[0]

def compute_lyapunov(N_steps, kappa, omega=None, D=6):
    if omega is None:
        omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    phi = np.random.uniform(0, 2*np.pi, D)
    v = np.random.randn(D); v /= np.linalg.norm(v)
    lyap_sum = 0.0
    for n in range(min(N_steps, 50000)):
        J = np.eye(D) + kappa * C_E6 @ np.diag(np.cos(phi))
        v = J @ v; norm_v = np.linalg.norm(v)
        lyap_sum += np.log(norm_v); v /= norm_v
        phi = (phi + omega + kappa * C_E6 @ np.sin(phi)) % (2*np.pi)
    return lyap_sum / min(N_steps, 50000)

# ================================================================
# GRAPH
# ================================================================
def build_graph(phases, target_degree=40, delta_min=10):
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    eps_lo, eps_hi = 0.1, 5.0
    for _ in range(15):
        eps_mid = (eps_lo + eps_hi) / 2
        sample = random.sample(range(N), min(500, N))
        degs = [sum(1 for j in tree.query_ball_point(coords[i], eps_mid)
                    if abs(j-i) > delta_min and j != i) for i in sample]
        avg = np.mean(degs)
        if avg < target_degree: eps_lo = eps_mid
        else: eps_hi = eps_mid
        if abs(avg - target_degree)/max(target_degree,1) < 0.05: break
    best_eps = (eps_lo + eps_hi) / 2
    pairs = tree.query_pairs(r=best_eps, output_type='ndarray')
    mask = np.abs(pairs[:,0] - pairs[:,1]) > delta_min
    pairs = pairs[mask]
    adj = {i: set() for i in range(N)}
    for i in range(N-1): adj[i].add(i+1); adj[i+1].add(i)
    for a,b in pairs: adj[a].add(b); adj[b].add(a)
    actual = sum(len(adj[v]) for v in adj)/N
    return adj, len(pairs), best_eps, actual

# ================================================================
# DIMENSION (fixed saturation)
# ================================================================
def measure_dimension(adj, N, num_obs=25, max_R=12):
    observers = random.sample(range(N//10, 9*N//10), min(num_obs, N//2))
    radii = list(range(1, max_R+1))
    all_vols = []
    for obs in observers:
        dist = {obs: 0}; q = deque([obs])
        while q:
            v = q.popleft()
            if dist[v] >= max_R: continue
            for u in adj[v]:
                if u not in dist: dist[u] = dist[v]+1; q.append(u)
        all_vols.append([sum(1 for d in dist.values() if d<=r) for r in radii])
    avg_v = np.mean(all_vols, axis=0)
    sat_R = max_R
    for i, vol in enumerate(avg_v):
        if vol > 0.3*N: sat_R = radii[i]; break
    local_D, valid_r = [], []
    for i in range(1, len(radii)):
        if radii[i] > sat_R: break
        if avg_v[i] > avg_v[i-1] > 0:
            local_D.append(np.log(avg_v[i]/avg_v[i-1]) / np.log(radii[i]/radii[i-1]))
            valid_r.append(radii[i])
    m = np.array(radii) <= sat_R; vr = np.array(radii)[m]; vv = avg_v[m]
    D_reg = np.polyfit(np.log(vr), np.log(vv), 1)[0] if len(vr)>=3 and all(v>0 for v in vv) else np.nan
    D_micro = np.nanmean(local_D[:2]) if len(local_D)>=2 else np.nan
    return {'valid_radii': valid_r, 'local_D': local_D, 'D_reg': D_reg,
            'D_micro': D_micro, 'sat_R': sat_R, 'avg_vols': avg_v, 'radii': radii}

# ================================================================
# CURVATURE
# ================================================================
def bfs_limited(adj, start, max_d=6):
    dist = {start: 0}; q = deque([start])
    while q:
        v = q.popleft()
        if dist[v] >= max_d: continue
        for u in adj[v]:
            if u not in dist: dist[u] = dist[v]+1; q.append(u)
    return dist

def ollivier_ricci(adj, a, b, alpha=0.5):
    na, nb = list(adj[a]), list(adj[b])
    if not na or not nb: return None
    sa, sb = [a]+na, [b]+nb
    da, db = len(na), len(nb)
    mu_a = {a: alpha};
    for v in na: mu_a[v] = mu_a.get(v,0) + (1-alpha)/da
    mu_b = {b: alpha};
    for v in nb: mu_b[v] = mu_b.get(v,0) + (1-alpha)/db
    n_a, n_b = len(sa), len(sb)
    dm = np.zeros((n_a, n_b))
    for i, u in enumerate(sa):
        d = bfs_limited(adj, u, max_d=6)
        for j, v in enumerate(sb): dm[i,j] = d.get(v, 20)
    c = dm.flatten(); nv = n_a*n_b
    A_rows, b_vec = [], []
    for i in range(n_a):
        row = np.zeros(nv)
        for j in range(n_b): row[i*n_b+j] = 1.0
        A_rows.append(row); b_vec.append(mu_a.get(sa[i],0.0))
    for j in range(n_b):
        row = np.zeros(nv)
        for i in range(n_a): row[i*n_b+j] = 1.0
        A_rows.append(row); b_vec.append(mu_b.get(sb[j],0.0))
    res = linprog(c, A_eq=np.array(A_rows), b_eq=np.array(b_vec),
                  bounds=[(0,None)]*nv, method='highs')
    return 1.0 - res.fun if res.success else None

def test_gravity(adj, N, n_samples=300):
    edges = [(v,u) for v in adj for u in adj[v] if u>v]
    sample = random.sample(edges, min(n_samples, len(edges)))
    ek = {}
    for i,(a,b) in enumerate(sample):
        k = ollivier_ricci(adj, a, b)
        if k is not None: ek[(a,b)] = k
        if (i+1)%100==0: print(f"        {i+1}/{len(sample)}")
    vc = {}
    for (a,b),k in ek.items(): vc[a]=vc.get(a,0)+k; vc[b]=vc.get(b,0)+k
    ad = sum(len(adj[v]) for v in adj)/N
    vd = {v: len(adj[v])/ad for v in vc}
    vs = list(vc.keys()); cu = np.array([vc[v] for v in vs]); de = np.array([vd[v] for v in vs])
    if len(cu)>2 and np.std(cu)>1e-10 and np.std(de)>1e-10:
        corr = np.corrcoef(cu, de)[0,1]
    else: corr = 0.0
    return corr, np.mean(cu) if len(cu)>0 else 0, cu, de, vs

# ================================================================
# BELL TEST (corrected + null control)
# ================================================================
def build_adj_sparse(adj, N):
    rows, cols = [], []
    for i in range(N):
        for j in adj[i]: rows.append(i); cols.append(j)
    return coo_matrix((np.ones(len(rows)), (rows,cols)), shape=(N,N)).tocsc()

def proper_bell_test(adj, N, n_eig=40):
    degrees = np.array([len(adj[i]) for i in range(N)], dtype=float)
    A_sp = build_adj_sparse(adj, N)
    D_inv = diags(1.0/np.sqrt(np.maximum(degrees, 1)))
    L = eye(N) - D_inv @ A_sp @ D_inv
    k = min(n_eig, N-2)
    evals, evecs = eigsh(L, k=k, which='SM', tol=1e-4, maxiter=5000)
    idx = np.argsort(evals); evals = evals[idx]; evecs = evecs[:,idx]

    center = N//2
    ds = bfs_limited(adj, center, max_d=100)
    A_n = [v for v,d in ds.items() if d%2==0]
    B_n = [v for v,d in ds.items() if d%2==1]
    if len(A_n)<10 or len(B_n)<10:
        A_n = list(range(N//2)); B_n = list(range(N//2, N))

    pA0 = evecs[np.array(A_n),1]; pA1 = evecs[np.array(A_n),2]
    pB0 = evecs[np.array(B_n),1]; pB1 = evecs[np.array(B_n),2]

    for p in [pA0, pA1, pB0, pB1]:
        n = np.linalg.norm(p)
        if n < 1e-10: return 0,0,0,0,0,0
        p /= n

    ov = np.dot(pA0, pA1); pA1 -= ov*pA0; n = np.linalg.norm(pA1)
    if n<1e-10: return 0,0,0,0,0,0
    pA1 /= n
    ov = np.dot(pB0, pB1); pB1 -= ov*pB0; n = np.linalg.norm(pB1)
    if n<1e-10: return 0,0,0,0,0,0
    pB1 /= n

    H_AB = np.zeros((2,2))
    B_set = set(B_n); B_list = list(B_n)
    B_index = {node: idx for idx, node in enumerate(B_n)}

    for i_idx, i_node in enumerate(A_n):
        for j_node in adj[i_node]:
            if j_node in B_set:
                jj = B_index[j_node]
                for kk in range(2):
                    pAk = [pA0, pA1][kk]
                    for ll in range(2):
                        pBl = [pB0, pB1][ll]
                        H_AB[kk,ll] += pAk[i_idx] * pBl[jj]

    H4 = np.zeros((4,4))
    for i in range(2):
        for j in range(2): H4[2*i+j, 2*i+j] = evals[1+i] + evals[1+j]
    for i in range(2):
        for j in range(2):
            for kk in range(2):
                for ll in range(2):
                    if (i,j) != (kk,ll):
                        H4[2*i+j, 2*kk+ll] += H_AB[i,ll]*(1 if j==kk else 0)
                        H4[2*i+j, 2*kk+ll] += H_AB[kk,j]*(1 if i==ll else 0)

    ev4, vec4 = np.linalg.eigh(H4)
    gs = vec4[:,0]
    a,b,c,d = gs
    conc = 2*abs(a*d - b*c)

    sz = np.array([[1,0],[0,-1]], dtype=float)
    sx = np.array([[0,1],[1,0]], dtype=float)
    def sig(th): return np.cos(th)*sz + np.sin(th)*sx
    def E(ta, tb):
        return float(gs @ np.kron(sig(ta), sig(tb)) @ gs)

    a1,a2 = 0, np.pi/4
    b1,b2 = np.pi/8, 3*np.pi/8
    E11,E12,E21,E22 = E(a1,b1), E(a1,b2), E(a2,b1), E(a2,b2)
    S = E11 - E12 + E21 + E22

    assert abs(S) <= 2*np.sqrt(2) + 0.01, f"|S|={abs(S):.4f} > Tsirelson!"
    return S, E11, E12, E21, E22, conc

# ================================================================
# DIRAC SPECTRUM
# ================================================================
def dirac_spectrum(adj, N, k=40):
    edges = [(i,j) for i in range(N) for j in adj[i] if j>i]
    ne = len(edges)
    inc = lil_matrix((N, ne), dtype=np.float64)
    for ei,(i,j) in enumerate(edges): inc[i,ei]=-1.0; inc[j,ei]=1.0
    inc = csc_matrix(inc); L0 = inc @ inc.T
    ev,_ = eigsh(L0, k=min(k,N-2), which='SM', tol=1e-4, maxiter=5000)
    return np.sort(np.sqrt(np.abs(ev)))

# ================================================================
# NULL MODEL
# ================================================================
def build_null(N, target_degree=40, D=6):
    ph = np.random.uniform(0, 2*np.pi, (N,D))
    adj, ne, eps, ad = build_graph(ph, target_degree=target_degree)
    return adj, ph, ne, ad

# ================================================================
# COXETER FREQUENCIES (Strategy C)
# ================================================================
def coxeter_omega():
    """E6 Coxeter exponents → intrinsic frequencies."""
    exp = np.array([1, 4, 5, 7, 8, 11])
    return 2 * np.sin(np.pi * exp / 12)

# ================================================================
# PHASE DIAGRAM (fine-grained)
# ================================================================
def phase_scan(N=10000, kappas=None, td=40, omega=None):
    if kappas is None:
        kappas = np.arange(0.0, 2.05, 0.1)
    results = []
    for kp in kappas:
        ph = generate_trajectory(N, kp, omega=omega)
        dc = estimate_correlation_dimension(ph)
        ly = compute_lyapunov(N, kp, omega=omega)
        adj, ne, eps, ad = build_graph(ph, target_degree=td)
        dm = measure_dimension(adj, N, num_obs=15, max_R=10)
        results.append({'kappa': kp, 'd_corr': dc, 'D_reg': dm['D_reg'],
                        'D_micro': dm['D_micro'], 'lyap': ly, 'deg': ad})
        print(f"    κ={kp:.2f}: D_corr={dc:.2f}, D_reg={dm['D_reg']:.2f}, "
              f"λ={ly:.3f}")
    return results

# ================================================================
# MASTER
# ================================================================
def run_v3():
    print("╔" + "═"*65 + "╗")
    print("║  SBE TEST SUITE v3                                        ║")
    print("║  + Null Bell control  + Coxeter frequencies  + Fine scan   ║")
    print("╚" + "═"*65 + "╝")

    N = 15000; KAPPA = 0.5; TD = 40

    # --- Standard frequencies ---
    omega_std = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.sqrt(7), np.sqrt(11), np.sqrt(13)])
    omega_cox = coxeter_omega()

    print(f"\n  Standard ω: {omega_std}")
    print(f"  Coxeter ω:  {omega_cox}")
    print(f"  Note: Coxeter has only 3 unique values!")

    for label, omega in [("STANDARD", omega_std), ("COXETER", omega_cox)]:
        print(f"\n{'='*65}")
        print(f"  FREQUENCY SET: {label}")
        print(f"{'='*65}")

        # Trajectory
        print(f"\n  [1] Trajectory (N={N}, κ={KAPPA})...")
        t0 = time.time()
        phases = generate_trajectory(N, KAPPA, omega=omega)
        dc = estimate_correlation_dimension(phases)
        ly = compute_lyapunov(N, KAPPA, omega=omega)
        print(f"      D_corr={dc:.3f}, λ_max={ly:.4f} "
              f"({'CHAOTIC' if ly>0.01 else 'REGULAR'})")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Graph
        print(f"\n  [2] Resonance graph...")
        t0 = time.time()
        adj, ne, eps, ad = build_graph(phases, target_degree=TD)
        print(f"      ε={eps:.3f}, edges={ne:,}, deg={ad:.1f}")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Null model
        print(f"\n  [3] Null model...")
        t0 = time.time()
        adj_null, ph_null, ne_null, ad_null = build_null(N, target_degree=TD)
        dc_null = estimate_correlation_dimension(ph_null)
        print(f"      D_corr(null)={dc_null:.3f}, deg={ad_null:.1f}")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Dimension
        print(f"\n  [4] Dimension measurement...")
        t0 = time.time()
        dim_sbe = measure_dimension(adj, N)
        dim_null = measure_dimension(adj_null, N)
        print(f"      SBE:  D_reg={dim_sbe['D_reg']:.2f}, D_micro={dim_sbe['D_micro']:.2f}")
        print(f"      NULL: D_reg={dim_null['D_reg']:.2f}, D_micro={dim_null['D_micro']:.2f}")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Curvature
        print(f"\n  [5] Ollivier-Ricci curvature...")
        t0 = time.time()
        corr_sbe, mc_sbe, cu, de, vs = test_gravity(adj, N, n_samples=200)
        corr_null, mc_null, _, _, _ = test_gravity(adj_null, N, n_samples=100)
        print(f"      SBE:  <κ>={mc_sbe:.4f}, corr={corr_sbe:.3f}")
        print(f"      NULL: <κ>={mc_null:.4f}, corr={corr_null:.3f}")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Bell test — BOTH models
        print(f"\n  [6] Bell test (SBE + NULL control)...")
        t0 = time.time()
        try:
            S_sbe, _, _, _, _, C_sbe = proper_bell_test(adj, N)
            print(f"      SBE:  |S|={abs(S_sbe):.4f}, C={C_sbe:.4f}")
        except Exception as e:
            S_sbe, C_sbe = 0, 0
            print(f"      SBE Bell error: {e}")

        try:
            S_null, _, _, _, _, C_null = proper_bell_test(adj_null, N)
            print(f"      NULL: |S|={abs(S_null):.4f}, C={C_null:.4f}")
        except Exception as e:
            S_null, C_null = 0, 0
            print(f"      NULL Bell error: {e}")

        bell_informative = abs(S_sbe) > 2 and abs(S_null) <= 2
        bell_generic = abs(S_sbe) > 2 and abs(S_null) > 2
        print(f"      Bell test {'INFORMATIVE' if bell_informative else 'GENERIC (both violate)' if bell_generic else 'INCONCLUSIVE'}")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Mass spectrum
        print(f"\n  [7] Dirac spectrum...")
        t0 = time.time()
        try:
            masses = dirac_spectrum(adj, N, k=40)
            nz = masses[masses > 1e-5]
            print(f"      Zero modes: {len(masses)-len(nz)}, gap: {nz[0]:.4f}" if len(nz)>0 else "      No gap")
            if len(nz) >= 5:
                print(f"      Ratios: {nz[:5]/nz[0]}")
        except Exception as e:
            nz = np.array([]); print(f"      Error: {e}")
        print(f"      Time: {time.time()-t0:.1f}s")

        # Verdict
        print(f"\n  {'─'*60}")
        print(f"  VERDICT for {label} frequencies:")
        tests = [
            (f"D_corr < 5.5", dc < 5.5, f"{dc:.2f}"),
            (f"D_reg(SBE) < D_reg(null)-0.3",
             dim_sbe['D_reg'] < dim_null['D_reg']-0.3
             if not np.isnan(dim_sbe['D_reg']) else False,
             f"{dim_sbe['D_reg']:.2f} vs {dim_null['D_reg']:.2f}"),
            (f"corr(κ,ρ) > 0.3", abs(corr_sbe) > 0.3, f"{corr_sbe:.3f}"),
            (f"corr(SBE) > corr(null)", corr_sbe > corr_null + 0.1,
             f"{corr_sbe:.3f} vs {corr_null:.3f}"),
            (f"Bell INFORMATIVE", bell_informative,
             f"SBE={abs(S_sbe):.2f}, null={abs(S_null):.2f}"),
        ]
        np_pass = 0
        for name, passed, det in tests:
            s = "✓" if passed else "✗"
            np_pass += int(passed)
            print(f"    {s} {name:<45s} {det}")
        print(f"  Score: {np_pass}/{len(tests)}")

    # Fine phase diagram for standard frequencies
    print(f"\n{'='*65}")
    print(f"  FINE PHASE DIAGRAM (Standard frequencies)")
    print(f"{'='*65}")
    pd = phase_scan(N=10000, kappas=np.arange(0.0, 1.55, 0.05), td=TD)

    # Phase diagram for Coxeter frequencies
    print(f"\n{'='*65}")
    print(f"  FINE PHASE DIAGRAM (Coxeter frequencies)")
    print(f"{'='*65}")
    pd_cox = phase_scan(N=10000, kappas=np.arange(0.0, 1.55, 0.05),
                         td=TD, omega=omega_cox)

    # ================================================================
    # PLOTS
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Phase diagram comparison
    ax = axes[0,0]
    ks = [r['kappa'] for r in pd]
    dc_std = [r['d_corr'] for r in pd]
    ks_c = [r['kappa'] for r in pd_cox]
    dc_cox_vals = [r['d_corr'] for r in pd_cox]
    ax.plot(ks, dc_std, 'o-', color='crimson', lw=2, label='Standard ω')
    ax.plot(ks_c, dc_cox_vals, 's-', color='navy', lw=2, label='Coxeter ω')
    ax.axhline(4, ls=':', color='green', alpha=0.7, label='4D target')
    ax.axhline(6, ls=':', color='gray', alpha=0.5, label='6D trivial')
    ax.set_xlabel('κ'); ax.set_ylabel('D_corr')
    ax.set_title('Attractor Dimension: Standard vs Coxeter')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. D_reg comparison
    ax = axes[0,1]
    dr_std = [r['D_reg'] for r in pd]
    dr_cox = [r['D_reg'] for r in pd_cox]
    ax.plot(ks, dr_std, 'o-', color='crimson', lw=2, label='Standard ω')
    ax.plot(ks_c, dr_cox, 's-', color='navy', lw=2, label='Coxeter ω')
    ax.axhline(4, ls=':', color='green', alpha=0.7)
    ax.set_xlabel('κ'); ax.set_ylabel('D_reg (graph)')
    ax.set_title('Graph Dimension: Standard vs Coxeter')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Lyapunov
    ax = axes[1,0]
    ly_std = [r['lyap'] for r in pd]
    ly_cox = [r['lyap'] for r in pd_cox]
    ax.plot(ks, ly_std, 'o-', color='crimson', lw=2, label='Standard ω')
    ax.plot(ks_c, ly_cox, 's-', color='navy', lw=2, label='Coxeter ω')
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('κ'); ax.set_ylabel('λ_max')
    ax.set_title('Lyapunov Exponent')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. D_corr vs D_reg correlation
    ax = axes[1,1]
    all_dc = dc_std + dc_cox_vals
    all_dr = dr_std + dr_cox
    valid_both = [(x,y) for x,y in zip(all_dc, all_dr)
                  if not np.isnan(x) and not np.isnan(y)]
    if valid_both:
        xv, yv = zip(*valid_both)
        ax.scatter(xv, yv, alpha=0.6, s=30)
        if len(xv) > 2:
            z = np.polyfit(list(xv), list(yv), 1)
            r_corr = np.corrcoef(list(xv), list(yv))[0,1]
            xx = np.linspace(min(xv), max(xv), 100)
            ax.plot(xx, np.polyval(z, xx), 'r-', lw=2,
                    label=f'r={r_corr:.3f}')
        ax.plot([3,7],[3,7], 'k--', alpha=0.3, label='D_corr = D_reg')
    ax.set_xlabel('D_corr (attractor)')
    ax.set_ylabel('D_reg (graph)')
    ax.set_title('Consistency: Attractor dim vs Graph dim')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('SBE v3: Standard vs Coxeter Frequencies',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sbe_v3_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_v3()