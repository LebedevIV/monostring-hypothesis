import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csc_matrix, eye, diags, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog, minimize_scalar
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
# CORE (from v4, unchanged)
# ================================================================
def generate_trajectory(N, kappa, omega):
    D = 6
    phases = np.zeros((N, D))
    phases[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N-1):
        phases[n+1] = (phases[n] + omega + kappa * C_E6 @ np.sin(phases[n])) % (2*np.pi)
    return phases

def correlation_dimension(phases, max_pairs=100000):
    N = len(phases)
    ia = np.random.randint(0, N, min(max_pairs, N*(N-1)//2))
    ib = np.random.randint(0, N, len(ia))
    m = ia != ib; ia, ib = ia[m], ib[m]
    diff = np.abs(phases[ia]-phases[ib])
    diff = np.minimum(diff, 2*np.pi-diff)
    dists = np.sqrt(np.sum(diff**2, axis=1))
    rv = np.logspace(np.log10(np.percentile(dists,1)),
                     np.log10(np.percentile(dists,90)), 60)
    Cr = np.array([np.mean(dists<r) for r in rv])
    v = (Cr>0.05)&(Cr<0.40)
    if np.sum(v)<5: v=(Cr>0.01)&(Cr<0.5)
    if np.sum(v)<5: return np.nan
    return np.polyfit(np.log(rv[v]), np.log(Cr[v]), 1)[0]

def full_lyapunov(N_steps, kappa, omega, D=6):
    phi = np.random.uniform(0, 2*np.pi, D)
    Q = np.eye(D); sums = np.zeros(D)
    ns = min(N_steps, 30000)
    for n in range(ns):
        J = np.eye(D) + kappa * C_E6 @ np.diag(np.cos(phi))
        Q, R = np.linalg.qr(J @ Q)
        sums += np.log(np.abs(np.diag(R)))
        phi = (phi + omega + kappa * C_E6 @ np.sin(phi)) % (2*np.pi)
    return np.sort(sums/ns)[::-1]

def kaplan_yorke(spectrum):
    s = np.sort(spectrum)[::-1]
    cs = np.cumsum(s); j = 0
    for i in range(len(s)):
        if cs[i] >= 0: j = i+1
        else: break
    if j == 0: return 0.0
    if j >= len(s): return float(len(s))
    return j + cs[j-1]/abs(s[j])

def build_graph(phases, target_degree=40, delta_min=10):
    N, D = phases.shape
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    lo, hi = 0.1, 5.0
    for _ in range(15):
        mid = (lo+hi)/2
        samp = random.sample(range(N), min(500,N))
        degs = [sum(1 for j in tree.query_ball_point(coords[i],mid)
                    if abs(j-i)>delta_min and j!=i) for i in samp]
        if np.mean(degs) < target_degree: lo=mid
        else: hi=mid
        if abs(np.mean(degs)-target_degree)/max(target_degree,1)<0.05: break
    eps = (lo+hi)/2
    pairs = tree.query_pairs(r=eps, output_type='ndarray')
    pairs = pairs[np.abs(pairs[:,0]-pairs[:,1])>delta_min]
    adj = {i: set() for i in range(N)}
    for i in range(N-1): adj[i].add(i+1); adj[i+1].add(i)
    for a,b in pairs: adj[a].add(b); adj[b].add(a)
    return adj, len(pairs), sum(len(adj[v]) for v in adj)/N

def bfs_limited(adj, start, max_d=6):
    d = {start:0}; q = deque([start])
    while q:
        v = q.popleft()
        if d[v]>=max_d: continue
        for u in adj[v]:
            if u not in d: d[u]=d[v]+1; q.append(u)
    return d

def ollivier_ricci(adj, a, b, alpha=0.5):
    na, nb = list(adj[a]), list(adj[b])
    if not na or not nb: return None
    sa, sb = [a]+na, [b]+nb
    da, db = len(na), len(nb)
    ma = {a:alpha}
    for v in na: ma[v]=ma.get(v,0)+(1-alpha)/da
    mb = {b:alpha}
    for v in nb: mb[v]=mb.get(v,0)+(1-alpha)/db
    na2, nb2 = len(sa), len(sb)
    dm = np.zeros((na2,nb2))
    for i,u in enumerate(sa):
        dd = bfs_limited(adj,u,max_d=6)
        for j,v in enumerate(sb): dm[i,j]=dd.get(v,20)
    c=dm.flatten(); nv=na2*nb2
    Ar,bv=[],[]
    for i in range(na2):
        row=np.zeros(nv)
        for j in range(nb2): row[i*nb2+j]=1.0
        Ar.append(row); bv.append(ma.get(sa[i],0.0))
    for j in range(nb2):
        row=np.zeros(nv)
        for i in range(na2): row[i*nb2+j]=1.0
        Ar.append(row); bv.append(mb.get(sb[j],0.0))
    res=linprog(c,A_eq=np.array(Ar),b_eq=np.array(bv),
                bounds=[(0,None)]*nv,method='highs')
    return 1.0-res.fun if res.success else None

# ================================================================
# NEW IN v5: ALL LIE ALGEBRAS OF RANK 6
# ================================================================
def cartan_matrices_rank6():
    """
    All simple Lie algebras of rank 6.
    If D=4 is specific to E6, other algebras should NOT give D≈4.
    This is the critical falsification test.
    """
    algebras = {}

    # A6 = SU(7)
    A6 = np.zeros((6,6))
    for i in range(6): A6[i,i] = 2
    for i in range(5): A6[i,i+1] = -1; A6[i+1,i] = -1
    algebras['A6_SU7'] = A6

    # B6 = SO(13)
    B6 = np.zeros((6,6))
    for i in range(6): B6[i,i] = 2
    for i in range(5): B6[i,i+1] = -1; B6[i+1,i] = -1
    B6[4,5] = -2  # Short root
    algebras['B6_SO13'] = B6

    # C6 = Sp(12)
    C6 = np.zeros((6,6))
    for i in range(6): C6[i,i] = 2
    for i in range(5): C6[i,i+1] = -1; C6[i+1,i] = -1
    C6[5,4] = -2  # Long root
    algebras['C6_Sp12'] = C6

    # D6 = SO(12)
    D6 = np.zeros((6,6))
    for i in range(6): D6[i,i] = 2
    for i in range(4): D6[i,i+1] = -1; D6[i+1,i] = -1
    D6[3,5] = -1; D6[5,3] = -1  # Branching at node 4
    algebras['D6_SO12'] = D6

    # E6 (reference)
    algebras['E6'] = C_E6.copy()

    # For comparison: non-simple (direct sums)
    # A2 + A2 + A2 = SU(3)^3
    A2pA2pA2 = np.zeros((6,6))
    for block in range(3):
        i = 2*block
        A2pA2pA2[i,i] = 2; A2pA2pA2[i+1,i+1] = 2
        A2pA2pA2[i,i+1] = -1; A2pA2pA2[i+1,i] = -1
    algebras['A2+A2+A2_SU3^3'] = A2pA2pA2

    # A3 + A2 + A1 = SU(4)×SU(3)×SU(2) ~ Pati-Salam-like
    PS = np.zeros((6,6))
    # A3 block
    for i in range(3): PS[i,i]=2
    PS[0,1]=-1; PS[1,0]=-1; PS[1,2]=-1; PS[2,1]=-1
    # A2 block
    PS[3,3]=2; PS[4,4]=2; PS[3,4]=-1; PS[4,3]=-1
    # A1 block
    PS[5,5]=2
    algebras['A3+A2+A1_PS'] = PS

    return algebras

def coxeter_frequencies(cartan_matrix):
    """
    Compute Coxeter-type frequencies from a Cartan matrix.
    Uses eigenvalues of Cartan as frequencies.
    """
    evals = np.sort(np.linalg.eigvalsh(cartan_matrix))
    return evals

def coxeter_exponent_frequencies(algebra_name, cartan_matrix):
    """
    For classical algebras, compute frequencies from Coxeter exponents.
    """
    rank = cartan_matrix.shape[0]

    # Coxeter numbers and exponents for rank-6 algebras
    coxeter_data = {
        'A6_SU7':      {'h': 7,  'exponents': [1,2,3,4,5,6]},
        'B6_SO13':     {'h': 11, 'exponents': [1,3,5,7,9,11]},
        'C6_Sp12':     {'h': 12, 'exponents': [1,3,5,7,9,11]},
        'D6_SO12':     {'h': 10, 'exponents': [1,3,5,5,7,9]},
        'E6':          {'h': 12, 'exponents': [1,4,5,7,8,11]},
        'A2+A2+A2_SU3^3': {'h': 3, 'exponents': [1,2,1,2,1,2]},
        'A3+A2+A1_PS': {'h': 4, 'exponents': [1,2,3,1,2,1]},
    }

    if algebra_name in coxeter_data:
        data = coxeter_data[algebra_name]
        h = data['h']
        exp = np.array(data['exponents'], dtype=float)
        return 2 * np.sin(np.pi * exp / h)

    # Fallback: use Cartan eigenvalues
    return coxeter_frequencies(cartan_matrix)

# ================================================================
# NEW IN v5: KOLMOGOROV-SINAI ENTROPY
# ================================================================
def ks_entropy(lyap_spectrum):
    """
    Kolmogorov-Sinai entropy = sum of positive Lyapunov exponents.
    (Pesin's theorem for smooth systems)
    """
    return np.sum(lyap_spectrum[lyap_spectrum > 0])

# ================================================================
# NEW IN v5: FIND κ* WHERE D_KY = 4 (analytical)
# ================================================================
def find_kappa_star(omega, target_D=4.0, N_steps=20000,
                    kappa_range=(0.01, 2.0), tol=0.01):
    """
    Binary search for κ* where D_KY = target_D.
    Returns κ* and the achieved D_KY.
    """
    def objective(kappa):
        spec = full_lyapunov(N_steps, kappa, omega)
        dky = kaplan_yorke(spec)
        return abs(dky - target_D)

    # Coarse scan first
    kappas = np.linspace(kappa_range[0], kappa_range[1], 30)
    dkys = []
    for k in kappas:
        spec = full_lyapunov(N_steps, k, omega)
        dkys.append(kaplan_yorke(spec))

    # Find region where D_KY crosses target
    dkys = np.array(dkys)
    diffs = dkys - target_D

    # Look for sign changes or closest approach
    best_idx = np.argmin(np.abs(diffs))

    # Fine search around best
    lo = max(kappa_range[0], kappas[max(0, best_idx-1)])
    hi = min(kappa_range[1], kappas[min(len(kappas)-1, best_idx+1)])

    for _ in range(20):
        mid = (lo + hi) / 2
        spec = full_lyapunov(N_steps, mid, omega)
        dky = kaplan_yorke(spec)

        if abs(dky - target_D) < tol:
            return mid, dky, spec

        # Use the coarse data to decide direction
        spec_lo = full_lyapunov(N_steps, lo, omega)
        dky_lo = kaplan_yorke(spec_lo)

        if (dky_lo - target_D) * (dky - target_D) < 0:
            hi = mid
        else:
            lo = mid

    return mid, dky, spec

# ================================================================
# NEW IN v5: SELF-TUNING κ (dynamical coupling)
# ================================================================
def self_tuning_trajectory(N, omega, kappa_init=1.0,
                           adaptation_rate=0.001, D=6):
    """
    Let κ evolve dynamically:
    κ(n+1) = κ(n) - η * ∂L/∂κ

    where L = (D_local - 4)^2 is a "dimension penalty"
    and D_local is estimated from local spreading rate.

    If the system self-tunes to κ* where D=4,
    this eliminates the fine-tuning problem.
    """
    phi = np.random.uniform(0, 2*np.pi, D)
    kappa = kappa_init
    eta = adaptation_rate

    kappa_history = [kappa]
    phase_history = [phi.copy()]

    # Running estimate of local dimension via tangent vectors
    tangent_vectors = [np.random.randn(D) for _ in range(D)]
    for v in tangent_vectors:
        v /= np.linalg.norm(v)

    local_dims = []
    window = 500  # Estimation window
    recent_norms = []

    for n in range(N - 1):
        # Standard dynamics
        J = np.eye(D) + kappa * C_E6 @ np.diag(np.cos(phi))
        phi_new = (phi + omega + kappa * C_E6 @ np.sin(phi)) % (2*np.pi)

        # Evolve tangent vectors (for local dimension estimate)
        new_tangents = []
        norms = []
        for v in tangent_vectors:
            w = J @ v
            norms.append(np.log(np.linalg.norm(w)))
            w /= np.linalg.norm(w)
            new_tangents.append(w)
        tangent_vectors = new_tangents
        recent_norms.append(norms)

        # Estimate local D_KY every 'window' steps
        if len(recent_norms) >= window and n % window == 0:
            avg_norms = np.mean(recent_norms[-window:], axis=0)
            local_lyap = np.sort(avg_norms)[::-1]
            local_D = kaplan_yorke(local_lyap)
            local_dims.append(local_D)

            # Adapt κ: push toward D=4
            error = local_D - 4.0
            # Gradient: if D > 4, increase κ (more chaos → lower D)
            # if D < 4, decrease κ (less chaos → higher D)
            # But this depends on the regime!
            # Simple heuristic: use sign of error
            kappa = max(0.01, kappa + eta * error)
            kappa = min(5.0, kappa)

        phi = phi_new
        kappa_history.append(kappa)
        if n % 1000 == 0:
            phase_history.append(phi.copy())

    return (np.array(kappa_history), np.array(phase_history),
            np.array(local_dims))

# ================================================================
# MASTER v5
# ================================================================
def run_v5():
    print("╔" + "═"*65 + "╗")
    print("║  SBE TEST SUITE v5 — FALSIFICATION & UNIVERSALITY         ║")
    print("║  + All rank-6 Lie algebras + Self-tuning κ                ║")
    print("║  + Analytical κ* search + KS entropy                     ║")
    print("╚" + "═"*65 + "╝")

    N = 15000

    # ============================================================
    # PART 1: IS D=4 SPECIFIC TO E6?
    # Critical falsification test: compare all rank-6 algebras
    # ============================================================
    print("\n" + "="*65)
    print("  PART 1: FALSIFICATION — Is D=4 specific to E6?")
    print("  Testing ALL simple Lie algebras of rank 6")
    print("="*65)

    algebras = cartan_matrices_rank6()
    algebra_results = {}

    for name, cartan in algebras.items():
        print(f"\n  ─── {name} ───")

        # Coxeter-type frequencies
        omega_cox = coxeter_exponent_frequencies(name, cartan)
        unique = len(set(np.round(omega_cox, 6)))
        print(f"    ω = [{', '.join(f'{w:.3f}' for w in omega_cox)}]")
        print(f"    Unique frequencies: {unique}/6")

        # Use CARTAN MATRIX of this algebra for coupling
        # (not E6's Cartan!)
        results_kappa = []

        for kappa in np.arange(0.0, 2.05, 0.1):
            try:
                # Generate trajectory with THIS algebra's Cartan
                phi = np.random.uniform(0, 2*np.pi, 6)
                phases = np.zeros((10000, 6))
                phases[0] = phi
                for n in range(9999):
                    coupling = kappa * cartan @ np.sin(phases[n])
                    phases[n+1] = (phases[n] + omega_cox + coupling) % (2*np.pi)

                dc = correlation_dimension(phases)
                spec = full_lyapunov(10000, kappa, omega_cox)
                # But wait - full_lyapunov uses C_E6 internally!
                # We need to use the correct Cartan matrix.
            except:
                dc = np.nan
                spec = np.zeros(6)

            results_kappa.append({'kappa': kappa, 'd_corr': dc})

        dcs = [r['d_corr'] for r in results_kappa
               if not np.isnan(r['d_corr'])]
        kps = [r['kappa'] for r in results_kappa
               if not np.isnan(r['d_corr'])]

        if dcs:
            i_min = np.argmin(dcs)
            d4_dists = [abs(d-4.0) for d in dcs]
            i_4 = np.argmin(d4_dists)

            algebra_results[name] = {
                'min_D': dcs[i_min], 'min_kappa': kps[i_min],
                'closest_4': dcs[i_4], 'closest_4_kappa': kps[i_4],
                'delta_4': d4_dists[i_4],
                'all': results_kappa
            }

            print(f"    Min D = {dcs[i_min]:.2f} at κ = {kps[i_min]:.2f}")
            print(f"    Closest to 4: D = {dcs[i_4]:.2f} at κ = {kps[i_4]:.2f} "
                  f"(Δ = {d4_dists[i_4]:.2f})")
        else:
            algebra_results[name] = {'min_D': np.nan, 'delta_4': np.nan}
            print(f"    FAILED to compute")

    # ============================================================
    # PART 2: FULL LYAPUNOV WITH CORRECT CARTAN MATRICES
    # ============================================================
    print("\n" + "="*65)
    print("  PART 2: LYAPUNOV SPECTRUM WITH CORRECT CARTAN MATRICES")
    print("="*65)

    def full_lyapunov_general(N_steps, kappa, omega, cartan, D=6):
        """Full Lyapunov spectrum using arbitrary Cartan matrix."""
        phi = np.random.uniform(0, 2*np.pi, D)
        Q = np.eye(D); sums = np.zeros(D)
        ns = min(N_steps, 20000)
        for n in range(ns):
            J = np.eye(D) + kappa * cartan @ np.diag(np.cos(phi))
            Q, R = np.linalg.qr(J @ Q)
            sums += np.log(np.abs(np.diag(R)))
            phi = (phi + omega + kappa * cartan @ np.sin(phi)) % (2*np.pi)
        return np.sort(sums/ns)[::-1]

    # Compare D_KY at critical points
    print(f"\n  {'Algebra':<25s} {'κ':>5s} {'D_corr':>7s} {'D_KY':>6s} "
          f"{'h_KS':>6s} {'λ_max':>7s}")
    print("  " + "─"*60)

    for name in algebra_results:
        if np.isnan(algebra_results[name].get('min_D', np.nan)):
            continue

        cartan = algebras[name]
        omega = coxeter_exponent_frequencies(name, cartan)

        # Test at the κ closest to D=4
        kappa = algebra_results[name]['closest_4_kappa']
        dc = algebra_results[name]['closest_4']

        spec = full_lyapunov_general(20000, kappa, omega, cartan)
        dky = kaplan_yorke(spec)
        hks = ks_entropy(spec)

        algebra_results[name]['D_KY'] = dky
        algebra_results[name]['h_KS'] = hks
        algebra_results[name]['lyap_spectrum'] = spec

        print(f"  {name:<25s} {kappa:>5.2f} {dc:>7.2f} {dky:>6.2f} "
              f"{hks:>6.4f} {spec[0]:>7.4f}")

    # ============================================================
    # PART 3: FIND κ* ANALYTICALLY FOR E6
    # ============================================================
    print("\n" + "="*65)
    print("  PART 3: ANALYTICAL κ* WHERE D_KY = 4 (E6 Coxeter)")
    print("="*65)

    omega_cox = 2 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12)

    # Fine scan of D_KY vs κ
    kappas_fine = np.linspace(0.01, 1.5, 100)
    dkys_fine = []
    hks_fine = []

    for k in kappas_fine:
        spec = full_lyapunov_general(20000, k, omega_cox, C_E6)
        dkys_fine.append(kaplan_yorke(spec))
        hks_fine.append(ks_entropy(spec))

    dkys_fine = np.array(dkys_fine)

    # Find all κ where D_KY ≈ 4
    mask_4 = np.abs(dkys_fine - 4.0) < 0.3
    if np.any(mask_4):
        kappa_4_range = kappas_fine[mask_4]
        print(f"  D_KY ∈ [3.7, 4.3] for κ ∈ [{kappa_4_range[0]:.3f}, "
              f"{kappa_4_range[-1]:.3f}]")
        print(f"  Width of D=4 plateau: Δκ = {kappa_4_range[-1]-kappa_4_range[0]:.3f}")

    # Find exact κ* (closest to D_KY = 4)
    idx_best = np.argmin(np.abs(dkys_fine - 4.0))
    kappa_star = kappas_fine[idx_best]
    dky_star = dkys_fine[idx_best]

    print(f"  κ* = {kappa_star:.4f}, D_KY(κ*) = {dky_star:.4f}")

    # Check if κ* = 1/4
    print(f"  κ* / (1/4) = {kappa_star / 0.25:.4f}")
    print(f"  κ* * h = {kappa_star * 12:.4f} (h=12 is Coxeter number)")
    print(f"  κ* * |W(E6)| = {kappa_star * 51840:.1f}")

    # ============================================================
    # PART 4: SELF-TUNING κ
    # ============================================================
    print("\n" + "="*65)
    print("  PART 4: SELF-TUNING κ (does it converge to κ*?)")
    print("="*65)

    for kappa_init in [0.1, 0.5, 1.0, 2.0]:
        print(f"\n  Starting from κ_init = {kappa_init}...")
        kh, ph, ld = self_tuning_trajectory(
            50000, omega_cox, kappa_init=kappa_init,
            adaptation_rate=0.002)

        # Final κ
        kappa_final = np.mean(kh[-5000:])
        kappa_std = np.std(kh[-5000:])
        final_D = np.mean(ld[-5:]) if len(ld) >= 5 else np.nan

        print(f"    κ_final = {kappa_final:.4f} ± {kappa_std:.4f}")
        print(f"    D_local(final) = {final_D:.2f}")
        print(f"    Converged to κ*={kappa_star:.3f}? "
              f"{'YES' if abs(kappa_final-kappa_star)<0.1 else 'NO'}")

    # ============================================================
    # PART 5: STABILITY OF D=4 (enlarged)
    # ============================================================
    print("\n" + "="*65)
    print("  PART 5: PRECISION MEASUREMENT OF D_corr AT κ*")
    print("="*65)

    d_corrs_20 = []
    for run in range(20):
        np.random.seed(run * 97 + 13)
        phases = generate_trajectory(15000, kappa_star, omega_cox)
        dc = correlation_dimension(phases)
        d_corrs_20.append(dc)
        if (run+1) % 5 == 0:
            print(f"    Runs 1-{run+1}: D_corr = {np.mean(d_corrs_20):.4f} "
                  f"± {np.std(d_corrs_20):.4f}")

    d_corrs_20 = np.array(d_corrs_20)
    mean_D = np.mean(d_corrs_20)
    std_D = np.std(d_corrs_20)
    sem_D = std_D / np.sqrt(len(d_corrs_20))

    print(f"\n  FINAL RESULT (20 runs):")
    print(f"  D_corr = {mean_D:.4f} ± {std_D:.4f}")
    print(f"  Standard error of mean: {sem_D:.4f}")
    print(f"  95% CI: [{mean_D - 1.96*sem_D:.4f}, {mean_D + 1.96*sem_D:.4f}]")
    print(f"  |D - 4.0| = {abs(mean_D - 4.0):.4f}")
    print(f"  |D - 4.0| / σ = {abs(mean_D - 4.0) / std_D:.2f} "
          f"({'CONSISTENT with 4.0' if abs(mean_D - 4.0) < 2*std_D else 'INCONSISTENT'})")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. D_KY vs κ (E6 Coxeter) — high resolution
    ax = axes[0,0]
    ax.plot(kappas_fine, dkys_fine, '-', color='navy', lw=2)
    ax.axhline(4, ls='--', color='red', alpha=0.7, label='D=4 target')
    ax.axvline(kappa_star, ls=':', color='green', alpha=0.7,
               label=f'κ*={kappa_star:.3f}')
    if np.any(mask_4):
        ax.axvspan(kappa_4_range[0], kappa_4_range[-1],
                   alpha=0.1, color='green', label='D≈4 region')
    ax.set_xlabel('κ'); ax.set_ylabel('D_KY')
    ax.set_title('E6 Coxeter: D_KY(κ)\nHigh-resolution scan')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. KS entropy vs κ
    ax = axes[0,1]
    ax.plot(kappas_fine, hks_fine, '-', color='darkred', lw=2)
    ax.axvline(kappa_star, ls=':', color='green', alpha=0.7)
    ax.set_xlabel('κ'); ax.set_ylabel('h_KS')
    ax.set_title('Kolmogorov-Sinai Entropy')
    ax.grid(True, alpha=0.3)

    # 3. Falsification: all algebras
    ax = axes[0,2]
    names = list(algebra_results.keys())
    deltas = [algebra_results[n].get('delta_4', np.nan) for n in names]
    colors_bar = ['green' if d < 0.2 else 'orange' if d < 0.5
                  else 'red' for d in deltas]
    bars = ax.barh(range(len(names)), deltas, color=colors_bar, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0.2, ls='--', color='green', alpha=0.5, label='Δ<0.2')
    ax.set_xlabel('|D_closest - 4.0|')
    ax.set_title('Falsification: Which algebras give D≈4?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Self-tuning κ(t) for different initial conditions
    ax = axes[1,0]
    for kappa_init in [0.1, 0.5, 1.0, 2.0]:
        kh, _, _ = self_tuning_trajectory(
            30000, omega_cox, kappa_init=kappa_init,
            adaptation_rate=0.002)
        ax.plot(kh[::100], alpha=0.7, label=f'κ₀={kappa_init}')
    ax.axhline(kappa_star, ls='--', color='red', alpha=0.7,
               label=f'κ*={kappa_star:.3f}')
    ax.set_xlabel('Step (×100)'); ax.set_ylabel('κ(t)')
    ax.set_title('Self-tuning: does κ → κ*?')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. D_corr histogram (20 runs)
    ax = axes[1,1]
    ax.hist(d_corrs_20, bins=10, color='navy', alpha=0.7, edgecolor='white')
    ax.axvline(4.0, ls='--', color='red', lw=2, label='D=4.0')
    ax.axvline(mean_D, ls='-', color='green', lw=2,
               label=f'mean={mean_D:.3f}')
    ax.axvspan(mean_D - 2*std_D, mean_D + 2*std_D,
               alpha=0.1, color='green')
    ax.set_xlabel('D_corr'); ax.set_ylabel('Count')
    ax.set_title(f'D_corr distribution (N=20 runs)\n'
                 f'{mean_D:.4f} ± {std_D:.4f}')
    ax.legend(fontsize=9)

    # 6. Phase diagram: all frequency families
    ax = axes[1,2]
    omega_families = {
        'Coxeter': 2*np.sin(np.pi*np.array([1,4,5,7,8,11])/12),
        'Cartan_eig': np.sort(np.linalg.eigvalsh(C_E6)),
        'Standard': np.array([np.sqrt(2),np.sqrt(3),np.sqrt(5),
                              np.sqrt(7),np.sqrt(11),np.sqrt(13)]),
    }
    for name, omega in omega_families.items():
        kks = np.arange(0.0, 1.55, 0.1)
        dcs_plot = []
        for k in kks:
            phases = generate_trajectory(8000, k, omega)
            dcs_plot.append(correlation_dimension(phases))
        ax.plot(kks, dcs_plot, 'o-', lw=2, markersize=4, label=name)
    ax.axhline(4, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('κ'); ax.set_ylabel('D_corr')
    ax.set_title('Frequency families: D(κ)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('SBE v5: Universality & Falsification Tests',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sbe_v5_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "╔" + "═"*70 + "╗")
    print("║  FINAL VERDICT v5                                             ║")
    print("╠" + "═"*70 + "╣")

    # Count how many algebras achieve D≈4
    n_D4 = sum(1 for n in algebra_results
               if algebra_results[n].get('delta_4', 99) < 0.2)
    n_total = len(algebra_results)

    verdicts = [
        f"D_corr(E6 Coxeter, κ*) = {mean_D:.4f} ± {std_D:.4f}",
        f"D_KY(E6 Coxeter, κ*) = {dky_star:.4f}",
        f"κ* = {kappa_star:.4f}",
        f"Algebras achieving D≈4: {n_D4}/{n_total}",
        f"{'E6 IS SPECIAL' if n_D4 <= 2 else 'D=4 is GENERIC for rank-6'}",
    ]

    for v in verdicts:
        print(f"║  {v:<68s} ║")

    print("╠" + "═"*70 + "╣")

    if n_D4 <= 2:
        print("║  CONCLUSION: D=4 is a SPECIFIC property of E6 algebra.       ║")
        print("║  This supports the SBE hypothesis with E6 internal structure. ║")
    else:
        print("║  CONCLUSION: D=4 is GENERIC for rank-6 algebras.             ║")
        print("║  E6 is not special — any rank-6 algebra gives D≈4.           ║")
        print("║  This WEAKENS the E6 motivation but STRENGTHENS the          ║")
        print("║  general result: rank-6 coupled maps → 4D attractors.        ║")

    print("╚" + "═"*70 + "╝")

    return {
        'algebra_results': algebra_results,
        'kappa_star': kappa_star,
        'D_KY_star': dky_star,
        'D_corr_20runs': d_corrs_20,
        'dky_fine': (kappas_fine, dkys_fine),
        'hks_fine': hks_fine
    }

if __name__ == "__main__":
    results = run_v5()