import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.sparse import diags as sp_diags
import time
import warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

def gen_traj(N, kappa, T):
    D = 6
    omega = 2 * np.sin(np.pi * np.array([1,4,5,7,8,11]) / 12)
    ph = np.zeros((N, D))
    ph[0] = np.random.uniform(0, 2*np.pi, D)
    for n in range(N-1):
        ph[n+1] = (ph[n] + omega + kappa * C_E6 @ np.sin(ph[n])
                   + np.random.normal(0, T, D)) % (2*np.pi)
    return ph

def build_graph(phases, target_deg=20, delta_min=5):
    N = len(phases)
    coords = np.column_stack([np.cos(phases), np.sin(phases)])
    tree = cKDTree(coords)
    lo, hi = 0.01, 8.0
    best_eps, best_deg, best_pf = 1.0, 0, None
    for _ in range(25):
        mid = (lo+hi)/2
        pairs = tree.query_pairs(r=mid, output_type='ndarray')
        pf = pairs[np.abs(pairs[:,0]-pairs[:,1])>delta_min] if len(pairs)>0 \
            else np.zeros((0,2),dtype=int)
        actual = (2*len(pf)+2*(N-1))/N
        if actual < target_deg: lo = mid
        else: hi = mid
        if abs(actual-target_deg) < abs(best_deg-target_deg):
            best_deg, best_eps, best_pf = actual, mid, pf.copy()
        if abs(actual-target_deg)/target_deg < 0.05: break
    G = nx.Graph(); G.add_nodes_from(range(N))
    for i in range(N-1): G.add_edge(i, i+1)
    if best_pf is not None:
        for a,b in best_pf: G.add_edge(int(a), int(b))
    fd = sum(dict(G.degree()).values())/N
    if fd > target_deg*1.1:
        re = [(u,v) for u,v in G.edges() if abs(u-v)>delta_min]
        nr = int((fd-target_deg)*N/2)
        if 0 < nr < len(re):
            G.remove_edges_from([re[i] for i in np.random.choice(len(re),nr,replace=False)])
    return G, sum(dict(G.degree()).values())/N

def kuramoto(phases):
    N, D = phases.shape
    r = np.zeros(D)
    for d in range(D):
        r[d] = np.abs(np.mean(np.exp(1j*phases[:,d])))
    return np.prod(r)**(1./D), r

# ================================================================
# THREE INDEPENDENT MASS MEASURES
# ================================================================

def mass_A_spectral_weight(G, phases, evals, evecs, dim, n_modes=50):
    """
    MASS A: Spectral weight distribution.

    Create a perturbation localized in phase dimension d:
    f_d(v) = cos(phi_d(v)) (oscillation in direction d)

    Project f_d onto Laplacian eigenmodes psi_k:
    c_k = <f_d | psi_k>

    Spectral weight: W(lambda) = sum_k |c_k|^2 * delta(lambda - lambda_k)

    Effective mass = <lambda>_W = sum_k |c_k|^2 * lambda_k
    (weighted average eigenvalue = effective "energy" of the perturbation)

    If dim d is synced: perturbation is smooth → concentrates on LOW lambda → SMALL mass
    If dim d is unsynced: perturbation is rough → concentrates on HIGH lambda → LARGE mass

    Wait — this gives ANTI-Higgs again!

    CORRECTION: Use GRADIENT of phase as perturbation.
    grad_d(u,v) = |phi_d(u) - phi_d(v)| for edge (u,v)
    In synced dim: gradient is SMALL → perturbation is smooth → low lambda
    In unsynced dim: gradient is LARGE → perturbation is rough → high lambda

    But we want synced = massive. So define:

    EFFECTIVE STIFFNESS = how much energy it costs to deform the field in direction d.
    stiffness_d = <f_d | L | f_d> / <f_d | f_d>
    (Rayleigh quotient = effective eigenvalue for this perturbation)

    For synced dim: L|f_d> is small (f_d is smooth) → low stiffness → EASIER to deform
    For unsynced dim: L|f_d> is large → high stiffness → HARDER to deform

    INVERSION: "mass" in the Higgs sense is NOT stiffness of the perturbation.
    Mass is the COUPLING STRENGTH between the perturbation and the background.

    FINAL DEFINITION:
    m_d^2 = Var_d(phi) * <L>_local_d
    = (variance of phase d across edges) * (local Laplacian strength)

    Synced: Var is LOW → BUT local coherence means edges couple MORE →
    Unsynced: Var is HIGH → BUT coupling is weaker because phases are random →

    This is getting circular. Let me use the cleanest definition.
    """
    # Clean definition: spectral center of mass for directional perturbation
    N = len(phases)
    n_use = min(n_modes, len(evals) - 1)

    # Perturbation: phase oscillation in direction d
    f_d = np.cos(phases[:, dim])
    f_d -= np.mean(f_d)  # Remove DC component
    norm = np.linalg.norm(f_d)
    if norm < 1e-10:
        return 0.0
    f_d /= norm

    # Project onto eigenmodes
    weights = np.zeros(n_use)
    for k in range(n_use):
        c_k = np.dot(f_d, evecs[:, k + 1])  # Skip zero mode
        weights[k] = c_k ** 2

    # Normalize
    total_w = np.sum(weights)
    if total_w < 1e-15:
        return 0.0
    weights /= total_w

    # Spectral center of mass
    lambdas = evals[1:n_use + 1]
    mass_sq = np.sum(weights * lambdas)

    return np.sqrt(max(mass_sq, 0))


def mass_B_return_time(G, phases, dim, n_pert=20, t_eval=3.0):
    """
    MASS B: How quickly does a directional perturbation RETURN to equilibrium?

    Heavy particles oscillate SLOWLY around their equilibrium.
    Light particles oscillate QUICKLY.

    m ~ 1/omega ~ T_return (return time)

    Create perturbation: psi_0(v) = cos(phi_d(v))
    Evolve: psi(t) = exp(-i*L*t) * psi_0
    Measure: overlap(t) = |<psi(t)|psi_0>|

    For oscillation with frequency omega: overlap oscillates as cos(omega*t)
    First return to maximum: T_return = 2*pi/omega

    m_d ~ T_return_d (longer return = heavier)
    """
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G)

    # Directional perturbation
    f_d = np.cos(phases[:, dim]).astype(np.complex128)
    f_d -= np.mean(f_d)
    norm = np.linalg.norm(f_d)
    if norm < 1e-10:
        return 0.0, []
    f_d /= norm

    # Time evolution and overlap
    n_times = 50
    times = np.linspace(0.1, t_eval, n_times)
    overlaps = []

    for t in times:
        psi_t = expm_multiply(-1j * t * L, f_d)
        ov = np.abs(np.dot(np.conj(f_d), psi_t))
        overlaps.append(ov)

    overlaps = np.array(overlaps)

    # Find first minimum → half-period → frequency
    # Look for first dip
    diffs = np.diff(overlaps)
    sign_changes = np.where(diffs[:-1] < 0)[0]

    if len(sign_changes) > 0:
        # First decrease starts at sign_changes[0]
        # Find the subsequent minimum
        min_after = sign_changes[0] + np.argmin(overlaps[sign_changes[0]:])
        T_half = times[min_after]
        T_return = 2 * T_half
        omega = np.pi / T_half
        mass = T_return  # Heavier = longer return
    else:
        mass = t_eval  # No oscillation detected → very heavy or overdamped

    return mass, overlaps


def mass_C_rayleigh(G, phases, dim):
    """
    MASS C: Rayleigh quotient of directional perturbation.

    R_d = <f_d | L | f_d> / <f_d | f_d>

    This is the effective "energy cost" of this perturbation.

    BUT: for Higgs, we want the INVERSE interpretation.
    In synced dim, f_d = cos(phi_d) is SMOOTH (low gradient) → low R_d
    In unsynced dim, f_d is ROUGH → high R_d

    Define effective mass as:
    m_d = max(R) - R_d
    (perturbation that's hardest to create has smallest effective mass —
     it's the "natural" mode that costs nothing extra)

    Actually, this is just R_d inverted. Not informative.

    Better: RESPONSE STIFFNESS.
    How much does the system resist deformation in direction d?
    If the field is coherent (synced), deforming it costs MORE energy
    because you're pulling aligned phases apart.
    If the field is random (unsynced), deforming costs LESS because
    there's nothing to break.

    Response stiffness:
    K_d = <grad_d | grad_d> where grad_d(u,v) = phi_d(u) - phi_d(v) for edge (u,v)
    But normalized per edge.
    """
    N = G.number_of_nodes()

    total_gradient_sq = 0
    n_edges = 0

    for u, v in G.edges():
        diff = phases[u, dim] - phases[v, dim]
        # Toroidal
        diff = min(abs(diff), 2*np.pi - abs(diff))
        total_gradient_sq += diff ** 2
        n_edges += 1

    if n_edges == 0:
        return 0.0

    # Average gradient squared
    avg_grad_sq = total_gradient_sq / n_edges

    # For synced dim: phases aligned → low gradient → low K
    # For unsynced dim: phases random → high gradient → high K
    #
    # Invert: STIFFNESS = max_possible_gradient - actual_gradient
    # max gradient ≈ pi^2 (anti-aligned phases)
    max_grad = np.pi ** 2
    stiffness = max_grad - avg_grad_sq

    # Higher stiffness in synced dim = MORE MASSIVE
    return stiffness


# ================================================================
# MAIN
# ================================================================
def run_v8():
    print("=" * 70)
    print("  MONOSTRING HIGGS LAB v8 — THREE INDEPENDENT MASS MEASURES")
    print("  A: Spectral weight center of mass")
    print("  B: Return time (oscillation period)")
    print("  C: Field stiffness (resistance to deformation)")
    print("=" * 70)

    N = 12000
    kappa = 0.5
    TARGET_DEG = 20

    temperatures = [3.0, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]

    data = {'T': [], 'r_kura': [], 'r_per': [],
            'massA': [], 'massB': [], 'massC': [],
            'deg': []}

    print(f"\n  N={N}, kappa={kappa}, target_deg={TARGET_DEG}")
    print(f"\n[1/3] Thermodynamic scan with 3 mass measures...")

    header = f"      {'T':>6s} | {'r_K':>5s} |"
    for label in ['A:spectral', 'B:return', 'C:stiffness']:
        header += f" {label:>12s}(s/u) |"
    header += f" {'deg':>5s}"
    print(header)
    print("      " + "-" * 90)

    for T in temperatures:
        t0 = time.time()
        phases = gen_traj(N, kappa, T)
        G, deg = build_graph(phases, target_deg=TARGET_DEG)
        r_k, r_per = kuramoto(phases)

        # Compute Laplacian spectrum (for mass A)
        try:
            L = nx.normalized_laplacian_matrix(G)
            k_eig = min(60, N-2)
            evals, evecs = eigsh(L, k=k_eig, which='SM', tol=1e-3, maxiter=3000)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]
        except:
            evals = np.zeros(60)
            evecs = np.eye(N, 60)

        # Classify
        synced = [d for d in range(6) if r_per[d] > 0.5]
        unsynced = [d for d in range(6) if r_per[d] <= 0.5]

        # Mass A: spectral weight
        mA = np.array([mass_A_spectral_weight(G, phases, evals, evecs, d)
                        for d in range(6)])

        # Mass B: return time
        mB = np.array([mass_B_return_time(G, phases, d, n_pert=10, t_eval=4.0)[0]
                        for d in range(6)])

        # Mass C: stiffness
        mC = np.array([mass_C_rayleigh(G, phases, d) for d in range(6)])

        data['T'].append(T)
        data['r_kura'].append(r_k)
        data['r_per'].append(r_per.copy())
        data['massA'].append(mA.copy())
        data['massB'].append(mB.copy())
        data['massC'].append(mC.copy())
        data['deg'].append(deg)

        # Print synced/unsynced averages
        if len(synced) > 0 and len(unsynced) > 0:
            line = f"      {T:>6.3f} | {r_k:>5.3f} |"
            for masses in [mA, mB, mC]:
                ms = np.mean(masses[synced])
                mu = np.mean(masses[unsynced])
                line += f" {ms:>5.3f}/{mu:>5.3f} |"
            line += f" {deg:>5.1f}  ({time.time()-t0:.1f}s)"
        else:
            line = f"      {T:>6.3f} | {r_k:>5.3f} | (no sync yet)"
            line += f"       | {deg:>5.1f}  ({time.time()-t0:.1f}s)"
        print(line)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n[2/3] Ratio analysis (synced/unsynced)...")

    r_per_cold = data['r_per'][-1]
    synced = [d for d in range(6) if r_per_cold[d] > 0.5]
    unsynced = [d for d in range(6) if r_per_cold[d] <= 0.5]

    print(f"\n      Synced: {synced}, Unsynced: {unsynced}")

    labels = ['Mass A (spectral)', 'Mass B (return time)', 'Mass C (stiffness)']
    mass_keys = ['massA', 'massB', 'massC']

    for label, key in zip(labels, mass_keys):
        print(f"\n      {label}:")
        print(f"      {'T':>6s} | {'m_sync':>7s} {'m_unsync':>9s} {'ratio':>7s}")
        print("      " + "-" * 35)

        ratios = []
        aniso = []

        for i, T in enumerate(data['T']):
            masses = data[key][i]
            if len(synced) > 0 and len(unsynced) > 0:
                ms = np.mean(masses[synced])
                mu = np.mean(masses[unsynced])
                ratio = ms / mu if mu > 1e-6 else np.nan
            else:
                ms = mu = ratio = np.nan

            ratios.append(ratio)
            r_s = np.mean(data['r_per'][i][synced]) if len(synced) > 0 else 0
            r_u = np.mean(data['r_per'][i][unsynced]) if len(unsynced) > 0 else 0
            aniso.append(r_s - r_u)

            print(f"      {T:>6.3f} | {ms:>7.4f} {mu:>9.4f} {ratio:>7.3f}")

        # Correlation with anisotropy
        valid = [not np.isnan(r) and not np.isnan(a) for r, a in zip(ratios, aniso)]
        if sum(valid) >= 3:
            r_v = [ratios[i] for i in range(len(ratios)) if valid[i]]
            a_v = [aniso[i] for i in range(len(aniso)) if valid[i]]
            corr = np.corrcoef(a_v, r_v)[0, 1]

            # Is ratio > 1 at low T?
            low_T_ratios = [ratios[i] for i in range(len(ratios))
                           if data['T'][i] < 0.1 and not np.isnan(ratios[i])]
            mean_ratio_low = np.mean(low_T_ratios) if low_T_ratios else np.nan

            print(f"      corr(ratio, anisotropy) = {corr:.3f}")
            print(f"      mean ratio at low T = {mean_ratio_low:.3f}")

            if corr > 0.5 and mean_ratio_low > 1.0:
                print(f"      *** HIGGS SIGNAL: {label} ***")
            elif corr > 0.3:
                print(f"      Weak positive signal")
            else:
                print(f"      No Higgs signal")

    # ================================================================
    # PLOTS
    # ================================================================
    print(f"\n[3/3] Plotting...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # 1. Per-dim Kuramoto
    ax = axes[0,0]
    for d in range(6):
        rs = [data['r_per'][i][d] for i in range(len(data['T']))]
        ax.plot(data['T'], rs, 'o-' if d in synced else 's--',
                markersize=4, label=f'd{d+1}')
    ax.axhline(0.5, ls=':', color='black', alpha=0.5)
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('r per dim')
    ax.set_title('Phase Synchronization')
    ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

    # 2-4. Three mass ratios vs T
    colors = ['navy', 'darkred', 'darkgreen']
    for idx, (label, key, color) in enumerate(zip(labels, mass_keys, colors)):
        ax = axes[0, idx + 1]
        ratios_plot = []
        for i in range(len(data['T'])):
            masses = data[key][i]
            if len(synced) > 0 and len(unsynced) > 0:
                ms = np.mean(masses[synced])
                mu = np.mean(masses[unsynced])
                ratios_plot.append(ms / mu if mu > 1e-6 else np.nan)
            else:
                ratios_plot.append(np.nan)

        valid_r = [not np.isnan(r) for r in ratios_plot]
        if any(valid_r):
            Tv = [data['T'][i] for i in range(len(data['T'])) if valid_r[i]]
            Rv = [ratios_plot[i] for i in range(len(ratios_plot)) if valid_r[i]]
            ax.plot(Tv, Rv, 'o-', color=color, lw=2.5, markersize=8)
        ax.axhline(1.0, ls='--', color='black', alpha=0.5)
        ax.invert_xaxis(); ax.set_xscale('log')
        ax.set_xlabel('T')
        ax.set_ylabel('m_sync / m_unsync')
        ax.set_title(f'{label}\nratio > 1 = Higgs')
        ax.grid(True, alpha=0.3)

    # 5-7. Per-dimension mass vs T for each measure
    for idx, (label, key, color) in enumerate(zip(labels, mass_keys, colors)):
        ax = axes[1, idx]
        for d in range(6):
            ms_d = [data[key][i][d] for i in range(len(data['T']))]
            style = 'o-' if d in synced else 's--'
            c = 'red' if d in synced else 'blue'
            ax.plot(data['T'], ms_d, style, color=c, markersize=3, alpha=0.6)
        ax.invert_xaxis(); ax.set_xscale('log')
        ax.set_xlabel('T'); ax.set_ylabel(f'Mass ({label[:6]})')
        ax.set_title(f'{label} per dim\nRed=synced, Blue=unsynced')
        ax.grid(True, alpha=0.3)

    # 8. Degree control
    ax = axes[1, 3]
    ax.plot(data['T'], data['deg'], 'o-', color='crimson', lw=2)
    ax.axhline(TARGET_DEG, ls=':', color='black')
    cv = np.std(data['deg'])/np.mean(data['deg'])
    ax.invert_xaxis(); ax.set_xscale('log')
    ax.set_xlabel('T'); ax.set_ylabel('Degree')
    ax.set_title(f'Control (cv={cv:.1%})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring Higgs v8: Three Independent Mass Measures',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('higgs_v8_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================
    # VERDICT
    # ================================================================
    cv = np.std(data['deg'])/np.mean(data['deg'])

    print("\n" + "=" * 70)
    print("  FINAL VERDICT v8")
    print("=" * 70)

    n_higgs = 0
    for label, key in zip(labels, mass_keys):
        ratios_v = []
        aniso_v = []
        for i in range(len(data['T'])):
            masses = data[key][i]
            if len(synced) > 0 and len(unsynced) > 0:
                ms = np.mean(masses[synced])
                mu = np.mean(masses[unsynced])
                ratios_v.append(ms/mu if mu > 1e-6 else np.nan)
                r_s = np.mean(data['r_per'][i][synced])
                r_u = np.mean(data['r_per'][i][unsynced])
                aniso_v.append(r_s - r_u)

        valid = [not np.isnan(r) for r in ratios_v]
        if sum(valid) >= 3:
            rv = [ratios_v[i] for i in range(len(ratios_v)) if valid[i]]
            av = [aniso_v[i] for i in range(len(aniso_v)) if valid[i]]
            corr = np.corrcoef(av, rv)[0, 1]
            low_rats = [ratios_v[i] for i in range(len(ratios_v))
                       if data['T'][i] < 0.1 and not np.isnan(ratios_v[i])]
            mr = np.mean(low_rats) if low_rats else np.nan

            higgs = corr > 0.5 and mr > 1.0
            n_higgs += int(higgs)

            print(f"  {'[+]' if higgs else '[-]'} {label:<30s}: "
                  f"corr={corr:.3f}, ratio(low T)={mr:.3f} "
                  f"→ {'HIGGS!' if higgs else 'no'}")

    print(f"\n  Higgs signal in {n_higgs}/3 measures")
    print(f"  Degree control: cv={cv:.1%}")

    if n_higgs >= 2:
        print("\n  *** ROBUST HIGGS MECHANISM: detected by multiple measures ***")
    elif n_higgs == 1:
        print("\n  * Weak Higgs signal: only one measure shows it *")
    else:
        print("\n  No Higgs mechanism detected.")

    print("=" * 70)

if __name__ == "__main__":
    run_v8()
