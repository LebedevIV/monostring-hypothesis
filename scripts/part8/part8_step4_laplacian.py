"""
Part VIII — Step 4: Laplacian Field Spectrum
=============================================
Question: What field theory does the E6 graph support?

Method:
1. Build weighted k-NN graph of E6 orbit
2. Compute graph Laplacian L = D - A
3. Find eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₙ
4. These ARE the field frequencies: ω²ₙ = λₙ
5. Fit dispersion: λₙ vs n (mode index)

Null hypothesis H0:
  λₙ ∝ n²/³ (random graph, diffusion-like)

H1 (massless field):
  λₙ ∝ n²/D for D-dimensional lattice
  (D=3: λ ∝ n^{2/3})

H2 (massive field):
  λₙ = m² + c²·f(n) with gap at n=0

H3 (E6 special):
  Eigenvalue spectrum has E6-specific degeneracies
  → "mass multiplets" matching Lie algebra structure

KEY: Compare E6 vs Random spectrum shape
  If spectra are identical → algebra irrelevant
  If E6 has special structure → algebra matters
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np  # Ensure imported at top
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# 1. ORBIT GENERATION
# ══════════════════════════════════════════════════════════════════

ALGEBRAS_STEP4 = {
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': [1, 4, 5, 7, 8, 11],
        'color': '#2196F3',
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
        'color': '#4CAF50',
    },
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': [1, 2, 3, 4, 5, 6],
        'color': '#9C27B0',
    },
    'Random': {
        'rank': 6, 'h': None,
        'exponents': None,
        'color': '#9E9E9E',
    },
}


def get_omega(alg_name, seed=42):
    alg = ALGEBRAS_STEP4[alg_name]
    if alg['exponents'] is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.5, 2.0, alg['rank'])
    m = np.array(alg['exponents'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])


def generate_orbit(omega, T=2000, kappa=0.05,
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
# 2. BUILD WEIGHTED LAPLACIAN
# ══════════════════════════════════════════════════════════════════

def build_laplacian(orbit, k=12, weight_type='gaussian'):
    """
    Build weighted graph Laplacian of orbit.

    weight_type:
    - 'gaussian':  w_ij = exp(-d²/σ²)  [standard]
    - 'inverse':   w_ij = 1/d          [simpler]
    - 'binary':    w_ij = 1            [unweighted]

    Returns: (L_sparse, sigma, n_edges)
    """
    T, rank = orbit.shape

    # k-NN on torus (wrap-around distance)
    # Approximation: use Euclidean on unwrapped coords
    # (good enough for dense orbit)
    nbrs = NearestNeighbors(n_neighbors=k+1,
                             algorithm='ball_tree',
                             metric='euclidean')
    nbrs.fit(orbit)
    distances, indices = nbrs.kneighbors(orbit)

    # Bandwidth: median distance
    sigma = np.median(distances[:, 1:k+1])

    rows, cols, weights = [], [], []

    for i in range(T):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]

            if weight_type == 'gaussian':
                w = np.exp(-d**2 / (2*sigma**2))
            elif weight_type == 'inverse':
                w = 1.0 / (d + 1e-10)
            else:
                w = 1.0

            rows.append(i); cols.append(j); weights.append(w)
            rows.append(j); cols.append(i); weights.append(w)

    # Build sparse adjacency matrix
    A = csr_matrix((weights, (rows, cols)),
                   shape=(T, T))

    # Normalize (symmetrize in case of duplicates)
    A = (A + A.T) / 2

    # Degree matrix
    deg = np.array(A.sum(axis=1)).flatten()
    D   = diags(deg)

    # Laplacian L = D - A
    L = D - A

    n_edges = len(set(zip(rows, cols))) // 2

    return L, sigma, n_edges


# ══════════════════════════════════════════════════════════════════
# 3. EIGENVALUE SPECTRUM
# ══════════════════════════════════════════════════════════════════

def compute_spectrum(L, n_eigs=200, seed=42):
    """
    Compute smallest n_eigs eigenvalues of L.

    λ₀ = 0 always (constant mode)
    λ₁ = Fiedler value (algebraic connectivity)
    λₙ ≈ 2*d_max for large n

    Returns: eigenvalues (sorted ascending)
    """
    T = L.shape[0]
    n_eigs = min(n_eigs, T - 2)

    # eigsh finds k smallest eigenvalues
    # sigma=0 → shift-invert for smallest
    vals, vecs = eigsh(L, k=n_eigs,
                   which='SM',
                   tol=1e-6,
                   maxiter=5000,
                   v0=np.random.RandomState(seed).rand(L.shape[0]))

    vals = np.sort(np.real(vals))
    vals = np.maximum(vals, 0)  # numerical cleanup

    return vals, np.real(vecs)


# ══════════════════════════════════════════════════════════════════
# 4. DISPERSION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyse_dispersion(vals, label=""):
    """
    Fit dispersion relation λₙ vs n.

    In D-dimensional space:
      λₙ ∝ n^{2/D}

    So: log(λₙ) = (2/D) log(n) + const
    → Slope in log-log gives D

    Models:
    1. Power law: λ = a·nᵅ    (continuous spectrum)
    2. Mass gap:  λ = m² + a·nᵅ  (gapped spectrum)
    3. Linear:    λ = a·n     (1D chain)
    """
    # Skip λ₀ = 0 (constant mode)
    vals_nonzero = vals[vals > 1e-10]
    n = np.arange(1, len(vals_nonzero) + 1)

    print(f"\n  [{label}] Dispersion analysis:")
    print(f"  Total eigenvalues: {len(vals)}")
    print(f"  Non-zero:          {len(vals_nonzero)}")
    print(f"  Fiedler value λ₁:  "
          f"{vals_nonzero[0]:.6f}")
    print(f"  Max eigenvalue:    "
          f"{vals_nonzero[-1]:.4f}")

    results = {}

    # Log-log fit: log(λ) = α·log(n) + c
    log_n   = np.log(n)
    log_lam = np.log(vals_nonzero + 1e-15)

    # Skip first few (might be special modes)
    fit_start = max(3, len(n)//10)
    sl, ic, r, p, se = stats.linregress(
        log_n[fit_start:],
        log_lam[fit_start:])

    alpha = sl  # power law exponent
    D_eff = 2.0 / alpha  # implied dimension

    print(f"\n  Power law fit (n > {fit_start}):")
    print(f"  λₙ ∝ nᵅ  with α = {alpha:.4f}")
    print(f"  R² = {r**2:.4f}  (p={p:.2e})")
    print(f"  → Implied dimension D = 2/α = {D_eff:.3f}")

    if abs(D_eff - 3.0) < 0.3:
        print(f"  → CONSISTENT WITH 3D SPACE ✓")
    elif abs(D_eff - 4.0) < 0.3:
        print(f"  → CONSISTENT WITH 4D SPACETIME ✓")
    elif abs(D_eff - 1.0) < 0.2:
        print(f"  → 1D CHAIN (no dimensional emergence)")
    else:
        print(f"  → D_eff = {D_eff:.1f} "
              f"(no simple interpretation)")

    results['alpha']  = alpha
    results['D_eff']  = D_eff
    results['r2']     = r**2
    results['ic']     = ic
    results['n']      = n
    results['vals_nz']= vals_nonzero
    results['fit_start'] = fit_start

    # Check for mass gap
    # Gap = ratio λ₁/λ₂
    if len(vals_nonzero) >= 2:
        gap = vals_nonzero[1] / vals_nonzero[0]
        results['gap'] = gap
        print(f"\n  Spectral gap ratio λ₂/λ₁ = {gap:.4f}")
        if gap > 3.0:
            print(f"  → Large gap: possible mass gap "
                  f"(isolated zero mode)")
        elif gap < 1.5:
            print(f"  → Small gap: continuous spectrum "
                  f"(gapless field)")
        else:
            print(f"  → Moderate gap")

    # Weyl law check
    # In D dimensions: N(λ) ∝ λ^{D/2}
    # If cumulative count N(λ) vs λ fits power D/2
    lambdas = vals_nonzero
    N_cumul = np.arange(1, len(lambdas) + 1)

    log_lam2 = np.log(lambdas[fit_start:])
    log_N    = np.log(N_cumul[fit_start:])
    sl2, _, r2w, _, _ = stats.linregress(
        log_lam2, log_N)
    D_weyl = 2.0 * sl2

    print(f"\n  Weyl law: N(λ) ∝ λ^(D/2)")
    print(f"  Weyl exponent D_weyl = {D_weyl:.3f}  "
          f"(R²={r2w**2:.4f})")
    results['D_weyl'] = D_weyl
    results['r2_weyl'] = r2w**2

    return results


# ══════════════════════════════════════════════════════════════════
# 5. E6 STRUCTURE IN EIGENVECTORS
# ══════════════════════════════════════════════════════════════════

def check_e6_structure(vecs, vals, omega, label=""):
    """
    Test if E6 Coxeter structure appears in eigenvectors.

    E6 has exponents [1,4,5,7,8,11], h=12.
    The frequencies ω_i = 2sin(πm_i/h) create
    specific resonances.

    Test: do eigenvectors of the Laplacian
    show E6-frequency modulation?

    Method: correlate eigenvector components
    with pure tones at ω_i frequencies.
    """
    if vecs is None or len(omega) != 6:
        return {}

    # Take first 20 non-trivial eigenvectors
    T = vecs.shape[0]
    n_check = min(20, vecs.shape[1])

    # For each eigenvector, compute
    # power at each natural frequency
    t = np.arange(T)
    freq_powers = np.zeros((n_check, len(omega)))

    for mode in range(n_check):
        v = vecs[:, mode]
        v = v - v.mean()
        for fi, om in enumerate(omega):
            # Correlation with cos/sin at omega_i
            cos_corr = np.abs(
                np.dot(v, np.cos(om * t)) / T)
            sin_corr = np.abs(
                np.dot(v, np.sin(om * t)) / T)
            freq_powers[mode, fi] = (
                cos_corr**2 + sin_corr**2)

    # Which frequency dominates each mode?
    dominant_freq = np.argmax(freq_powers, axis=1)

    print(f"\n  [{label}] E6 frequency structure "
          f"in eigenvectors:")
    print(f"  Mode  DomFreq  ω_dom  power")
    for m in range(min(10, n_check)):
        fi  = dominant_freq[m]
        pow = freq_powers[m, fi]
        print(f"  {m+1:>4}  {fi+1:>7}  "
              f"{omega[fi]:.4f}  {pow:.6f}")

    return {
        'freq_powers':    freq_powers,
        'dominant_freq':  dominant_freq,
    }


# ══════════════════════════════════════════════════════════════════
# 6. COMPARISON: SPEED OF LIGHT FROM SPECTRUM
# ══════════════════════════════════════════════════════════════════

def extract_speed_of_light(disp_result, orbit,
                            k_graph=12, label=""):
    """
    In a D-dimensional graph, the "speed of light"
    for massless modes is:

    c = √(λₙ/k²ₙ) where k²ₙ ∝ n^{2/D} / L²

    L = characteristic length scale
    (= RMS distance between orbit points)

    This gives: c_eff = L × √(α-law prefactor)
    """
    # RMS scale
    T, rank = orbit.shape
    # Sample distances
    rng = np.random.default_rng(42)
    idx = rng.choice(T, size=500, replace=False)
    sub = orbit[idx]
    # Pairwise distances (subsample)
    dists = []
    for i in range(0, len(idx)-1, 5):
        d = np.linalg.norm(sub[i] - sub[i+1:i+50])
        if hasattr(d, 'tolist'):
        # Safe extension for scalars/arrays
            print(f"DEBUG: d={d}, type={type(d)}, isscalar={np.isscalar(d)}")
            if np.isscalar(d):
                dists.append(float(d))
            else:
                dists.extend(d.tolist())
        else:
             dists.append(float(d))

    L = np.median(dists) if dists else 1.0

    alpha = disp_result['alpha']
    ic    = disp_result['ic']
    D_eff = disp_result['D_eff']

    # For massless field in D dims:
    # λ_n = (2π/L)² × c² × n^{2/D}
    # → c² = exp(ic) × (L/2π)² × 1/n^0 at n=1
    c2_eff = np.exp(ic) * (L / (2*np.pi))**2
    c_eff  = np.sqrt(max(c2_eff, 0))

    print(f"\n  [{label}] Effective speed of light:")
    print(f"  Orbital scale L = {L:.4f}")
    print(f"  c_eff = {c_eff:.6f}")
    print(f"  c_eff/L = {c_eff/L:.4f}")

    return {'c_eff': c_eff, 'L': L}


# ══════════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_step4(T=2000, k_nn=12, n_eigs=150, seed=42):
    print("=" * 62)
    print("PART VIII Step 4: Laplacian Field Spectrum")
    print("=" * 62)
    print(f"T={T}, k={k_nn}, n_eigs={n_eigs}")

    all_results = {}

    for alg_name in ['E6', 'A6', 'E8', 'Random']:
        alg   = ALGEBRAS_STEP4[alg_name]
        omega = get_omega(alg_name, seed=seed)
        rank  = alg['rank']

        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}  rank={rank}  "
              f"h={alg['h']}")
        print(f"  omega = {np.round(omega, 4).tolist()}")

        # Generate orbit
        orbit = generate_orbit(omega, T=T,
                                kappa=0.05,
                                warmup=500,
                                seed=seed)
        print(f"  Orbit shape: {orbit.shape}")

        # Build Laplacian
        print(f"  Building Laplacian (k={k_nn})...")
        L, sigma, n_edges = build_laplacian(
            orbit, k=k_nn, weight_type='gaussian')
        print(f"  σ={sigma:.4f}, edges={n_edges}")

        # Compute spectrum
        print(f"  Computing {n_eigs} eigenvalues...")
        vals, vecs = compute_spectrum(
            L, n_eigs=n_eigs, seed=seed)
        print(f"  λ_min={vals[0]:.2e}, "
              f"λ_max={vals[-1]:.4f}")

        # Dispersion analysis
        disp = analyse_dispersion(vals, label=alg_name)

        # Speed of light
        c_out = extract_speed_of_light(
            disp, orbit, k_graph=k_nn,
            label=alg_name)

        # E6 structure (only for rank-6 with Coxeter)
        e6_struct = {}
        if rank == 6 and alg['exponents'] is not None:
            e6_struct = check_e6_structure(
                vecs, vals, omega, label=alg_name)

        all_results[alg_name] = {
            'omega':    omega,
            'rank':     rank,
            'orbit':    orbit,
            'vals':     vals,
            'vecs':     vecs,
            'disp':     disp,
            'c_out':    c_out,
            'e6_struct':e6_struct,
            'color':    alg['color'],
            'L':        L,
            'sigma':    sigma,
        }

    return all_results


# ══════════════════════════════════════════════════════════════════
# 8. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(all_results):
    alg_list = ['E6', 'A6', 'E8', 'Random']

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Part VIII — Step 4: Laplacian Field Spectrum\n"
        r"$\lambda_n$ vs $n$: What field theory does "
        "the E6 graph support?",
        fontsize=14, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.40)

    # ── Row 0: Raw spectrum λₙ vs n ───────────────────
    for col, alg_name in enumerate(alg_list):
        res  = all_results[alg_name]
        vals = res['vals']
        disp = res['disp']
        c    = res['color']

        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#f0f4f8')

        n = np.arange(len(vals))
        ax.plot(n, vals, '.', color=c,
                ms=3, alpha=0.7)

        # Mark Fiedler value
        fiedler_idx = np.argmax(vals > 1e-10)
        ax.axvline(fiedler_idx, color='red',
                   ls=':', lw=1.5, alpha=0.7,
                   label=f'Fiedler={vals[fiedler_idx]:.3f}')

        ax.set_xlabel('Mode index n', fontsize=9)
        ax.set_ylabel('Eigenvalue λₙ', fontsize=9)
        ax.set_title(
            f'{alg_name}: Eigenvalue spectrum\n'
            f'D_eff={disp["D_eff"]:.2f}  '
            f'D_Weyl={disp["D_weyl"]:.2f}',
            fontsize=9, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Row 1: Log-log spectrum ────────────────────────
    for col, alg_name in enumerate(alg_list):
        res  = all_results[alg_name]
        disp = res['disp']
        c    = res['color']

        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#f0f4f8')

        n_nz  = disp['n']
        lam_nz = disp['vals_nz']
        alpha  = disp['alpha']
        ic     = disp['ic']
        r2     = disp['r2']
        D_eff  = disp['D_eff']

        ax.loglog(n_nz, lam_nz, '.', color=c,
                  ms=4, alpha=0.7, label='data')

        # Power law fit line
        fit_start = disp['fit_start']
        n_fit = n_nz[fit_start:]
        lam_fit = np.exp(ic) * n_fit**alpha
        ax.loglog(n_fit, lam_fit, 'r-',
                  lw=2.5,
                  label=f'α={alpha:.3f}\nD={D_eff:.2f}\nR²={r2:.3f}')

        # Reference slopes
        n_ref = np.array([n_nz[fit_start],
                           n_nz[-1]])
        scale = np.exp(ic) * n_ref[0]**alpha / \
                n_ref[0]**(2/3)
        ax.loglog(n_ref, scale * n_ref**(2/3),
                  'g--', lw=1.5, alpha=0.6,
                  label='D=3 ref')
        scale4 = np.exp(ic) * n_ref[0]**alpha / \
                  n_ref[0]**(2/4)
        ax.loglog(n_ref, scale4 * n_ref**(2/4),
                  'b--', lw=1.5, alpha=0.6,
                  label='D=4 ref')

        ax.set_xlabel('log n', fontsize=9)
        ax.set_ylabel('log λₙ', fontsize=9)
        ax.set_title(
            f'{alg_name}: Log-log dispersion\n'
            f'λ ∝ nᵅ, D=2/α',
            fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')

    # ── Row 2: Comparison summary ──────────────────────
    ax_sum = fig.add_subplot(gs[2, 0:2])
    ax_sum.set_facecolor('#f0f4f8')

    names   = alg_list
    d_effs  = [all_results[n]['disp']['D_eff']
                for n in names]
    d_weyls = [all_results[n]['disp']['D_weyl']
                for n in names]
    r2s     = [all_results[n]['disp']['r2']
                for n in names]
    clrs    = [all_results[n]['color'] for n in names]

    x = np.arange(len(names))
    w = 0.35

    ax_sum.bar(x - w/2, d_effs, width=w,
               color=clrs, alpha=0.8,
               edgecolor='k', lw=1.2,
               label='D_eff = 2/α')
    ax_sum.bar(x + w/2, d_weyls, width=w,
               color=clrs, alpha=0.4,
               edgecolor='k', lw=1.2,
               hatch='//', label='D_Weyl')

    ax_sum.axhline(3.0, color='blue', ls='--',
                   lw=2, alpha=0.7, label='D=3')
    ax_sum.axhline(4.0, color='red', ls='--',
                   lw=2, alpha=0.7, label='D=4')

    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(names, fontsize=11)
    ax_sum.set_ylabel('Effective dimension', fontsize=11)
    ax_sum.set_title(
        'Effective dimension from Laplacian spectrum\n'
        'Solid=D_eff=2/α, Hatched=D_Weyl',
        fontsize=10, fontweight='bold')
    ax_sum.legend(fontsize=9)
    ax_sum.grid(True, alpha=0.3, axis='y')
    ax_sum.set_ylim([0, 8])

    # ── Row 2: Spectral gap ────────────────────────────
    ax_gap = fig.add_subplot(gs[2, 2:4])
    ax_gap.set_facecolor('#f0f4f8')

    gaps = []
    for n in names:
        g = all_results[n]['disp'].get('gap', 1.0)
        gaps.append(g)

    clrs2 = [all_results[n]['color'] for n in names]
    bars  = ax_gap.bar(x, gaps, color=clrs2,
                        alpha=0.8,
                        edgecolor='k', lw=1.2)

    ax_gap.axhline(3.0, color='orange', ls='--',
                   lw=2, alpha=0.7,
                   label='gap=3 (significant)')

    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(names, fontsize=11)
    ax_gap.set_ylabel('Spectral gap λ₂/λ₁', fontsize=11)
    ax_gap.set_title(
        'Spectral gap (mass gap analog)\n'
        'Large gap = gapped field (massive)',
        fontsize=10, fontweight='bold')
    ax_gap.legend(fontsize=9)
    ax_gap.grid(True, alpha=0.3, axis='y')

    plt.savefig('part8_step4_laplacian.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: part8_step4_laplacian.png")


# ══════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VIII Step 4: Laplacian Field Spectrum\n")

    all_results = run_step4(T=2000, k_nn=12,
                             n_eigs=150, seed=42)
    make_figure(all_results)

    # Final summary
    print("\n" + "="*62)
    print("SUMMARY: Step 4")
    print("="*62)

    print(f"\n{'Algebra':<10} {'D_eff':>8} "
          f"{'D_Weyl':>8} {'R²':>8} "
          f"{'Gap':>8}  verdict")
    print("-"*65)

    for alg_name in ['E6', 'A6', 'E8', 'Random']:
        res  = all_results[alg_name]
        disp = res['disp']
        gap  = disp.get('gap', float('nan'))

        D  = disp['D_eff']
        Dw = disp['D_weyl']
        r2 = disp['r2']

        if abs(D - 3.0) < 0.5 and r2 > 0.95:
            v = "3D FIELD ✓"
        elif abs(D - 4.0) < 0.5 and r2 > 0.95:
            v = "4D FIELD ✓"
        elif r2 < 0.8:
            v = "no clean dispersion ✗"
        else:
            v = f"D={D:.1f} (unusual)"

        print(f"{alg_name:<10} {D:>8.3f} "
              f"{Dw:>8.3f} {r2:>8.4f} "
              f"{gap:>8.3f}  {v}")

    print("\nPhysical interpretation:")
    print("  D_eff = 2/α where λₙ ∝ nᵅ")
    print("  D_eff ≈ 3: field lives in 3D (photon, "
          "gluon)")
    print("  D_eff ≈ 4: field lives in 4D spacetime")
    print("  Gap >> 1: massive field (Higgs-like)")
    print("  Gap ≈ 1:  massless field (photon-like)")
