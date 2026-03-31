import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time, random, warnings
warnings.filterwarnings("ignore")

# ================================================================
# CARTAN MATRICES FOR MULTIPLE RANKS
# ================================================================
def cartan_An(n):
    """SU(n+1) Cartan matrix, rank n."""
    C = np.zeros((n, n))
    for i in range(n): C[i,i] = 2
    for i in range(n-1): C[i,i+1] = -1; C[i+1,i] = -1
    return C

def cartan_Dn(n):
    """SO(2n) Cartan matrix, rank n (n >= 3)."""
    C = np.zeros((n, n))
    for i in range(n): C[i,i] = 2
    for i in range(n-2): C[i,i+1] = -1; C[i+1,i] = -1
    C[n-3,n-1] = -1; C[n-1,n-3] = -1
    return C

def cartan_E(n):
    """E6, E7, E8 Cartan matrices."""
    if n == 6:
        return np.array([
            [ 2,-1, 0, 0, 0, 0],
            [-1, 2,-1, 0, 0, 0],
            [ 0,-1, 2,-1, 0,-1],
            [ 0, 0,-1, 2,-1, 0],
            [ 0, 0, 0,-1, 2, 0],
            [ 0, 0,-1, 0, 0, 2]
        ], dtype=np.float64)
    elif n == 7:
        return np.array([
            [ 2,-1, 0, 0, 0, 0, 0],
            [-1, 2,-1, 0, 0, 0, 0],
            [ 0,-1, 2,-1, 0, 0,-1],
            [ 0, 0,-1, 2,-1, 0, 0],
            [ 0, 0, 0,-1, 2,-1, 0],
            [ 0, 0, 0, 0,-1, 2, 0],
            [ 0, 0,-1, 0, 0, 0, 2]
        ], dtype=np.float64)
    elif n == 8:
        return np.array([
            [ 2,-1, 0, 0, 0, 0, 0, 0],
            [-1, 2,-1, 0, 0, 0, 0, 0],
            [ 0,-1, 2,-1, 0, 0, 0,-1],
            [ 0, 0,-1, 2,-1, 0, 0, 0],
            [ 0, 0, 0,-1, 2,-1, 0, 0],
            [ 0, 0, 0, 0,-1, 2,-1, 0],
            [ 0, 0, 0, 0, 0,-1, 2, 0],
            [ 0, 0,-1, 0, 0, 0, 0, 2]
        ], dtype=np.float64)

def coxeter_data_all():
    """
    Coxeter exponents and numbers for various algebras.
    Returns dict: name -> (rank, coxeter_number, exponents, cartan)
    """
    data = {}

    # A-series: SU(n+1), rank n
    for r in [3, 4, 5, 6, 7, 8, 10, 12]:
        h = r + 1
        exponents = list(range(1, r + 1))
        data[f'A{r}_SU{r+1}'] = (r, h, exponents, cartan_An(r))

    # D-series: SO(2n), rank n
    for r in [4, 5, 6, 7, 8]:
        h = 2 * (r - 1)
        # D_n exponents: 1, 3, 5, ..., 2n-3, n-1
        exponents = list(range(1, 2*r-2, 2)) + [r-1]
        exponents = sorted(exponents)[:r]
        data[f'D{r}_SO{2*r}'] = (r, h, exponents, cartan_Dn(r))

    # Exceptional
    data['E6'] = (6, 12, [1,4,5,7,8,11], cartan_E(6))
    data['E7'] = (7, 18, [1,5,7,9,11,13,17], cartan_E(7))
    data['E8'] = (8, 30, [1,7,11,13,17,19,23,29], cartan_E(8))

    return data

def coxeter_frequencies(exponents, h):
    """ω_k = 2 sin(π m_k / h)"""
    return 2 * np.sin(np.pi * np.array(exponents) / h)

# ================================================================
# LYAPUNOV ANALYSIS
# ================================================================
def full_lyapunov_general(N_steps, kappa, omega, cartan):
    """Full Lyapunov spectrum for arbitrary Cartan matrix."""
    D = cartan.shape[0]
    phi = np.random.uniform(0, 2*np.pi, D)
    Q = np.eye(D)
    sums = np.zeros(D)
    ns = min(N_steps, 25000)
    for n in range(ns):
        J = np.eye(D) + kappa * cartan @ np.diag(np.cos(phi))
        Q, R = np.linalg.qr(J @ Q)
        sums += np.log(np.abs(np.diag(R)))
        phi = (phi + omega + kappa * cartan @ np.sin(phi)) % (2*np.pi)
    return np.sort(sums / ns)[::-1]

def kaplan_yorke(spectrum):
    s = np.sort(spectrum)[::-1]
    cs = np.cumsum(s)
    j = 0
    for i in range(len(s)):
        if cs[i] >= 0: j = i + 1
        else: break
    if j == 0: return 0.0
    if j >= len(s): return float(len(s))
    return j + cs[j-1] / abs(s[j])

def correlation_dimension(phases, max_pairs=100000):
    N = len(phases)
    ia = np.random.randint(0, N, min(max_pairs, N*(N-1)//2))
    ib = np.random.randint(0, N, len(ia))
    m = ia != ib; ia, ib = ia[m], ib[m]
    diff = np.abs(phases[ia] - phases[ib])
    diff = np.minimum(diff, 2*np.pi - diff)
    dists = np.sqrt(np.sum(diff**2, axis=1))
    rv = np.logspace(np.log10(np.percentile(dists, 1)),
                     np.log10(np.percentile(dists, 90)), 60)
    Cr = np.array([np.mean(dists < r) for r in rv])
    v = (Cr > 0.05) & (Cr < 0.40)
    if np.sum(v) < 5: v = (Cr > 0.01) & (Cr < 0.5)
    if np.sum(v) < 5: return np.nan
    return np.polyfit(np.log(rv[v]), np.log(Cr[v]), 1)[0]

# ================================================================
# MAIN ANALYSIS: D_KY vs RANK
# ================================================================
def rank_analysis():
    print("╔" + "═"*70 + "╗")
    print("║  CRITICAL TEST: Does D_min depend on rank?                    ║")
    print("║  Hypothesis A (trivial): D_min ≈ rank - 2                     ║")
    print("║  Hypothesis B (non-trivial): D_min ≈ 4 for all ranks         ║")
    print("╚" + "═"*70 + "╝")

    algebras = coxeter_data_all()

    kappa_range = np.arange(0.0, 2.05, 0.1)
    results = {}

    print(f"\n{'Algebra':<20s} {'rank':>4s} {'h':>4s} "
          f"{'#unique ω':>9s} {'D_KY min':>9s} {'κ(min)':>7s} "
          f"{'D_KY≈4 κ':>9s} {'Δ from 4':>9s}")
    print("─" * 80)

    for name in sorted(algebras.keys(),
                       key=lambda x: (algebras[x][0], x)):
        rank, h, exponents, cartan = algebras[name]
        omega = coxeter_frequencies(exponents, h)
        n_unique = len(set(np.round(omega, 6)))

        dkys = []
        for kappa in kappa_range:
            try:
                spec = full_lyapunov_general(20000, kappa, omega, cartan)
                dkys.append(kaplan_yorke(spec))
            except:
                dkys.append(float(rank))

        dkys = np.array(dkys)

        # Find minimum D_KY (excluding κ=0 where D_KY=rank trivially)
        nonzero_mask = kappa_range > 0.05
        if np.any(nonzero_mask):
            valid_dkys = dkys[nonzero_mask]
            valid_kappas = kappa_range[nonzero_mask]
            i_min = np.argmin(valid_dkys)
            dky_min = valid_dkys[i_min]
            kappa_min = valid_kappas[i_min]
        else:
            dky_min = float(rank)
            kappa_min = 0.0

        # Find κ closest to D_KY = 4
        diffs_to_4 = np.abs(dkys - 4.0)
        i_4 = np.argmin(diffs_to_4)
        delta_4 = diffs_to_4[i_4]
        kappa_4 = kappa_range[i_4]

        results[name] = {
            'rank': rank, 'h': h, 'n_unique': n_unique,
            'dky_min': dky_min, 'kappa_min': kappa_min,
            'delta_4': delta_4, 'kappa_4': kappa_4,
            'dkys': dkys, 'kappas': kappa_range
        }

        print(f"{name:<20s} {rank:>4d} {h:>4d} "
              f"{n_unique:>9d} {dky_min:>9.2f} {kappa_min:>7.2f} "
              f"{kappa_4:>9.2f} {delta_4:>9.2f}")

    # ============================================================
    # KEY ANALYSIS: D_min vs rank
    # ============================================================
    print("\n" + "="*70)
    print("  KEY ANALYSIS: D_KY(min) vs rank")
    print("="*70)

    ranks_simple = []
    dky_mins_simple = []
    names_simple = []

    for name, data in results.items():
        # Only simple algebras of A and D series (clean comparison)
        if name.startswith('A') or name.startswith('D') or name.startswith('E'):
            ranks_simple.append(data['rank'])
            dky_mins_simple.append(data['dky_min'])
            names_simple.append(name)

    ranks_arr = np.array(ranks_simple)
    dky_arr = np.array(dky_mins_simple)

    # Test hypothesis A: D_min = a * rank + b
    if len(ranks_arr) >= 3:
        coeffs = np.polyfit(ranks_arr, dky_arr, 1)
        a, b = coeffs
        residuals = dky_arr - np.polyval(coeffs, ranks_arr)
        r_squared = 1 - np.sum(residuals**2) / np.sum((dky_arr - np.mean(dky_arr))**2)

        print(f"\n  Linear fit: D_min = {a:.3f} * rank + ({b:.3f})")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Prediction: D_min(rank=6) = {a*6+b:.2f}")

        # Test hypothesis A: D = rank - 2
        residuals_A = dky_arr - (ranks_arr - 2)
        rms_A = np.sqrt(np.mean(residuals_A**2))
        print(f"\n  Hypothesis A (D = rank - 2):")
        print(f"    RMS residual = {rms_A:.3f}")

        # Test hypothesis B: D = 4 constant
        residuals_B = dky_arr - 4.0
        rms_B = np.sqrt(np.mean(residuals_B**2))
        print(f"\n  Hypothesis B (D = 4 always):")
        print(f"    RMS residual = {rms_B:.3f}")

        # Test hypothesis C: D = 2/3 * rank
        residuals_C = dky_arr - (2/3 * ranks_arr)
        rms_C = np.sqrt(np.mean(residuals_C**2))
        print(f"\n  Hypothesis C (D = 2r/3):")
        print(f"    RMS residual = {rms_C:.3f}")

        best = min([('A: D=r-2', rms_A), ('B: D=4', rms_B),
                     ('C: D=2r/3', rms_C),
                     (f'Linear: D={a:.2f}r+{b:.2f}', np.sqrt(np.mean(residuals**2)))],
                    key=lambda x: x[1])
        print(f"\n  BEST FIT: {best[0]} (RMS = {best[1]:.3f})")

    # ============================================================
    # DETAILED COMPARISON FOR EACH RANK
    # ============================================================
    print("\n" + "="*70)
    print("  D_KY(min) GROUPED BY RANK")
    print("="*70)

    for rank in sorted(set(r['rank'] for r in results.values())):
        algebras_this_rank = {n: d for n, d in results.items()
                              if d['rank'] == rank}
        dkys = [d['dky_min'] for d in algebras_this_rank.values()]
        print(f"\n  Rank {rank}: {len(algebras_this_rank)} algebras")
        for name, data in sorted(algebras_this_rank.items()):
            print(f"    {name:<20s} D_min={data['dky_min']:.2f} "
                  f"at κ={data['kappa_min']:.2f}")
        print(f"    Mean D_min = {np.mean(dkys):.2f} ± {np.std(dkys):.2f}")

    # ============================================================
    # PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1. D_KY(κ) for all algebras, colored by rank
    ax = axes[0]
    cmap = plt.cm.viridis
    all_ranks = sorted(set(r['rank'] for r in results.values()))
    rank_colors = {r: cmap(i / len(all_ranks))
                   for i, r in enumerate(all_ranks)}

    for name, data in results.items():
        rank = data['rank']
        ax.plot(data['kappas'], data['dkys'], '-',
                color=rank_colors[rank], alpha=0.6, lw=1.5)
        # Mark minimum
        ax.plot(data['kappa_min'], data['dky_min'], 'o',
                color=rank_colors[rank], markersize=6)

    ax.axhline(4, ls='--', color='red', alpha=0.7, label='D=4')
    ax.set_xlabel('κ', fontsize=12)
    ax.set_ylabel('D_KY', fontsize=12)
    ax.set_title('D_KY(κ) for all algebras\n(color = rank)')

    # Add rank legend
    for r in all_ranks:
        ax.plot([], [], '-', color=rank_colors[r], lw=3, label=f'rank {r}')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. D_min vs rank
    ax = axes[1]
    for name, data in results.items():
        marker = 'o' if name.startswith('A') else \
                 's' if name.startswith('D') else \
                 'D' if name.startswith('E') else '^'
        ax.plot(data['rank'], data['dky_min'], marker, markersize=10,
                label=name, alpha=0.8)

    # Plot hypothesis lines
    r_line = np.linspace(2.5, 13, 100)
    ax.plot(r_line, r_line - 2, 'r--', lw=2, alpha=0.5, label='D = r-2')
    ax.plot(r_line, np.full_like(r_line, 4.0), 'g--', lw=2,
            alpha=0.5, label='D = 4')
    ax.plot(r_line, 2/3 * r_line, 'b--', lw=2, alpha=0.5, label='D = 2r/3')
    if len(ranks_arr) >= 3:
        ax.plot(r_line, np.polyval(coeffs, r_line), 'k-', lw=2,
                alpha=0.7, label=f'Fit: D={a:.2f}r+{b:.2f}')

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('D_KY(min)', fontsize=12)
    ax.set_title('Minimum attractor dimension vs rank')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. |D-4| vs rank (how close to 4)
    ax = axes[2]
    for name, data in results.items():
        marker = 'o' if name.startswith('A') else \
                 's' if name.startswith('D') else \
                 'D' if name.startswith('E') else '^'
        ax.plot(data['rank'], data['delta_4'], marker, markersize=10,
                label=name, alpha=0.8)

    ax.axhline(0.3, ls=':', color='green', alpha=0.5, label='Δ=0.3 threshold')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('|D_closest - 4.0|', fontsize=12)
    ax.set_title('How close does each algebra get to D=4?')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('SBE v6: Does D=4 depend on rank?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sbe_v6_rank_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "╔" + "═"*70 + "╗")
    print("║  CRITICAL RESULT: D_min vs RANK                              ║")
    print("╠" + "═"*70 + "╣")

    if len(ranks_arr) >= 3:
        if rms_A < rms_B and rms_A < rms_C:
            print("║  D_min = rank - 2 (trivial scaling)                           ║")
            print("║  → D=4 from rank 6 is NOT special, just r-2                   ║")
            print("║  → SBE needs rank=6 assumption, nothing more                  ║")
        elif rms_B < rms_A and rms_B < rms_C:
            print("║  D_min ≈ 4 INDEPENDENT OF RANK (non-trivial!)                 ║")
            print("║  → D=4 is a UNIVERSAL ATTRACTOR of coupled map dynamics       ║")
            print("║  → This is a NEW result in dynamical systems theory            ║")
        else:
            print(f"║  D_min ≈ {a:.2f} * rank + {b:.2f} (intermediate scaling)"
                  + " "*(70-55) + "║")
            print("║  → Partial support for SBE                                    ║")

    print("╚" + "═"*70 + "╝")

    return results

if __name__ == "__main__":
    results = rank_analysis()
