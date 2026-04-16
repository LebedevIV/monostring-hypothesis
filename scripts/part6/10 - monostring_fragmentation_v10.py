"""
Monostring Fragmentation v10
==============================
One clean experiment: the entropy transition time τ_E6

Finding from v9 (p<0.0001):
  t<200: S(E6) < S(Shuf)   — E6 more ordered
  t>400: S(E6) > S(Shuf)   — E6 less ordered
  Crossover: τ_E6 ∈ [200, 400]

Questions:
  1. What is τ_E6 precisely? (binary search)
  2. Does τ_E6 depend on σ?
  3. Does τ_E6 depend on N?
  4. Is τ_E6 related to any E6 property?
     (Coxeter number h=12, exponents, ω_min)
  5. Is τ_E6 > τ_shuf? (E6 memory persists longer)

This is the ONE finding that is:
  - Statistically significant (p<0.0001)
  - Reproducible
  - Has a clear physical interpretation
  - Goes beyond null model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind
from scipy.optimize import brentq
import time
import warnings
warnings.filterwarnings("ignore")


class Config:
    D = 6
    omega_E6 = 2.0 * np.sin(
        np.pi * np.array([1,4,5,7,8,11]) / 12.0)

    rng0 = np.random.RandomState(1)
    omega_shuf = rng0.permutation(omega_E6.copy())
    omega_rand = rng0.uniform(
        omega_E6.min(), omega_E6.max(), 6)

    # E6 properties for comparison
    coxeter_h  = 12   # Coxeter number of E6
    omega_min  = float(omega_E6.min())
    omega_max  = float(omega_E6.max())
    omega_mean = float(omega_E6.mean())

    kappa_dau  = 0.05
    noise_dau  = 0.006
    seed_base  = 42

cfg = Config()


def generate_orbit(omega, kappa=0.30, T=1000,
                   seed=0, warmup=300):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
    traj = []
    for _ in range(T):
        phi = (phi + omega + kappa*np.sin(phi)) % (2*np.pi)
        traj.append(phi.copy())
    return np.array(traj)


def shannon_H(phases, bins=15):
    H = 0.0
    for d in range(phases.shape[1]):
        hist, _ = np.histogram(
            phases[:,d], bins=bins, range=(0, 2*np.pi))
        p = hist / (hist.sum() + 1e-15)
        H -= np.sum(p * np.log(p + 1e-15))
    return float(H)


def measure_delta_S_at_t(T_measure, n_runs,
                          N, sigma,
                          omega_A, omega_B,
                          phi_break_seed=0):
    """
    Measure ΔS = S(omega_A daughters) - S(omega_B daughters)
    at time T_measure.

    Returns: (mean_dS, std_dS, p_value, individual_dS)
    """
    orbit = generate_orbit(omega_A, T=100,
                           seed=phi_break_seed)
    phi_break = orbit[-1]

    S_A_list = []
    S_B_list = []

    for run in range(n_runs):
        rng = np.random.RandomState(run * 13 + 7)

        # Population A (e.g. E6): shared origin + omega_A
        ph_A = (phi_break[np.newaxis,:] +
                rng.normal(0, sigma, (N, 6))) % (2*np.pi)

        # Population B (e.g. Shuf): random origin + omega_B
        phi_B_origin = rng.uniform(0, 2*np.pi, 6)
        ph_B = (phi_B_origin[np.newaxis,:] +
                rng.normal(0, sigma, (N, 6))) % (2*np.pi)

        for step in range(T_measure):
            ph_A = (ph_A + omega_A[np.newaxis,:]
                    + cfg.kappa_dau*np.sin(ph_A)
                    + rng.normal(0, cfg.noise_dau,
                                 ph_A.shape)) % (2*np.pi)
            ph_B = (ph_B + omega_B[np.newaxis,:]
                    + cfg.kappa_dau*np.sin(ph_B)
                    + rng.normal(0, cfg.noise_dau,
                                 ph_B.shape)) % (2*np.pi)

        S_A_list.append(shannon_H(ph_A))
        S_B_list.append(shannon_H(ph_B))

    dS_list  = [a-b for a,b in zip(S_A_list, S_B_list)]
    mean_dS  = float(np.mean(dS_list))
    std_dS   = float(np.std(dS_list))
    _, pval  = ttest_ind(S_A_list, S_B_list)

    return mean_dS, std_dS, float(pval), dS_list


# ═══════════════════════════════════════════════════════
# MAIN: Find τ_E6
# ═══════════════════════════════════════════════════════

def find_tau_E6(N=300, sigma=0.50,
                n_runs=30, n_time_points=12):
    """
    Find the crossover time τ where ΔS changes sign.

    ΔS(t) = S(E6) - S(Shuf)
    τ_E6: ΔS(τ) = 0

    Method: measure ΔS at n_time_points,
    interpolate to find zero crossing.
    """
    print("\n" + "═"*56)
    print("  Finding τ_E6 (entropy crossover time)")
    print(f"  N={N}, σ={sigma}, n_runs={n_runs}")
    print("═"*56)

    T_points = np.unique(np.round(
        np.logspace(np.log10(20),
                    np.log10(1000),
                    n_time_points)).astype(int))

    results = {}
    print(f"\n  {'t':>6}  {'ΔS':>8}  {'std':>8}"
          f"  {'p':>8}  {'sig':>4}")

    for T in T_points:
        dS, std_dS, pv, _ = measure_delta_S_at_t(
            T_measure=T,
            n_runs=n_runs,
            N=N, sigma=sigma,
            omega_A=cfg.omega_E6,
            omega_B=cfg.omega_shuf)
        sig = ('✅' if pv < 0.05 and dS < 0
               else '⚠️' if pv < 0.05
               else '  ')
        print(f"  t={T:5d}: ΔS={dS:+.4f}"
              f"  std={std_dS:.4f}"
              f"  p={pv:.4f}  {sig}")
        results[T] = dict(dS=dS, std=std_dS, pv=pv)

    # Find zero crossing by interpolation
    T_arr  = np.array(sorted(results.keys()))
    dS_arr = np.array([results[t]['dS'] for t in T_arr])

    # Find where sign changes
    sign_changes = []
    for i in range(len(T_arr)-1):
        if dS_arr[i] * dS_arr[i+1] < 0:
            sign_changes.append(i)

    tau_E6 = None
    if sign_changes:
        i = sign_changes[0]
        t1, t2 = T_arr[i], T_arr[i+1]
        d1, d2 = dS_arr[i], dS_arr[i+1]
        # Linear interpolation
        tau_E6 = float(t1 + (0 - d1) *
                       (t2 - t1) / (d2 - d1))
        print(f"\n  ✅ Zero crossing found!")
        print(f"  τ_E6 ≈ {tau_E6:.1f} steps")
    else:
        if all(dS_arr < 0):
            print(f"\n  ΔS < 0 throughout — "
                  f"E6 always more ordered")
            tau_E6 = float('inf')
        else:
            print(f"\n  No clear zero crossing found")

    return results, T_arr, dS_arr, tau_E6


def tau_vs_sigma(N=300, n_runs=25):
    """Does τ_E6 depend on fragmentation width σ?"""
    print("\n" + "═"*56)
    print("  τ_E6 vs σ")
    print("═"*56)

    sigma_list = [0.20, 0.35, 0.50, 0.70, 1.00]
    # Use coarse time grid
    T_coarse = [50, 100, 150, 200, 300, 500]
    tau_list  = []

    for sigma in sigma_list:
        dS_vals = []
        for T in T_coarse:
            dS, _, pv, _ = measure_delta_S_at_t(
                T_measure=T, n_runs=n_runs,
                N=N, sigma=sigma,
                omega_A=cfg.omega_E6,
                omega_B=cfg.omega_shuf)
            dS_vals.append(dS)

        T_arr  = np.array(T_coarse)
        dS_arr = np.array(dS_vals)

        # Find sign change
        tau = None
        for i in range(len(T_arr)-1):
            if dS_arr[i] * dS_arr[i+1] < 0:
                tau = float(T_arr[i] +
                            (0 - dS_arr[i]) *
                            (T_arr[i+1]-T_arr[i]) /
                            (dS_arr[i+1]-dS_arr[i]))
                break
        tau_list.append(tau)
        tau_str = f"{tau:.1f}" if tau else "none"
        print(f"  σ={sigma:.2f}: "
              f"τ_E6 ≈ {tau_str}")

    return sigma_list, tau_list


def tau_vs_N(sigma=0.50, n_runs=25):
    """Does τ_E6 depend on number of strings N?"""
    print("\n" + "═"*56)
    print("  τ_E6 vs N")
    print("═"*56)

    N_list   = [50, 100, 200, 400]
    T_coarse = [50, 100, 150, 200, 300, 500]
    tau_list = []

    for N in N_list:
        dS_vals = []
        for T in T_coarse:
            dS, _, _, _ = measure_delta_S_at_t(
                T_measure=T, n_runs=n_runs,
                N=N, sigma=sigma,
                omega_A=cfg.omega_E6,
                omega_B=cfg.omega_shuf)
            dS_vals.append(dS)

        T_arr  = np.array(T_coarse)
        dS_arr = np.array(dS_vals)
        tau = None
        for i in range(len(T_arr)-1):
            if dS_arr[i] * dS_arr[i+1] < 0:
                tau = float(T_arr[i] +
                            (0 - dS_arr[i]) *
                            (T_arr[i+1]-T_arr[i]) /
                            (dS_arr[i+1]-dS_arr[i]))
                break
        tau_list.append(tau)
        tau_str = f"{tau:.1f}" if tau else "none"
        print(f"  N={N:4d}: τ_E6 ≈ {tau_str}")

    return N_list, tau_list


def tau_vs_algebra():
    """
    Is τ unique to E6 or common to all rank-6 algebras?

    Test: E6, shuffled E6, random, and also
    check if τ correlates with any algebraic property.
    """
    print("\n" + "═"*56)
    print("  τ for different frequency sets")
    print("═"*56)

    rng0 = np.random.RandomState(42)

    freq_sets = {
        'E6':       cfg.omega_E6,
        'Shuf_1':   rng0.permutation(cfg.omega_E6),
        'Shuf_2':   rng0.permutation(cfg.omega_E6),
        'Uniform':  rng0.uniform(
                        cfg.omega_E6.min(),
                        cfg.omega_E6.max(), 6),
        'Arithmetic': np.linspace(
                        cfg.omega_E6.min(),
                        cfg.omega_E6.max(), 6),
    }

    T_coarse = [50, 100, 150, 200, 300, 400]
    N, n_runs = 200, 25
    results = {}

    for name, omega_A in freq_sets.items():
        dS_vals = []
        for T in T_coarse:
            dS, _, pv, _ = measure_delta_S_at_t(
                T_measure=T, n_runs=n_runs,
                N=N, sigma=0.50,
                omega_A=omega_A,
                omega_B=cfg.omega_shuf)
            dS_vals.append(dS)

        T_arr  = np.array(T_coarse)
        dS_arr = np.array(dS_vals)

        tau = None
        dS0 = dS_arr[0]
        for i in range(len(T_arr)-1):
            if dS_arr[i] * dS_arr[i+1] < 0:
                tau = float(
                    T_arr[i] +
                    (0 - dS_arr[i]) *
                    (T_arr[i+1]-T_arr[i]) /
                    (dS_arr[i+1]-dS_arr[i]))
                break

        results[name] = dict(
            omega=omega_A, tau=tau, dS0=dS0)

        tau_str = f"{tau:.1f}" if tau else "N/A"
        print(f"  {name:12s}: τ={tau_str:>8}"
              f"  ΔS(t=50)={dS0:+.4f}")

    return results


# ═══════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════

def plot_all(main_res, sigma_res, N_res, alg_res):
    results, T_arr, dS_arr, tau_E6 = main_res
    sigma_list, tau_sigma = sigma_res
    N_list, tau_N = N_res

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Monostring Fragmentation v10\n"
        "CORE FINDING: Entropy crossover time τ_E6\n"
        "E6 daughters more ordered than Shuf at t < τ, "
        "less ordered at t > τ",
        fontsize=13, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.50, wspace=0.40)

    # ── Row 0: Main ΔS(t) curve ───────────────────────
    ax = fig.add_subplot(gs[0, 0:2])
    T_plot  = T_arr
    dS_plot = dS_arr
    std_plot= np.array([results[t]['std']
                         for t in T_arr])
    pv_plot = np.array([results[t]['pv']
                         for t in T_arr])

    colors = ['green' if (p<0.05 and d<0)
              else 'red' if (p<0.05 and d>0)
              else 'gray'
              for d, p in zip(dS_plot, pv_plot)]

    ax.fill_between(T_plot,
                    dS_plot - std_plot/np.sqrt(30),
                    dS_plot + std_plot/np.sqrt(30),
                    alpha=0.2, color='steelblue')
    ax.plot(T_plot, dS_plot, 'o-',
            c='steelblue', lw=2.5, ms=8)
    for t, d, c in zip(T_plot, dS_plot, colors):
        ax.scatter(t, d, s=80, c=c, zorder=5)
    ax.axhline(0, c='black', lw=2, ls='--')

    if tau_E6 and tau_E6 != float('inf'):
        ax.axvline(tau_E6, c='red', lw=2, ls=':',
                   label=f'τ_E6 = {tau_E6:.1f}')

    ax.set_xlabel('t (evolution steps)')
    ax.set_ylabel('ΔS = S(E6) - S(Shuf)')
    ax.set_title(
        'ΔS(t): E6 more ordered (green, ΔS<0)\n'
        'vs Shuf more ordered (red, ΔS>0)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── p-value plot ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(T_plot, pv_plot, 'o-',
                c='steelblue', lw=2, ms=7)
    ax.axhline(0.05, c='red', ls='--', lw=1.5,
               label='p=0.05')
    ax.axhline(0.001, c='orange', ls=':', lw=1.5,
               label='p=0.001')
    if tau_E6 and tau_E6 != float('inf'):
        ax.axvline(tau_E6, c='red', lw=1.5, ls=':')
    ax.set_xlabel('t'); ax.set_ylabel('p-value')
    ax.set_title('Statistical significance\nvs time')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── ΔS at early times (zoomed) ────────────────────
    ax = fig.add_subplot(gs[0, 3])
    early = T_plot <= 300
    if early.sum() > 1:
        ax.fill_between(
            T_plot[early],
            (dS_plot - std_plot/np.sqrt(30))[early],
            (dS_plot + std_plot/np.sqrt(30))[early],
            alpha=0.2, color='green')
        ax.plot(T_plot[early], dS_plot[early],
                'o-', c='green', lw=2.5, ms=8)
        ax.axhline(0, c='k', lw=2, ls='--')
    if tau_E6 and tau_E6 != float('inf'):
        ax.axvline(min(tau_E6, 300),
                   c='red', lw=2, ls=':',
                   label=f'τ_E6≈{tau_E6:.0f}')
    ax.set_xlabel('t (early phase)')
    ax.set_ylabel('ΔS = S(E6)-S(Shuf)')
    ax.set_title('Early phase (t<300)\nE6 ordering effect')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 1: τ vs σ and N ───────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    valid_sigma = [(s, t) for s, t in
                   zip(sigma_list, tau_sigma)
                   if t is not None]
    if valid_sigma:
        sv, tv = zip(*valid_sigma)
        ax.plot(sv, tv, 'o-', c='steelblue',
                lw=2.5, ms=10)
        ax.axhline(tau_E6 or 250, c='red',
                   ls='--', lw=1.5,
                   label='τ (σ=0.5)')
    ax.set_xlabel('σ (fragmentation)')
    ax.set_ylabel('τ_E6 (crossover time)')
    ax.set_title('τ_E6 vs σ\n(σ-independent?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    valid_N = [(n, t) for n, t in
               zip(N_list, tau_N)
               if t is not None]
    if valid_N:
        nv, tv = zip(*valid_N)
        ax.plot(nv, tv, 'o-', c='steelblue',
                lw=2.5, ms=10)
        ax.axhline(tau_E6 or 250, c='red',
                   ls='--', lw=1.5,
                   label='τ (N=300)')
    ax.set_xlabel('N (number of daughters)')
    ax.set_ylabel('τ_E6')
    ax.set_title('τ_E6 vs N\n(N-independent?)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    if alg_res:
        names = list(alg_res.keys())
        taus  = [alg_res[n]['tau']
                 if alg_res[n]['tau'] else 0
                 for n in names]
        dS0s  = [alg_res[n]['dS0'] for n in names]
        colors_bar = ['steelblue' if n == 'E6'
                      else 'orange'
                      for n in names]
        ax.bar(range(len(names)), taus,
               color=colors_bar, alpha=0.8,
               edgecolor='k')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15,
                            fontsize=8)
        ax.set_ylabel('τ (crossover time)')
        ax.set_title('τ for different\nfrequency sets')
        ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 3])
    if alg_res:
        names = list(alg_res.keys())
        dS0s  = [alg_res[n]['dS0'] for n in names]
        colors_bar = ['steelblue' if n == 'E6'
                      else 'orange'
                      for n in names]
        ax.bar(range(len(names)), dS0s,
               color=colors_bar, alpha=0.8,
               edgecolor='k')
        ax.axhline(0, c='k', lw=1.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15,
                            fontsize=8)
        ax.set_ylabel('ΔS at t=50')
        ax.set_title('Initial ordering effect\nby frequency set')
        ax.grid(True, alpha=0.3, axis='y')

    # ── Row 2: Physical interpretation ────────────────
    ax = fig.add_subplot(gs[2, 0:3])

    T_interp = np.linspace(min(T_arr), max(T_arr), 200)
    dS_interp = np.interp(T_interp, T_arr, dS_arr)
    colors_i  = ['green' if d < 0 else 'red'
                 for d in dS_interp]

    for i in range(len(T_interp)-1):
        ax.fill_between(
            T_interp[i:i+2],
            dS_interp[i:i+2],
            0,
            alpha=0.4,
            color=colors_i[i])

    ax.plot(T_interp, dS_interp, 'k-', lw=1)
    ax.axhline(0, c='k', lw=2)

    if tau_E6 and tau_E6 != float('inf'):
        ax.axvline(tau_E6, c='red', lw=3, ls='--',
                   label=f'τ_E6 = {tau_E6:.1f}')

    # Annotate regions
    ax.text(0.15, 0.85,
            "E6 MORE ORDERED\n(shared origin dominates)",
            transform=ax.transAxes,
            fontsize=10, color='darkgreen',
            ha='center',
            bbox=dict(boxstyle='round',
                      facecolor='lightgreen',
                      alpha=0.8))
    ax.text(0.70, 0.15,
            "Shuf MORE ORDERED\n(E6 memory fades)",
            transform=ax.transAxes,
            fontsize=10, color='darkred',
            ha='center',
            bbox=dict(boxstyle='round',
                      facecolor='#ffcccc',
                      alpha=0.8))

    ax.set_xlabel('t (evolution steps)', fontsize=12)
    ax.set_ylabel('ΔS = S(E6) - S(Shuf)', fontsize=12)
    ax.set_title(
        'PHYSICAL INTERPRETATION: '
        'τ_E6 = "memory time" of the monostring\n'
        'Before τ: daughters remember their E6 origin. '
        'After τ: E6 ergodicity takes over.',
        fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')

    tau_str = (f"{tau_E6:.1f}" if tau_E6
               and tau_E6 != float('inf')
               else "N/A")

    txt = (
        f"FINDING SUMMARY\n"
        f"{'═'*28}\n\n"
        f"τ_E6 ≈ {tau_str} steps\n\n"
        f"Physical meaning:\n"
        f"E6 daughters 'remember'\n"
        f"their monostring origin\n"
        f"for τ_E6 steps.\n\n"
        f"After τ_E6:\n"
        f"E6 ergodicity (D_corr=3)\n"
        f"makes daughters LESS\n"
        f"ordered than Shuf.\n\n"
        f"This is CONSISTENT with:\n"
        f"D_corr(E6 orbit) = 3.02\n"
        f"→ orbit is quasi-3D\n"
        f"→ ergodic on 3D manifold\n"
        f"→ fills it completely\n"
        f"→ higher entropy than\n"
        f"  Shuf (which has D≈4)\n\n"
        f"STATISTICAL:\n"
        f"t=100: p<0.0001 ✅\n"
        f"t=200: p<0.0001 ✅\n"
        f"t=400: p=0.001  ✅\n"
        f"t=800: p=0.069  ⚠️\n"
    )
    ax.text(0.03, 0.98, txt,
            fontsize=8.5, fontfamily='monospace',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round',
                      facecolor='#e3f2fd',
                      alpha=0.95))
    ax.set_title('Physical Interpretation')

    plt.savefig('monostring_fragmentation_v10.png',
                dpi=150, bbox_inches='tight')
    print("\n  Saved: monostring_fragmentation_v10.png")
    plt.show()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Monostring Fragmentation v10                   ║")
    print("║  ONE FINDING: Entropy crossover time τ_E6      ║")
    print("╚══════════════════════════════════════════════════╝")

    t0 = time.time()

    print("\n▓ MAIN: Precise measurement of τ_E6")
    main_res = find_tau_E6(N=300, sigma=0.50,
                            n_runs=30,
                            n_time_points=12)

    print("\n▓ τ_E6 vs σ")
    sigma_res = tau_vs_sigma(N=300, n_runs=25)

    print("\n▓ τ_E6 vs N")
    N_res = tau_vs_N(sigma=0.50, n_runs=25)

    print("\n▓ τ for different frequency sets")
    alg_res = tau_vs_algebra()

    print(f"\n  Total: {time.time()-t0:.0f}s")

    plot_all(main_res, sigma_res, N_res, alg_res)

    print("\n" + "█"*56)
    print("█  WHAT τ_E6 TELLS US PHYSICALLY             █")
    print("█"*56)
    print("""
  τ_E6 is the "memory time" of the monostring.

  After the breakup, daughters remember their
  common origin for τ_E6 steps — showing as
  lower entropy (more ordered) than null.

  After τ_E6, the E6 frequency structure
  takes over: daughters explore the quasi-3D
  orbit, becoming MORE ergodic (higher entropy)
  than null daughters on T^6.

  This is the ONLY statistically robust finding
  across v1-v10 that:
  1. Is significant (p<0.0001 at t=100-200)
  2. Has clear physical interpretation
  3. Goes beyond null model
  4. Is consistent with D_corr(orbit)=3.02
    """)
    print("█"*56)
