"""
Part VII v8: CORRECT REPRODUCTION + τ ∝ h TEST
================================================
Now we know EXACTLY what Part VI measured:

omega_E6 = 2*sin(π*m/h)  for m in [1,4,5,7,8,11], h=12
dynamics: phi += omega + kappa*sin(phi) + noise
null: E6 daughters share phi_break; Shuf daughters get RANDOM phi_B

Cross-algebra test:
  omega_X = 2*sin(π*m/h_X)  for each algebra X
  null: shuffled omega_X (same formula, different assignment)

Prediction: τ(X) ≈ 20 × h(X)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_ind, pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 1. ALGEBRA DEFINITIONS — Part VI convention
#    omega_i = 2 * sin(π * m_i / h)
#    This gives BOUNDED frequencies in [0, 2]
#    and encodes algebraic structure via ratios m_i/h
# ══════════════════════════════════════════════════════════════════

def coxeter_omega(exponents, h):
    """
    Part VI formula: ω_i = 2·sin(π·m_i/h)
    Bounded in [0,2], encodes m_i/h ratios.
    """
    m = np.array(exponents, dtype=float)
    return 2.0 * np.sin(np.pi * m / h)


ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': [1, 2, 3, 4, 5, 6],
        'color': '#9C27B0', 'tau_pred': 140,
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': [1, 4, 5, 7, 8, 11],
        'color': '#2196F3', 'tau_pred': 240,
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': [1, 5, 7, 9, 11, 13, 17],
        'color': '#FF9800', 'tau_pred': 360,
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
        'color': '#4CAF50', 'tau_pred': 600,
    },
}

# Dynamics parameters (Part VI values)
KAPPA = 0.05
NOISE = 0.006
N_BINS = 15   # Part VI uses bins=15


# ══════════════════════════════════════════════════════════════════
# 2. DYNAMICS — exact Part VI
# ══════════════════════════════════════════════════════════════════

def generate_orbit(omega, kappa=0.30, T=1000,
                   seed=0, warmup=300):
    """
    Part VI: generate monostring orbit for phi_break.
    Uses kappa=0.30 (stronger than daughter kappa=0.05).
    """
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)
    traj = []
    for _ in range(T):
        phi = (phi + omega
               + kappa * np.sin(phi)) % (2*np.pi)
        traj.append(phi.copy())
    return np.array(traj)


def shannon_H(phases, bins=N_BINS):
    """
    Part VI Shannon entropy: sum over dims (NOT averaged).
    phases: (N, rank)
    """
    H = 0.0
    for d in range(phases.shape[1]):
        hist, _ = np.histogram(
            phases[:, d], bins=bins,
            range=(0, 2*np.pi))
        p = hist / (hist.sum() + 1e-15)
        H -= np.sum(p * np.log(p + 1e-15))
    return float(H)


def measure_delta_S(T_measure, n_runs, N, sigma,
                    omega_A, omega_B,
                    kappa=KAPPA, noise=NOISE,
                    seed_base=42):
    """
    EXACT Part VI metric:
    - Population A: N daughters near phi_break, evolve with omega_A
    - Population B: N daughters near RANDOM phi_B, evolve with omega_B

    ΔS = S(A) - S(B)
    Negative ΔS → A more ordered (remembers origin)
    """
    rank_A = len(omega_A)
    rank_B = len(omega_B)

    # Get phi_break from monostring orbit
    orbit = generate_orbit(omega_A, T=100, seed=seed_base)
    phi_break = orbit[-1]

    S_A_list = []
    S_B_list = []

    for run in range(n_runs):
        rng = np.random.RandomState(run * 13 + 7 + seed_base)

        # Population A: shared origin phi_break
        ph_A = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N, rank_A))
               ) % (2*np.pi)

        # Population B: RANDOM origin (Part VI convention)
        phi_B_origin = rng.uniform(0, 2*np.pi, rank_B)
        ph_B = (phi_B_origin[np.newaxis, :]
                + rng.normal(0, sigma, (N, rank_B))
               ) % (2*np.pi)

        # Evolve
        for _ in range(T_measure):
            noise_A = rng.normal(0, noise, ph_A.shape)
            noise_B = rng.normal(0, noise, ph_B.shape)
            ph_A = (ph_A + omega_A[np.newaxis, :]
                    + kappa * np.sin(ph_A)
                    + noise_A) % (2*np.pi)
            ph_B = (ph_B + omega_B[np.newaxis, :]
                    + kappa * np.sin(ph_B)
                    + noise_B) % (2*np.pi)

        S_A_list.append(shannon_H(ph_A))
        S_B_list.append(shannon_H(ph_B))

    dS_list = [a - b for a, b in
               zip(S_A_list, S_B_list)]
    mean_dS = float(np.mean(dS_list))
    std_dS  = float(np.std(dS_list))
    _, pval = ttest_ind(S_A_list, S_B_list)

    return mean_dS, std_dS, float(pval), dS_list


# ══════════════════════════════════════════════════════════════════
# 3. τ MEASUREMENT — exact Part VI method
# ══════════════════════════════════════════════════════════════════

def find_tau(omega_A, omega_B,
             N=300, sigma=0.5,
             n_runs=30, n_points=12,
             T_min=20, T_max=1000,
             label="",
             verbose=True):
    """
    Find τ = zero crossing of ΔS(t).
    Returns (t_arr, dS_arr, pv_arr, tau)
    """
    T_points = np.unique(np.round(
        np.logspace(np.log10(T_min),
                    np.log10(T_max),
                    n_points)
    ).astype(int))

    t_arr  = []
    dS_arr = []
    pv_arr = []

    if verbose:
        print(f"\n  {label}")
        print(f"  {'t':>6}  {'ΔS':>9}  "
              f"{'p':>9}  sig")
        print(f"  {'-'*38}")

    for T in T_points:
        dS, std, pv, _ = measure_delta_S(
            T_measure=T, n_runs=n_runs,
            N=N, sigma=sigma,
            omega_A=omega_A, omega_B=omega_B)
        t_arr.append(T)
        dS_arr.append(dS)
        pv_arr.append(pv)

        if verbose:
            sig = ("***" if pv < 0.001 else
                   "**"  if pv < 0.01  else
                   "*"   if pv < 0.05  else
                   "."   if pv < 0.10  else "")
            print(f"  t={T:5d}: ΔS={dS:+.4f}"
                  f"  p={pv:.4f}  {sig}")

    t_arr  = np.array(t_arr)
    dS_arr = np.array(dS_arr)
    pv_arr = np.array(pv_arr)

    # Find zero crossing
    tau = np.nan
    for i in range(len(t_arr) - 1):
        if dS_arr[i] * dS_arr[i+1] < 0:
            # Linear interpolation
            tau = float(
                t_arr[i]
                + (0 - dS_arr[i])
                * (t_arr[i+1] - t_arr[i])
                / (dS_arr[i+1] - dS_arr[i]))
            break

    if np.isnan(tau):
        # Fallback: last significant negative
        sig_neg = (dS_arr < 0) & (pv_arr < 0.05)
        if sig_neg.any():
            tau = float(t_arr[sig_neg][-1])

    if verbose:
        if not np.isnan(tau):
            print(f"\n  τ ≈ {tau:.1f} steps")
        else:
            print(f"\n  τ = NaN (no zero crossing)")

    return t_arr, dS_arr, pv_arr, tau


# ══════════════════════════════════════════════════════════════════
# 4. STEP 0: VALIDATE E6 → τ ≈ 237
# ══════════════════════════════════════════════════════════════════

def validate_e6(rng_seed=42):
    print("\n" + "="*62)
    print("STEP 0: VALIDATE E6 (must give τ≈237)")
    print("="*62)

    alg   = ALGEBRAS['E6']
    omega = coxeter_omega(alg['exponents'], alg['h'])
    rng0  = np.random.RandomState(rng_seed)
    omega_shuf = rng0.permutation(omega.copy())

    print(f"\n  E6 frequencies: "
          f"{np.round(omega, 4).tolist()}")
    print(f"  Shuf frequencies: "
          f"{np.round(omega_shuf, 4).tolist()}")

    t_arr, dS_arr, pv_arr, tau = find_tau(
        omega_A=omega, omega_B=omega_shuf,
        N=300, sigma=0.5,
        n_runs=30, n_points=12,
        T_min=20, T_max=1000,
        label="E6 validation",
        verbose=True)

    print(f"\n  E6 τ = {tau:.1f}  [expected: ~237]")
    ok = not np.isnan(tau) and 150 < tau < 400
    print(f"  {'[OK]' if ok else '[WARN]'} "
          f"Validation {'passed' if ok else 'FAILED'}")

    return t_arr, dS_arr, pv_arr, tau, omega


# ══════════════════════════════════════════════════════════════════
# 5. CROSS-ALGEBRA TEST
# ══════════════════════════════════════════════════════════════════

def run_cross_algebra(seed=2025):
    """
    Test τ ∝ h for A6, E6, E7, E8.

    For each algebra X:
      omega_X = 2*sin(π*m_i/h_X)
      null    = shuffled omega_X
      τ_X     = zero crossing of ΔS(t)
    """
    rng = np.random.RandomState(seed)
    results = {}

    print("\n" + "="*62)
    print("CROSS-ALGEBRA TEST: τ ∝ h?")
    print("  omega_X = 2*sin(π*m/h)")
    print("  null = shuffled omega_X")
    print("="*62)

    for alg_name, alg in ALGEBRAS.items():
        h    = alg['h']
        rank = alg['rank']

        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}  h={h}  rank={rank}")

        omega = coxeter_omega(alg['exponents'], h)
        omega_shuf = rng.permutation(omega.copy())

        print(f"  omega    = {np.round(omega,4).tolist()}")
        print(f"  omega_sh = {np.round(omega_shuf,4).tolist()}")

        # Special case: A6 [1..6] with h=7
        # omega = 2*sin(π*[1..6]/7) — NOT arithmetic,
        # so shuffle IS different
        # Check:
        if np.allclose(np.sort(omega),
                       np.sort(omega_shuf)):
            print(f"  NOTE: shuffle = same set "
                  f"(permutation), testing order effect")

        T_max = max(600, 30 * h)
        t_arr, dS_arr, pv_arr, tau = find_tau(
            omega_A=omega, omega_B=omega_shuf,
            N=200, sigma=0.5,
            n_runs=25, n_points=14,
            T_min=20, T_max=T_max,
            label=f"{alg_name} (h={h})",
            verbose=True)

        tau_pred = alg['tau_pred']
        ratio    = tau / h if not np.isnan(tau) else np.nan

        if not np.isnan(ratio) and 10 < ratio < 40:
            status = "CONFIRMS"
        elif not np.isnan(ratio):
            status = "REJECTS"
        else:
            status = "NO SIGNAL"

        results[alg_name] = {
            'h': h, 'rank': rank,
            'tau_pred': tau_pred,
            'tau_obs':  tau,
            'tau_ratio': ratio,
            'omega': omega,
            'omega_shuf': omega_shuf,
            't_arr':  t_arr,
            'dS_arr': dS_arr,
            'pv_arr': pv_arr,
            'color':  alg['color'],
            'status': status,
        }

        print(f"\n  RESULT: "
              f"τ={'N/A' if np.isnan(tau) else f'{tau:.1f}'}, "
              f"pred={tau_pred}, "
              f"τ/h={'N/A' if np.isnan(ratio) else f'{ratio:.1f}'}, "
              f"status={status}")

    return results


# ══════════════════════════════════════════════════════════════════
# 6. STATISTICS
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results, tau_e6_ref=None):
    print("\n" + "="*62)
    print("STATISTICAL ANALYSIS: τ ∝ h?")
    print("="*62)

    names   = list(results.keys())
    h_arr   = np.array([results[n]['h']       for n in names])
    rank_arr= np.array([results[n]['rank']     for n in names])
    tau_arr = np.array([results[n]['tau_obs']  for n in names])

    valid = np.isfinite(tau_arr)
    n_v   = valid.sum()
    print(f"Valid τ: {n_v}/{len(names)}")
    for n in names:
        t = results[n]['tau_obs']
        print(f"  {n}: τ={'NaN' if np.isnan(t) else f'{t:.1f}'}")

    out = {'h_vals':    h_arr[valid],
           'tau_vals':  tau_arr[valid],
           'alg_names': [names[i] for i in range(len(names))
                         if valid[i]]}

    if n_v < 2:
        print("Insufficient data.")
        return out

    h_v, r_v, tau_v = (h_arr[valid], rank_arr[valid],
                        tau_arr[valid])

    # Forced-origin τ = k*h
    k0   = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    res0 = tau_v - k0 * h_v
    ss_t = np.sum((tau_v - tau_v.mean())**2)
    r2_0 = 1 - np.sum(res0**2)/ss_t if ss_t > 0 else np.nan
    print(f"\nτ = k*h (origin): "
          f"k={k0:.2f} [expect ~20], R²={r2_0:.4f}")
    out.update({'k0': k0, 'r2_h0': r2_0})

    if n_v >= 3:
        sl, ic, r, p, _ = linregress(h_v, tau_v)
        print(f"τ = a*h+b: a={sl:.2f}, b={ic:.1f}, "
              f"R²={r**2:.4f}, p={p:.4f}")
        out.update({'sl_h': sl, 'ic_h': ic,
                    'r2_h': r**2, 'p_h': p})
        sl2, _, r2, p2, _ = linregress(r_v, tau_v)
        print(f"τ ~ rank:  R²={r2**2:.4f}, p={p2:.4f}")
        out['r2_rank'] = r2**2
        if n_v >= 3:
            rho, pp = pearsonr(h_v, tau_v)
            print(f"Pearson r(τ,h) = {rho:.3f}, p={pp:.4f}")
            out.update({'pearson_r': rho, 'pearson_p': pp})

        # Include Part VI reference point?
        if tau_e6_ref is not None:
            print(f"\nWith Part VI E6 ref (τ={tau_e6_ref:.0f}):")
            h_ext   = np.append(h_v, 12.)
            tau_ext = np.append(tau_v, tau_e6_ref)
            sl_e, ic_e, r_e, p_e, _ = linregress(
                h_ext, tau_ext)
            print(f"  a={sl_e:.2f}, b={ic_e:.1f}, "
                  f"R²={r_e**2:.4f}, p={p_e:.4f}")
    else:
        out.update({'r2_h': np.nan, 'p_h': np.nan,
                    'r2_rank': np.nan,
                    'pearson_r': np.nan,
                    'pearson_p': np.nan,
                    'sl_h': np.nan, 'ic_h': np.nan})

    return out


# ══════════════════════════════════════════════════════════════════
# 7. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(val_result, cross_results, stats_out):
    t_val, dS_val, pv_val, tau_val, omega_e6 = val_result
    alg_list = list(cross_results.keys())

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII v8\n"
        r"$\tau \propto h_{\rm Coxeter}$: "
        "Correct Part VI dynamics\n"
        r"$\omega_i = 2\sin(\pi m_i/h)$, "
        r"$\phi_{n+1}=\phi_n+\omega+\kappa\sin\phi+\xi$, "
        r"null=shuffled$\,\omega$",
        fontsize=13, fontweight='bold', y=0.995,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.38)

    # Row 0 col 0: E6 validation
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('#e3f2fd')
    ax.fill_between(t_val,
                    np.array(dS_val) - 0.01,
                    np.array(dS_val) + 0.01,
                    alpha=0.2, color='#2196F3')
    ax.plot(t_val, dS_val, 'o-',
            c='#2196F3', lw=2.5, ms=8)
    ax.axhline(0, c='k', lw=2)
    if not np.isnan(tau_val):
        ax.axvline(tau_val, c='red', ls='--', lw=2,
                   label=f'τ={tau_val:.0f}')
    ax.axvline(237, c='orange', ls=':', lw=2,
               label='Part VI: 237')
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel('ΔS = S(E6)−S(Shuf)', fontsize=9)
    ax.set_title(
        'VALIDATION: E6\n'
        f'τ={"N/A" if np.isnan(tau_val) else f"{tau_val:.0f}"}'
        f'  [exp:237]',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 0 cols 1-3: ΔS per algebra
    for i, alg_name in enumerate(alg_list):
        res = cross_results[alg_name]
        ax  = fig.add_subplot(gs[0, i+1] if i < 3
                              else gs[1, i-3])
        ax.set_facecolor('#f0f4f8')

        t  = res['t_arr']
        dS = res['dS_arr']
        pv = res['pv_arr']
        c  = res['color']

        ax.plot(t, dS, 'o-', color=c, lw=2.5, ms=7)
        ax.axhline(0, c='k', lw=1.8)

        for j in range(len(t)):
            if pv[j] < 0.01 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=70,
                           c='#2E7D32', zorder=5,
                           edgecolors='k', lw=0.7)
            elif pv[j] < 0.05 and dS[j] < 0:
                ax.scatter(t[j], dS[j], s=45,
                           c='#81C784', zorder=5,
                           edgecolors='k', lw=0.7)

        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.axvline(tau, c='red', ls='--', lw=2,
                       label=f"τ={tau:.0f}")
        ax.axvline(res['tau_pred'], c='orange',
                   ls=':', lw=2,
                   label=f"pred={res['tau_pred']}")

        n_sig = np.sum((pv < 0.05) & (dS < 0))
        ax.set_title(
            f"{alg_name}: h={res['h']}, "
            f"rank={res['rank']}\n"
            f"τ={'N/A' if np.isnan(tau) else f'{tau:.0f}'}  "
            f"n_sig={n_sig}\n{res['status']}",
            fontsize=9, fontweight='bold')
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel('ΔS', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 1: τ vs h scatter
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    h_line = np.linspace(5, 35, 200)
    ax.plot(h_line, 20*h_line, '--', c='gray',
            lw=2, alpha=0.5,
            label='τ=20h (Part VI prediction)')
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.06, color='gray',
                    label='10h–40h band')

    # Part VI reference
    if not np.isnan(tau_val):
        ax.scatter(12, tau_val, s=300, marker='*',
                   c='blue', zorder=7, edgecolors='k',
                   label=f'E6 validation (τ={tau_val:.0f})')
    ax.scatter(12, 237, s=150, marker='D',
               c='navy', zorder=6, edgecolors='k',
               alpha=0.5, label='Part VI: τ=237')

    for alg_name, res in cross_results.items():
        tau = res['tau_obs']
        if not np.isnan(tau):
            ax.scatter(res['h'], tau, s=220,
                       c=res['color'], zorder=5,
                       edgecolors='k', lw=2)
            ax.annotate(
                f"{alg_name}\nτ={tau:.0f}\n{res['status']}",
                xy=(res['h'], tau),
                xytext=(res['h']+0.5, tau+15),
                fontsize=9, fontweight='bold',
                color=res['color'])

    if (stats_out and
            not np.isnan(stats_out.get('r2_h', np.nan)) and
            stats_out.get('r2_h', 0) > 0.3 and
            len(stats_out.get('h_vals', [])) >= 3):
        sl = stats_out['sl_h']
        ic = stats_out['ic_h']
        r2 = stats_out['r2_h']
        ax.plot(h_line, sl*h_line+ic, '-', c='red',
                lw=2,
                label=f"fit:{sl:.1f}h+{ic:.0f} "
                      f"R²={r2:.3f}")

    ax.set_xlabel('Coxeter number h', fontsize=12)
    ax.set_ylabel('Memory time τ (steps)', fontsize=12)
    ax.set_title('KEY: τ vs h\n'
                 'Prediction: τ ≈ 20h',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # Row 1 right: n_sig comparison
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    alg_names_p = list(cross_results.keys())
    n_sigs = [np.sum((cross_results[n]['pv_arr'] < 0.05)
                     & (cross_results[n]['dS_arr'] < 0))
              for n in alg_names_p]
    h_vals_p = [cross_results[n]['h'] for n in alg_names_p]
    clrs_p   = [cross_results[n]['color']
                for n in alg_names_p]

    bars = ax.bar(range(len(alg_names_p)), n_sigs,
                  color=clrs_p, alpha=0.85,
                  edgecolor='k', lw=1.5)
    for bar, ns in zip(bars, n_sigs):
        ax.text(bar.get_x() + bar.get_width()/2,
                ns + 0.2, str(ns),
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(alg_names_p)))
    ax.set_xticklabels(
        [f"{n}\n(h={cross_results[n]['h']})"
         for n in alg_names_p])
    ax.set_ylabel('n_sig (sig. negative ΔS)', fontsize=11)
    ax.set_title('Signal strength per algebra\n'
                 'n_sig: # time points with p<0.05, ΔS<0',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Row 2: Summary + statistics
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    ax.text(0.01, 0.97, "RESULTS TABLE",
            fontsize=12, fontweight='bold',
            transform=ax.transAxes, color='#1a1a2e')
    hdr = (f"{'Alg':<5} {'h':<4} {'rank':<5} "
           f"{'pred':>6} {'τ_obs':>8} "
           f"{'τ/h':>7} {'n_sig':>6}  status")
    ax.text(0.01, 0.85, hdr,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', fontweight='bold',
            color='#1a1a2e')
    ax.text(0.01, 0.80, "-"*65,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', color='gray')

    y = 0.73
    for alg_name, res in cross_results.items():
        tau  = res['tau_obs']
        n_sg = np.sum((res['pv_arr'] < 0.05)
                      & (res['dS_arr'] < 0))
        ts   = f"{tau:>8.0f}" if not np.isnan(tau) else f"{'N/A':>8}"
        rs   = (f"{tau/res['h']:>7.1f}"
                if not np.isnan(tau) else f"{'N/A':>7}")
        st   = res['status']
        c    = ('#2E7D32' if st == 'CONFIRMS' else
                '#C62828' if st == 'REJECTS'  else '#888')
        row  = (f"{alg_name:<5} {res['h']:<4} {res['rank']:<5}"
                f"{res['tau_pred']:>6}{ts}{rs}"
                f"{n_sg:>6}  {st}")
        ax.text(0.01, y, row,
                transform=ax.transAxes, fontsize=10,
                fontfamily='monospace', color=c)
        y -= 0.10

    # Stats
    if stats_out:
        r2h = stats_out.get('r2_h', np.nan)
        k0  = stats_out.get('k0',   np.nan)
        r20 = stats_out.get('r2_h0',np.nan)
        rho = stats_out.get('pearson_r', np.nan)
        pp  = stats_out.get('pearson_p', np.nan)

        if not np.isnan(r2h):
            if r2h > 0.85:
                verd = "STRONG: tau ∝ h CONFIRMED"
                vc   = '#2E7D32'
            elif r2h > 0.5:
                verd = "MODERATE: tau correlates with h"
                vc   = '#FF9800'
            else:
                verd = "WEAK/NONE: tau NOT ∝ h"
                vc   = '#C62828'
        else:
            verd, vc = "INSUFFICIENT DATA", '#888'

        sl_s = (f"tau=a*h+b: a={stats_out.get('sl_h',np.nan):.1f},"
                f" b={stats_out.get('ic_h',np.nan):.0f},"
                f" R²={r2h:.3f}")
        lines = [
            f"tau=k*h(origin): k={k0:.1f} [exp~20],"
            f" R²={r20:.3f}",
            sl_s,
            f"tau~rank:        "
            f"R²={stats_out.get('r2_rank',np.nan):.3f}",
            f"Pearson r={rho:.3f}, p={pp:.4f}",
        ]
        ax.text(0.63, 0.95, "STATISTICS",
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color='#1a1a2e')
        for i, line in enumerate(lines):
            ax.text(0.63, 0.85-i*0.09, line,
                    transform=ax.transAxes,
                    fontsize=10, fontfamily='monospace',
                    color='#1a1a2e')
        ax.text(0.63, 0.85-len(lines)*0.09,
                f"VERDICT: {verd}",
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=vc)

    ax.text(0.50, 0.01,
            r"$\omega_i = 2\sin(\pi m_i/h)$  |  "
            r"$\phi_{n+1}=\phi_n+\omega+0.05\sin\phi+\mathcal{N}(0,0.006)$"
            "  |  null: shuffled ω",
            transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_v8.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_v8.png")


# ══════════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VII v8: Correct Part VI dynamics + τ∝h test")
    print(f"omega = 2*sin(π*m/h), "
          f"kappa={KAPPA}, noise={NOISE}\n")

    # Step 0: Validate E6
    val_result = validate_e6(rng_seed=42)
    tau_e6_val = val_result[3]

    # Step 1: Cross-algebra test
    cross_results = run_cross_algebra(seed=2025)

    # Step 2: Statistics
    stats_out = statistical_analysis(
        cross_results, tau_e6_ref=tau_e6_val)

    # Step 3: Figure
    make_figure(val_result, cross_results, stats_out)

    # Final table
    print("\n" + "="*62)
    print("FINAL TABLE")
    print("="*62)
    print(f"{'Alg':<5} {'h':<4} {'pred':>6} "
          f"{'τ_obs':>8} {'τ/h':>7}  status")
    print("-"*62)
    for n, res in cross_results.items():
        tau = res['tau_obs']
        print(f"{n:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6} "
              f"{tau:>8.1f} "
              f"{tau/res['h']:>7.1f}  "
              f"{res['status']}"
              if not np.isnan(tau) else
              f"{n:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6} "
              f"{'NaN':>8} {'NaN':>7}  "
              f"{res['status']}")

    r2 = stats_out.get('r2_h', np.nan)
    k0 = stats_out.get('k0',   np.nan)
    print(f"\nConclusion: R²={r2:.3f}, k={k0:.1f}")
    if not np.isnan(r2):
        if r2 > 0.85:
            print("CONFIRMED: τ ∝ h")
        elif r2 > 0.5:
            print("PARTIAL: τ correlates with h")
        else:
            print("REJECTED: τ not ∝ h")
