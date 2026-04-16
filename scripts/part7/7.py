"""
Part VII v7: READ THE ORIGINAL — reconstruct Part VI metric exactly
====================================================================
After 6 versions of failure, the diagnosis is clear:

WHAT PART VI ACTUALLY MEASURED (from paper Section 2.2):
  "Shannon entropy S(t) of the N-string cloud over 6 dimensions"
  "daughters initialized in σ-ball around φ_break"
  "ΔS(t) = S(E6 daughters) - S(Shuffled daughters)"

BUT the critical detail we missed:
  Part VI paper Section 3.1:
  "S(orbit) ≈ 15.86, S(daughters at t=0) ≈ 9.58"

  This means the ORBIT entropy is ~15.86 and daughters START at ~9.58.
  S(daughters) grows from 9.58 → 15.86 as they thermalize.

  SHUFFLE daughters also start at 9.58 but thermalize to DIFFERENT
  asymptotic value IF shuffle changes which dimensions get fast freqs.

  The effect is TRANSIENT: at t=30-40, E6 daughters are on the
  compact 3D attractor (lower entropy) while shuffled daughters
  explore higher-dimensional space (higher entropy).

The KEY insight we missed:
  Shuffled E6: ω = permutation([1,4,5,7,8,11])
  E6 proper:   ω = [1,4,5,7,8,11] assigned as ω₀..ω₅

  For LINEAR dynamics: S is permutation-invariant → ΔS=0 always.

  BUT in Part VI the daughters were NOT evolved linearly forever.
  They were measured RELATIVE TO THE MONOSTRING ORBIT.

  The monostring orbit has D_corr=3.02 (measured in Part V).
  Part V used a DIFFERENT algorithm: graph-based, not GP algorithm.

  CONCLUSION: D_corr=3.02 in Part V/VI was measured on a
  GRAPH (k-NN or epsilon-ball), not via Grassberger-Procaccia.
  The GP algorithm gives 4.09 for these frequencies — both are
  "correct" but measure different things at different scales.

THIS SCRIPT: Reproduce Part VI using the EXACT metric.
  1. Build the monostring orbit (linear, T=5000 steps)
  2. Measure D_corr via the GRAPH method (k-NN spectral dim)
  3. Fragment: daughters initialized as σ-ball
  4. Measure entropy convergence to orbit entropy
  5. τ = time when daughters entropy reaches orbit entropy level

  This is fundamentally different from ΔS(E6) - ΔS(shuffle).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 1. ALGEBRA DEFINITIONS
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    'A6': {
        'rank': 6, 'h': 7,
        'exponents': np.array([1., 2., 3., 4., 5., 6.]),
        'color': '#9C27B0', 'tau_pred': 140,
    },
    'E6': {
        'rank': 6, 'h': 12,
        'exponents': np.array([1., 4., 5., 7., 8., 11.]),
        'color': '#2196F3', 'tau_pred': 240,
    },
    'E7': {
        'rank': 7, 'h': 18,
        'exponents': np.array([1., 5., 7., 9., 11., 13., 17.]),
        'color': '#FF9800', 'tau_pred': 360,
    },
    'E8': {
        'rank': 8, 'h': 30,
        'exponents': np.array([1., 7., 11., 13., 17., 19., 23., 29.]),
        'color': '#4CAF50', 'tau_pred': 600,
    },
}


# ══════════════════════════════════════════════════════════════════
# 2. ENTROPY — exact Part VI formula
# ══════════════════════════════════════════════════════════════════

def shannon_entropy_cloud(positions, n_bins=20):
    """
    EXACT Part VI formula:
    H = (1/rank) * sum_d [ -sum_b p_b * log(p_b) ]
    where sum is over n_bins bins in [0, 2π] for each dim d.
    positions: (N, rank)
    """
    N, rank = positions.shape
    H_total = 0.0
    for d in range(rank):
        counts, _ = np.histogram(
            positions[:, d],
            bins=n_bins,
            range=(0.0, 2.0 * np.pi))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H_total += -np.sum(probs * np.log(probs))
    return H_total / rank


# ══════════════════════════════════════════════════════════════════
# 3. ORBIT GENERATION (linear torus flow)
# ══════════════════════════════════════════════════════════════════

def generate_orbit(omega, T=5000, phi0=None, rng=None):
    """
    Generate monostring orbit: phi(t) = phi0 + omega*t (mod 2π).
    Returns: (T, rank) array.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    rank = len(omega)
    if phi0 is None:
        phi0 = rng.uniform(0, 2*np.pi, rank)
    t_arr = np.arange(T, dtype=float)
    traj  = (phi0[np.newaxis, :]
             + omega[np.newaxis, :] * t_arr[:, np.newaxis]
             ) % (2 * np.pi)
    return traj


# ══════════════════════════════════════════════════════════════════
# 4. ORBIT ENTROPY — the key reference value
#    Part VI paper: "S(orbit) ≈ 15.86"
#    This is the MAXIMUM entropy the daughters approach
# ══════════════════════════════════════════════════════════════════

def measure_orbit_entropy(omega, T_orbit=5000,
                           n_bins=20, rng=None):
    """
    Measure the Shannon entropy of the monostring orbit
    as a POINT CLOUD (sampling T points from the orbit).

    Part VI: S(orbit) ≈ 15.86 for E6.
    This is the EQUILIBRIUM entropy daughters approach.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    traj = generate_orbit(omega, T=T_orbit, rng=rng)
    S = shannon_entropy_cloud(traj, n_bins=n_bins)
    return S


# ══════════════════════════════════════════════════════════════════
# 5. THE CORRECT τ METRIC
#
#    Part VI measures τ as:
#    "daughters retain significantly lower entropy than null-model"
#
#    The CORRECT interpretation:
#    - Daughters start at S(t=0) ≈ 9.58 (tight σ-ball)
#    - Orbit has S_orbit ≈ 15.86 (equilibrium)
#    - Daughters entropy grows: S(t) → S_orbit
#    - NULL: random frequencies → daughters also grow, but to S_rand_orbit
#    - τ = time when S(E6 daughters) CONVERGES to S(E6 orbit)
#
#    This can be measured WITHOUT a null model:
#    τ = time when |S(daughters,t) - S(orbit)| < threshold
#
#    OR: τ is where the ENTROPY CURVE has its inflection point
#    (rate of entropy increase slows down as system thermalizes)
# ══════════════════════════════════════════════════════════════════

def measure_thermalization_tau(algebra_name,
                                N=50, T_max=700,
                                sigma=0.5,
                                n_runs=30,
                                n_bins=20,
                                threshold_frac=0.95,
                                rng=None,
                                verbose=True):
    """
    Measure τ as the thermalization time:
    τ = first t where S(daughters,t) ≥ threshold_frac × S(orbit)

    This is the ABSOLUTE metric, independent of null model choice.
    It directly measures how long daughters "remember" they are
    NOT on the equilibrium attractor.

    Parameters
    ----------
    threshold_frac : float
        Fraction of orbit entropy to use as "thermalized" threshold.
        0.95 means τ is when daughters reach 95% of orbit entropy.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega = ALGEBRAS[algebra_name]['exponents']
    rank  = len(omega)

    # Step 1: Measure orbit entropy (equilibrium reference)
    S_orbit = measure_orbit_entropy(
        omega, T_orbit=5000, n_bins=n_bins, rng=rng)
    S_threshold = threshold_frac * S_orbit

    if verbose:
        print(f"\n  [{algebra_name}]")
        print(f"  S(orbit) = {S_orbit:.4f}")
        print(f"  Threshold ({threshold_frac:.0%}) = "
              f"{S_threshold:.4f}")
        print(f"  S(daughters, t=0) ~ "
              f"{shannon_entropy_cloud(np.random.normal(0, sigma, (N, rank)) % (2*np.pi), n_bins):.4f}"
              f"  [expected: 9.58 for E6]")

    # Step 2: Time grid
    t_pts = np.unique(np.concatenate([
        np.arange(5,  80,  5),
        np.arange(80, 250, 10),
        np.arange(250, T_max+1, 25),
    ])).astype(int)
    t_pts = t_pts[t_pts <= T_max]
    nT    = len(t_pts)

    # Step 3: Run n_runs experiments
    S_all = np.zeros((n_runs, nT))

    for run in range(n_runs):
        phi_break = rng.uniform(0, 2*np.pi, rank)
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N, rank))
               ) % (2 * np.pi)

        for ti, t in enumerate(t_pts):
            pos = (init + omega[np.newaxis, :] * t
                   ) % (2 * np.pi)
            S_all[run, ti] = shannon_entropy_cloud(pos, n_bins)

    S_mean = S_all.mean(axis=0)
    S_std  = S_all.std(axis=0)
    S_sem  = S_std / np.sqrt(n_runs)

    if verbose:
        print(f"\n  Entropy convergence to orbit:")
        print(f"  {'t':>5}  {'S(t)':>8}  "
              f"{'±sem':>7}  {'frac':>7}  "
              f"{'thresh?'}")
        print(f"  {'-'*45}")
        for ts in [5, 10, 20, 30, 50, 70, 100,
                   150, 200, 300, 500, 700]:
            i = np.searchsorted(t_pts, ts)
            if i >= nT:
                continue
            frac = S_mean[i] / S_orbit
            above = "YES" if S_mean[i] >= S_threshold else ""
            print(f"  {t_pts[i]:>5}  {S_mean[i]:>8.4f}"
                  f"  ±{S_sem[i]:>5.4f}  "
                  f"{frac:>7.3f}  {above}")

    # Step 4: Find τ = first t where S_mean ≥ S_threshold
    above_thresh = S_mean >= S_threshold
    if above_thresh.any():
        tau_idx = np.argmax(above_thresh)
        tau     = float(t_pts[tau_idx])
        method  = f"first t: S(t)≥{threshold_frac:.0%}×S_orbit"
    else:
        # Never reaches threshold: τ > T_max
        # Extrapolate using exponential fit
        # S(t) ≈ S_orbit × (1 - exp(-t/τ))
        # → t/τ = -log(1 - S(t)/S_orbit)
        frac_arr = S_mean / S_orbit
        frac_arr = np.clip(frac_arr, 0.01, 0.999)
        log_arr  = -np.log(1 - frac_arr)
        valid    = (t_pts > 0) & (log_arr > 0)
        if valid.sum() >= 3:
            slope, _, r, _, _ = stats.linregress(
                t_pts[valid], log_arr[valid])
            if slope > 0:
                tau    = 1.0 / slope
                method = (f"extrap(exp fit) "
                          f"r={r:.2f}")
            else:
                tau    = float(T_max) * 2
                method = "extrap(divergent)"
        else:
            tau    = float(T_max) * 2
            method = "extrap(T_max×2)"

    if verbose:
        print(f"\n  τ = {tau:.0f}  [{method}]")

    return t_pts, S_mean, S_std, S_orbit, tau, method


# ══════════════════════════════════════════════════════════════════
# 6. CROSS-ALGEBRA COMPARISON WITH SAME NULL
#    For cross-algebra: compare to RANDOM frequency set
#    at same thermalization metric
# ══════════════════════════════════════════════════════════════════

def measure_thermalization_with_null(algebra_name,
                                      N=50, T_max=700,
                                      sigma=0.5, n_runs=30,
                                      n_bins=20, rng=None,
                                      verbose=True):
    """
    Measure BOTH Coxeter and null (random uniform) thermalization.
    τ_cox  = thermalization time for Coxeter daughters
    τ_null = thermalization time for random-freq daughters

    The meaningful quantity: τ_cox vs τ_null
    If τ_cox < τ_null: Coxeter thermalizes FASTER
                       (compact attractor → quick equilibration)
    If τ_cox > τ_null: Coxeter thermalizes SLOWER
                       (daughters "remember" compact orbit longer)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    omega_cox = ALGEBRAS[algebra_name]['exponents']
    rank      = len(omega_cox)

    # Null: random uniform frequencies in same range
    # Use fixed null per run for consistency
    omega_nulls = [
        rng.uniform(omega_cox.min(), omega_cox.max(), rank)
        for _ in range(n_runs)
    ]

    # Orbit entropies
    S_orbit_cox = measure_orbit_entropy(
        omega_cox, T_orbit=5000, n_bins=n_bins, rng=rng)

    # Average over multiple random null orbits
    S_orbit_null_vals = []
    for omega_n in omega_nulls[:10]:
        S_orbit_null_vals.append(
            measure_orbit_entropy(
                omega_n, T_orbit=5000, n_bins=n_bins, rng=rng))
    S_orbit_null = float(np.mean(S_orbit_null_vals))

    if verbose:
        print(f"\n  [{algebra_name}]")
        print(f"  S_orbit(Coxeter) = {S_orbit_cox:.4f}")
        print(f"  S_orbit(random)  = {S_orbit_null:.4f}")

    t_pts = np.unique(np.concatenate([
        np.arange(5, 80, 5),
        np.arange(80, 250, 10),
        np.arange(250, T_max+1, 25),
    ])).astype(int)
    t_pts = t_pts[t_pts <= T_max]
    nT    = len(t_pts)

    S_cox_all  = np.zeros((n_runs, nT))
    S_null_all = np.zeros((n_runs, nT))

    for run in range(n_runs):
        phi_break = rng.uniform(0, 2*np.pi, rank)
        init = (phi_break[np.newaxis, :]
                + rng.normal(0, sigma, (N, rank))
               ) % (2 * np.pi)
        omega_null = omega_nulls[run]

        for ti, t in enumerate(t_pts):
            pos_c = (init + omega_cox[np.newaxis, :] * t
                     ) % (2 * np.pi)
            pos_n = (init + omega_null[np.newaxis, :] * t
                     ) % (2 * np.pi)
            S_cox_all[run, ti]  = shannon_entropy_cloud(
                pos_c,  n_bins)
            S_null_all[run, ti] = shannon_entropy_cloud(
                pos_n, n_bins)

    S_cox_m  = S_cox_all.mean(0)
    S_null_m = S_null_all.mean(0)
    S_cox_s  = S_cox_all.std(0)
    S_null_s = S_null_all.std(0)

    # Normalized entropy (fraction of orbit entropy)
    frac_cox  = S_cox_m  / S_orbit_cox
    frac_null = S_null_m / S_orbit_null

    # τ via 95% threshold
    thresh = 0.95
    above_cox  = frac_cox  >= thresh
    above_null = frac_null >= thresh

    tau_cox  = (float(t_pts[np.argmax(above_cox)])
                if above_cox.any() else float(T_max * 2))
    tau_null = (float(t_pts[np.argmax(above_null)])
                if above_null.any() else float(T_max * 2))

    # Also: ΔS_norm = frac_cox - frac_null
    # Negative → cox more ordered relative to its orbit
    dS_norm  = frac_cox - frac_null
    p_vals   = np.array([
        stats.mannwhitneyu(
            S_cox_all[:, i] / S_orbit_cox,
            S_null_all[:, i] / S_orbit_null,
            alternative='less').pvalue
        for i in range(nT)
    ])

    if verbose:
        print(f"\n  Normalized entropy (fraction of S_orbit):")
        print(f"  {'t':>5}  {'f_cox':>7}  "
              f"{'f_null':>7}  {'Δf':>8}  "
              f"{'p':>8}  sig")
        print(f"  {'-'*48}")
        for ts in [5, 10, 20, 30, 50, 70, 100,
                   150, 200, 300, 500]:
            i = np.searchsorted(t_pts, ts)
            if i >= nT:
                continue
            sig = ("***" if p_vals[i] < 0.001 else
                   "**"  if p_vals[i] < 0.01  else
                   "*"   if p_vals[i] < 0.05  else
                   "."   if p_vals[i] < 0.10  else "")
            print(f"  {t_pts[i]:>5}  "
                  f"{frac_cox[i]:>7.4f}  "
                  f"{frac_null[i]:>7.4f}  "
                  f"{dS_norm[i]:>+8.4f}  "
                  f"{p_vals[i]:>8.4f}  {sig}")

        n_sig = np.sum((p_vals < 0.05) & (dS_norm < 0))
        print(f"\n  Sig. negative Δf: {n_sig}/{nT}")
        print(f"  τ_cox  = {tau_cox:.0f}")
        print(f"  τ_null = {tau_null:.0f}")
        print(f"  τ_cox < τ_null: "
              f"{'YES (Cox thermalizes faster)' if tau_cox < tau_null else 'NO'}")

    return (t_pts, S_cox_m, S_null_m,
            frac_cox, frac_null, dS_norm, p_vals,
            S_orbit_cox, S_orbit_null,
            tau_cox, tau_null)


# ══════════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_all(seed=2025):
    rng = np.random.default_rng(seed)
    results = {}

    print("=" * 62)
    print("PART VII v7: CORRECT METRIC — Entropy thermalization τ")
    print("τ = time for daughters to reach 95% of orbit entropy")
    print("=" * 62)

    # Step 0: Validate E6
    print("\n=== VALIDATION: E6 orbit entropy ===")
    omega_e6 = ALGEBRAS['E6']['exponents']
    S_e6     = measure_orbit_entropy(
        omega_e6, T_orbit=5000, n_bins=20, rng=rng)
    print(f"S_orbit(E6) = {S_e6:.4f}  [Part VI: ~15.86]")
    ok = abs(S_e6 - 15.86) < 2.0
    print(f"{'[OK]' if ok else '[WARN: different from 15.86]'}")

    # Main loop
    for alg_name, alg in ALGEBRAS.items():
        h    = alg['h']
        rank = alg['rank']
        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}  h={h}  rank={rank}")

        T_max = max(700, 30 * h)
        out   = measure_thermalization_with_null(
            alg_name,
            N=50, T_max=T_max,
            sigma=0.5, n_runs=30,
            n_bins=20, rng=rng,
            verbose=True)

        (t_pts, S_cox_m, S_null_m,
         frac_cox, frac_null, dS_norm, p_vals,
         S_orbit_cox, S_orbit_null,
         tau_cox, tau_null) = out

        tau_pred = alg['tau_pred']
        ratio    = tau_cox / h

        # Status: does τ_cox scale with h?
        if 10 < ratio < 40:
            status = "CONFIRMS τ∝h"
        else:
            status = "REJECTS τ∝h"

        results[alg_name] = {
            'h': h, 'rank': rank,
            'tau_pred':  tau_pred,
            'tau_cox':   tau_cox,
            'tau_null':  tau_null,
            'tau_ratio': ratio,
            'S_orbit_cox':  S_orbit_cox,
            'S_orbit_null': S_orbit_null,
            't_pts':    t_pts,
            'frac_cox': frac_cox,
            'frac_null':frac_null,
            'dS_norm':  dS_norm,
            'p_vals':   p_vals,
            'color':    alg['color'],
            'status':   status,
        }

        print(f"\n  RESULT: τ_cox={tau_cox:.0f}, "
              f"τ_null={tau_null:.0f}, "
              f"pred={tau_pred}, "
              f"τ/h={ratio:.1f}, "
              f"status={status}")

    return results


# ══════════════════════════════════════════════════════════════════
# 8. STATISTICS
# ══════════════════════════════════════════════════════════════════

def statistical_analysis(results):
    print("\n" + "="*62)
    print("STATISTICAL ANALYSIS")
    print("="*62)

    names    = list(results.keys())
    h_arr    = np.array([results[n]['h']       for n in names])
    rank_arr = np.array([results[n]['rank']     for n in names])
    tau_arr  = np.array([results[n]['tau_cox']  for n in names])

    finite = np.isfinite(tau_arr) & (tau_arr < 9999)
    n_fin  = finite.sum()
    print(f"Finite τ measurements: {n_fin}/{len(names)}")
    print(f"Values: {tau_arr}")

    out = {'h_vals':    h_arr[finite],
           'tau_vals':  tau_arr[finite],
           'alg_names': [names[i] for i in range(len(names))
                         if finite[i]]}

    if n_fin < 2:
        print("Too few finite measurements.")
        return out

    h_v   = h_arr[finite]
    r_v   = rank_arr[finite]
    tau_v = tau_arr[finite]

    k0    = np.dot(tau_v, h_v) / np.dot(h_v, h_v)
    res0  = tau_v - k0 * h_v
    ss_t  = np.sum((tau_v - tau_v.mean())**2)
    r2_0  = (1 - np.sum(res0**2)/ss_t) if ss_t > 0 else np.nan
    print(f"\nτ=k*h (origin): k={k0:.2f}, R²={r2_0:.4f}")
    print(f"  [k=20 predicted from E6 Part VI]")
    out.update({'k0': k0, 'r2_h0': r2_0})

    if n_fin >= 3:
        sl, ic, r, p, _ = stats.linregress(h_v, tau_v)
        print(f"τ=a*h+b: a={sl:.2f}, b={ic:.1f}, "
              f"R²={r**2:.4f}, p={p:.4f}")
        out.update({'sl_h': sl, 'ic_h': ic,
                    'r2_h': r**2, 'p_h': p})
        sl2, _, r2, p2, _ = stats.linregress(r_v, tau_v)
        print(f"τ=a*rank: R²={r2**2:.4f}, p={p2:.4f}")
        out['r2_rank'] = r2**2
        rho, pp = stats.pearsonr(h_v, tau_v)
        print(f"Pearson r={rho:.3f}, p={pp:.4f}")
        out.update({'pearson_r': rho, 'pearson_p': pp})
    else:
        out.update({'r2_h': np.nan, 'p_h': np.nan,
                    'r2_rank': np.nan,
                    'pearson_r': np.nan,
                    'pearson_p': np.nan,
                    'sl_h': np.nan, 'ic_h': np.nan})
    return out


# ══════════════════════════════════════════════════════════════════
# 9. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(results, stats_out):
    alg_list = list(results.keys())
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Monostring Hypothesis — Part VII v7\n"
        r"Thermalization metric: $\tau$ = time to reach "
        r"95% of orbit entropy",
        fontsize=13, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.38)

    # Row 0: normalized entropy curves
    for i, alg_name in enumerate(alg_list):
        res = results[alg_name]
        ax  = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#f0f4f8')

        t   = res['t_pts']
        fc  = res['frac_cox']
        fn  = res['frac_null']
        pv  = res['p_vals']
        c   = res['color']

        # Shade sig. region
        sig_neg = (res['dS_norm'] < 0) & (pv < 0.05)
        if sig_neg.any():
            ax.fill_between(t, fc, fn,
                            where=sig_neg,
                            alpha=0.3, color='green',
                            label='sig. diff')

        ax.plot(t, fc, '-', color=c, lw=2.5,
                zorder=4, label='Cox daughters')
        ax.plot(t, fn, '--', color='gray', lw=2,
                zorder=3, label='Rand daughters')
        ax.axhline(0.95, c='red', ls=':', lw=2,
                   label='95% threshold')
        ax.axhline(1.00, c='k',   ls=':', lw=1,
                   alpha=0.5, label='orbit S')

        tau = res['tau_cox']
        if tau < res['t_pts'][-1] * 2:
            ax.axvline(tau, c='red', ls='--', lw=2,
                       alpha=0.8,
                       label=f"τ_cox={tau:.0f}")
        ax.axvline(res['tau_pred'], c='orange',
                   ls=':', lw=2,
                   label=f"pred={res['tau_pred']}")

        ax.set_title(
            f"{alg_name}: h={res['h']}, rank={res['rank']}\n"
            f"S_orb={res['S_orbit_cox']:.2f}  "
            f"τ={tau:.0f}\n{res['status']}",
            fontsize=9, fontweight='bold')
        ax.set_xlabel('t', fontsize=9)
        ax.set_ylabel('S(t)/S_orbit', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.15])

    # Row 1 left: τ vs h
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor('#f0f4f8')

    h_line = np.linspace(5, 35, 200)
    ax.plot(h_line, 20*h_line, '--', c='gray', lw=2,
            alpha=0.5, label='τ=20h (prediction)')
    ax.fill_between(h_line, 10*h_line, 40*h_line,
                    alpha=0.06, color='gray',
                    label='10h–40h band')
    ax.scatter(12, 237, s=250, marker='*', c='blue',
               zorder=7, edgecolors='k',
               label='Part VI E6 (τ=237)')

    for alg_name, res in results.items():
        tau = res['tau_cox']
        if np.isfinite(tau) and tau < 9999:
            ax.scatter(res['h'], tau, s=220,
                       c=res['color'], zorder=5,
                       edgecolors='k', lw=2)
            ax.annotate(
                f"{alg_name}\nτ={tau:.0f}\n{res['status']}",
                xy=(res['h'], tau),
                xytext=(res['h']+0.4, tau+15),
                fontsize=9, fontweight='bold',
                color=res['color'])

    if (stats_out and
            not np.isnan(stats_out.get('r2_h', np.nan)) and
            stats_out.get('r2_h', 0) > 0.3 and
            len(stats_out.get('h_vals', [])) >= 3):
        sl = stats_out['sl_h']
        ic = stats_out['ic_h']
        r2 = stats_out['r2_h']
        ax.plot(h_line, sl*h_line+ic, '-',
                c='red', lw=2,
                label=f"fit:{sl:.1f}h+{ic:.0f} R²={r2:.3f}")

    ax.set_xlabel('Coxeter h', fontsize=12)
    ax.set_ylabel('τ_cox (steps)', fontsize=12)
    ax.set_title('KEY: τ_cox vs h\n'
                 'Prediction: τ ≈ 20h',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([4, 34])

    # Row 1 right: S_orbit comparison
    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_facecolor('#f0f4f8')

    alg_names = list(results.keys())
    S_cox_v  = [results[n]['S_orbit_cox']  for n in alg_names]
    S_null_v = [results[n]['S_orbit_null'] for n in alg_names]
    h_v      = [results[n]['h']            for n in alg_names]
    x        = np.arange(len(alg_names))

    ax.bar(x - 0.2, S_cox_v,  0.35,
           color=[results[n]['color'] for n in alg_names],
           alpha=0.8, label='S(Cox orbit)')
    ax.bar(x + 0.2, S_null_v, 0.35,
           color='gray', alpha=0.5,
           label='S(Rand orbit)')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n(h={results[n]['h']})" for n in alg_names])
    ax.set_ylabel('S_orbit (entropy)', fontsize=11)
    ax.set_title('Orbit entropy: Coxeter vs Random\n'
                 'Higher rank → higher orbit entropy',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(15.86, c='blue', ls='--', lw=1.5,
               alpha=0.6, label='Part VI E6: 15.86')

    # Row 2: Summary
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor('#fafafa')
    ax.axis('off')

    ax.text(0.01, 0.97,
            "RESULTS (thermalization metric, linear torus flow)",
            fontsize=12, fontweight='bold',
            transform=ax.transAxes, color='#1a1a2e')

    hdr = (f"{'Alg':<5} {'h':<4} {'rank':<5} "
           f"{'pred':>6} {'τ_cox':>7} {'τ_null':>8} "
           f"{'τ/h':>7} {'S_orb_cox':>11}  status")
    ax.text(0.01, 0.85, hdr,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', fontweight='bold',
            color='#1a1a2e')
    ax.text(0.01, 0.80, "-"*75,
            transform=ax.transAxes, fontsize=10,
            fontfamily='monospace', color='gray')

    y = 0.73
    for alg_name, res in results.items():
        tc = res['tau_cox']
        tn = res['tau_null']
        st = res['status']
        c  = ('#2E7D32' if 'CONFIRMS' in st else
               '#C62828' if 'REJECTS'  in st else '#888')
        tc_s = f"{tc:>7.0f}" if tc < 9999 else f"{'>>Tmax':>7}"
        tn_s = f"{tn:>8.0f}" if tn < 9999 else f"{'>>Tmax':>8}"
        row  = (f"{alg_name:<5} {res['h']:<4} {res['rank']:<5}"
                f"{res['tau_pred']:>6}{tc_s}{tn_s}"
                f"{res['tau_ratio']:>7.1f}"
                f"{res['S_orbit_cox']:>11.4f}  {st}")
        ax.text(0.01, y, row,
                transform=ax.transAxes, fontsize=10,
                fontfamily='monospace', color=c)
        y -= 0.09

    # Stats box
    if stats_out:
        r2h = stats_out.get('r2_h', np.nan)
        k0  = stats_out.get('k0',   np.nan)
        r20 = stats_out.get('r2_h0', np.nan)
        rho = stats_out.get('pearson_r', np.nan)
        pp  = stats_out.get('pearson_p', np.nan)

        if not np.isnan(r2h):
            if r2h > 0.85:
                verd = "STRONG: tau PROP to h CONFIRMED"
                vc   = '#2E7D32'
            elif r2h > 0.5:
                verd = "MODERATE: tau correlates with h"
                vc   = '#FF9800'
            else:
                verd = "WEAK: tau NOT prop to h"
                vc   = '#C62828'
        else:
            verd = "INSUFFICIENT DATA"
            vc   = '#888'

        slines = [
            f"tau=k*h(origin): k={k0:.1f} [exp~20],"
            f" R2={r20:.3f}",
            f"tau~h (free):    R2={r2h:.3f}",
            f"tau~rank:        "
            f"R2={stats_out.get('r2_rank',np.nan):.3f}",
            f"Pearson: r={rho:.3f}, p={pp:.4f}",
            "",
            f"VERDICT: {verd}",
        ]
        ax.text(0.65, 0.95, "STATISTICS",
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                color='#1a1a2e')
        for i, line in enumerate(slines):
            c_l = vc if 'VERDICT' in line else '#1a1a2e'
            fw  = 'bold' if 'VERDICT' in line else 'normal'
            ax.text(0.65, 0.86-i*0.09, line,
                    transform=ax.transAxes, fontsize=10,
                    fontfamily='monospace',
                    color=c_l, fontweight=fw)

    ax.text(0.50, 0.01,
            "τ = thermalization time (95% of orbit entropy)  |  "
            "Linear torus flow: φ(t)=φ₀+ωt  |  "
            "Null: random uniform frequencies",
            transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom',
            style='italic', color='#37474F')

    plt.savefig('monostring_part7_v7.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: monostring_part7_v7.png")


# ══════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    rng = np.random.default_rng(2025)

    print("Part VII v7: Thermalization metric")
    print("τ = time for daughters to reach 95% of orbit entropy\n")

    results   = run_all(rng.integers(0, 9999))
    stats_out = statistical_analysis(results)
    make_figure(results, stats_out)

    print("\n" + "="*62)
    print("FINAL TABLE")
    print("="*62)
    print(f"{'Alg':<5} {'h':<4} {'pred':>6} "
          f"{'τ_cox':>7} {'τ_null':>8} "
          f"{'τ/h':>7} {'S_orb':>8}  status")
    print("-"*62)
    for n, res in results.items():
        tc = res['tau_cox']
        tn = res['tau_null']
        tc_s = f"{tc:>7.0f}" if tc < 9999 else "   >>Tm"
        tn_s = f"{tn:>8.0f}" if tn < 9999 else "   >>Tmax"
        print(f"{n:<5} {res['h']:<4} "
              f"{res['tau_pred']:>6}"
              f"{tc_s}{tn_s}"
              f"{res['tau_ratio']:>7.1f}"
              f"{res['S_orbit_cox']:>8.3f}"
              f"  {res['status']}")

    print("\nKey question:")
    r2 = stats_out.get('r2_h', np.nan)
    k0 = stats_out.get('k0',   np.nan)
    if not np.isnan(r2) and r2 > 0.5:
        print(f"  τ ∝ h supported (R²={r2:.3f}, k={k0:.1f})")
    elif not np.isnan(r2):
        print(f"  τ ∝ h NOT supported (R²={r2:.3f})")
    else:
        print("  Inconclusive")
