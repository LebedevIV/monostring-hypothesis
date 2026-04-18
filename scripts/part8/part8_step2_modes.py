"""
Part VIII — Step 2: Collective Modes and Dispersion Relation
=============================================================
Question: Do N daughter strings develop wave-like collective modes?

Method:
1. Generate N=100 daughter strings near phi_break
2. Track center-of-mass oscillations over time
3. Compute Fourier spectrum of collective coordinate q(t)
4. Fit dispersion relation ω(k)

Physical interpretation:
- ω² ∝ k²        → massless field (photon-like, c=const)
- ω² ∝ k² + m²   → massive scalar field (Higgs-like)
- ω = const       → no propagation (frozen mode)
- No clear pattern → no field interpretation

Key insight from Step 1:
E8 has 4+4 eigenvalue split → two distinct "sectors"
Do collective modes also split into two groups?
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# 1. SETUP (same as Step 1)
# ══════════════════════════════════════════════════════════════════

ALGEBRAS = {
    'E6': {'rank': 6, 'h': 12,
            'exponents': [1, 4, 5, 7, 8, 11],
            'color': '#2196F3'},
    'E8': {'rank': 8, 'h': 30,
            'exponents': [1, 7, 11, 13, 17, 19, 23, 29],
            'color': '#4CAF50'},
    'Random': {'rank': 6, 'h': None,
                'exponents': None,
                'color': '#9E9E9E'},
}

KAPPA = 0.05
NOISE = 0.006


def get_omega(alg_name, seed=42):
    alg = ALGEBRAS[alg_name]
    if alg['exponents'] is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.5, 2.0, alg['rank'])
    m = np.array(alg['exponents'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])


def get_phi_break(omega, seed=42, warmup=500):
    """Get attractor point for fragmentation."""
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omega))
    for _ in range(warmup):
        phi = (phi + omega
               + 0.30 * np.sin(phi)) % (2*np.pi)
    return phi


# ══════════════════════════════════════════════════════════════════
# 2. GENERATE DAUGHTER CLOUD TRAJECTORY
# ══════════════════════════════════════════════════════════════════

def generate_cloud_trajectory(omega, N=100,
                               sigma=0.3, T=2000,
                               seed=42):
    """
    Generate trajectory of N daughter strings.

    Returns:
    - traj: (T, N, rank) — full trajectory
    - q_cm: (T, rank)   — center of mass
    - q_rel: (T, N, rank) — relative positions
    """
    rng = np.random.RandomState(seed)
    rank = len(omega)

    phi_break = get_phi_break(omega, seed=seed)

    # Initialize daughters in sigma-ball
    phi = (phi_break[np.newaxis, :]
           + rng.normal(0, sigma, (N, rank))
           ) % (2*np.pi)

    traj  = np.zeros((T, N, rank))
    q_cm  = np.zeros((T, rank))
    q_std = np.zeros((T, rank))

    for t in range(T):
        phi = (phi + omega[np.newaxis, :]
               + KAPPA * np.sin(phi)
               + rng.normal(0, NOISE, phi.shape)
               ) % (2*np.pi)
        traj[t]  = phi
        q_cm[t]  = phi.mean(axis=0)
        q_std[t] = phi.std(axis=0)

    return traj, q_cm, q_std


# ══════════════════════════════════════════════════════════════════
# 3. SPECTRAL ANALYSIS OF COLLECTIVE MODES
# ══════════════════════════════════════════════════════════════════

def spectral_analysis(q_cm, dt=1.0,
                       nperseg=256, label=""):
    """
    Compute power spectral density of each collective coordinate.

    q_cm: (T, rank) — center of mass trajectory

    Returns:
    - freqs: frequency array
    - psd:   (rank, len(freqs)) power spectral density
    - dominant_freqs: dominant frequency per dimension
    """
    T, rank = q_cm.shape
    freqs_list = []
    psd_list   = []

    for d in range(rank):
        signal = q_cm[:, d]
        # Remove mean (DC component)
        signal = signal - signal.mean()
        f, Pxx = welch(signal, fs=1.0/dt,
                       nperseg=min(nperseg, T//4),
                       scaling='density')
        freqs_list.append(f)
        psd_list.append(Pxx)

    freqs = freqs_list[0]  # same for all dims
    psd   = np.array(psd_list)  # (rank, n_freqs)

    # Dominant frequency per dimension
    dom_freqs = freqs[np.argmax(psd, axis=1)]

    return freqs, psd, dom_freqs


# ══════════════════════════════════════════════════════════════════
# 4. DISPERSION RELATION
#
# Key idea: treat each dimension d as a "spatial mode"
# with "wavevector" k_d = omega_d (natural frequency)
# and "frequency" f_d = dominant PSD peak.
#
# If ω_collective ∝ k_natural → linear dispersion
# (massless, photon-like)
# ══════════════════════════════════════════════════════════════════

def fit_dispersion(omega, dom_freqs, alg_name):
    """
    Fit dispersion relation:
    ω_collective(d) vs k_d = omega_d

    Models:
    1. Linear:   ω = c * k          (massless)
    2. Massive:  ω² = c²k² + m²     (massive scalar)
    3. Flat:     ω = const           (non-dispersive)
    """
    k    = omega.copy()        # natural frequencies as k
    w    = dom_freqs.copy()    # collective freqs as ω

    results = {}

    # Sort by k for plotting
    idx = np.argsort(k)
    k_s = k[idx]
    w_s = w[idx]

    # Model 1: linear ω = c*k
    if len(k) >= 2:
        slope, intercept, r_lin, p_lin, _ = \
            stats.linregress(k_s, w_s)
        results['linear'] = {
            'slope': slope, 'intercept': intercept,
            'r2': r_lin**2, 'p': p_lin,
        }

    # Model 2: massless ω² = c²k²
    if len(k) >= 2:
        slope2, _, r2, p2, _ = \
            stats.linregress(k_s**2, w_s**2)
        results['massless'] = {
            'c2': slope2, 'r2': r2**2, 'p': p2,
        }

    # Model 3: massive ω² = c²k² + m²
    if len(k) >= 3:
        try:
            def massive(k, c, m):
                return np.sqrt(np.maximum(
                    c**2 * k**2 + m**2, 0))
            popt, pcov = curve_fit(
                massive, k_s, w_s,
                p0=[1.0, 0.1],
                maxfev=5000)
            w_pred = massive(k_s, *popt)
            ss_res = np.sum((w_s - w_pred)**2)
            ss_tot = np.sum((w_s - w_s.mean())**2)
            r2_m   = 1 - ss_res/ss_tot if ss_tot > 0 \
                     else np.nan
            results['massive'] = {
                'c': popt[0], 'm': popt[1],
                'r2': r2_m,
            }
        except Exception:
            results['massive'] = {
                'c': np.nan, 'm': np.nan,
                'r2': np.nan,
            }

    # Best model
    r2_vals = {
        'linear':   results.get('linear', {}).get('r2', 0),
        'massless': results.get('massless', {}).get('r2', 0),
        'massive':  results.get('massive', {}).get('r2', 0),
    }
    best = max(r2_vals, key=r2_vals.get)

    print(f"\n  [{alg_name}] Dispersion relation:")
    print(f"  {'k (omega)':>12}  {'ω_collective':>14}")
    for ki, wi in zip(k_s, w_s):
        print(f"  {ki:>12.4f}  {wi:>14.6f}")
    print(f"\n  Linear   (ω=c·k):     R²="
          f"{r2_vals['linear']:.4f}")
    print(f"  Massless (ω²=c²k²):   R²="
          f"{r2_vals['massless']:.4f}")
    print(f"  Massive  (ω²=c²k²+m²):R²="
          f"{r2_vals['massive']:.4f}")
    print(f"  Best model: {best}")

    results['best']  = best
    results['k_s']   = k_s
    results['w_s']   = w_s
    results['r2vals']= r2_vals

    return results


# ══════════════════════════════════════════════════════════════════
# 5. EIGENMODE DECOMPOSITION
#
# Beyond center-of-mass: decompose cloud fluctuations
# into eigenmodes using PCA of the (N, rank) cloud.
# Each eigenmode is a collective oscillation pattern.
# ══════════════════════════════════════════════════════════════════

def eigenmode_analysis(traj, omega, label=""):
    """
    Decompose cloud trajectory into collective eigenmodes.

    traj: (T, N, rank)

    At each time t, the cloud has shape (N, rank).
    We compute the covariance matrix of the cloud
    and track how its eigenvalues evolve.

    This gives:
    - Mean eigenvalue per mode over time
    - Variance of each mode (= amplitude of oscillation)
    - Whether modes are coupled or independent
    """
    T, N, rank = traj.shape

    # Compute cloud covariance at each timestep
    # Shape: (T, rank)
    cloud_std = np.zeros((T, rank))
    cloud_cov_trace = np.zeros(T)

    for t in range(T):
        snapshot = traj[t]  # (N, rank)
        centered = snapshot - snapshot.mean(0)
        C = (centered.T @ centered) / N
        ev = np.linalg.eigvalsh(C)
        cloud_std[t]       = np.sort(ev)[::-1]
        cloud_cov_trace[t] = ev.sum()

    # Spectral analysis of each eigenvalue mode
    mode_freqs = []
    mode_amps  = []
    for mode in range(rank):
        signal = cloud_std[:, mode]
        signal = signal - signal.mean()
        f, Pxx = welch(signal, fs=1.0,
                       nperseg=min(256, T//4))
        dom_f  = f[np.argmax(Pxx)]
        amp    = signal.std()
        mode_freqs.append(dom_f)
        mode_amps.append(amp)

    print(f"\n  [{label}] Eigenmode oscillations:")
    print(f"  {'Mode':>5}  {'Amplitude':>10}"
          f"  {'Dominant f':>12}")
    print(f"  {'-'*32}")
    for i, (a, f) in enumerate(
            zip(mode_amps, mode_freqs)):
        print(f"  {i+1:>5}  {a:>10.6f}  {f:>12.6f}")

    return cloud_std, mode_freqs, mode_amps


# ══════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_step2(seed=42):
    rng_seed = seed

    print("=" * 62)
    print("PART VIII Step 2: Collective Modes")
    print("=" * 62)

    all_results = {}

    for alg_name in ['E6', 'E8', 'Random']:
        alg   = ALGEBRAS[alg_name]
        omega = get_omega(alg_name, seed=rng_seed)
        rank  = alg['rank']

        print(f"\n{'='*55}")
        print(f"ALGEBRA: {alg_name}  rank={rank}")
        print(f"  omega = {np.round(omega, 4).tolist()}")

        # Generate cloud trajectory
        print(f"\n  Generating N=100 daughters, T=2000...")
        traj, q_cm, q_std = generate_cloud_trajectory(
            omega, N=100, sigma=0.3,
            T=2000, seed=rng_seed)

        # Spectral analysis of center of mass
        print(f"\n  Spectral analysis (center of mass):")
        freqs, psd, dom_freqs = spectral_analysis(
            q_cm, dt=1.0, nperseg=256,
            label=alg_name)

        # Dispersion relation
        disp = fit_dispersion(omega, dom_freqs, alg_name)

        # Eigenmode analysis
        print(f"\n  Eigenmode decomposition:")
        cloud_std, mode_freqs, mode_amps = \
            eigenmode_analysis(traj, omega,
                               label=alg_name)

        all_results[alg_name] = {
            'omega':      omega,
            'rank':       rank,
            'q_cm':       q_cm,
            'q_std':      q_std,
            'freqs':      freqs,
            'psd':        psd,
            'dom_freqs':  dom_freqs,
            'disp':       disp,
            'cloud_std':  cloud_std,
            'mode_freqs': mode_freqs,
            'mode_amps':  mode_amps,
            'color':      alg['color'],
        }

    return all_results


# ══════════════════════════════════════════════════════════════════
# 7. FIGURE
# ══════════════════════════════════════════════════════════════════

def make_figure(all_results):
    alg_list = list(all_results.keys())

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        "Part VIII — Step 2: Collective Modes\n"
        "Dispersion relation ω(k) of daughter string cloud\n"
        "Massless (photon-like) vs Massive vs No dispersion",
        fontsize=13, fontweight='bold', y=0.99,
        color='#1a1a2e')

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.52, wspace=0.38)

    colors_model = {
        'linear':   'red',
        'massless': 'blue',
        'massive':  'green',
    }

    for col, alg_name in enumerate(alg_list):
        res   = all_results[alg_name]
        omega = res['omega']
        disp  = res['disp']
        c     = res['color']

        # ── Row 0: PSD per dimension ───────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#f0f4f8')

        freqs = res['freqs']
        psd   = res['psd']
        rank  = res['rank']

        for d in range(rank):
            alpha = 0.9 if d == 0 else 0.4
            lw    = 2.0 if d == 0 else 1.0
            ax.semilogy(freqs, psd[d],
                        alpha=alpha, lw=lw,
                        label=f'd={d+1}')

        ax.set_xlabel('Frequency f', fontsize=9)
        ax.set_ylabel('PSD (log)', fontsize=9)
        ax.set_title(
            f'{alg_name}: Power Spectrum\n'
            f'per dimension',
            fontsize=10, fontweight='bold')
        if rank <= 6:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ── Row 1: Dispersion relation ─────────────────
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#f0f4f8')

        k_s = disp['k_s']
        w_s = disp['w_s']

        ax.scatter(k_s, w_s, s=150, c=c,
                   zorder=5, edgecolors='k',
                   lw=1.5, label='data')

        # Plot best fit
        k_fit = np.linspace(k_s.min() * 0.9,
                             k_s.max() * 1.1, 100)
        best  = disp['best']
        r2    = disp['r2vals'][best]

        if best == 'linear':
            sl = disp['linear']['slope']
            ic = disp['linear']['intercept']
            w_fit = sl * k_fit + ic
            lbl = f"linear: R²={r2:.3f}"
        elif best == 'massless':
            c2 = disp['massless']['c2']
            w_fit = np.sqrt(np.maximum(c2, 0)) * k_fit
            lbl = f"massless: R²={r2:.3f}"
        else:
            c_ = disp['massive']['c']
            m_ = disp['massive']['m']
            if not (np.isnan(c_) or np.isnan(m_)):
                w_fit = np.sqrt(np.maximum(
                    c_**2 * k_fit**2 + m_**2, 0))
                lbl = (f"massive c={c_:.3f},"
                       f"m={m_:.3f}: R²={r2:.3f}")
            else:
                w_fit = None
                lbl = "massive: fit failed"

        if w_fit is not None:
            ax.plot(k_fit, w_fit, '-',
                    color=colors_model[best],
                    lw=2.5, label=lbl)

        ax.set_xlabel('k = ω_natural', fontsize=10)
        ax.set_ylabel('ω_collective (dominant f)',
                      fontsize=9)
        ax.set_title(
            f'{alg_name}: Dispersion ω(k)\n'
            f'Best: {best}, R²={r2:.3f}',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Row 2: Eigenmode amplitudes ────────────────
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor('#f0f4f8')

        mode_amps  = np.array(res['mode_amps'])
        mode_freqs = np.array(res['mode_freqs'])
        modes      = np.arange(1, rank + 1)

        sc = ax.scatter(modes, mode_amps,
                        c=mode_freqs,
                        s=150, cmap='coolwarm',
                        zorder=5, edgecolors='k',
                        lw=1.5)
        plt.colorbar(sc, ax=ax,
                     label='Dominant freq',
                     shrink=0.8)

        ax.set_xlabel('Eigenmode index', fontsize=10)
        ax.set_ylabel('Amplitude (std)',  fontsize=10)
        ax.set_title(
            f'{alg_name}: Mode amplitudes\n'
            f'Color = dominant frequency',
            fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.savefig('part8_step2_modes.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\nSaved: part8_step2_modes.png")


# ══════════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Part VIII Step 2: Collective Modes\n")

    all_results = run_step2(seed=42)
    make_figure(all_results)

    # Summary
    print("\n" + "=" * 62)
    print("SUMMARY: Dispersion relations")
    print("=" * 62)
    print(f"\n{'Algebra':<10} {'Best model':>12} "
          f"{'R²':>8}  interpretation")
    print("-" * 55)

    for alg_name, res in all_results.items():
        disp = res['disp']
        best = disp['best']
        r2   = disp['r2vals'][best]

        if r2 > 0.85:
            if best == 'massless':
                interp = "PHOTON-LIKE (massless)"
            elif best == 'massive':
                m = disp['massive']['m']
                interp = f"MASSIVE SCALAR (m={m:.3f})"
            else:
                interp = "LINEAR dispersion"
        elif r2 > 0.5:
            interp = f"weak {best} signal"
        else:
            interp = "NO clear dispersion"

        print(f"{alg_name:<10} {best:>12} "
              f"{r2:>8.4f}  {interp}")

    print("\nPhysical meaning:")
    print("  Massless dispersion → field propagates at c=const")
    print("  Massive dispersion  → field has rest mass")
    print("  No dispersion       → no field interpretation")
    print("\nNext: Step 3 — Shortest paths as light rays")
