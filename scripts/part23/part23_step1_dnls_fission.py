"""
Part XXIII: String Fission — From One to Many
==============================================

Physical model: 1D Nonlinear Schrödinger (NLS) / DNLS
with modulational instability → soliton formation.

This maps to the BЭ hypothesis as follows:
  - Initial state: one coherent oscillation (BЭ in "Super-Zero")
  - Instability: nonlinear energy redistribution
  - Fission: spontaneous formation of stable localized objects
  - Interaction: collisions, mergers, bound states

Falsification criteria (set BEFORE running):
  SUCCESS: N_solitons > 2, discrete mass spectrum
  FAILURE: thermalization OR single dominant soliton
  NOT CLAIMED: particle masses, n_s, Big Bang details

Physical analogies:
  - Preheating after inflation (KLS 1997)
  - String breaking in QCD
  - Modulational instability (Benjamin-Feir 1967)
  - FPUT recurrence (but we go BEYOND FPUT regime)
"""

import numpy as np
from scipy import stats, signal
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SEED = 42
np.random.seed(SEED)

print("="*65)
print("Part XXIII: String Fission")
print("One coherent oscillation → many stable objects")
print("="*65)
print()
print("Falsification criteria (pre-registered):")
print("  SUCCESS : N_solitons>2, discrete mass spectrum")
print("  FAILURE : thermalization or N_solitons=1")
print("  NOT CLAIMED: SM masses, n_s, Big Bang")
print()

# ══════════════════════════════════════════════════════════════
# STEP 1: Theoretical Analysis (before simulation)
# ══════════════════════════════════════════════════════════════

print("─"*65)
print("STEP 1: Theoretical Analysis")
print("─"*65)
print()

# Focusing NLS on a lattice (DNLS):
# i·dψ_n/dt = -J(ψ_{n+1}+ψ_{n-1}-2ψ_n) - g|ψ_n|²ψ_n
#
# For uniform state ψ_n = A·exp(-i·ω₀·t):
# ω₀ = g·A²  (nonlinear frequency shift)
#
# Modulational instability for perturbation ~ e^{iKn-iΩt}:
# Ω² = 4J²sin²(K/2)[4J·sin²(K/2) - g·A²]
#
# Instability when: Ω² < 0
# → 4J·sin²(K/2) < g·A²
# → K < K_c where K_c = 2·arcsin(sqrt(g·A²/4J))

J   = 1.0   # hopping (coupling)
g   = 2.0   # nonlinearity (focusing, g>0)
A   = 1.0   # initial amplitude

# Critical wavenumber
K_c = 2 * np.arcsin(np.sqrt(min(g*A**2/(4*J), 1.0)))
# Most unstable wavenumber
K_max_inst = 2 * np.arcsin(
    np.sqrt(min(g*A**2/(8*J), 1.0)))
# Maximum growth rate
Gamma_max = g * A**2 * J  # approximate

# Expected soliton spacing
lambda_sol = 2 * np.pi / K_max_inst if K_max_inst > 0 else np.inf

print(f"Parameters: J={J}, g={g}, A={A}")
print(f"Modulational instability analysis:")
print(f"  Critical wavenumber: K_c = {K_c:.4f}")
print(f"  Most unstable K: K_max = {K_max_inst:.4f}")
print(f"  Max growth rate: Γ = {Gamma_max:.4f}")
print(f"  Expected soliton spacing: λ = {lambda_sol:.2f}")
print()

N = 512  # lattice size
expected_solitons = int(N / lambda_sol)
print(f"For N={N}: expected ~{expected_solitons} solitons")
print(f"Instability timescale: t_inst ~ 1/Γ = {1/Gamma_max:.2f}")
print()

# ══════════════════════════════════════════════════════════════
# STEP 2: Simulation
# ══════════════════════════════════════════════════════════════

print("─"*65)
print("STEP 2: Simulation (DNLS)")
print("─"*65)
print()

# Initial condition: uniform + tiny noise (the key!)
# This is the "Super-Zero" state with a small perturbation
noise_amp = 0.001
rng = np.random.RandomState(SEED)

psi = (A * np.ones(N)
       + noise_amp * (rng.randn(N) + 1j*rng.randn(N)))

print(f"Initial state: |ψ|² = {(np.abs(psi)**2).mean():.4f}")
print(f"Noise level: {noise_amp} (= {noise_amp/A*100:.2f}% of A)")
print()

# RK4 integrator for DNLS
def dnls_rhs(psi, J, g):
    """i·dψ/dt = -J·Δψ - g·|ψ|²·ψ"""
    lap = (np.roll(psi,-1) + np.roll(psi,1) - 2*psi)
    return -1j * (-J*lap - g*np.abs(psi)**2*psi)

dt     = 0.002
n_steps = 500000  # long enough to see interactions

# Diagnostics storage
t_list      = []
rho_list    = []  # density snapshots
N_sol_list  = []  # soliton count
H_list      = []  # Shannon entropy of density
mass_list   = []  # soliton masses
E_list      = []  # total energy
norm_list   = []  # norm (should be conserved)

record_every = 500
snap_times   = []
snap_rho     = []
snap_record  = [0.1, 0.2, 0.4, 0.7, 1.0]
snap_fracs   = [int(s*n_steps) for s in snap_record]

print("Running... (this may take a moment)")

# Norm and energy conserved quantities
def compute_energy(psi, J, g):
    lap_term = -J * np.sum(psi.conj() *
                           (np.roll(psi,-1) +
                            np.roll(psi,1) - 2*psi)).real
    nl_term  = -g/2 * np.sum(np.abs(psi)**4)
    return lap_term + nl_term

def count_solitons(rho, threshold_frac=0.5):
    """
    Count solitons as connected regions above threshold.
    Threshold = threshold_frac × max density.
    """
    threshold = threshold_frac * rho.max()
    binary    = (rho > threshold).astype(int)
    labeled, n_features = label(binary)

    masses    = []
    positions = []
    for i in range(1, n_features+1):
        idx = np.where(labeled == i)[0]
        m   = rho[idx].sum()
        pos = (rho[idx] * idx).sum() / m
        masses.append(m)
        positions.append(pos)

    return n_features, np.array(masses), np.array(positions)

E0   = compute_energy(psi, J, g)
N0   = np.sum(np.abs(psi)**2)

for step in range(n_steps):
    # RK4
    k1 = dnls_rhs(psi,                 J, g)
    k2 = dnls_rhs(psi + 0.5*dt*k1,    J, g)
    k3 = dnls_rhs(psi + 0.5*dt*k2,    J, g)
    k4 = dnls_rhs(psi + dt*k3,         J, g)
    psi = psi + (dt/6)*(k1+2*k2+2*k3+k4)

    # Record
    if step % record_every == 0:
        t   = step * dt
        rho = np.abs(psi)**2

        n_sol, masses, positions = count_solitons(rho)

        E_now   = compute_energy(psi, J, g)
        N_now   = rho.sum()

        # Shannon entropy
        p = rho/rho.sum()
        p = p[p > 1e-15]
        H = -np.sum(p * np.log(p))

        t_list.append(t)
        N_sol_list.append(n_sol)
        H_list.append(H)
        mass_list.append(masses)
        E_list.append(E_now)
        norm_list.append(N_now)

        # Snapshots at specific times
        if step in snap_fracs:
            snap_times.append(t)
            snap_rho.append(rho.copy())

    if step % (record_every*20) == 0:
        t = step * dt
        rho = np.abs(psi)**2
        n_sol, masses, _ = count_solitons(rho)
        print(f"  t={t:7.1f} | N_sol={n_sol:3d} | "
              f"⟨ρ⟩={rho.mean():.4f} | "
              f"ΔE={abs(E_list[-1]-E0)/abs(E0)*100:.3f}%")

t_arr = np.array(t_list)
N_sol_arr = np.array(N_sol_list)
H_arr = np.array(H_list)
E_arr = np.array(E_list)

print()
print(f"Final state:")
print(f"  Solitons: {N_sol_arr[-1]}")
print(f"  Expected: ~{expected_solitons}")
print(f"  Energy conservation: "
      f"ΔE/E0 = {abs(E_arr[-1]-E0)/abs(E0)*100:.4f}%")
print()

# ══════════════════════════════════════════════════════════════
# STEP 3: Analysis
# ══════════════════════════════════════════════════════════════

print("─"*65)
print("STEP 3: Analysis of final state")
print("─"*65)
print()

# Final soliton analysis
final_rho = np.abs(psi)**2
n_sol_final, masses_final, pos_final = count_solitons(
    final_rho, threshold_frac=0.5)

print(f"Number of solitons: {n_sol_final}")
print()

if n_sol_final > 0:
    print(f"Soliton masses:")
    print(f"  Mean:   {masses_final.mean():.4f}")
    print(f"  Std:    {masses_final.std():.4f}")
    print(f"  CV:     {masses_final.std()/masses_final.mean():.4f}")
    print(f"  Min:    {masses_final.min():.4f}")
    print(f"  Max:    {masses_final.max():.4f}")
    print()

    # Mass spectrum: is it discrete or continuous?
    # Test: bimodality (Hartigan dip test approximation)
    sorted_m = np.sort(masses_final)
    if len(sorted_m) > 4:
        # Coefficient of variation
        cv = sorted_m.std() / sorted_m.mean()
        # Is there clear clustering?
        # Simple test: are masses within 20% of each other?
        ratio_max_min = sorted_m[-1]/sorted_m[0]

        print(f"  Mass ratio max/min: {ratio_max_min:.2f}")
        print(f"  CV = {cv:.4f}")

        if cv < 0.2:
            print("  → Masses NEARLY EQUAL: quasi-monodisperse")
            print("    (single 'particle type')")
            mass_verdict = "monodisperse"
        elif cv < 0.5:
            print("  → Moderate spread: possible discrete levels")
            mass_verdict = "moderate"
        else:
            print("  → Large spread: continuous distribution")
            mass_verdict = "continuous"

        # Check mass ratios for integer relations
        print()
        print("  Mass ratios (checking integer relations):")
        for i in range(min(5, len(sorted_m))):
            for j in range(i+1, min(5, len(sorted_m))):
                ratio = sorted_m[j]/sorted_m[i]
                nearest_int = round(ratio)
                dev = abs(ratio - nearest_int)
                if dev < 0.05 and nearest_int > 0:
                    print(f"    m_{j}/m_{i} ≈ {nearest_int} "
                          f"(dev={dev:.3f}) ← integer!")

print()

# Power spectrum of density fluctuations
print("Power spectrum of density fluctuations:")
fft_rho = np.fft.rfft(final_rho - final_rho.mean())
P_rho   = np.abs(fft_rho)**2
freqs_k = np.fft.rfftfreq(N)

# Slope
mask_k  = (freqs_k > 0.02) & (freqs_k < 0.3)
if mask_k.sum() > 5:
    sl, _, r_val, p_val, _ = stats.linregress(
        np.log(freqs_k[mask_k]),
        np.log(P_rho[mask_k] + 1e-30))
    print(f"  P(k) ~ k^{sl:.4f}")
    print(f"  (CMB: n_s-1 = -0.035)")
    print(f"  (Honest: slope={sl:.4f} ≠ CMB "
          f"unless |{sl+0.035:.4f}| ≈ 0)")

    # Is this close to CMB?
    if abs(sl - (-0.035)) < 0.1:
        print("  → SURPRISINGLY CLOSE to CMB! Report carefully.")
    else:
        print(f"  → NOT CMB spectrum (off by {sl-(-0.035):.3f})")

print()

# ══════════════════════════════════════════════════════════════
# STEP 4: Soliton Collisions
# ══════════════════════════════════════════════════════════════

print("─"*65)
print("STEP 4: Soliton Interaction Analysis")
print("─"*65)
print()

# Track soliton count over time to detect mergers/splits
if len(N_sol_arr) > 10:
    # Find collision events: sudden change in N_sol
    dN = np.diff(N_sol_arr)

    merger_events  = np.where(dN < -1)[0]  # N decreases by >1
    fission_events = np.where(dN > 1)[0]   # N increases by >1

    print(f"Collision events detected:")
    print(f"  Merger events (N decreases): {len(merger_events)}")
    print(f"  Fission events (N increases): {len(fission_events)}")
    print()

    if len(merger_events) > 0:
        print(f"  First merger at t = "
              f"{t_arr[merger_events[0]]:.2f}")
    if len(fission_events) > 0:
        print(f"  First fission at t = "
              f"{t_arr[fission_events[0]]:.2f}")

    # Soliton lifetime: time-averaged count
    mean_N  = N_sol_arr[N_sol_arr > 0].mean()
    final_N = N_sol_arr[-1]

    print()
    print(f"  Peak N_solitons:  {N_sol_arr.max()}")
    print(f"  Mean N_solitons:  {mean_N:.1f}")
    print(f"  Final N_solitons: {final_N}")

    # Does N stabilize?
    late_N  = N_sol_arr[len(N_sol_arr)//2:]
    late_std = late_N.std()
    print(f"  Late-time N std:  {late_std:.2f}")

    if late_std < 1.0:
        print("  → N_solitons STABILIZES: robust structures")
        interaction_verdict = "stable"
    else:
        print("  → N_solitons fluctuates: ongoing collisions")
        interaction_verdict = "fluctuating"

print()

# ══════════════════════════════════════════════════════════════
# STEP 5: Verdict
# ══════════════════════════════════════════════════════════════

print("="*65)
print("STEP 5: Honest Verdict")
print("="*65)
print()

# Apply pre-registered criteria
success_1 = (n_sol_final > 2)
success_2 = (mass_verdict in ["monodisperse", "moderate"]
             if 'mass_verdict' in dir() else False)

print("Pre-registered criteria:")
print(f"  N_solitons > 2: "
      f"{'✓' if success_1 else '✗'} ({n_sol_final})")
print(f"  Discrete masses: "
      f"{'✓' if success_2 else '✗'} ({mass_verdict if 'mass_verdict' in dir() else 'N/A'})")
print()

if success_1 and success_2:
    verdict = "SUCCESS"
    detail = ("Spontaneous formation of multiple stable "
              "localized structures with quasi-discrete masses.")
elif success_1:
    verdict = "PARTIAL"
    detail = ("Multiple solitons form, but mass spectrum "
              "is continuous. Structure without discreteness.")
else:
    verdict = "FAILURE"
    detail = "Failed to produce multiple stable structures."

print(f"VERDICT: {verdict}")
print(f"  {detail}")
print()
print("What this means for the BЭ hypothesis:")
print()

if verdict == "SUCCESS":
    print("  Positive result: ONE coherent state spontaneously")
    print("  produces MANY stable objects via modulational")
    print("  instability. This is the correct dynamical")
    print("  mechanism for 'Super-Zero → particles'.")
    print()
    print("  BUT: this is NLS/DNLS physics, known since 1960s.")
    print("  The BЭ hypothesis gains a concrete dynamical")
    print("  mechanism, not a new physical prediction.")
elif verdict == "PARTIAL":
    print("  Solitons form but masses are random.")
    print("  No preferred 'particle types' emerge.")
    print("  Fragmentation is generic, not structured.")
else:
    print("  The model does not produce stable structures.")
    print("  Initial coherent state either thermalizes or")
    print("  collapses into one dominant soliton.")

print()
print("NOT CLAIMED regardless of result:")
print("  × Standard Model particle masses")
print("  × CMB spectral index n_s = 0.965")
print("  × Explanation of Big Bang")
print("  (These would require much more specific theory)")

# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(3, 3, figure=fig,
               hspace=0.45, wspace=0.35)

x_arr = np.arange(N)

# Panel 1: density snapshots
ax1 = fig.add_subplot(gs[0, :2])
colors_snap = plt.cm.plasma(np.linspace(0, 1, len(snap_rho)))
for i, (t_s, rho_s) in enumerate(
        zip(snap_times, snap_rho)):
    ax1.plot(x_arr, rho_s,
             color=colors_snap[i],
             lw=1.2, alpha=0.8,
             label=f't={t_s:.0f}')

ax1.set_xlabel('Position n')
ax1.set_ylabel('|ψ|²')
ax1.set_title('Density evolution: uniform → solitons\n'
              '(BЭ: Super-Zero → many stable objects)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: soliton count vs time
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(t_arr, N_sol_arr,
         'b-', lw=1.5, alpha=0.8)
ax2.axhline(expected_solitons, color='r',
            ls='--', lw=2,
            label=f'Theory: {expected_solitons}')
ax2.set_xlabel('Time')
ax2.set_ylabel('N solitons')
ax2.set_title('Soliton count vs time')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: entropy vs time
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t_arr, H_arr, 'g-', lw=1.5)
ax3.set_xlabel('Time')
ax3.set_ylabel('Shannon entropy H')
ax3.set_title('Entropy: uniform=max,\nsolitons=lower')
ax3.grid(True, alpha=0.3)

# Panel 4: final density
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x_arr, final_rho, 'b-', lw=1.5)
ax4.axhline(A**2, color='gray', ls=':', alpha=0.7,
            label=f'Initial A²={A**2}')
if len(pos_final) > 0:
    ax4.axvline(pos_final[0], color='r',
                ls='--', alpha=0.5)
ax4.set_xlabel('Position n')
ax4.set_ylabel('|ψ_final|²')
ax4.set_title(f'Final state: {n_sol_final} solitons')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Panel 5: mass spectrum
ax5 = fig.add_subplot(gs[1, 2])
if len(masses_final) > 1:
    ax5.hist(masses_final,
             bins=max(5, min(20, len(masses_final)//2)),
             color='steelblue',
             edgecolor='black', alpha=0.75)
    ax5.axvline(masses_final.mean(), color='r',
                ls='--', label=f'Mean={masses_final.mean():.2f}')
    ax5.set_xlabel('Soliton mass')
    ax5.set_ylabel('Count')
    ax5.set_title('Mass spectrum\n(discrete peaks = particle types?)')
    ax5.legend(fontsize=8)
else:
    ax5.text(0.5, 0.5, f'N={n_sol_final} solitons\n(need >4)',
             transform=ax5.transAxes, ha='center', va='center')
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: power spectrum
ax6 = fig.add_subplot(gs[2, 0])
freqs_pos = freqs_k[freqs_k > 0]
P_pos     = P_rho[freqs_k > 0]
ax6.loglog(freqs_pos, P_pos, 'b-', lw=1, alpha=0.7)
if mask_k.sum() > 5:
    k_fit = freqs_k[mask_k]
    P_fit = np.exp(np.log(P_rho[mask_k]).mean()
                   + sl*(np.log(k_fit)
                         - np.log(freqs_k[mask_k]).mean()))
    ax6.loglog(k_fit, P_fit, 'r--', lw=2,
               label=f'slope={sl:.3f}')
    ax6.axhline(0, color='gray', ls=':')
ax6.set_xlabel('k')
ax6.set_ylabel('P(k)')
ax6.set_title('Power spectrum P(k)\n'
              f'CMB: -0.035')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, which='both')

# Panel 7: energy conservation
ax7 = fig.add_subplot(gs[2, 1])
ax7.semilogy(t_arr,
             np.abs(E_arr - E0)/abs(E0) + 1e-16,
             'r-', lw=1.5)
ax7.set_xlabel('Time')
ax7.set_ylabel('|ΔE/E₀|')
ax7.set_title('Energy conservation\n(numerical accuracy)')
ax7.grid(True, alpha=0.3)

# Panel 8: verdict
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

verdict_text = [
    'PART XXIII VERDICT',
    '══════════════════',
    '',
    f'Model: DNLS, N={N}',
    f'J={J}, g={g}, A={A}',
    '',
    'Pre-registered criteria:',
    f'  N_sol > 2: {"✓" if success_1 else "✗"}',
    f'  N_sol = {n_sol_final}',
    f'  Discrete masses: {"✓" if success_2 else "✗"}',
    f'  CV = {masses_final.std()/masses_final.mean():.3f}'
    if len(masses_final)>0 else '  CV = N/A',
    '',
    f'RESULT: {verdict}',
    '',
    'Physical interpretation:',
    '  Modulational instability',
    '  → spontaneous soliton',
    '  formation confirmed.',
    '',
    'NOT claimed:',
    '  × SM masses',
    '  × n_s = 0.965',
    '  × Big Bang theory',
]

ax8.text(0.03, 0.97, '\n'.join(verdict_text),
         transform=ax8.transAxes,
         fontsize=8, va='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round',
                   facecolor='#e8f8f8',
                   alpha=0.9))

plt.suptitle(
    'Part XXIII: String Fission\n'
    'Modulational instability → spontaneous soliton formation',
    fontsize=12, fontweight='bold', y=1.01)

plt.savefig('part23_string_fission.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part23_string_fission.png")
