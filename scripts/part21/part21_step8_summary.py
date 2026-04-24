# scripts/part21/part21_summary.py
"""
Part XXI Summary Script
Reproduces all key numbers from Part XXI in one run.
Expected runtime: ~3 minutes.
"""

import numpy as np
from scipy import stats

SEED = 42
np.random.seed(SEED)

print("="*60)
print("Part XXI: Stochastic BЭ — Summary Reproduction")
print("="*60)
print()

m2 = 0.01
m  = np.sqrt(m2)
N  = 512

# ── KEY RESULT 1: Metropolis → KG ────────────────────────────

def metropolis_kg(N, m2, T, n_sweeps, seed=SEED):
    rng   = np.random.RandomState(seed)
    phi   = rng.randn(N) * np.sqrt(T/(m2+4))
    delta = np.sqrt(T/(m2+4))
    for _ in range(n_sweeps):
        sites = rng.permutation(N)
        for i in sites:
            l = (i-1)%N; r = (i+1)%N
            pn = phi[i] + delta*rng.randn()
            dS = (0.5*m2*(pn**2-phi[i]**2)
                  +0.5*((pn-phi[l])**2-(phi[i]-phi[l])**2)
                  +0.5*((pn-phi[r])**2-(phi[i]-phi[r])**2))
            if dS<0 or rng.rand()<np.exp(-dS/T):
                phi[i] = pn
    return phi

phi = metropolis_kg(N, m2, 1.0, 3000)
phi_c = phi - phi.mean()
fft_p = np.fft.rfft(phi_c)
P_mc  = np.abs(fft_p)**2/N
freqs = np.fft.rfftfreq(N)

mask  = (freqs>0.08)&(freqs<0.10)
sl_mc, _, _, _, _ = stats.linregress(
    np.log(freqs[mask]), np.log(P_mc[mask]+1e-30))

corr = np.correlate(phi_c, phi_c, mode='full')
corr = corr[N-1:]/corr[N-1]
lc   = np.where(corr<1/np.e)[0]
lc   = lc[0] if len(lc)>0 else N

print("KEY RESULT 1: Metropolis MC")
print(f"  slope [0.08,0.10] = {sl_mc:.4f}  (expect ≈-1.88)")
print(f"  l_corr            = {lc:.0f}      (expect ≈10)")
print()

# ── KEY RESULT 2: Exact KG slope ─────────────────────────────

k_idx = np.arange(N//2+1)
freqs_ex = k_idx/N
lam_ex = 2-2*np.cos(2*np.pi*freqs_ex)+m2
P_ex   = 1.0/lam_ex

mask2  = (freqs_ex>0.08)&(freqs_ex<0.10)
sl_ex, _, _, _, _ = stats.linregress(
    np.log(freqs_ex[mask2]), np.log(P_ex[mask2]))

print("KEY RESULT 2: Exact KG lattice spectrum")
print(f"  slope [0.08,0.10] = {sl_ex:.4f}  (expect ≈-1.88)")
print()

# ── KEY RESULT 3: CMB gap ────────────────────────────────────

slope_cmb = -0.035
gap = abs(sl_ex - slope_cmb)

print("KEY RESULT 3: CMB gap")
print(f"  slope_KG  = {sl_ex:.4f}")
print(f"  slope_CMB = {slope_cmb:.4f}  (Planck n_s-1)")
print(f"  gap       = {gap:.4f}  (fundamental)")
print()

# ── KEY RESULT 4: Lattice artifact ───────────────────────────

f_test = 0.10
theta  = np.pi*f_test
slope_analytic = (-2*np.pi*f_test*np.cos(theta)/np.sin(theta)
                  / (1+(m/(2*np.sin(theta)))**2))

print("KEY RESULT 4: Lattice slope formula")
print(f"  At f=0.10:")
print(f"  slope_analytic = {slope_analytic:.4f}  (≠ -2.000)")
print()

# ── FINAL SCORECARD ───────────────────────────────────────────

print("="*60)
print("FINAL SCORECARD (Parts I-XXI)")
print("="*60)
print(f"  Experiments : 21")
print(f"  Signals     : 0")
print(f"  Theorems    : 6")
print(f"  Artifacts   : 13 (11 prev + 2 in XXI)")
print(f"  Frameworks  : 8 (classical, gauge, spectral,")
print(f"                   graph, Cayley, XXZ, inflation,")
print(f"                   stochastic)")
print()
print("  Philosophical idea   : SURVIVES")
print("  New physical content : NONE")
print()
print("BЭ ≡ Euclidean QFT path integral (theorem).")
print("No new predictions beyond QFT.")
