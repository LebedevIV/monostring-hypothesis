#!/usr/bin/env python3
"""
Part XXIV Patch-3: Финальный эксперимент double SG
IC верифицированы: Q=0, φ(±∞)=0
Используем kink_profile_inverse() из диагностики
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import eigsh
import time, os

jax.config.update("jax_enable_x64", True)
os.makedirs("figures/part24", exist_ok=True)

print("="*60)
print("Part XXIV Patch-3: Double SG — финальный эксперимент")
print("JAX:", jax.devices())
print("="*60)

# ============================================================
# ПОТЕНЦИАЛ
# ============================================================

def V_np(phi, r):
    return 1.0 - r*np.cos(phi/2) - (1-r)*np.cos(phi)

def dV_dphi_jax(phi, r):
    return (r/2.0)*jnp.sin(phi/2.0) + (1.0-r)*jnp.sin(phi)

def V_jax(phi, r):
    return 1.0 - r*jnp.cos(phi/2.0) - (1.0-r)*jnp.cos(phi)

# ============================================================
# ВЕРИФИЦИРОВАННЫЙ ПРОФИЛЬ КИНКА
# ============================================================

def kink_profile_inverse(x_arr, x_center, r, sign=+1):
    """
    Надёжный профиль через инверсию x(φ).
    ВЕРИФИЦИРОВАН: Q=0, φ(±∞) правильные.
    sign=+1: кинк     (0→4π)
    sign=-1: антикинк (4π→0)
    """
    N_phi = 200000
    phi_vals = np.linspace(1e-8, 4*np.pi - 1e-8, N_phi)
    V_vals = np.maximum(V_np(phi_vals, r), 1e-14)
    integrand = 1.0 / np.sqrt(2.0 * V_vals)

    dphi = phi_vals[1] - phi_vals[0]
    x_of_phi_from_0 = np.cumsum(integrand) * dphi

    idx_2pi = np.argmin(np.abs(phi_vals - 2*np.pi))
    x_at_2pi = x_of_phi_from_0[idx_2pi]
    x_of_phi = x_center + sign * (x_of_phi_from_0 - x_at_2pi)

    if sign > 0:
        phi_out = np.interp(x_arr, x_of_phi, phi_vals,
                             left=0.0, right=4*np.pi)
    else:
        phi_out = np.interp(x_arr, x_of_phi[::-1], phi_vals[::-1],
                             left=4*np.pi, right=0.0)
    return phi_out

def kink_mass(r):
    """M = ∫₀^{4π} √(2V) dφ"""
    phi_arr = np.linspace(1e-8, 4*np.pi - 1e-8, 200000)
    V_arr = np.maximum(V_np(phi_arr, r), 0.0)
    return np.trapezoid(np.sqrt(2.0 * V_arr), phi_arr)

# ============================================================
# IC ДЛЯ СТОЛКНОВЕНИЯ (верифицированные)
# ============================================================

def make_ic(x_arr, x_K, x_AK, v, r):
    """
    K+AK с Lorentz boost.
    Верифицировано: Q=0, φ(±∞)=0.

    Lorentz boost скорости v для кинка:
      φ_K(x,t) = φ_K^static(γ(x - x_K - vt))
    При t=0: φ_K(x,0) = φ_K^static(γ(x - x_K))
    π_K = ∂_t φ_K|_{t=0} = -v·γ · φ'_K^static(γ(x-x_K))
                           = -v · dφ_K/dx
    (γ сокращается: dφ/dx уже содержит γ от boosted сетки)
    """
    dx = x_arr[1] - x_arr[0]
    gamma = 1.0 / np.sqrt(max(1.0 - v**2, 1e-8))

    # Boosted сетки
    x_K_boosted  = gamma * (x_arr - x_K)  + x_K
    x_AK_boosted = gamma * (x_arr - x_AK) + x_AK

    # Широкая сетка для интерполяции
    x_ext = np.linspace(x_arr[0]-80, x_arr[-1]+80, len(x_arr)*6)

    phi_K_ext  = kink_profile_inverse(x_ext, x_K,  r, sign=+1)
    phi_AK_ext = kink_profile_inverse(x_ext, x_AK, r, sign=-1)

    phi_K  = np.interp(x_K_boosted,  x_ext, phi_K_ext,
                        left=0.0,    right=4*np.pi)
    phi_AK = np.interp(x_AK_boosted, x_ext, phi_AK_ext,
                        left=4*np.pi, right=0.0)

    phi_ic = phi_K + phi_AK - 4.0*np.pi

    # π = -v·dφ_K/dx + v·dφ_AK/dx
    dphi_K  = np.gradient(phi_K,  dx)
    dphi_AK = np.gradient(phi_AK, dx)
    pi_ic = -v * dphi_K + v * dphi_AK

    return jnp.array(phi_ic), jnp.array(pi_ic)

# ============================================================
# ИНТЕГРАТОР
# ============================================================

def make_stepper(dx, r):
    r_ = float(r)
    @jit
    def step(phi, pi, dt):
        def F(p):
            p_xx = (jnp.roll(p,-1)+jnp.roll(p,1)-2*p)/dx**2
            return p_xx - dV_dphi_jax(p, r_)
        F0 = F(phi)
        pi_h = pi + (dt/2)*F0
        phi_n = phi + dt*pi_h
        F1 = F(phi_n)
        pi_n = pi_h + (dt/2)*F1
        return phi_n, pi_n
    return step

def total_energy(phi, pi, dx, r):
    phi_x = jnp.gradient(phi, dx)
    return float(jnp.sum(pi**2/2 + phi_x**2/2 + V_jax(phi,r))*dx)

# ============================================================
# ДЕТЕКТОР РАЗДЕЛЕНИЯ
# ============================================================

def kink_separation(phi_np, dx, L):
    """
    Позиции K и AK через центр масс |dφ/dx|.
    Возвращает разделение и позиции.
    """
    N = len(phi_np)
    x_arr = np.linspace(-L/2, L/2, N, endpoint=False)
    dphi = np.gradient(phi_np, dx)
    dphi_s = gaussian_filter1d(dphi, sigma=max(1.0, 1.5/dx))

    pos = np.maximum(dphi_s, 0)
    neg = np.abs(np.minimum(dphi_s, 0))

    norm_p = pos.sum() + 1e-10
    norm_n = neg.sum() + 1e-10

    x_K  = (x_arr * pos).sum() / norm_p
    x_AK = (x_arr * neg).sum() / norm_n

    return abs(x_K - x_AK), x_K, x_AK

def count_bounces(sep_arr, sep_initial, frac=0.45, min_gap_idx=15):
    """
    Отскок = вход в зону sep < sep_initial*frac
    с минимальным интервалом min_gap_idx между отскоками.
    """
    threshold = sep_initial * frac
    n = 0
    last_entry = -min_gap_idx * 2
    in_col = False
    for i, s in enumerate(sep_arr):
        if s < threshold:
            if not in_col and (i - last_entry) > min_gap_idx:
                n += 1
                last_entry = i
            in_col = True
        else:
            in_col = False
    return n

# ============================================================
# ВЕРИФИКАЦИЯ IC (быстрая)
# ============================================================

print("\n--- Верификация IC ---")
N0, L0, r0 = 512, 100.0, 0.5
dx0 = L0/N0
x0_arr = np.linspace(-L0/2, L0/2, N0, endpoint=False)
sep0 = 20.0
v0 = 0.3

phi_v, pi_v = make_ic(x0_arr, -sep0/2, +sep0/2, v0, r0)
phi_v_np = np.array(phi_v)
Q_v = (phi_v_np[-1]-phi_v_np[0])/(4*np.pi)
E_v = total_energy(phi_v, pi_v, dx0, r0)
M_v = kink_mass(r0)
gamma_v = 1/np.sqrt(1-v0**2)

print(f"Q_total  = {Q_v:.4f}  (expect: 0)")
print(f"φ(-∞)    = {phi_v_np[:20].mean():.4f}  (expect: 0)")
print(f"φ(+∞)    = {phi_v_np[-20:].mean():.4f}  (expect: 0)")
print(f"E_IC     = {E_v:.3f}")
print(f"2·γ·M    = {2*gamma_v*M_v:.3f}  (expect)")
print(f"E/(2γM)  = {E_v/(2*gamma_v*M_v):.4f}  (expect: ~1)")

sep_v, xK_v, xAK_v = kink_separation(phi_v_np, dx0, L0)
print(f"sep_init = {sep_v:.2f}  (expect: ~{sep0})")
print(f"x_K={xK_v:.2f}, x_AK={xAK_v:.2f}")

assert Q_v < 0.1, f"IC сломаны! Q={Q_v}"
assert sep_v > 10, f"sep слишком мала: {sep_v}"
print("IC: OK ✓")

# ============================================================
# STEP 2: WOBBLE MODE (повтор с правильной сеткой)
# ============================================================

print("\n" + "="*60)
print("STEP 2: Wobble mode")
print("="*60)

N2 = 1024
L2 = 80.0
dx2 = L2/N2
x2 = np.linspace(-L2/2, L2/2, N2, endpoint=False)

def find_wobble(phi_kink, dx, r_val, n_modes=10):
    """Спектр малых колебаний вокруг кинка."""
    def V_pp(phi):
        return (r_val/4)*np.cos(phi/2) + (1-r_val)*np.cos(phi)
    U = V_pp(phi_kink)
    M_sq_vac = r_val/4 + (1-r_val)
    M_vac = np.sqrt(M_sq_vac)

    diag    = 2.0/dx**2 + U
    offdiag = -np.ones(len(phi_kink)-1)/dx**2
    L_mat   = sp_diags([offdiag, diag, offdiag], [-1,0,1],
                        format='csr')
    evals, evecs = eigsh(L_mat, k=n_modes, which='SM', tol=1e-10)
    evals = np.sort(evals)
    omegas = np.sqrt(np.maximum(evals, 0.0))
    return omegas, M_vac, evecs

print(f"\n{'r':>5} {'M_vac':>7} {'ω₀(trans)':>11} "
      f"{'ω₁(wobble)':>12} {'ω₁/M':>8} {'Bound?':>7}")
print("-"*55)

wobble_results = {}
for r_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    phi_k = kink_profile_inverse(x2, 0.0, r_val, sign=+1)
    omegas, M_vac, _ = find_wobble(phi_k, dx2, r_val)

    # ω₀ ≈ 0 (трансляционная), ω₁ = wobble
    omega0 = omegas[0]
    omega1 = omegas[1] if len(omegas) > 1 else 0.0
    is_bound = (omega1 < M_vac) and (omega1 > 0.01)

    wobble_results[r_val] = {
        'omega_shape': omega1,
        'M_vac': M_vac,
        'is_bound': is_bound
    }
    print(f"{r_val:>5.1f} {M_vac:>7.4f} {omega0:>11.4f} "
          f"{omega1:>12.4f} {omega1/M_vac:>8.4f} "
          f"{'YES' if is_bound else 'NO':>7}")

S3_pass = sum(1 for r in wobble_results
               if wobble_results[r]['is_bound']) >= 3

# ============================================================
# STEP 3: СКАНИРОВАНИЕ v
# ============================================================

print("\n" + "="*60)
print("STEP 3: Сканирование v — столкновения K+AK")
print("="*60)

N3   = 512
L3   = 120.0
dx3  = L3/N3
x3   = np.linspace(-L3/2, L3/2, N3, endpoint=False)
r3   = 0.5
sep3 = 22.0
dt3  = 0.02
tmax3 = 350.0

# Диапазон скоростей
v_scan = np.round(np.arange(0.05, 0.62, 0.02), 4)

print(f"\n{'v':>6} {'Q':>6} {'sep₀':>6} "
      f"{'N_b':>5} {'Cap':>5} "
      f"{'sep_min':>8} {'sep_fin':>8}")
print("-"*52)

scan_results = {}
t_scan_start = time.time()

for v_val in v_scan:
    phi0, pi0 = make_ic(x3, -sep3/2, +sep3/2, v_val, r3)

    # Быстрая проверка IC
    phi_np0 = np.array(phi0)
    Q_ic = (phi_np0[-1]-phi_np0[0])/(4*np.pi)
    sep_ic, _, _ = kink_separation(phi_np0, dx3, L3)

    step_fn = make_stepper(dx3, r3)
    phi, pi = phi0, pi0

    sep_hist = []
    t_hist   = []
    N_steps  = int(tmax3/dt3)
    rec      = 5

    for i in range(N_steps):
        phi, pi = step_fn(phi, pi, dt3)
        if i % rec == 0:
            s, _, _ = kink_separation(np.array(phi), dx3, L3)
            sep_hist.append(s)
            t_hist.append(i*dt3)

    sep_arr = np.array(sep_hist)
    nb  = count_bounces(sep_arr, sep3)
    cap = (len(sep_arr) > 30 and
           sep_arr[-30:].mean() < sep3*0.4)

    scan_results[v_val] = {
        'nb': nb, 'cap': cap,
        'sep': sep_hist, 't': t_hist,
        'Q_ic': Q_ic, 'sep0': sep_ic
    }

    s_min = sep_arr.min() if len(sep_arr) else 0
    s_fin = sep_arr[-1]  if len(sep_arr) else 0
    print(f"{v_val:>6.2f} {Q_ic:>6.3f} {sep_ic:>6.1f} "
          f"{nb:>5} {'Y' if cap else 'N':>5} "
          f"{s_min:>8.2f} {s_fin:>8.2f}")

print(f"\nВремя сканирования: {time.time()-t_scan_start:.1f}s")

nb_arr  = np.array([scan_results[v]['nb']  for v in v_scan])
cap_arr = np.array([scan_results[v]['cap'] for v in v_scan])
sep0_arr = np.array([scan_results[v]['sep0'] for v in v_scan])

# Диагностика sep0
print(f"\nsep0 (ожидается ~{sep3}):")
print(f"  mean={sep0_arr.mean():.2f}, "
      f"min={sep0_arr.min():.2f}, "
      f"max={sep0_arr.max():.2f}")

bad_ic = sep0_arr < sep3 * 0.5
if bad_ic.any():
    print(f"  ВНИМАНИЕ: {bad_ic.sum()} точек с sep0 < {sep3*0.5:.1f}")
    print(f"  (плохие IC при v={v_scan[bad_ic]})")

# Критерии
cap_v = v_scan[cap_arr]
esc_v = v_scan[~cap_arr]
v_cr  = float(cap_v.max()) if len(cap_v) > 0 else None

S1_pass = (v_cr is not None and v_cr < 0.58)
S2_pass = bool(np.any(nb_arr > 1))
n_multi = int(np.sum(nb_arr > 1))

print(f"\nv_cr = {v_cr}")
print(f"S1 (bound state, v_cr<0.58): "
      f"{'PASS' if S1_pass else 'FAIL'}")
print(f"S2 (N_bounces>1 exists): "
      f"{'PASS' if S2_pass else 'FAIL'}, n_multi={n_multi}")

# ============================================================
# ГРАФИКИ STEP 3
# ============================================================

fig3, ax3 = plt.subplots(2, 3, figsize=(15, 9))

# N_bounces(v)
a = ax3[0,0]
colors_v = ['green' if c else 'steelblue' for c in cap_arr]
a.bar(v_scan, nb_arr, width=0.016, color=colors_v, alpha=0.8)
a.set_xlabel('v'); a.set_ylabel('N_bounces')
a.set_title(f'Bounce structure r={r3}')
a.grid(True, alpha=0.3, axis='y')
if v_cr:
    a.axvline(x=v_cr, color='r', ls='--',
               label=f'v_cr={v_cr:.2f}')
    a.legend()

# Separation(t) для нескольких v
a = ax3[0,1]
for v_show in [0.10, 0.20, 0.30, 0.40, 0.50]:
    vc = min(v_scan, key=lambda x: abs(x-v_show))
    res = scan_results[vc]
    lbl = f"v={vc:.2f} ({'cap' if res['cap'] else 'esc'})"
    a.plot(res['t'], res['sep'], lw=1.5, label=lbl)
a.axhline(y=sep3*0.4, color='r', ls='--', alpha=0.7,
           label='capture threshold')
a.set_xlabel('t'); a.set_ylabel('Separation')
a.set_title('K-AK separation(t)')
a.legend(fontsize=7); a.grid(True, alpha=0.3)

# sep_final(v)
a = ax3[0,2]
sep_finals = [np.array(scan_results[v]['sep'])[-1]
              if scan_results[v]['sep'] else 0
              for v in v_scan]
a.scatter(v_scan, sep_finals, c=colors_v, s=60, zorder=5)
a.axhline(y=sep3*0.4, color='r', ls='--', label='capture')
a.set_xlabel('v'); a.set_ylabel('Final separation')
a.set_title('Final state')
a.legend(); a.grid(True, alpha=0.3)

# sep_min(v)
a = ax3[1,0]
sep_mins = [np.array(scan_results[v]['sep']).min()
            if scan_results[v]['sep'] else 0
            for v in v_scan]
a.plot(v_scan, sep_mins, 'ko-', ms=5, lw=1.5)
a.set_xlabel('v'); a.set_ylabel('Min separation')
a.set_title('Minimum approach distance')
a.grid(True, alpha=0.3)

# sep0(v) — проверка IC
a = ax3[1,1]
a.plot(v_scan, sep0_arr, 'bs-', ms=5, lw=1.5)
a.axhline(y=sep3, color='g', ls='--', label=f'Expected {sep3}')
a.axhline(y=sep3*0.5, color='r', ls='--', label='Bad IC threshold')
a.set_xlabel('v'); a.set_ylabel('Initial separation')
a.set_title('IC quality: sep0(v)')
a.legend(fontsize=8); a.grid(True, alpha=0.3)

# Резонансные окна
a = ax3[1,2]
multi_v = v_scan[nb_arr > 1]
multi_nb = nb_arr[nb_arr > 1]
if len(multi_v) > 0:
    a.bar(multi_v, multi_nb, width=0.016,
           color='darkgreen', alpha=0.8)
    a.set_title(f'Resonance windows (N>1): {len(multi_v)} found')
else:
    a.text(0.5, 0.5, 'No resonance windows\n(N_bounces ≤ 1)',
            ha='center', va='center',
            transform=a.transAxes, fontsize=14, color='red')
    a.set_title('Resonance windows')
a.set_xlabel('v'); a.set_ylabel('N_bounces')
a.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/part24/step3_patch3.png', dpi=150)
plt.close()
print("→ Saved: figures/part24/step3_patch3.png")

# ============================================================
# STEP 4: ДОЛГОЖИВУЩИЕ СОСТОЯНИЯ
# ============================================================

print("\n" + "="*60)
print("STEP 4: Долгоживущие состояния")
print("="*60)

# Выбираем v для долгого прогона
if v_cr is not None and v_cr > 0.08:
    v_long = round(v_cr * 0.6, 2)
else:
    v_long = 0.10
v_long = max(0.05, v_long)
print(f"v_long = {v_long:.2f}, r={r3}, t_max=2000")

N4   = 1024
L4   = 140.0
dx4  = L4/N4
x4   = np.linspace(-L4/2, L4/2, N4, endpoint=False)
sep4 = 24.0
dt4  = 0.015

phi4, pi4 = make_ic(x4, -sep4/2, +sep4/2, v_long, r3)

# Проверка IC
phi4_np = np.array(phi4)
Q4 = (phi4_np[-1]-phi4_np[0])/(4*np.pi)
sep4_ic, _, _ = kink_separation(phi4_np, dx4, L4)
E4_0 = total_energy(phi4, pi4, dx4, r3)
print(f"IC: Q={Q4:.3f}, sep0={sep4_ic:.2f}, E0={E4_0:.3f}")

step4 = make_stepper(dx4, r3)
phi4, pi4 = jnp.array(phi4), jnp.array(pi4)

sep_hist4  = []
e_cen_hist = []
t_hist4    = []
E_hist     = []

N_steps4 = int(2000.0/dt4)
rec4 = 10

for i in range(N_steps4):
    phi4, pi4 = step4(phi4, pi4, dt4)
    if i % rec4 == 0:
        phi4_np = np.array(phi4)
        s, _, _ = kink_separation(phi4_np, dx4, L4)
        sep_hist4.append(s)
        t_hist4.append(i*dt4)

        # Центральная энергия
        lo = int(N4*0.35); hi = int(N4*0.65)
        phi4_x = jnp.gradient(phi4, dx4)
        e_d = pi4**2/2 + phi4_x**2/2 + V_jax(phi4, r3)
        Ec = float(jnp.sum(e_d[lo:hi])*dx4)
        Et = float(jnp.sum(e_d)*dx4)
        e_cen_hist.append(Ec/(Et+1e-10))
        E_hist.append(Et)

sep4_arr  = np.array(sep_hist4)
e4_arr    = np.array(e_cen_hist)
E4_arr    = np.array(E_hist)

nb4  = count_bounces(sep4_arr, sep4)
cap4 = sep4_arr[-50:].mean() < sep4*0.4
e_late4 = e4_arr[-50:].mean() if len(e4_arr) > 50 else 0.0
dE_frac = abs(E4_arr[-1]-E4_arr[0])/(E4_arr[0]+1e-10)*100

print(f"N_bounces   = {nb4}")
print(f"Captured    = {cap4}")
print(f"E_cen_late  = {e_late4:.3f}")
print(f"sep_final   = {sep4_arr[-1]:.2f}")
print(f"ΔE/E₀       = {dE_frac:.4f}%  "
      f"({'OK' if dE_frac < 0.1 else 'WARNING'})")

# FFT
dt_rec4  = t_hist4[1]-t_hist4[0] if len(t_hist4)>1 else 1.0
e4_det   = e4_arr - e4_arr.mean()
freqs4   = np.fft.rfftfreq(len(e4_det), d=dt_rec4)
power4   = np.abs(np.fft.rfft(e4_det))**2
if len(power4) > 1:
    omega_QB = freqs4[1:][np.argmax(power4[1:])]
    print(f"ω_QB        = {omega_QB:.4f}")
else:
    omega_QB = 0.0

S1_pass = cap4 and (e_late4 > 0.15 or nb4 >= 2)

fig4, ax4 = plt.subplots(2, 2, figsize=(13, 9))

a = ax4[0,0]
a.plot(t_hist4, sep_hist4, 'b-', lw=1)
a.axhline(y=sep4*0.4, color='r', ls='--', label='Capture threshold')
a.set_xlabel('t'); a.set_ylabel('Separation')
a.set_title(f'K-AK separation: v={v_long:.2f}, r={r3}')
a.legend(); a.grid(True, alpha=0.3)

a = ax4[0,1]
a.plot(t_hist4, e_cen_hist, 'r-', lw=1)
a.axhline(y=0.3, color='k', ls='--', label='S1=0.3')
a.set_xlabel('t'); a.set_ylabel('E_center/E_total')
a.set_title('Central energy (S1 criterion)')
a.legend(); a.grid(True, alpha=0.3)

a = ax4[1,0]
cutoff = min(200, len(freqs4))
a.plot(freqs4[1:cutoff], power4[1:cutoff], 'b-', lw=1)
a.axvline(x=omega_QB, color='r', ls='--',
           label=f'ω_QB={omega_QB:.4f}')
if r3 in wobble_results:
    ws = wobble_results[r3]['omega_shape']
    a.axvline(x=ws, color='g', ls='--',
               label=f'ω_shape={ws:.4f}')
a.set_xlabel('Frequency'); a.set_ylabel('Power')
a.set_title('FFT(E_center) vs wobble mode')
a.legend(fontsize=8); a.grid(True, alpha=0.3)

a = ax4[1,1]
a.plot(t_hist4, E_hist, 'g-', lw=1)
a.set_xlabel('t'); a.set_ylabel('E_total')
a.set_title(f'Energy conservation ΔE/E₀={dE_frac:.4f}%')
a.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/part24/step4_patch3.png', dpi=150)
plt.close()
print("→ Saved: figures/part24/step4_patch3.png")

# ============================================================
# STEP 5: СКАНИРОВАНИЕ r
# ============================================================

print("\n" + "="*60)
print("STEP 5: Сканирование r")
print("="*60)

r_scan = [0.1, 0.3, 0.5, 0.7, 0.9]
v_test_list = [0.10, 0.15, 0.20, 0.25, 0.30]

print(f"\n{'r':>5} {'M_kink':>8} {'ω_shape':>9} "
      f"{'ω/M':>7} {'v_cr':>7}")
print("-"*43)

r_scan_results = {}

for r_val in r_scan:
    M_k = kink_mass(r_val)
    wd  = wobble_results.get(r_val, {})
    omega_sh = wd.get('omega_shape', 0.0)
    M_vac    = wd.get('M_vac', 1.0)

    # Быстрый поиск v_cr
    vcr_found = None
    for v_t in v_test_list:
        phi_t, pi_t = make_ic(x3, -sep3/2, +sep3/2, v_t, r_val)
        phi_t_np = np.array(phi_t)
        sep_t0, _, _ = kink_separation(phi_t_np, dx3, L3)

        if sep_t0 < sep3 * 0.5:
            continue  # плохие IC — пропускаем

        step_t  = make_stepper(dx3, r_val)
        phi_t, pi_t = jnp.array(phi_t), jnp.array(pi_t)
        sep_h_t = []
        for i in range(int(200.0/dt3)):
            phi_t, pi_t = step_t(phi_t, pi_t, dt3)
            if i % 5 == 0:
                s, _, _ = kink_separation(
                    np.array(phi_t), dx3, L3)
                sep_h_t.append(s)

        sep_t_arr = np.array(sep_h_t)
        cap_t = (len(sep_t_arr) > 20 and
                 sep_t_arr[-20:].mean() < sep3*0.4)
        if cap_t:
            vcr_found = v_t

    r_scan_results[r_val] = {
        'M_kink': M_k, 'omega_shape': omega_sh,
        'M_vac': M_vac, 'v_cr': vcr_found
    }
    print(f"{r_val:>5.1f} {M_k:>8.3f} {omega_sh:>9.4f} "
          f"{omega_sh/M_vac:>7.4f} "
          f"{str(vcr_found) if vcr_found else 'None':>7}")

# ============================================================
# ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================

print("\n" + "="*65)
print("PART XXIV — ФИНАЛЬНЫЙ ВЕРДИКТ")
print("="*65)

S3_pass = sum(1 for r in wobble_results
               if wobble_results[r]['is_bound']) >= 3
S4_pass = False  # физически верно: exp(-M/T) ≈ 0

print(f"\n{'Критерий':<45} {'Результат':>10}")
print("-"*57)
criteria = [
    ("S1: bound state (cap + E_late>0.15 или N_b≥2)", S1_pass),
    ("S2: resonance windows (N_bounces>1 при каком-то v)", S2_pass),
    ("S3: wobble mode (ω_shape < M_vac, ≥3/5 r-значений)", S3_pass),
    ("S4: thermal nucleation T < M",                       S4_pass),
]
for name, result in criteria:
    print(f"  {name:<43} "
          f"{'✓ PASS' if result else '✗ FAIL':>10}")

n_pass = sum(r for _, r in criteria)

if n_pass >= 3:
    verdict = "CONFIRMED"
    detail  = "неинтегрируемость создаёт связанные состояния"
elif n_pass == 2:
    verdict = "PARTIAL"
    detail  = "базовые механизмы работают"
else:
    verdict = "FALSIFIED"
    detail  = "double SG недостаточен для модели BЭ"

print(f"\n{'='*40}")
print(f"ИТОГО:   {n_pass}/4")
print(f"ВЕРДИКТ: {verdict} — {detail}")
print(f"{'='*40}")

# Wobble подробнее
print("\nWobble mode (Step 2):")
for r_val, wd in sorted(wobble_results.items()):
    print(f"  r={r_val}: ω_shape={wd['omega_shape']:.4f}, "
          f"M={wd['M_vac']:.4f}, "
          f"ω/M={wd['omega_shape']/wd['M_vac']:.4f}, "
          f"bound={'YES' if wd['is_bound'] else 'NO'}")

print(f"\nS4=FAIL — физически верно:")
print(f"  P_nucleation ~ exp(-M/T) = exp(-{kink_mass(0.5):.1f}/0.5)"
      f" ≈ {np.exp(-kink_mass(0.5)/0.5):.2e}")
print(f"  Нуклеация подавлена при T << M_kink")

# Артефакты
print(f"""
ДОКУМЕНТИРОВАННЫЕ АРТЕФАКТЫ Part XXIV:
  #19: N_bounces=0 (детектор φ=2π) → sep(t) детектор
  #20: M_analytic=16 — не баг, физически верно (0→4π кинк)
  #21: AxisError thermal IC → исправлен размер sigma_pi
  #22: IC Q=2 вместо Q=0 (static_antikink сломан) → kink_profile_inverse
  #23: N_bounces~50 = шум → count_bounces через sep threshold
  #24: sep_init=0.1 (кинки накладывались) → верифицированы sep0≈{sep3}
""")

print("Part XXIV Patch-3 завершён.")
