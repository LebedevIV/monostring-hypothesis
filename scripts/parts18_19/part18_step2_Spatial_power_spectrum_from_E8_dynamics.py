"""
Part XVIII Step 2: Spatial power spectrum from E8 dynamics

Исправление: вычисляем ПРОСТРАНСТВЕННЫЙ спектр P(|k|),
а не временной спектр Welch.

Физика:
  Каждый корень αᵢ ∈ R^8 — точка в 8D пространстве.
  φᵢ(t) — значение поля в этой точке.

  Пространственный спектр:
  φ̃(k,t) = Σᵢ δφᵢ(t) · exp(ik·αᵢ)
  P(|k|,t) = |φ̃(k,t)|²
  P_avg(|k|) = ⟨P(|k|,t)⟩_t
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ── ПОВТОРЯЕМ ГЕНЕРАЦИЮ (та же что в Step 1) ────────────────

def generate_E8_roots():
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in [-1,1]:
                for sj in [-1,1]:
                    v = np.zeros(8)
                    v[i]=si; v[j]=sj
                    roots.append(v)
    for bits in range(256):
        signs = np.array([((bits>>i)&1)*2-1
                          for i in range(8)],dtype=float)
        if np.sum(signs<0)%2==1:
            roots.append(0.5*signs)
    return np.array(roots)

def run_E8_dynamics(roots, C, kappa=1.0, T=30000,
                    warmup=3000, dt=0.001, seed=SEED):
    """dt=0.001 (уменьшен для стабильности)"""
    rng = np.random.RandomState(seed)
    n   = len(roots)
    phi = rng.uniform(0, 2*np.pi, n)
    p   = rng.normal(0, 0.1, n)

    def force(phi_):
        diff = phi_[:,None] - phi_[None,:]
        return kappa*(C*np.sin(diff)).sum(axis=1)

    # Прогрев
    for _ in range(warmup):
        F   = force(phi)
        phi = phi + p*dt + 0.5*F*dt**2
        F2  = force(phi)
        p   = p + 0.5*(F+F2)*dt

    # Сбор орбиты
    orbit = np.zeros((T, n))
    E_arr = []
    for t in range(T):
        F   = force(phi)
        phi = phi + p*dt + 0.5*F*dt**2
        F2  = force(phi)
        p   = p + 0.5*(F+F2)*dt
        orbit[t] = phi

        if t % 1000 == 0:
            Ek = 0.5*(p**2).sum()
            diff = phi[:,None]-phi[None,:]
            Ep = kappa*(C*np.cos(diff)).sum()
            E_arr.append(Ek+Ep)

    return orbit, np.array(E_arr)

# ── ПРОСТРАНСТВЕННЫЙ СПЕКТР ──────────────────────────────────

def spatial_power_spectrum(roots, orbit,
                            n_k_bins=30, n_t_sample=200):
    """
    Пространственный спектр P(|k|):

    1. Берём n_t_sample временных срезов
    2. Для каждого: δφᵢ = φᵢ - ⟨φᵢ⟩
    3. Создаём набор волновых векторов k
    4. Вычисляем φ̃(k) = Σᵢ δφᵢ exp(ik·αᵢ)
    5. P(|k|) = ⟨|φ̃(k)|²⟩_t
    """
    n_roots = len(roots)
    T_total = orbit.shape[0]

    # Среднее по времени
    phi_mean = orbit.mean(axis=0)  # (240,)

    # Временные индексы для выборки
    t_indices = np.linspace(T_total//4, T_total-1,
                             n_t_sample, dtype=int)

    # Создаём волновые векторы k в R^8
    # Берём случайные направления, радиусы логарифмически
    np.random.seed(42)
    k_magnitudes = np.logspace(-1, 1, n_k_bins)  # 0.1 до 10

    P_by_k = {k_mag: [] for k_mag in k_magnitudes}

    # Для каждого |k|: усредняем по нескольким направлениям
    n_dirs = 20  # направлений в R^8
    directions = np.random.randn(n_dirs, 8)
    directions /= np.linalg.norm(directions,
                                  axis=1, keepdims=True)

    for t_idx in t_indices:
        delta_phi = orbit[t_idx] - phi_mean  # (240,)

        for k_mag in k_magnitudes:
            P_k_dirs = []
            for d in directions:
                k_vec = k_mag * d  # (8,)
                # φ̃(k) = Σᵢ δφᵢ · exp(ik·αᵢ)
                phases  = roots @ k_vec  # (240,)
                phi_k   = np.sum(delta_phi *
                                  np.exp(1j*phases))
                P_k_dirs.append(abs(phi_k)**2)
            P_by_k[k_mag].append(np.mean(P_k_dirs))

    # Усредняем по времени
    k_arr = np.array(k_magnitudes)
    P_arr = np.array([np.mean(P_by_k[k])
                       for k in k_magnitudes])

    # Спектральный индекс: P(k) ~ k^(n_s-1)
    log_k = np.log(k_arr)
    log_P = np.log(P_arr + 1e-30)

    # Линейная регрессия на средних k
    # (исключаем края где шум)
    mid = slice(n_k_bins//4, 3*n_k_bins//4)
    if np.isfinite(log_P[mid]).sum() > 5:
        coeff  = np.polyfit(log_k[mid], log_P[mid], 1)
        slope  = coeff[0]
        ns_est = 1.0 + slope
    else:
        ns_est = np.nan

    return k_arr, P_arr, ns_est

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("="*65)
print("Part XVIII Step 2: Spatial Power Spectrum")
print("="*65)
print()

roots = generate_E8_roots()
C     = roots @ roots.T

print("── Stability test (dt=0.001 vs dt=0.01) ──")
for dt in [0.01, 0.001]:
    _, E = run_E8_dynamics(roots, C, kappa=0.5,
                           T=5000, warmup=500, dt=dt)
    drift = (E[-1]-E[0])/abs(E[0])*100
    print(f"  dt={dt}: E_drift={drift:.2f}%")
print()

# Выбираем стабильный dt
DT = 0.001

print("── Spatial spectrum for E8 (κ=0.5, T=20000) ──")
print("  Computing orbit... (может занять 5-10 мин)")
orbit_e8, E_e8 = run_E8_dynamics(
    roots, C, kappa=0.5,
    T=20000, warmup=2000, dt=DT)
print(f"  E_drift = {(E_e8[-1]-E_e8[0])/abs(E_e8[0])*100:.2f}%")
print()

print("  Computing spatial power spectrum...")
k_arr, P_e8, ns_e8 = spatial_power_spectrum(
    roots, orbit_e8)
print(f"  E8 spatial n_s = {ns_e8:.4f}")
print(f"  Planck n_s     = 0.9649")
print(f"  Δ              = {ns_e8-0.9649:+.4f}")
print()

# Случайный контроль: random coupling matrix
print("── Random coupling control ──")
rng_ctrl = np.random.RandomState(777)
roots_rand = rng_ctrl.randn(240, 8)
roots_rand /= np.linalg.norm(roots_rand,
                               axis=1, keepdims=True)
roots_rand *= np.sqrt(2)  # та же норма |α|²=2
C_rand = roots_rand @ roots_rand.T

orbit_rand, _ = run_E8_dynamics(
    roots_rand, C_rand, kappa=0.5,
    T=20000, warmup=2000, dt=DT, seed=123)

_, P_rand, ns_rand = spatial_power_spectrum(
    roots_rand, orbit_rand)
print(f"  Random spatial n_s = {ns_rand:.4f}")
print()

# Сравнение
print("── Summary ──")
print(f"  E8     n_s = {ns_e8:.4f}")
print(f"  Random n_s = {ns_rand:.4f}")
print(f"  Δ(E8-rand) = {ns_e8-ns_rand:+.4f}")
planck_ns = 0.9649
print(f"  Δ(E8-Planck) = {ns_e8-planck_ns:+.4f} "
      f"({abs(ns_e8-planck_ns)/0.0042:.1f}σ)")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'Part XVIII Step 2: Spatial Power Spectrum\n'
    'E8 root lattice — correct n_s estimation',
    fontsize=13)

# 0,0: P(k) в пространстве
ax = axes[0,0]
ax.loglog(k_arr, P_e8, 'o-', color='#e74c3c',
          lw=2, ms=8, label=f'E8 n_s={ns_e8:.3f}')
ax.loglog(k_arr, P_rand, 's--', color='gray',
          lw=2, ms=8, label=f'Random n_s={ns_rand:.3f}')

# Линии P∝k^(n_s-1)
k_line = np.logspace(-1, 1, 50)
# Нормировка для отображения
P0 = P_e8[len(P_e8)//2]
k0 = k_arr[len(k_arr)//2]
ax.loglog(k_line,
          P0*(k_line/k0)**(ns_e8-1),
          'r:', lw=1.5, alpha=0.7,
          label=f'k^(n_s-1) fit')
ax.loglog(k_line,
          P0*(k_line/k0)**(planck_ns-1),
          'g--', lw=1.5, alpha=0.7,
          label=f'Planck k^(0.965-1)')

ax.set_xlabel('|k| (spatial)')
ax.set_ylabel('P(k) = |φ̃(k)|²')
ax.set_title('Spatial power spectrum')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# 0,1: Временная эволюция нескольких мод
ax = axes[0,1]
for i in [0, 60, 120, 180]:
    ax.plot(orbit_e8[:500, i] % (2*np.pi),
            alpha=0.7, lw=1,
            label=f'root {i}')
ax.set_xlabel('Time step')
ax.set_ylabel('φ (mod 2π)')
ax.set_title('E8 mode trajectories (κ=0.5)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,0: Флуктуации δφ
ax = axes[1,0]
phi_mean = orbit_e8.mean(axis=0)
delta_phi = orbit_e8[-1000:] - phi_mean
ax.hist(delta_phi.flatten(), bins=50,
        color='#e74c3c', alpha=0.7,
        density=True, label='E8 δφ')
# Гауссово распределение для сравнения
x = np.linspace(-3, 3, 100)
sigma = delta_phi.std()
ax.plot(x, np.exp(-x**2/(2*sigma**2)) /
        (sigma*np.sqrt(2*np.pi)),
        'k--', lw=2, label='Gaussian')
ax.set_xlabel('δφᵢ')
ax.set_ylabel('Density')
ax.set_title('Fluctuation distribution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1,1: Итоговый вердикт
ax = axes[1,1]
ax.axis('off')
lines = [
    'STEP 2 VERDICT',
    '─'*28,
    '',
    'Spatial spectrum n_s:',
    f'  E8:     {ns_e8:.4f}',
    f'  Random: {ns_rand:.4f}',
    f'  Planck: {planck_ns:.4f}',
    '',
    f'  Δ(E8-Random) = {ns_e8-ns_rand:+.4f}',
    f'  Δ(E8-Planck) = '
    f'{ns_e8-planck_ns:+.4f}',
    f'  Distance: '
    f'{abs(ns_e8-planck_ns)/0.0042:.1f}σ',
    '',
    'Step 1 problem:',
    '  Welch temporal spectrum',
    '  always gives n_s < 0',
    '  (Brownian motion artifact)',
    '',
    'Step 2 fix:',
    '  Spatial Fourier transform',
    '  on E8 root lattice',
    '',
    'VERDICT:',
]

if not np.isnan(ns_e8):
    if abs(ns_e8 - planck_ns) < 0.0042:
        lines += ['*** IN PLANCK 1σ ***']
        color = '#e0ffe0'
    elif abs(ns_e8 - planck_ns) < 0.05:
        lines += [f'Within 0.05 of Planck.',
                  '→ Fine-tune κ']
        color = '#fffde0'
    else:
        lines += ['Outside Planck range.',
                  '→ Different H or params']
        color = '#ffe0e0'
else:
    lines += ['No convergence.']
    color = '#ffe0e0'

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('part18_step2_spatial.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part18_step2_spatial.png")

print()
print("="*65)
print("ЧЕСТНЫЙ ИТОГ Part XVIII")
print("="*65)
print()
print("Step 1 проблема:")
print("  Welch-спектр временного ряда φ(t)")
print("  всегда даёт n_s < 0 (броуновское движение)")
print("  Это НЕ инфляционный n_s")
print()
print("Step 2 исправление:")
print("  Пространственный Фурье-спектр на решётке E8")
print("  φ̃(k) = Σᵢ δφᵢ exp(ik·αᵢ)")
print("  P(k) = |φ̃(k)|² — правильный аналог CMB спектра")
print()
print("Если spatial n_s ≈ 0.965 → реальный сигнал")
print("Если spatial n_s ≈ random → H₀")
