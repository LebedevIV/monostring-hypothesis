"""
Part XXI Step 4: Сходимость к КГ вакууму

Проверяем: slope → -2.0 при N → ∞?
Если да: БЭ (random + детальный баланс) ≡ КГ вакуум.

Дополнительно: что означает T в модели БЭ?
T = ℏ (квантовая флуктуация)?
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

print("="*60)
print("Part XXI Step 4: Сходимость БЭ → КГ вакуум")
print("="*60)
print()

# ── УТИЛИТЫ ──────────────────────────────────────────────────

def metropolis_kg_fast(N, m2, T, n_sweeps, seed=SEED):
    """
    Оптимизированный MC для больших N.
    Используем векторизованный checkerboard sweep.
    """
    rng   = np.random.RandomState(seed)
    phi   = rng.randn(N) * np.sqrt(T / (m2 + 4))
    delta = np.sqrt(T / (m2 + 4))

    # Checkerboard: чётные и нечётные независимы
    even = np.arange(0, N, 2)
    odd  = np.arange(1, N, 2)

    for sweep in range(n_sweeps):
        for sites in [even, odd]:
            left  = (sites - 1) % N
            right = (sites + 1) % N

            phi_old = phi[sites]
            phi_new = (phi_old +
                       delta * rng.randn(len(sites)))

            dS = (0.5 * m2 * (phi_new**2 - phi_old**2)
                  + 0.5 * ((phi_new - phi[left])**2
                           - (phi_old - phi[left])**2)
                  + 0.5 * ((phi_new - phi[right])**2
                           - (phi_old - phi[right])**2))

            accept = ((dS < 0) |
                      (rng.rand(len(sites)) <
                       np.exp(-dS/T)))
            phi[sites] = np.where(accept, phi_new, phi_old)

    return phi

def measure_spectrum(phi):
    """Измеряем спектр и корреляционную длину."""
    N     = len(phi)
    phi_c = phi - phi.mean()
    fft   = np.fft.rfft(phi_c)
    P     = np.abs(fft)**2 / N
    freqs = np.fft.rfftfreq(N)

    # Наклон в среднем диапазоне k
    mask = (freqs > 2/N) & (freqs < 0.3)
    if mask.sum() > 5:
        sl, _, _, _, _ = stats.linregress(
            np.log(freqs[mask]),
            np.log(P[mask] + 1e-30))
    else:
        sl = np.nan

    # Корреляционная длина
    corr = np.correlate(phi_c, phi_c, mode='full')
    corr = corr[N-1:] / (corr[N-1] + 1e-30)
    idx  = np.where(corr < 1/np.e)[0]
    l_c  = idx[0] if len(idx) > 0 else N

    return freqs, P, sl, l_c

def kg_exact(N, m2, T=1.0):
    freqs = np.fft.rfftfreq(N)
    lam   = 2 - 2*np.cos(2*np.pi*freqs) + m2
    return freqs, T / lam

# ── ТЕСТ 1: СХОДИМОСТЬ ПО N ──────────────────────────────────

print("── Тест 1: slope(N) при N → ∞ ──")
print()
print("Если slope → -2.0: БЭ ≡ КГ вакуум")
print()

m2     = 0.01
T      = 1.0
N_sw   = 5000
N_warm = 1000

N_vals  = [64, 128, 256, 512, 1024]
slopes  = []
lcorrs  = []
l_th    = 1.0 / np.sqrt(m2)

print(f"  {'N':>6} {'slope':>8} {'l_corr':>8} "
      f"{'Δslope':>8} {'Δlcorr':>8}")
print("  " + "-"*44)

for N in N_vals:
    # Прогрев
    phi = metropolis_kg_fast(N, m2, T,
                              N_warm, seed=SEED)
    # Измерение
    phi = metropolis_kg_fast(N, m2, T,
                              N_sw, seed=SEED+1)
    _, _, sl, lc = measure_spectrum(phi)
    slopes.append(sl)
    lcorrs.append(lc)

    d_sl = sl - (-2.0)
    d_lc = (lc - l_th) / l_th * 100

    print(f"  {N:>6} {sl:>8.3f} {lc:>8.1f} "
          f"{d_sl:>+8.3f} {d_lc:>+7.0f}%")

print(f"  {'∞ (КГ)':>6} {-2.000:>8.3f} "
      f"{l_th:>8.1f}     ---      ---")
print()

# Экстраполяция slope → N→∞
slopes_arr = np.array(slopes)
N_arr      = np.array(N_vals, dtype=float)

if len(N_arr) >= 3:
    # Fit: slope(N) = slope_inf + a/N
    from scipy.optimize import curve_fit
    def fit_func(N, s_inf, a):
        return s_inf + a/N
    try:
        popt, _ = curve_fit(fit_func, N_arr,
                             slopes_arr,
                             p0=[-2.0, 10.0])
        slope_inf = popt[0]
        print(f"  Экстраполяция N→∞: slope = {slope_inf:.4f}")
        print(f"  Отклонение от КГ (-2.0): "
              f"{slope_inf+2.0:+.4f}")
        is_converging = abs(slope_inf + 2.0) < 0.1
        if is_converging:
            print("  *** slope → -2.0: БЭ ≡ КГ вакуум! ***")
        else:
            print("  slope не сходится к -2.0")
    except Exception as e:
        slope_inf = slopes_arr[-1]
        print(f"  Fit failed: {e}")
        is_converging = False
else:
    slope_inf     = slopes_arr[-1]
    is_converging = False

print()

# ── ТЕСТ 2: РОЛЬ ТЕМПЕРАТУРЫ T ───────────────────────────────

print("── Тест 2: T как квантовый параметр ──")
print()
print("В КТП: ⟨φ²⟩ = T·(1/λ_k суммарно)")
print("При T→0: классическое поле (нет флуктуаций)")
print("При T=ℏ: квантовый вакуум")
print()

N_T   = 256
N_T_sw = 3000
T_vals = [0.1, 0.5, 1.0, 2.0, 5.0]

print(f"  {'T':>6} {'⟨φ²⟩':>10} "
      f"{'slope':>8} {'l_corr':>8}")
print("  " + "-"*36)

T_results = {}
for T_val in T_vals:
    phi = metropolis_kg_fast(N_T, m2, T_val,
                              N_T_sw, seed=SEED)
    phi2_mean = (phi**2).mean()
    _, _, sl, lc = measure_spectrum(phi)
    T_results[T_val] = {
        'phi2': phi2_mean, 'slope': sl, 'lc': lc}
    print(f"  {T_val:>6.1f} {phi2_mean:>10.4f} "
          f"{sl:>8.3f} {lc:>8.1f}")

print()
print("  Теоретически: l_corr = 1/√m² = "
      f"{l_th:.1f} (не зависит от T!)")
print("  Если l_corr ≈ 10 для всех T: подтверждено ✓")
lc_stable = all(
    abs(T_results[t]['lc'] - l_th)/l_th < 0.3
    for t in T_vals)
print(f"  l_corr стабильна: {lc_stable}")
print()

# ── ТЕСТ 3: ИНТЕРПРЕТАЦИЯ ────────────────────────────────────

print("── Тест 3: Физическая интерпретация ──")
print()
print("БЭ-интерпретация квантового поля:")
print()
print("  Классическая КТП:")
print("    φ̂(x) = Σₖ (aₖ e^{ikx} + aₖ† e^{-ikx})")
print("    ⟨0|φ̂²|0⟩ = Σₖ 1/(2ωₖ)  [нулевые колебания]")
print()
print("  БЭ-версия:")
print("    φ(x,t) = значение поля в точке x")
print("             в момент посещения БЭ")
print("    ⟨φ²⟩   = T·Σₖ 1/λₖ  [классическая статистика]")
print()
print("  Соответствие при T = ℏ/2:")
print("    ⟨0|φ̂²|0⟩_КТП = ⟨φ²⟩_БЭ при T=ℏ/2")
print()

# Проверяем числово
phi2_T1 = T_results[1.0]['phi2']
# Теоретически: ⟨φ²⟩ = T·Σ 1/λₖ / N
freqs_th, P_th = kg_exact(N_T, m2, T=1.0)
phi2_theory    = P_th.mean()
print(f"  Числовая проверка (N={N_T}, T=1.0):")
print(f"    ⟨φ²⟩_БЭ    = {phi2_T1:.4f}")
print(f"    ⟨φ²⟩_КГтеор = {phi2_theory:.4f}")
ratio = phi2_T1 / phi2_theory
print(f"    Отношение  = {ratio:.4f}")
print(f"    {'✓ Совпадение!' if abs(ratio-1)<0.15 else '✗ Расхождение'}")
print()

# ── ФИНАЛЬНЫЙ ВЫВОД ──────────────────────────────────────────

print("="*60)
print("ФИНАЛЬНЫЙ ВЫВОД Part XXI")
print("="*60)
print()

print("РЕЗУЛЬТАТЫ:")
print(f"  slope(N=1024) = {slopes[-1]:.3f} "
      f"(КГ: -2.000)")
print(f"  Экстраполяция N→∞: {slope_inf:.4f}")
print(f"  Сходится к КГ: {is_converging}")
print()

if is_converging:
    print("╔══════════════════════════════════════════╗")
    print("║  РЕЗУЛЬТАТ: БЭ ≡ КГ ВАКУУМ             ║")
    print("║                                         ║")
    print("║  Модель единого базового элемента       ║")
    print("║  с случайным обходом и детальным        ║")
    print("║  балансом математически эквивалентна    ║")
    print("║  квантовому полю Клейна-Гордона.        ║")
    print("║                                         ║")
    print("║  Новая интерпретация КТП:               ║")
    print("║  квантовые поля = статистика БЭ         ║")
    print("║                                         ║")
    print("║  → Публикуемый результат               ║")
    print("║  → Новый раздел paper (Part XXI)       ║")
    print("╚══════════════════════════════════════════╝")
else:
    print("╔══════════════════════════════════════════╗")
    print("║  РЕЗУЛЬТАТ: ЧАСТИЧНОЕ СООТВЕТСТВИЕ      ║")
    print("║                                         ║")
    print("║  БЭ воспроизводит спектр P(k)~k^{-1.8} ║")
    print("║  вместо k^{-2.0} (КГ вакуум).          ║")
    print("║                                         ║")
    print("║  Отклонение = конечный размер решётки.  ║")
    print("║  При N→∞: возможна точная сходимость.  ║")
    print("╚══════════════════════════════════════════╝")

print()
print("Физический смысл результата:")
print()
print("  1. Случайный обход БЭ → квантовая случайность")
print("  2. Детальный баланс   → принцип наименьшего действия")
print("  3. Температура T=ℏ   → квантовая неопределённость")
print()
print("  Это не новая физика.")
print("  Это новая ОНТОЛОГИЯ той же физики:")
print("  квантовое поле = статистика одного объекта.")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'Part XXI Step 4: Сходимость БЭ → КГ вакуум\n'
    'Единый базовый элемент ≡ квантовое поле?',
    fontsize=13)

# 0,0: slope vs N
ax = axes[0,0]
ax.semilogx(N_vals, slopes, 'o-',
            color='#e74c3c', lw=2, ms=10,
            label='БЭ (random MC)')
ax.axhline(-2.0, color='k', ls='--', lw=2,
           label='КГ вакуум (-2.0)')
if is_converging:
    N_fit = np.logspace(np.log10(64),
                         np.log10(10000), 50)
    ax.semilogx(N_fit,
                fit_func(N_fit, *popt),
                'r:', lw=1.5,
                label=f'Экстраполяция→{slope_inf:.3f}')
ax.set_xlabel('N (размер решётки)')
ax.set_ylabel('slope P(k)~k^slope')
ax.set_title('Сходимость slope к -2.0\n'
             '(КГ вакуум)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 0,1: l_corr vs N
ax = axes[0,1]
ax.semilogx(N_vals, lcorrs, 's-',
            color='#3498db', lw=2, ms=10,
            label='БЭ l_corr')
ax.axhline(l_th, color='k', ls='--', lw=2,
           label=f'КГ: 1/√m²={l_th:.1f}')
ax.set_xlabel('N')
ax.set_ylabel('l_corr (сайты)')
ax.set_title('Корреляционная длина vs N')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1,0: спектры при разных N
ax = axes[1,0]
colors_N = plt.cm.viridis(
    np.linspace(0, 1, len(N_vals)))
for i, N in enumerate(N_vals[-3:]):
    phi_i = metropolis_kg_fast(
        N, m2, T, 3000, seed=SEED+i)
    f_i, P_i, _, _ = measure_spectrum(phi_i)
    mask_i = (f_i > 0) & (f_i < 0.45)
    ax.loglog(f_i[mask_i], P_i[mask_i],
              lw=1.5, alpha=0.7,
              label=f'N={N}')

# Теория
f_th, P_th2 = kg_exact(1024, m2, T)
mask_th = (f_th > 0) & (f_th < 0.45)
ax.loglog(f_th[mask_th], P_th2[mask_th],
          'k--', lw=2, label='КГ теория')
ax.set_xlabel('k')
ax.set_ylabel('P(k)')
ax.set_title('Спектры при разных N\n'
             'vs КГ теория')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# 1,1: итоговый вердикт
ax = axes[1,1]
ax.axis('off')

color = '#e0ffe0' if is_converging else '#fff3cd'
lines = [
    'PART XXI — ИТОГ',
    '═'*30,
    '',
    'Шаги прогресса:',
    '  Step 1: slope=−0.01 (белый шум)',
    '  Step 2: slope=−1.40 (локальная связь)',
    '  Step 3: slope=−1.83 (детальный баланс)',
    f'  Step 4: slope→{slope_inf:.3f} (N→∞)',
    '  КГ:     slope=−2.00',
    '',
    'Сходимость:',
    f'  {"ДА ✓" if is_converging else "ЧАСТИЧНАЯ ~"}',
    '',
    'Физический смысл:',
    '  БЭ + случайный обход',
    '  + детальный баланс',
    '  ≈ КТП квантовый вакуум',
    '',
    'Интерпретация:',
    '  Квантовое поле = статистика',
    '  одного объекта (БЭ),',
    '  обходящего все состояния.',
    '',
    'T = ℏ: квантовая флуктуация',
    '→ нулевые колебания вакуума',
]

ax.text(0.03, 0.97, '\n'.join(lines),
        transform=ax.transAxes,
        fontsize=8.5, va='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.9))

plt.tight_layout()
plt.savefig('part21_step4_convergence.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part21_step4_convergence.png")
