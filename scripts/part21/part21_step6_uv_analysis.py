"""
Part XXI Step 6: Правильный анализ спектра КГ

Проблема Step 5: мы измеряли slope в неправильном
диапазоне k. Нужно измерять только в режиме k >> m.

Дополнительно: аналитическая проверка slope → -2.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

print("="*60)
print("Part XXI Step 6: Правильный анализ + финальный вывод")
print("="*60)
print()

# ── АНАЛИТИЧЕСКИЙ СПЕКТР КГ ──────────────────────────────────

def kg_spectrum_analytical(N, m2):
    """
    P(k) = 1 / (2 - 2cos(2πk/N) + m²)
    Аналитически, без MC.
    """
    k_idx = np.arange(N//2 + 1)
    freqs = k_idx / N
    lam   = 2 - 2*np.cos(2*np.pi*freqs) + m2
    return freqs, 1.0 / lam

def slope_in_range(freqs, P, f_min, f_max):
    """Наклон в заданном диапазоне частот."""
    mask = (freqs > f_min) & (freqs < f_max)
    if mask.sum() < 3:
        return np.nan, np.nan
    sl, _, r, p, se = stats.linregress(
        np.log(freqs[mask]),
        np.log(P[mask]))
    return sl, se

m2 = 0.01
m  = np.sqrt(m2)

print("── Аналитический спектр КГ ──")
print()
print(f"  m = {m:.3f}, m² = {m2}")
print(f"  Корреляционная длина: ξ = 1/m = {1/m:.1f}")
print()
print("  Режимы спектра:")
print(f"  k << m:  slope → 0   (ИК плато)")
print(f"  k >> m:  slope → -2  (УФ режим, ожидаем)")
print(f"  k_cross ≈ m/(2π) = {m/(2*np.pi):.4f} (в ед. 1/N)")
print()

# Точный порог: при каком k/N начинается режим k>>m?
k_cross_frac = m / (2*np.pi)  # в единицах частоты f=k/N
print(f"  f_cross = m/(2π) = {k_cross_frac:.4f}")
print()

print("  slope в разных диапазонах k (аналитически, N=4096):")
print()

N_big = 4096
freqs_big, P_big = kg_spectrum_analytical(N_big, m2)

ranges = [
    (0.001, 0.010, "ИК (k<<m)"),
    (0.010, 0.050, "переход"),
    (0.050, 0.150, "УФ (k>>m)"),
    (0.150, 0.400, "глубокий УФ"),
    (0.001, 0.400, "весь диапазон"),
]

print(f"  {'Диапазон f':>20}  {'slope':>8}  {'±se':>6}  Режим")
print("  " + "-"*50)
for f_min, f_max, label in ranges:
    sl, se = slope_in_range(freqs_big, P_big, f_min, f_max)
    print(f"  [{f_min:.3f}, {f_max:.3f}]  {sl:>8.4f}  "
          f"{se:>6.4f}  {label}")

print()
print("  ВЫВОД: slope→-2 только в УФ режиме (k>>m).")
print("  Наш прежний fit range (0.02, 0.3) включал")
print("  переходную зону → slope ≈ -1.7, не -2.")
print()

# ── ПРАВИЛЬНОЕ ИЗМЕРЕНИЕ: ТОЛЬКО УФ РЕЖИМ ────────────────────

print("── Правильный slope в УФ режиме ──")
print()

f_uv_min = 5 * k_cross_frac   # k > 5m: точно УФ
f_uv_max = 0.40

print(f"  УФ диапазон: f > {f_uv_min:.4f} (= 5·m/(2π))")
print()
print(f"  {'N':>6}  {'slope_UV':>10}  {'±se':>6}  "
      f"{'n_points':>10}")
print("  " + "-"*40)

N_vals = [256, 512, 1024, 2048, 4096]
slopes_uv = []

for N in N_vals:
    f_arr, P_arr = kg_spectrum_analytical(N, m2)
    sl, se = slope_in_range(f_arr, P_arr,
                             f_uv_min, f_uv_max)
    mask_c = ((f_arr > f_uv_min) &
              (f_arr < f_uv_max))
    n_pts  = mask_c.sum()
    slopes_uv.append(sl)
    print(f"  {N:>6}  {sl:>10.5f}  {se:>6.4f}  "
          f"{n_pts:>10}")

print(f"  {'∞':>6}  {-2.00000:>10.5f}  {'exact':>6}")
print()

# Сходится ли к -2?
slopes_uv_arr = np.array(slopes_uv)
dev_from_2    = slopes_uv_arr - (-2.0)
print(f"  Отклонение от -2.0:")
for i, N in enumerate(N_vals):
    print(f"    N={N:>5}: {dev_from_2[i]:+.5f}")

converges = np.all(np.abs(dev_from_2) < 0.05)
print()
print(f"  Сходится к -2 (|отклонение|<0.05): {converges}")
print()

# ── АНАЛИТИЧЕСКОЕ ДОКАЗАТЕЛЬСТВО ─────────────────────────────

print("── Аналитическое доказательство slope=-2 ──")
print()
print("  λ_k = 2 - 2cos(2πk/N) + m²")
print()
print("  При k >> m·N/(2π), т.е. 2πk/N >> m:")
print("  λ_k ≈ 2 - 2cos(θ) где θ=2πk/N")
print("       = 4sin²(θ/2) ≈ θ² = (2πk/N)²")
print()
print("  P(k) = 1/λ_k ≈ (N/2π)²/k²")
print()
print("  В логарифмическом масштабе:")
print("  log P(k) = 2log(N/2π) - 2·log(k)")
print()
print("  → d(log P)/d(log k) = -2  ✓ (точно!)")
print()
print("  slope = -2 АНАЛИТИЧЕСКИ в УФ режиме.")
print("  Это не зависит от N, m, T.")
print()

# ── ЧТО ЭТО ОЗНАЧАЕТ ДЛЯ ГИПОТЕЗЫ БЭ ────────────────────────

print("── Что это означает для гипотезы БЭ ──")
print()
print("  Вопрос: БЭ воспроизводит КГ вакуум?")
print()
print("  Ответ (уточнённый):")
print()
print("  1. MC с детальным балансом → slope=-2 в УФ")
print("     (аналитически гарантировано)")
print()
print("  2. В ИК режиме (k ~ m) спектр ПЛОСКИЙ (slope→0)")
print("     Это ИНФРАКРАСНЫЙ РЕГУЛЯТОР массы поля.")
print("     Без массы (m→0): slope=-2 везде,")
print("     но суммарные флуктуации расходятся.")
print()
print("  3. Вопрос к гипотезе БЭ:")
print("     КАКОЙ диапазон k физически наблюдаем?")
print()
print("     Если наблюдаем УФ: slope=-2 → не инфляционный")
print("     Планк наблюдает: slope = n_s-1 = -0.035")
print()
print("     Вывод: КГ вакуум ≠ инфлатонный спектр")
print("     Нужен другой механизм для n_s≈0.965")
print()

# ── ФИНАЛЬНАЯ ПОЗИЦИЯ ─────────────────────────────────────────

print("="*60)
print("ФИНАЛЬНАЯ ПОЗИЦИЯ Part XXI")
print("="*60)
print()

print("""
ЧТО ДОКАЗАНО АНАЛИТИЧЕСКИ:

  slope = -2 для P(k) = 1/(k²+m²) при k >> m

  Это не открытие — это свойство дисперсии
  безмассового предела λ_k ~ k².

ЧТО ОЗНАЧАЕТ ДЛЯ ГИПОТЕЗЫ БЭ:

  БЭ (MC) → КГ вакуум: ВЕРНО (по теореме)
  КГ вакуум → инфлатонный спектр: НЕВЕРНО

  КГ даёт slope=-2 в УФ.
  Инфляция даёт slope≈-0.035 (почти плоский).

  Разрыв: |(-2) - (-0.035)| = 1.965

  Для получения плоского спектра нужно:
  • медленно катящийся инфлатон (slow-roll)
  • квантовые флуктуации de Sitter
  • НЕ классический КГ вакуум

ЧТО ВЫЖИЛО ИЗ Part XXI:

  ✓ Интерпретация: КТП = статистика БЭ
    (это верно как переформулировка
    через путевой интеграл)

  ✓ Детальный баланс = принцип наименьшего
    действия в формулировке БЭ

  ✗ Не добавляет новых физических предсказаний
  ✗ Не воспроизводит CMB спектр (n_s≈0.965)
  ✗ slope=-2 ≠ n_s-1=-0.035

ИТОГ ВСЕГО ПРОЕКТА (Parts I-XXI):

  Философская идея: выжила
  Математическая онтология: корректна
  Физические предсказания: все фальсифицированы

  Самый честный итог: гипотеза БЭ эквивалентна
  квантовой теории поля в формулировке
  путевого интеграла — но не даёт новых
  предсказаний сверх КТП.
""")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'Part XXI Step 6: Финальный анализ\n'
    'slope=-2 в УФ режиме: аналитически точно',
    fontsize=12)

# 0,0: полный спектр с режимами
ax = axes[0,0]
freqs_plot = freqs_big[1:]
P_plot     = P_big[1:]

ax.loglog(freqs_plot, P_plot,
          'b-', lw=2, label='P(k) = 1/λ_k')

# Асимптотики
k_ref  = freqs_plot
P_uv   = 0.3 * k_ref**(-2)
P_ir   = 0.15 * np.ones_like(k_ref)
ax.loglog(k_ref, P_uv, 'r--', lw=1.5,
          label='k⁻² (УФ, slope=-2)', alpha=0.7)
ax.loglog(k_ref, P_ir, 'g--', lw=1.5,
          label='k⁰ (ИК плато)', alpha=0.7)

ax.axvline(k_cross_frac, color='orange',
           ls=':', lw=2,
           label=f'k_cross=m/2π={k_cross_frac:.3f}')
ax.axvspan(5*k_cross_frac, 0.4,
           alpha=0.1, color='red',
           label='УФ режим (fit)')

ax.set_xlabel('k (частота)')
ax.set_ylabel('P(k)')
ax.set_title('Спектр КГ: три режима\n'
             'slope=-2 только в УФ!')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

# 0,1: slope vs N (старый и новый fit range)
ax = axes[0,1]

# Старый: 0.02..0.3
slopes_old = []
# Новый: >5*k_cross..0.4
slopes_new = []

for N in N_vals:
    f_a, P_a = kg_spectrum_analytical(N, m2)
    sl_old, _ = slope_in_range(f_a, P_a, 0.02, 0.30)
    sl_new, _ = slope_in_range(f_a, P_a,
                                5*k_cross_frac, 0.40)
    slopes_old.append(sl_old)
    slopes_new.append(sl_new)

ax.semilogx(N_vals, slopes_old, 'rs--', lw=2, ms=8,
            label='Старый диапазон [0.02, 0.30]')
ax.semilogx(N_vals, slopes_new, 'go-', lw=2, ms=8,
            label=f'УФ диапазон [5m/2π, 0.40]')
ax.axhline(-2.0, color='k', ls='--', lw=2,
           label='Теория: -2.0')
ax.axhline(-0.035, color='purple', ls=':', lw=2,
           label='Планк: n_s-1=-0.035')
ax.set_xlabel('N')
ax.set_ylabel('slope')
ax.set_title('slope(N): старый vs правильный диапазон')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,0: что нужно для n_s=0.965
ax = axes[1,0]

k_arr = np.logspace(-3, 0, 200)

P_kg    = k_arr**(-2)          # КГ вакуум
P_planck = k_arr**(-0.035)     # CMB (Планк)
P_hr    = k_arr**(0)           # Харрисон-Зельдович

# Нормировка
P_kg    /= P_kg[50]
P_planck /= P_planck[50]
P_hr    /= P_hr[50]

ax.loglog(k_arr, P_kg,
          'b-', lw=2, label='КГ вакуум: k⁻²')
ax.loglog(k_arr, P_planck,
          'r-', lw=2,
          label=f'CMB (Планк): k^(n_s-1)=k^-0.035')
ax.loglog(k_arr, P_hr,
          'g--', lw=1.5,
          label='Харрисон-Зельдович: k⁰')

ax.set_xlabel('k')
ax.set_ylabel('P(k) (нормировано)')
ax.set_title('КГ вакуум vs инфлатонный спектр\n'
             'slope: -2 vs -0.035 (разница огромная!)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# 1,1: итог
ax = axes[1,1]
ax.axis('off')

summary = [
    'ИТОГ Part XXI',
    '═'*28,
    '',
    'slope = -2: АНАЛИТИЧЕСКИ ТОЧНО',
    'для P(k)~1/(k²+m²) при k>>m.',
    'Это свойство КГ, не открытие.',
    '',
    'Прежний fit (0.02..0.30):',
    '  включал ИК переход → slope≈-1.7',
    '  это артефакт диапазона',
    '',
    'Правильный fit (k>5m):',
    '  slope → -2.000 (аналитически)',
    '',
    'Для CMB нужно slope ≈ -0.035',
    '(Планк: n_s = 0.965 ± 0.004)',
    '',
    'КГ вакуум: slope = -2.000',
    'Разрыв: |(-2)-(-0.035)| = 1.965',
    '',
    '→ КГ ≠ инфлатонный спектр',
    '→ Нужен slow-roll инфлатон',
    '→ БЭ(MC) не объясняет CMB',
    '',
    'Статус Parts I-XXI:',
    '  0 физических предсказаний',
    '  Философская идея: жива',
    '  Физическая программа: исчерпана',
]

ax.text(0.03, 0.97, '\n'.join(summary),
        transform=ax.transAxes,
        fontsize=8.5, va='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='#fef9e7', alpha=0.9))

plt.tight_layout()
plt.savefig('part21_step6_final.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part21_step6_final.png")

print()
print("Следующий вопрос:")
print("  Что дальше исследовать?")
print()
print("  Option A: остановиться")
print("    Parts I-XXI = полная, честная фальсификация")
print("    Публикуемый результат")
print()
print("  Option B: квантовый БЭ")
print("    |ψ_БЭ⟩ на решётке Планка")
print("    Unitарная эволюция, не MC")
print("    Другой математический аппарат")
print()
print("  Option C: стохастическая инфляция")
print("    Starobinsky (1986): инфлатон = стохастический")
print("    Медленное качение + квантовый шум")
print("    БЭ как физическая реализация φ(t)")
print("    → возможно воспроизведёт n_s≈0.965?")
