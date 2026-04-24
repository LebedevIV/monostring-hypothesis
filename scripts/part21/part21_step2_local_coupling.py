"""
Part XXI Step 2: БЭ с локальной связью

Ключевое улучшение:
При посещении точки x, БЭ обновляет φ(x)
с учётом соседей φ(x±1) — это воспроизводит
уравнение движения поля (дискретный лапласиан).

Физика:
  Обычное поле: dφ/dt = -δH/δφ = ∇²φ - m²φ
  БЭ-версия: при посещении x →
    φ(x) ← φ(x) + α·[φ(x+1) + φ(x-1) - 2φ(x)]
                 - β·φ(x) + η  (шум)

Это дискретизованное уравнение Клейна-Гордона!
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SEED = 42
np.random.seed(SEED)

# ── ПАРАМЕТРЫ ────────────────────────────────────────────────

N_sites  = 512    # точек пространства
N_cycles = 50000  # циклов БЭ
m2       = 0.01   # масса² поля
alpha    = 0.3    # сила лапласиана (связь с соседями)
noise    = 0.1    # амплитуда вакуумных флуктуаций

# Три варианта порядка обхода:
ORDERS = ['random',      # случайный каждый цикл
          'sequential',  # 0,1,2,...,N-1
          'energy']      # по убыванию |φ| (высокоэн. первые)

print("="*60)
print("Part XXI Step 2: БЭ с локальной связью (KG-подобный)")
print("="*60)
print()
print(f"N_sites={N_sites}, N_cycles={N_cycles}")
print(f"m²={m2}, α={alpha}, noise={noise}")
print()

results = {}

for order_type in ORDERS:
    rng   = np.random.RandomState(SEED)
    field = rng.randn(N_sites) * noise  # начальные условия

    # Сбор статистики
    snapshots     = []
    sample_every  = 500

    for cycle in range(N_cycles):

        # Выбор порядка обхода
        if order_type == 'random':
            order = rng.permutation(N_sites)
        elif order_type == 'sequential':
            order = np.arange(N_sites)
        else:  # energy: сначала высокоэнергетические
            order = np.argsort(-np.abs(field))

        # Обход всех точек за один цикл
        for i in order:
            left  = (i - 1) % N_sites
            right = (i + 1) % N_sites

            # Дискретный лапласиан
            laplacian = (field[left] + field[right]
                         - 2*field[i])

            # Обновление: уравнение КГ + шум
            field[i] += (alpha * laplacian
                         - m2 * field[i]
                         + noise * rng.randn())

        if cycle % sample_every == 0:
            snapshots.append(field.copy())

    snapshots = np.array(snapshots)

    # ── СТАТИСТИКА ───────────────────────────────────────────

    # Пространственный коррелятор
    field_final = field - field.mean()
    corr_full   = np.correlate(field_final,
                                field_final,
                                mode='full')
    corr        = corr_full[N_sites-1:]
    corr       /= corr[0]

    # Корреляционная длина (e-folding)
    try:
        l_corr = np.where(corr < 1/np.e)[0][0]
    except IndexError:
        l_corr = N_sites

    # Спектр мощности
    fft   = np.fft.rfft(field_final)
    power = np.abs(fft)**2
    freqs = np.fft.rfftfreq(N_sites)

    # Наклон в логарифмическом масштабе
    mask = (freqs > freqs[1]) & (freqs < 0.4)
    if mask.sum() > 5:
        slope, intercept, r, p, se = stats.linregress(
            np.log(freqs[mask]),
            np.log(power[mask] + 1e-30))
    else:
        slope = np.nan

    # Ожидаемый спектр КГ в 1D:
    # P(k) = 1/(k² + m²)  → наклон ≈ -2 при k >> m
    k_arr      = 2*np.pi*freqs[mask]
    P_KG_theor = 1.0 / (k_arr**2 + m2)
    P_KG_theor *= (power[mask].mean() /
                   P_KG_theor.mean())

    results[order_type] = {
        'field':    field,
        'corr':     corr,
        'l_corr':   l_corr,
        'power':    power,
        'freqs':    freqs,
        'slope':    slope,
        'P_KG':     P_KG_theor,
        'k_KG':     k_arr,
        'mask':     mask,
        'snaps':    snapshots,
    }

    print(f"Order: {order_type:>12}")
    print(f"  ⟨φ⟩  = {field.mean():+.4f}")
    print(f"  σ(φ) = {field.std():.4f}")
    print(f"  l_corr = {l_corr} sites")
    print(f"  P(k) ~ k^{slope:.3f}  "
          f"(КГ ожидает ≈ -2)")
    print()

# ── СРАВНЕНИЕ С КВАНТОВЫМ ВАКУУМОМ ──────────────────────────

print("── Сравнение с теоретическим вакуумом КГ ──")
print()
print("Квантовый вакуум 1D Клейна-Гордона:")
print(f"  P(k) ~ 1/(k²+m²)")
print(f"  При m²={m2}: наклон ≈ -2 для k >> {np.sqrt(m2):.3f}")
print(f"  Корреляционная длина: l_corr = 1/m = "
      f"{1/np.sqrt(m2):.1f} sites")
print()

theor_l_corr = 1.0 / np.sqrt(m2)
print(f"  Теоретическая l_corr = {theor_l_corr:.1f}")
for order_type in ORDERS:
    r   = results[order_type]
    dev = abs(r['l_corr'] - theor_l_corr) / theor_l_corr
    match = "✓" if dev < 0.3 else "✗"
    print(f"  {order_type:>12}: l_corr={r['l_corr']:5.1f} "
          f"(отклонение {dev:.0%}) {match}")

print()

# Какой порядок лучше воспроизводит КГ?
slopes = {o: results[o]['slope'] for o in ORDERS}
target_slope = -2.0
best_order = min(slopes,
                 key=lambda o: abs(slopes[o]-target_slope))
print(f"Лучшее соответствие КГ (наклон -2): {best_order}")
print(f"  slope = {slopes[best_order]:.3f}")
print()

# ── КЛЮЧЕВОЙ ВОПРОС: ПОРЯДОК ОБХОДА ─────────────────────────

print("── Ключевой вопрос: порядок обхода БЭ ──")
print()
print("Три физических интерпретации:")
print()
print("  random:     БЭ обходит точки в случайном порядке")
print("              → квантовая случайность как следствие")
print("              → нарушение причинности?")
print()
print("  sequential: БЭ обходит 0→1→2→...→N")
print("              → выделенное пространственное")
print("                направление → нарушение изотропии")
print()
print("  energy:     БЭ сначала посещает высокоэнерг. точки")
print("              → приоритет нестабильных состояний")
print("              → аналог принципа наименьшего действия?")
print()

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(
    'Part XXI: БЭ как последовательный автомат\n'
    'Три варианта порядка обхода vs квантовый вакуум',
    fontsize=13)

colors = {'random': '#e74c3c',
           'sequential': '#3498db',
           'energy': '#2ecc71'}

for col, order_type in enumerate(ORDERS):
    r = results[order_type]

    # Row 0: профиль поля
    ax = axes[0, col]
    ax.plot(r['field'][:200],
            color=colors[order_type],
            lw=1, alpha=0.8)
    ax.set_title(f'Order: {order_type}\n'
                 f'σ={r["field"].std():.3f}',
                 fontsize=10)
    ax.set_xlabel('x (планковские единицы)')
    ax.set_ylabel('φ(x)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    # Row 1: коррелятор
    ax = axes[1, col]
    max_lag = min(100, len(r['corr']))
    lags    = np.arange(max_lag)
    ax.plot(lags, r['corr'][:max_lag],
            color=colors[order_type], lw=2)

    # Теоретический: exp(-x/l_corr)
    l_th = 1.0/np.sqrt(m2)
    ax.plot(lags, np.exp(-lags/l_th),
            'k--', lw=1.5, alpha=0.7,
            label=f'КГ: exp(-x/{l_th:.0f})')
    ax.axhline(1/np.e, color='gray',
               ls=':', lw=1, label='1/e')
    ax.axvline(r['l_corr'], color=colors[order_type],
               ls=':', lw=1,
               label=f"l_corr={r['l_corr']}")
    ax.set_xlabel('Лаг (точки)')
    ax.set_ylabel('C(r)')
    ax.set_title(f'Коррелятор ⟨φ(x)φ(x+r)⟩',
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 1.1)

    # Row 2: спектр мощности
    ax = axes[2, col]
    freqs = r['freqs']
    power = r['power']
    mask  = (freqs > freqs[1]) & (freqs < 0.45)

    ax.loglog(freqs[mask], power[mask],
              color=colors[order_type],
              lw=1.5, alpha=0.8,
              label=f'БЭ: k^{r["slope"]:.2f}')

    # Теоретический КГ
    ax.loglog(r['k_KG']/(2*np.pi),
              r['P_KG'], 'k--',
              lw=2, alpha=0.7,
              label='КГ: 1/(k²+m²)')

    ax.set_xlabel('k (1/сайт)')
    ax.set_ylabel('P(k)')
    ax.set_title(f'Спектр мощности\n'
                 f'наклон={r["slope"]:.2f} '
                 f'(КГ≈-2)',
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('part21_step2_local.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part21_step2_local.png")

# ── ФИНАЛЬНЫЙ ВЫВОД ──────────────────────────────────────────

print()
print("="*60)
print("ВЫВОД Part XXI Step 2")
print("="*60)
print()
print("Модель БЭ с локальной связью (дискретный КГ):")
print()

for order_type in ORDERS:
    r     = results[order_type]
    slope = r['slope']
    l_c   = r['l_corr']
    l_th  = 1.0/np.sqrt(m2)

    slope_ok = abs(slope - (-2.0)) < 0.5
    lcorr_ok = abs(l_c - l_th) / l_th < 0.3

    status = "✓ СОГЛАСУЕТСЯ с КГ" \
             if (slope_ok and lcorr_ok) \
             else "✗ не согласуется"

    print(f"  {order_type:>12}: {status}")
    print(f"    slope={slope:.2f} "
          f"({'✓' if slope_ok else '✗'}), "
          f"l_corr={l_c:.0f} "
          f"({'✓' if lcorr_ok else '✗'} "
          f"теория={l_th:.0f})")

print()
print("Если порядок 'energy' воспроизводит КГ лучше других:")
print("→ это означает что принцип наименьшего действия")
print("  СЛЕДУЕТ из порядка обхода БЭ!")
print("→ Физический закон как алгоритм.")
print()
print("Следующий шаг:")
print("  Step 3: проверить в 2D/3D")
print("  Step 4: сравнить с наблюдениями CMB")
