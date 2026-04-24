"""
Part XXI Step 3: Стационарное распределение БЭ

Вместо динамики обхода — прямое сравнение
стационарных распределений.

Вопрос: какое стационарное распределение φ(x)
производит БЭ с локальной связью?

Метод: Monte Carlo (метрополис) — это
mathematically equivalent to sequential БЭ
с правильным балансом.

Проверяем:
1. Воспроизводит ли MC правильный КГ спектр?
2. Как порядок обхода влияет на термализацию?
3. Есть ли E8-специфичный сигнал в 8D версии?
"""

import numpy as np
from scipy import stats, linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

print("="*60)
print("Part XXI Step 3: Стационарное распределение")
print("="*60)
print()

# ── АНАЛИТИЧЕСКОЕ РЕШЕНИЕ КГ НА РЕШЁТКЕ ─────────────────────

def kg_spectrum_exact(N, m2, T=1.0):
    """
    Точный спектр мощности КГ поля на 1D решётке:
    P(k) = T / (2 - 2cos(2πk/N) + m²)

    Это и есть то, что должен воспроизвести БЭ
    в стационарном состоянии.
    """
    k_arr = np.arange(N//2 + 1)
    freqs = k_arr / N
    lam   = 2 - 2*np.cos(2*np.pi*freqs) + m2
    P     = T / lam
    return freqs, P

# ── METROPOLIS MC = "ПРАВИЛЬНЫЙ БЭ" ─────────────────────────

def metropolis_kg(N, m2, T, n_sweeps,
                   order='random', seed=SEED):
    """
    Metropolis MC для КГ поля.
    Один sweep = N обновлений (один "цикл БЭ").

    Это mathematically exact:
    стационарное распределение = P[φ] ~ exp(-S[φ]/T)
    где S[φ] = Σ [(∇φ)² + m²φ²]/2

    order: порядок обхода сайтов за sweep
    """
    rng   = np.random.RandomState(seed)
    phi   = rng.randn(N) * np.sqrt(T / m2)
    delta = 0.5 * np.sqrt(T)  # шаг предложения

    n_acc     = 0
    snapshots = []
    E_history = []

    for sweep in range(n_sweeps):

        # Порядок обхода
        if order == 'random':
            sites = rng.permutation(N)
        elif order == 'sequential':
            sites = np.arange(N)
        else:  # checkerboard: чётные потом нечётные
            sites = np.concatenate([
                np.arange(0, N, 2),
                np.arange(1, N, 2)])

        for i in sites:
            left  = (i - 1) % N
            right = (i + 1) % N

            # Текущая локальная энергия
            dS_old = (0.5 * m2 * phi[i]**2
                      + (phi[i]-phi[left])**2/2
                      + (phi[i]-phi[right])**2/2)

            # Предложение
            phi_new = phi[i] + delta * rng.randn()

            # Новая локальная энергия
            dS_new = (0.5 * m2 * phi_new**2
                      + (phi_new-phi[left])**2/2
                      + (phi_new-phi[right])**2/2)

            # Metropolis acceptance
            if (dS_new < dS_old or
                    rng.rand() < np.exp(-(dS_new-dS_old)/T)):
                phi[i] = phi_new
                n_acc += 1

        if sweep % 100 == 0:
            snapshots.append(phi.copy())
            E = (0.5 * m2 * (phi**2).sum()
                 + 0.5 * ((np.roll(phi,1)-phi)**2).sum())
            E_history.append(E)

    acc_rate = n_acc / (n_sweeps * N)
    return phi, np.array(snapshots), acc_rate

# ── ПАРАМЕТРЫ ────────────────────────────────────────────────

N      = 256
m2     = 0.01
T      = 1.0      # "температура" = ℏ в единицах где ℏ=1
N_sw   = 20000    # sweeps (циклов БЭ)
N_warm = 2000     # прогрев

print(f"N={N} сайтов, m²={m2}, T={T}")
print(f"N_sweeps={N_sw}, warmup={N_warm}")
print()

# Точный спектр
freqs_ex, P_ex = kg_spectrum_exact(N, m2, T)

# ── СРАВНЕНИЕ ТРЁХ ПОРЯДКОВ ───────────────────────────────────

orders = ['random', 'sequential', 'checkerboard']
mc_results = {}

print(f"{'Order':>14} {'Acc.rate':>10} "
      f"{'slope':>8} {'l_corr':>8} {'χ²/dof':>10}")
print("-"*54)

for order in orders:
    # Запуск MC
    phi_warm, _, _ = metropolis_kg(
        N, m2, T, N_warm, order=order, seed=SEED)

    phi_final, snaps, acc = metropolis_kg(
        N, m2, T, N_sw, order=order,
        seed=SEED+1)

    # Спектр из последнего снимка
    phi_c = phi_final - phi_final.mean()
    fft   = np.fft.rfft(phi_c)
    P_mc  = np.abs(fft)**2 / N

    # Наклон
    fmask = (freqs_ex > 0.02) & (freqs_ex < 0.4)
    if fmask.sum() > 5:
        sl, _, r_sl, _, _ = stats.linregress(
            np.log(freqs_ex[fmask]),
            np.log(P_mc[fmask] + 1e-30))
    else:
        sl = np.nan

    # Корреляционная длина
    corr = np.correlate(phi_c, phi_c, mode='full')
    corr = corr[N-1:] / corr[N-1]
    try:
        l_c = np.where(corr < 1/np.e)[0][0]
    except IndexError:
        l_c = N

    # χ² сравнение с теорией
    P_th_interp = np.interp(freqs_ex[fmask],
                             freqs_ex, P_ex)
    P_mc_norm   = (P_mc[fmask] *
                   P_th_interp.mean() /
                   (P_mc[fmask].mean() + 1e-30))
    chi2 = (((P_mc_norm - P_th_interp)**2) /
             (P_th_interp**2 + 1e-30)).mean()

    mc_results[order] = {
        'phi': phi_final, 'snaps': snaps,
        'P_mc': P_mc, 'slope': sl,
        'l_corr': l_c, 'acc': acc,
        'chi2': chi2
    }

    l_th = 1.0/np.sqrt(m2)
    print(f"{order:>14} {acc:>10.3f} "
          f"{sl:>8.3f} {l_c:>8.1f} {chi2:>10.4f}")

print(f"{'ТЕОРИЯ КГ':>14} {'1.000':>10} "
      f"{-2.0:>8.3f} {1/np.sqrt(m2):>8.1f} {'0.0000':>10}")
print()

# ── КЛЮЧЕВОЙ РЕЗУЛЬТАТ ───────────────────────────────────────

print("── Насколько близко к КГ вакууму? ──")
print()

best_order = min(orders,
                 key=lambda o: mc_results[o]['chi2'])
print(f"Лучший порядок: {best_order} "
      f"(χ²={mc_results[best_order]['chi2']:.4f})")
print()

# Физический смысл
l_th = 1.0/np.sqrt(m2)
for order in orders:
    r     = mc_results[order]
    sl_ok = abs(r['slope'] - (-2.0)) < 0.3
    lc_ok = abs(r['l_corr'] - l_th) / l_th < 0.2
    both  = sl_ok and lc_ok
    status = "✓ КГ вакуум!" if both else "≈ частично"
    print(f"  {order:>14}: slope={r['slope']:.2f} "
          f"({'✓' if sl_ok else '✗'}), "
          f"l_corr={r['l_corr']:.0f} "
          f"({'✓' if lc_ok else '✗'}) → {status}")

print()

# ── СВЯЗЬ С ГИПОТЕЗОЙ БЭ ─────────────────────────────────────

print("── Связь с гипотезой БЭ ──")
print()
print("Metropolis MC = БЭ с условием детального баланса:")
print()
print("  P(принять s→s') = min(1, exp(-ΔS/T))")
print()
print("Физический смысл:")
print("  БЭ переходит в новое состояние s'")
print("  с вероятностью, зависящей от ΔS.")
print("  При ΔS<0: всегда принимает (энергетически выгодно)")
print("  При ΔS>0: принимает с exp(-ΔS/T)")
print()
print("Это НЕ случайный обход!")
print("Это ФИЗИЧЕСКИ МОТИВИРОВАННЫЙ обход:")
print("  → БЭ предпочитает низкоэнергетические состояния")
print("  → но допускает флуктуации (квантовый туннель)")
print()
print("Если MC воспроизводит КГ спектр точно:")
print("  → гипотеза БЭ с принципом детального баланса")
print("    математически эквивалентна КТП!")
print("  → это не новая физика, но новая интерпретация")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Part XXI Step 3: БЭ = Metropolis MC\n'
    'Стационарное распределение vs КГ вакуум',
    fontsize=13)

colors_o = {'random': '#e74c3c',
             'sequential': '#3498db',
             'checkerboard': '#2ecc71'}

for col, order in enumerate(orders):
    r = mc_results[order]

    # Row 0: спектр мощности
    ax = axes[0, col]
    fmask2 = freqs_ex > 0

    ax.loglog(freqs_ex[fmask2], r['P_mc'][fmask2],
              color=colors_o[order],
              lw=1.5, alpha=0.7,
              label=f'БЭ ({order})\nslope={r["slope"]:.2f}')
    ax.loglog(freqs_ex[fmask2], P_ex[fmask2],
              'k--', lw=2,
              label='КГ теория\nslope=-2')

    ax.set_xlabel('k')
    ax.set_ylabel('P(k)')
    ax.set_title(f'{order}\nAcc={r["acc"]:.2f}, '
                 f'χ²={r["chi2"]:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Row 1: коррелятор
    ax = axes[1, col]
    phi_c  = r['phi'] - r['phi'].mean()
    corr   = np.correlate(phi_c, phi_c, mode='full')
    corr   = corr[N-1:] / corr[N-1]
    lags   = np.arange(min(50, N))

    ax.plot(lags, corr[:len(lags)],
            color=colors_o[order], lw=2,
            label=f'БЭ l_corr={r["l_corr"]:.0f}')
    ax.plot(lags, np.exp(-lags * np.sqrt(m2)),
            'k--', lw=2,
            label=f'КГ l_corr={l_th:.0f}')
    ax.axhline(1/np.e, color='gray', ls=':')
    ax.set_xlabel('Лаг')
    ax.set_ylabel('C(r)/C(0)')
    ax.set_title('Пространственный коррелятор')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part21_step3_mc.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part21_step3_mc.png")

print()
print("="*60)
print("ИТОГ Part XXI")
print("="*60)
print()
print("Step 1: белый шум → нужна локальная связь")
print("Step 2: slope=-1.4, l_corr=5 → близко но не КГ")
print("Step 3: MC (детальный баланс) → ?")
print()
print("Если Step 3 даёт slope≈-2, l_corr≈10:")
print("  → БЭ с детальным балансом ≡ КТП")
print("  → новая интерпретация квантового вакуума")
print("  → публикуемый философско-физический результат")
print()
print("Если нет:")
print("  → классическая стохастическая модель")
print("  → не эквивалентна КТП без квантования")
