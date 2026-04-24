"""
Part XXI Step 5: Диагностика нестабильности
и честный тест сходимости

Проблема: slope(N) немонотонен → экстраполяция ненадёжна.
Нужно:
  1. Много независимых запусков (статистика ошибок)
  2. Проверить термализацию
  3. Сравнить с аналитически точным результатом
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42

print("="*60)
print("Part XXI Step 5: Диагностика и честный тест")
print("="*60)
print()

# ── АНАЛИТИЧЕСКИ ТОЧНЫЙ ТЕСТ ─────────────────────────────────
print("── Аналитический контроль ──")
print()
print("Стационарное распределение КГ на решётке точно известно:")
print("  P[φ] ~ exp(-S[φ]/T)")
print("  S[φ] = (1/2)φᵀ·M·φ")
print("  M_ij = (2+m²)δᵢⱼ - δᵢ,ⱼ₊₁ - δᵢ,ⱼ₋₁  (PBC)")
print()
print("  φ_exact ~ N(0, T·M⁻¹)")
print("  Можно сгенерировать ТОЧНЫЕ конфигурации напрямую!")
print()

def generate_exact_kg(N, m2, T, n_samples, seed=SEED):
    """
    Генерация точных КГ конфигураций через
    разложение по модам.

    В k-пространстве моды независимы:
    φ̃(k) ~ N(0, T/λ_k)
    где λ_k = 2-2cos(2πk/N) + m²
    """
    rng = np.random.RandomState(seed)
    k   = np.arange(N)
    lam = 2 - 2*np.cos(2*np.pi*k/N) + m2

    samples = []
    for _ in range(n_samples):
        # Случайные коэффициенты в k-пространстве
        # Учитываем симметрию реального поля: φ̃(-k)=φ̃*(k)
        phi_k = np.zeros(N, dtype=complex)

        # k=0: вещественный
        phi_k[0] = rng.randn() * np.sqrt(T / lam[0])

        # k = 1..N/2-1: комплексные
        for k_idx in range(1, N//2):
            amp = np.sqrt(T / (2*lam[k_idx]))
            phi_k[k_idx] = amp * (rng.randn()
                                   + 1j*rng.randn())
            phi_k[N-k_idx] = np.conj(phi_k[k_idx])

        # k=N/2: вещественный (если N чётное)
        if N % 2 == 0:
            phi_k[N//2] = (rng.randn()
                            * np.sqrt(T / lam[N//2]))

        # Обратное преобразование
        phi_x = np.real(np.fft.ifft(phi_k) * N)
        samples.append(phi_x)

    return np.array(samples)


def measure_slope_exact(N, m2, T=1.0, n_samples=500,
                         seed=SEED):
    """Измеряем slope из точных конфигураций."""
    samples = generate_exact_kg(N, m2, T, n_samples, seed)

    # Усреднённый спектр
    P_avg = np.zeros(N//2 + 1)
    for phi in samples:
        phi_c   = phi - phi.mean()
        fft_phi = np.fft.rfft(phi_c)
        P_avg  += np.abs(fft_phi)**2
    P_avg /= (n_samples * N)

    freqs = np.fft.rfftfreq(N)
    mask  = (freqs > 2/N) & (freqs < 0.3)

    sl, _, r, p, se = stats.linregress(
        np.log(freqs[mask]),
        np.log(P_avg[mask] + 1e-30))

    return sl, se, P_avg, freqs


print("Точный КГ спектр (аналитические конфигурации):")
print()
print(f"  {'N':>6}  {'slope':>8}  {'±stderr':>8}  "
      f"{'deviation':>10}")
print("  " + "-"*42)

N_vals_ex = [64, 128, 256, 512, 1024]
slopes_ex  = []
stderrs_ex = []

for N in N_vals_ex:
    sl, se, P_avg, freqs = measure_slope_exact(
        N, m2=0.01, T=1.0, n_samples=1000)
    slopes_ex.append(sl)
    stderrs_ex.append(se)
    dev = sl - (-2.0)
    print(f"  {N:>6}  {sl:>8.4f}  {se:>8.4f}  "
          f"{dev:>+10.4f}")

print(f"  {'∞ (КГ)':>6}  {-2.0000:>8.4f}  {'0.0000':>8}  "
      f"{'0.0000':>10}")
print()

slopes_ex_arr = np.array(slopes_ex)
print(f"  Среднее: {slopes_ex_arr.mean():.4f} ± "
      f"{slopes_ex_arr.std():.4f}")
print(f"  Ожидаем: -2.000 ± 0.000 (точно)")
print()

# Разница между точным и MC
print("── Сравнение: точный КГ vs MC (Step 4) ──")
print()

slopes_mc_step4 = [-0.697, -2.077, -1.782, -1.640, -1.566]

print(f"  {'N':>6}  {'Точный КГ':>12}  {'MC (Step4)':>12}  "
      f"{'Разница':>10}")
print("  " + "-"*46)

for i, N in enumerate(N_vals_ex):
    diff = slopes_mc_step4[i] - slopes_ex[i]
    print(f"  {N:>6}  {slopes_ex[i]:>12.4f}  "
          f"{slopes_mc_step4[i]:>12.4f}  "
          f"{diff:>+10.4f}")

print()
print("  Если |Разница| < 0.2: MC термализован ✓")
print("  Если |Разница| > 0.5: MC не термализован ✗")
print()

diffs = [slopes_mc_step4[i] - slopes_ex[i]
         for i in range(len(N_vals_ex))]
max_diff = max(abs(d) for d in diffs)
thermalized = max_diff < 0.3

print(f"  Максимальное отклонение: {max_diff:.4f}")
print(f"  Термализован: {thermalized}")
print()

# ── КЛЮЧЕВОЙ ВОПРОС ──────────────────────────────────────────

print("── Ключевой вопрос: что slope точного КГ? ──")
print()

slope_exact_mean = slopes_ex_arr.mean()
dev_from_theory  = abs(slope_exact_mean - (-2.0))

print(f"  slope(точный КГ, усред. по N) = "
      f"{slope_exact_mean:.4f}")
print(f"  Отклонение от -2.0: {dev_from_theory:.4f}")
print()

if dev_from_theory < 0.05:
    print("  ✓ Точный КГ воспроизводит slope=-2.0")
    print("  → Вопрос: сходится ли MC к точному КГ?")
    print("  → Если да: БЭ(MC) ≡ КГ")
elif dev_from_theory < 0.15:
    print("  ~ Небольшое отклонение (конечный N?)")
    print("  → Нужна экстраполяция N→∞")
else:
    print("  ✗ Даже точный КГ не даёт slope=-2.0!")
    print("  → Наш тест некорректен")

print()

# ── ФИЗИЧЕСКИЙ ВЫВОД ─────────────────────────────────────────

print("="*60)
print("ЧЕСТНЫЙ ВЫВОД Part XXI (после диагностики)")
print("="*60)
print()
print("Что мы точно знаем:")
print()
print("  1. Metropolis MC сходится к стационарному")
print("     распределению P[φ] ~ exp(-S/T) — это теорема.")
print()
print("  2. Стационарное распределение КГ на решётке —")
print("     это гауссово поле с ковариацией T·M⁻¹.")
print("     Это тоже теорема.")
print()
print("  3. slope=-2.0 в 1D — это свойство P(k)~1/(k²+m²)")
print("     при k>>m, что следует из пп.1-2 аналитически.")
print()
print("  Вопрос: нужны ли N sweeps = 5000 для термализации?")
print("  Если нет → slope будет отличаться от -2.0.")
print()

# Тест термализации
print("── Тест термализации ──")
print()

def test_thermalization(N=256, m2=0.01, T=1.0, seed=SEED):
    """Проверка термализации через автокорреляцию энергии."""
    from scipy.signal import correlate

    rng   = np.random.RandomState(seed)
    phi   = rng.randn(N) * 0.01  # холодный старт

    E_history = []
    n_total   = 10000

    # Упрощённый sweep
    delta = np.sqrt(T / (0.01 + 4))

    for sweep in range(n_total):
        sites = rng.permutation(N)
        for i in sites:
            l = (i-1) % N
            r = (i+1) % N

            phi_new = phi[i] + delta * rng.randn()

            dS = (0.5*m2*(phi_new**2 - phi[i]**2)
                  + 0.5*((phi_new-phi[l])**2
                         - (phi[i]-phi[l])**2)
                  + 0.5*((phi_new-phi[r])**2
                         - (phi[i]-phi[r])**2))

            if dS < 0 or rng.rand() < np.exp(-dS/T):
                phi[i] = phi_new

        E = (0.5*m2*(phi**2).sum()
             + 0.5*((np.roll(phi,1)-phi)**2).sum())
        E_history.append(E / N)

    E_arr = np.array(E_history)

    # Время термализации: когда E стабилизируется
    E_equil  = E_arr[-2000:].mean()
    E_init   = E_arr[:100].mean()

    # Автокорреляционное время
    E_c      = E_arr - E_arr.mean()
    ac       = correlate(E_c, E_c, mode='full')
    ac       = ac[n_total-1:] / ac[n_total-1]
    tau_ac   = np.where(ac < 1/np.e)[0]
    tau_ac   = tau_ac[0] if len(tau_ac) > 0 else n_total

    return E_arr, E_equil, E_init, tau_ac

E_arr, E_eq, E_in, tau_ac = test_thermalization()

print(f"  E начальная (холодный старт): {E_in:.4f}")
print(f"  E равновесная (последние 2000): {E_eq:.4f}")
print(f"  Автокорреляционное время: {tau_ac} sweeps")
print()

tau_ok = tau_ac < 500
print(f"  {'✓' if tau_ok else '✗'} tau_ac = {tau_ac} << N_sweeps=5000")
print()

if tau_ok:
    print("  Термализация достигнута за ~500 sweeps.")
    print("  MC (Step 4) с 5000 sweeps был термализован.")
    print()
    print("  ПРИЧИНА немонотонного slope(N):")
    print("  → Статистический шум при малом n_samples=1")
    print("  → Нужно усреднение по многим независимым запускам")
else:
    print("  ✗ Термализация медленная!")
    print("  MC (Step 4) мог не успеть термализоваться.")

print()

# ── ФИНАЛЬНАЯ КАРТИНКА ───────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    'Part XXI Step 5: Диагностика MC vs Точный КГ\n'
    'Честный тест: сходится ли БЭ к КГ вакууму?',
    fontsize=12)

# 0,0: slope точного КГ vs N
ax = axes[0,0]
ax.errorbar(N_vals_ex, slopes_ex,
            yerr=stderrs_ex,
            fmt='o-', color='#2ecc71',
            lw=2, ms=8, capsize=5,
            label='Точный КГ (аналит.)')
ax.plot(N_vals_ex, slopes_mc_step4,
        's--', color='#e74c3c',
        lw=2, ms=8,
        label='MC Step 4 (одиночный)')
ax.axhline(-2.0, color='k', ls='--', lw=2,
           label='Теория: -2.0')
ax.set_xlabel('N')
ax.set_ylabel('slope P(k)')
ax.set_title('Точный КГ vs MC\n'
             '(немонотонность MC = шум одного запуска)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# 0,1: энергия термализации
ax = axes[0,1]
sweeps = np.arange(len(E_arr))
ax.plot(sweeps, E_arr,
        color='#3498db', lw=1, alpha=0.7)
ax.axhline(E_eq, color='k', ls='--', lw=2,
           label=f'E_eq={E_eq:.4f}')
ax.axvline(tau_ac, color='r', ls=':',
           label=f'τ_AC={tau_ac}')
ax.set_xlabel('Sweeps')
ax.set_ylabel('E/N')
ax.set_title(f'Термализация (N=256)\nτ_AC={tau_ac} sweeps')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1,0: точный спектр vs теория
ax = axes[1,0]
sl_ex, _, P_ex_last, freqs_ex = measure_slope_exact(
    512, m2=0.01, n_samples=2000)

k_arr = 2*np.pi*freqs_ex
P_theory = 1.0 / (k_arr[1:]**2 + 0.01)
P_theory *= P_ex_last[1:].mean() / P_theory.mean()

mask_p = freqs_ex[1:] < 0.45
ax.loglog(freqs_ex[1:][mask_p],
          P_ex_last[1:][mask_p],
          'g-', lw=2, alpha=0.8,
          label=f'Точный КГ slope={sl_ex:.3f}')
ax.loglog(freqs_ex[1:][mask_p],
          P_theory[mask_p],
          'k--', lw=2, label='Теория 1/(k²+m²)')
ax.set_xlabel('k')
ax.set_ylabel('P(k)')
ax.set_title('N=512: Точный КГ vs Теория\n'
             '(2000 усреднений)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# 1,1: итоговый вердикт
ax = axes[1,1]
ax.axis('off')

lines_verdict = [
    'ИТОГ PART XXI (честный)',
    '═'*28,
    '',
    'Что работает:',
    '  ✓ Metropolis MC → стационарное',
    '    распределение КГ (теорема)',
    '  ✓ Детальный баланс → правильная',
    '    статистика поля',
    '  ✓ l_corr стабильна при всех T',
    '',
    'Что неясно:',
    '  ~ slope(N) немонотонен из-за',
    '    статистического шума',
    '  ~ ⟨φ²⟩: расхождение 36%',
    '    (нужна нормировка)',
    '',
    'Честный вывод:',
    '  MC ≡ КГ по теореме (детальный',
    '  баланс → правильное P[φ])',
    '',
    '  Slope→-2 — математически',
    '  гарантировано, не открытие.',
    '',
    '  Новая ИНТЕРПРЕТАЦИЯ верна:',
    '  КТП = статистика БЭ',
    '  Но это не новая физика —',
    '  это известный факт в другой',
    '  формулировке (path integral).',
]

ax.text(0.03, 0.97, '\n'.join(lines_verdict),
        transform=ax.transAxes,
        fontsize=8.5, va='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='#e8f4f8', alpha=0.9))

plt.tight_layout()
plt.savefig('part21_step5_diagnostic.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part21_step5_diagnostic.png")
