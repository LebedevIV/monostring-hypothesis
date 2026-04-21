"""
Part XIII: KAM Threshold κ_c(algebra)

H₀: Порог хаоса κ_c не зависит от алгебры Коксетера

Метод: максимальный показатель Ляпунова λ_max(κ)
  λ_max < 0 → регулярный режим (КАМ торы)
  λ_max = 0 → граница хаоса κ_c
  λ_max > 0 → хаос

Для каждой алгебры находим κ_c методом бисекции.
Сравниваем с N=200 случайными наборами частот.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

SEED    = 42
T_LYA   = 10000   # шаги для Ляпунова
WARMUP  = 1000
N_CTRL  = 200
N_KAPPA = 50      # точек по κ

np.random.seed(SEED)

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7},
    'F4': {'m': [1,5,7,11],              'h': 12},
    'G2': {'m': [1,5],                   'h':  6},
}

def coxeter_freqs(name):
    alg = ALGEBRAS[name]
    m = np.array(alg['m'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])

def random_freqs(rank, seed):
    return np.random.RandomState(seed).uniform(0.1, 2.0, rank)

# ── ЛЯПУНОВ ──────────────────────────────────────────────────

def lyapunov_max(omegas, kappa, T=T_LYA,
                 warmup=WARMUP, seed=SEED):
    """
    Максимальный показатель Ляпунова для
    многомерного стандартного отображения.

    Метод Бенеттина: следим за расстоянием
    между двумя близкими траекториями.
    """
    rng = np.random.RandomState(seed)
    r   = len(omegas)

    # Основная траектория
    phi = rng.uniform(0, 2*np.pi, r)

    # Возмущённая траектория
    delta0 = 1e-8
    dphi   = rng.normal(0, 1, r)
    dphi   = dphi / np.linalg.norm(dphi) * delta0
    phi_p  = phi + dphi

    lyap_sum = 0.0
    n_steps  = 0

    for t in range(T + warmup):
        # Основная
        phi = (phi + omegas + kappa*np.sin(phi)) % (2*np.pi)
        # Возмущённая
        phi_p = (phi_p + omegas + kappa*np.sin(phi_p)) % (2*np.pi)

        if t >= warmup:
            # Расстояние (на торе)
            diff = phi_p - phi
            # Периодическая метрика
            diff = diff - 2*np.pi * np.round(diff/(2*np.pi))
            dist = np.linalg.norm(diff)

            if dist > 0:
                lyap_sum += np.log(dist / delta0)
                # Ренормировка
                phi_p = phi + diff/dist * delta0
                n_steps += 1

    if n_steps == 0:
        return 0.0
    return lyap_sum / n_steps

def find_kam_threshold(omegas, kappa_range,
                       n_points=N_KAPPA):
    """
    Вычисляет λ_max(κ) и находит κ_c
    методом линейной интерполяции λ=0.
    """
    lambdas = []
    for kap in kappa_range:
        lya = lyapunov_max(omegas, kap)
        lambdas.append(lya)

    lambdas = np.array(lambdas)

    # Находим κ_c: первый κ где λ_max > threshold
    threshold = 0.001
    crossings = np.where(lambdas > threshold)[0]

    if len(crossings) == 0:
        kc = kappa_range[-1]   # не нашли → κ_c > max
    elif crossings[0] == 0:
        kc = kappa_range[0]    # хаос сразу
    else:
        # Линейная интерполяция
        i  = crossings[0] - 1
        k0, k1 = kappa_range[i], kappa_range[i+1]
        l0, l1 = lambdas[i],     lambdas[i+1]
        if l1 > l0:
            kc = k0 + (threshold - l0)/(l1-l0) * (k1-k0)
        else:
            kc = k0

    return lambdas, kc

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

# Диапазон κ
kappa_range = np.linspace(0.01, 2.0, N_KAPPA)

print("="*65)
print("Part XIII: KAM Thresholds κ_c(algebra)")
print("="*65)
print(f"T_lyapunov={T_LYA}, κ ∈ [{kappa_range[0]:.2f}, "
      f"{kappa_range[-1]:.2f}], N_points={N_KAPPA}")
print(f"N_random={N_CTRL}")
print()

# Коксетеровы алгебры
print("── Coxeter algebras ──")
cox_kc     = {}
cox_lambda = {}

for name in ALGEBRAS:
    ω = coxeter_freqs(name)
    lambdas, kc = find_kam_threshold(ω, kappa_range)
    cox_kc[name]     = kc
    cox_lambda[name] = lambdas

    # λ_max при стандартном κ=0.05
    lya_std = lambdas[np.argmin(np.abs(kappa_range - 0.05))]

    print(f"  {name:4s} (rank={len(ω):1d}, h={ALGEBRAS[name]['h']:2d}): "
          f"κ_c = {kc:.4f}, "
          f"λ(κ=0.05) = {lya_std:.4f}")

print()

# Случайные контроли — rank=8 (для E8)
print(f"── Random controls rank=8 (N={N_CTRL}) ──")
rand8_kc = []

for i in range(N_CTRL):
    ω = random_freqs(rank=8, seed=1000+i)
    _, kc = find_kam_threshold(ω, kappa_range)
    rand8_kc.append(kc)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_CTRL} done...")

rand8_kc = np.array(rand8_kc)
print(f"  κ_c: mean={rand8_kc.mean():.4f} "
      f"± {rand8_kc.std():.4f}, "
      f"range=[{rand8_kc.min():.4f}, {rand8_kc.max():.4f}]")
print()

# Случайные контроли — rank=6 (для E6, A6)
print(f"── Random controls rank=6 (N={N_CTRL}) ──")
rand6_kc = []

for i in range(N_CTRL):
    ω = random_freqs(rank=6, seed=2000+i)
    _, kc = find_kam_threshold(ω, kappa_range)
    rand6_kc.append(kc)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_CTRL} done...")

rand6_kc = np.array(rand6_kc)
print(f"  κ_c: mean={rand6_kc.mean():.4f} "
      f"± {rand6_kc.std():.4f}")
print()

# Случайные контроли — rank=7 (для E7)
print(f"── Random controls rank=7 (N={N_CTRL}) ──")
rand7_kc = []

for i in range(N_CTRL):
    ω = random_freqs(rank=7, seed=3000+i)
    _, kc = find_kam_threshold(ω, kappa_range)
    rand7_kc.append(kc)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_CTRL} done...")

rand7_kc = np.array(rand7_kc)
print(f"  κ_c: mean={rand7_kc.mean():.4f} "
      f"± {rand7_kc.std():.4f}")
print()

# ── СТАТИСТИКА ───────────────────────────────────────────────

print("── Statistical comparison (rank-matched) ──")
print()

rank_ctrl = {6: rand6_kc, 7: rand7_kc, 8: rand8_kc}

for name in ALGEBRAS:
    ω    = coxeter_freqs(name)
    rank = len(ω)
    kc   = cox_kc[name]
    ref  = rank_ctrl.get(rank, rand8_kc)

    pct  = float(np.mean(ref < kc)) * 100
    pval = float(np.mean(ref >= kc))

    print(f"  {name:4s} (rank={rank}): κ_c={kc:.4f}  "
          f"| ref={ref.mean():.4f}±{ref.std():.4f}  "
          f"| pct={pct:.0f}%  p={pval:.3f}  "
          f"{'*** SIGNAL' if pval<0.01 else '* marginal' if pval<0.05 else 'H0'}")

# Mann-Whitney E8 vs random
e8_kc_val = cox_kc['E8']
u_stat, mw_pval = mannwhitneyu(
    [e8_kc_val]*10,   # псевдо-выборка
    rand8_kc,
    alternative='greater'
)
pval_e8 = float(np.mean(rand8_kc >= e8_kc_val))
print()
print(f"E8 κ_c={e8_kc_val:.4f} vs random κ_c={rand8_kc.mean():.4f}")
print(f"p(E8 κ_c > random) = {pval_e8:.4f}")

# ── ЗАВИСИМОСТЬ κ_c от h ─────────────────────────────────────

print()
print("── κ_c vs Coxeter number h ──")
h_vals  = []
kc_vals = []
for name in ALGEBRAS:
    h  = ALGEBRAS[name]['h']
    kc = cox_kc[name]
    h_vals.append(h)
    kc_vals.append(kc)
    print(f"  {name}: h={h:2d}, κ_c={kc:.4f}")

corr_h_kc = np.corrcoef(h_vals, kc_vals)[0,1]
print(f"\n  r(h, κ_c) = {corr_h_kc:.3f}")
if abs(corr_h_kc) > 0.8:
    print("  СИЛЬНАЯ корреляция: κ_c ∝ h(Coxeter)!")
    print("  Физически: большее h → более устойчив к хаосу")
elif abs(corr_h_kc) > 0.5:
    print("  Умеренная корреляция")
else:
    print("  Слабая корреляция: κ_c не связан с h")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Part XIII: KAM Thresholds κ_c(algebra)\n'
             'Does chaos onset depend on Coxeter structure?',
             fontsize=13)

cox_colors = {'E8':'#e74c3c','E6':'#3498db',
              'E7':'#2ecc71','A6':'#9b59b6',
              'F4':'#f39c12','G2':'#1abc9c'}

# ── λ_max(κ) для всех алгебр
ax1 = fig.add_subplot(2, 3, 1)
for name in ALGEBRAS:
    ax1.plot(kappa_range, cox_lambda[name],
             color=cox_colors[name], lw=2,
             label=f'{name} (κ_c={cox_kc[name]:.3f})')
ax1.axhline(0, color='k', lw=1, ls='--')
ax1.axhline(0.001, color='gray', lw=1, ls=':',
            label='threshold')
ax1.set_xlabel('κ')
ax1.set_ylabel('λ_max (Lyapunov)')
ax1.set_title('Lyapunov exponent vs κ')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 0.5)

# ── κ_c distribution: rank=8
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(rand8_kc, bins=25, color='gray',
         alpha=0.6, label=f'Random rank=8\n'
         f'({rand8_kc.mean():.3f}±{rand8_kc.std():.3f})')
for name in ['E8', 'E7']:
    ax2.axvline(cox_kc[name], color=cox_colors[name],
               lw=2.5, label=f'{name} κ_c={cox_kc[name]:.3f}')
ax2.set_xlabel('κ_c (KAM threshold)')
ax2.set_ylabel('Count')
ax2.set_title('κ_c distribution: rank=8')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── κ_c distribution: rank=6
ax3 = fig.add_subplot(2, 3, 3)
ax3.hist(rand6_kc, bins=25, color='gray',
         alpha=0.6, label=f'Random rank=6\n'
         f'({rand6_kc.mean():.3f}±{rand6_kc.std():.3f})')
for name in ['E6', 'A6', 'F4', 'G2']:
    if name in cox_kc:
        ax3.axvline(cox_kc[name], color=cox_colors[name],
                   lw=2.5, label=f'{name} κ_c={cox_kc[name]:.3f}')
ax3.set_xlabel('κ_c')
ax3.set_ylabel('Count')
ax3.set_title('κ_c distribution: rank=6')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── κ_c vs h (Coxeter number)
ax4 = fig.add_subplot(2, 3, 4)
h_arr  = np.array(h_vals)
kc_arr = np.array(kc_vals)
names_arr = list(ALGEBRAS.keys())

for i, name in enumerate(ALGEBRAS):
    ax4.scatter(h_arr[i], kc_arr[i],
               color=cox_colors[name], s=150,
               zorder=5, label=name,
               edgecolors='k', linewidths=1)
    ax4.annotate(name, (h_arr[i], kc_arr[i]),
                textcoords="offset points",
                xytext=(5,5), fontsize=8)

# Линия тренда
if abs(corr_h_kc) > 0.3:
    z = np.polyfit(h_arr, kc_arr, 1)
    p = np.poly1d(z)
    h_line = np.linspace(h_arr.min(), h_arr.max(), 100)
    ax4.plot(h_line, p(h_line), 'k--',
             alpha=0.5, label=f'trend r={corr_h_kc:.2f}')

ax4.set_xlabel('Coxeter number h')
ax4.set_ylabel('κ_c (KAM threshold)')
ax4.set_title(f'κ_c vs h  |  r={corr_h_kc:.3f}')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# ── λ_max(κ) с random envelope
ax5 = fig.add_subplot(2, 3, 5)

# Random envelope (несколько кривых)
for i in range(0, min(20, N_CTRL)):
    ω = random_freqs(rank=8, seed=1000+i)
    lams = []
    for kap in kappa_range:
        lams.append(lyapunov_max(ω, kap,
                                  T=T_LYA//2))
    ax5.plot(kappa_range, lams,
             color='lightgray', lw=0.5, alpha=0.5)

ax5.plot(kappa_range, cox_lambda['E8'],
         color='#e74c3c', lw=2.5, label='E8', zorder=5)
ax5.axhline(0, color='k', lw=1, ls='--')
ax5.axvline(cox_kc['E8'], color='#e74c3c',
            lw=1.5, ls=':', label=f'E8 κ_c={cox_kc["E8"]:.3f}')
ax5.set_xlabel('κ')
ax5.set_ylabel('λ_max')
ax5.set_title('E8 vs random trajectories (λ_max)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(-0.2, 0.8)

# ── Сводная таблица
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

lines = ['PART XIII RESULTS', '─'*30, '']
for name in ALGEBRAS:
    ω    = coxeter_freqs(name)
    rank = len(ω)
    kc   = cox_kc[name]
    ref  = rank_ctrl.get(rank, rand8_kc)
    pct  = float(np.mean(ref < kc))*100
    pv   = float(np.mean(ref >= kc))
    sig  = '***' if pv<0.01 else '*' if pv<0.05 else ''
    lines.append(f'{name}(r={rank},h={ALGEBRAS[name]["h"]:2d}): '
                 f'κ_c={kc:.3f} '
                 f'p={pv:.3f} {sig}')

lines += [
    '',
    f'r(h, κ_c) = {corr_h_kc:.3f}',
    '',
    'Random refs:',
    f'  rank=8: {rand8_kc.mean():.3f}±{rand8_kc.std():.3f}',
    f'  rank=6: {rand6_kc.mean():.3f}±{rand6_kc.std():.3f}',
    f'  rank=7: {rand7_kc.mean():.3f}±{rand7_kc.std():.3f}',
]

ax6.text(0.02, 0.98, '\n'.join(lines),
         transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round',
                   facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part13_kam_thresholds.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part13_kam_thresholds.png")

# ── ФИНАЛ ────────────────────────────────────────────────────

print()
print("="*65)
print("PART XIII VERDICT")
print("="*65)
print()
print(f"r(h, κ_c) = {corr_h_kc:.3f}")
print()

if abs(corr_h_kc) > 0.85:
    print("СИЛЬНЫЙ РЕЗУЛЬТАТ:")
    print("  κ_c систематически зависит от h(Coxeter)")
    print("  Большее h → более устойчив к хаосу")
    print()
    print("Физическая интерпретация:")
    print("  E8 (h=30) наиболее устойчива")
    print("  → ранняя Вселенная в E8-фазе")
    print("  → переход в хаос при κ > κ_c(E8)")
    print("  → Большой Взрыв = КАМ-переход?")
elif abs(corr_h_kc) > 0.5:
    print("УМЕРЕННЫЙ РЕЗУЛЬТАТ:")
    print("  Частичная корреляция κ_c с h")
    print("  Нужно больше алгебр для подтверждения")
else:
    print("ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ:")
    print("  κ_c не коррелирует с h(Coxeter)")
    print("  → КАМ-порог определяется rank/spread,")
    print("    не алгеброй")
    print("  → Переходим к направлению №3")
    print("    (Комплексные частоты)")
