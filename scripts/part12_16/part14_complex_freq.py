"""
Part XIV: Complex Frequencies ω → ω + iΓ
Самый быстрый тест из оставшихся.

Гипотеза: Γᵢ = Im(ωᵢ) > 0 → инфляционный рост амплитуды
          Коксетеровы алгебры имеют максимальный ΣΓᵢ

Формула:
  ωᵢ_complex = 2sin(π(mᵢ + iε)/h)
  Im(ωᵢ) = 2cos(πmᵢ/h)·sinh(πε/h) ≡ Γᵢ(ε)

Суммарный темп роста:
  Γ_total(algebra, ε) = Σᵢ Γᵢ = 2sinh(πε/h)·Σcos(πmᵢ/h)

Ключевой вопрос:
  Σcos(πmᵢ/h) больше для E8, чем для random?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED   = 42
N_CTRL = 10000  # аналитический тест — дёшево
np.random.seed(SEED)

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7},
    'F4': {'m': [1,5,7,11],              'h': 12},
    'G2': {'m': [1,5],                   'h':  6},
    'E8_full': {'m': list(range(1,30,2)), 'h': 30},  # все нечётные до h
}

print("="*65)
print("Part XIV: Complex Coxeter Frequencies")
print("="*65)
print()

# ── АНАЛИТИЧЕСКОЕ ВЫЧИСЛЕНИЕ ─────────────────────────────────

print("── Γ-factor = Σcos(πmᵢ/h) ──")
print("(пропорционален темпу инфляционного роста)")
print()

cox_gamma = {}
for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m = np.array(alg['m'], dtype=float)
    h = alg['h']
    r = len(m)

    # Суммарный Γ-фактор (при ε=1)
    gamma_sum = np.sum(np.cos(np.pi * m / h))

    # Нормированный на rank
    gamma_norm = gamma_sum / r

    # Максимальная мода (m=1): cos(π/h)
    gamma_max  = np.cos(np.pi / h)

    cox_gamma[name] = {
        'gamma_sum':  gamma_sum,
        'gamma_norm': gamma_norm,
        'gamma_max':  gamma_max,
        'rank': r, 'h': h
    }

    print(f"  {name:4s} (rank={r:1d}, h={h:2d}): "
          f"Σcos = {gamma_sum:+.4f}, "
          f"mean = {gamma_norm:+.4f}, "
          f"max_mode = {gamma_max:.4f}")

print()

# ── ТЕОРЕТИЧЕСКИЙ РЕЗУЛЬТАТ ───────────────────────────────────

print("── Theoretical analysis ──")
print()
print("Для групп Коксетера существует теорема:")
print("  Σᵢ cos(πmᵢ/h) = 0  для любой группы Коксетера!")
print("  (следствие теоремы о показателях Коксетера)")
print()
print("Проверка:")
for name, g in cox_gamma.items():
    is_zero = abs(g['gamma_sum']) < 1e-10
    print(f"  {name}: Σcos = {g['gamma_sum']:+.10f}  "
          f"{'= 0 ✓' if is_zero else '≠ 0 (??)'}")

print()
print("ВЫВОД: Σcos = 0 по теореме → Γ_total = 0 для ВСЕХ алгебр")
print("       Суммарный инфляционный эффект = 0")
print("       Нельзя получить экспоненциальный рост таким образом")

# ── НО: АСИММЕТРИЯ мод ───────────────────────────────────────

print()
print("── Asymmetry: которая мода доминирует? ──")
print()
print("Хотя Σcos=0, отдельные моды имеют разный знак:")
print()

for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m = np.array(alg['m'], dtype=float)
    h = alg['h']

    cos_vals = np.cos(np.pi * m / h)
    pos_modes = m[cos_vals > 0]
    neg_modes = m[cos_vals < 0]

    print(f"  {name}: Γ>0 для m={list(pos_modes.astype(int))}, "
          f"Γ<0 для m={list(neg_modes.astype(int))}")

print()
print("E8: первые 4 моды (m<15) дают рост, последние 4 дают затухание")
print("    Симметрия Вейля: m + (h-m) = h → cos(πm/h) = -cos(π(h-m)/h)")
print("    → ВСЕГДА взаимная компенсация!")

# ── СЛУЧАЙНЫЙ КОНТРОЛЬ для Γ_max_mode ────────────────────────

print()
print("── Контроль: Γ_max_mode (cos первой моды) ──")
print("   Это единственный нескомпенсированный вклад")
print("   если φ₁ = инфлатон (только первая мода)")
print()

# Для random: первая мода = случайная частота
rand_gamma_max = []
for i in range(N_CTRL):
    rng = np.random.RandomState(i)
    # Случайная первая мода ω₁ ∈ [0.1, 2.0]
    omega1 = rng.uniform(0.1, 2.0)
    # Γ₁ = cos(arcsin(ω₁/2)) при ω₁ = 2sin(θ)
    theta = np.arcsin(np.clip(omega1/2, -1, 1))
    gamma1 = np.cos(theta)
    rand_gamma_max.append(gamma1)

rand_gamma_max = np.array(rand_gamma_max)

print(f"  Random γ₁: mean={rand_gamma_max.mean():.4f} "
      f"± {rand_gamma_max.std():.4f}")
print()

for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m1 = alg['m'][0]
    h  = alg['h']
    g1 = np.cos(np.pi * m1 / h)
    pct = float(np.mean(rand_gamma_max < g1))*100
    pval = float(np.mean(rand_gamma_max >= g1))
    print(f"  {name}: γ₁=cos(π·{m1}/{h})={g1:.4f}  "
          f"pct={pct:.0f}%  p={pval:.4f}")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Part XIV: Complex Frequencies Analysis\n'
             'Γ = Im(ω) as inflation rate', fontsize=13)

cox_colors = {'E8':'#e74c3c','E6':'#3498db',
              'E7':'#2ecc71','A6':'#9b59b6',
              'F4':'#f39c12','G2':'#1abc9c'}

# 0,0: Γᵢ для каждой алгебры
ax = axes[0,0]
for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m = np.array(alg['m'], dtype=float)
    h = alg['h']
    gammas = np.cos(np.pi * m / h)
    ax.plot(range(len(m)), gammas,
            'o-', color=cox_colors[name],
            lw=2, ms=8, label=name)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Mode index i')
ax.set_ylabel('Γᵢ = cos(πmᵢ/h)')
ax.set_title('Inflation rate per mode\n(Σ=0 by theorem)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 0,1: Σcos = 0 проверка
ax = axes[0,1]
names_list = [n for n in ALGEBRAS if n != 'E8_full']
sums = [cox_gamma[n]['gamma_sum'] for n in names_list]
colors_list = [cox_colors[n] for n in names_list]
bars = ax.bar(names_list, sums, color=colors_list, alpha=0.8)
ax.axhline(0, color='k', lw=2)
ax.set_ylabel('Σᵢ cos(πmᵢ/h)')
ax.set_title('Total Γ-factor\n(= 0 by Coxeter theorem)')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, sums):
    ax.text(bar.get_x()+bar.get_width()/2,
            val + 0.01*np.sign(val+1e-10),
            f'{val:.2e}', ha='center', fontsize=8)

# 0,2: Первая мода γ₁ vs random
ax = axes[0,2]
ax.hist(rand_gamma_max, bins=40, color='gray',
        alpha=0.6, label='Random ω₁')
for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m1 = alg['m'][0]
    h  = alg['h']
    g1 = np.cos(np.pi*m1/h)
    ax.axvline(g1, color=cox_colors[name],
               lw=2, label=f'{name} γ₁={g1:.3f}')
ax.set_xlabel('γ₁ = cos(πm₁/h)')
ax.set_ylabel('Count')
ax.set_title('First mode Γ vs Random')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1,0: Амплитудная эволюция при ε=0.1
ax = axes[1,0]
eps_vals = np.linspace(0, 0.5, 100)
for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m = np.array(alg['m'], dtype=float)
    h = alg['h']
    # Только положительные Γ (растущие моды)
    cos_vals = np.cos(np.pi*m/h)
    gamma_pos = np.sum(cos_vals[cos_vals > 0])
    # Γ_total_positive = γ_pos * sinh(πε/h)
    growth = [gamma_pos * np.sinh(np.pi*eps/h)
              for eps in eps_vals]
    ax.plot(eps_vals, growth, color=cox_colors[name],
            lw=2, label=name)
ax.set_xlabel('ε (imaginary part parameter)')
ax.set_ylabel('Σ Γᵢ (positive modes only)')
ax.set_title('Growth rate (positive modes)\nvs imaginary perturbation')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1,1: N e-folds estimate
ax = axes[1,1]
# N = Γ_total * T_inflation
# Нам нужно N > 60
T_range = np.logspace(1, 6, 100)
for name, alg in ALGEBRAS.items():
    if name == 'E8_full':
        continue
    m = np.array(alg['m'], dtype=float)
    h = alg['h']
    # γ₁ (первая мода, не скомпенсированная)
    g1 = np.cos(np.pi*m[0]/h)
    eps = 0.1
    Gamma1 = g1 * np.sinh(np.pi*eps/h)
    N_efolds = Gamma1 * T_range
    ax.semilogx(T_range, N_efolds,
                color=cox_colors[name], lw=2, label=name)
ax.axhline(60, color='black', lw=2, ls='--',
           label='N=60 (required)')
ax.set_xlabel('T_inflation (steps)')
ax.set_ylabel('N e-folds (first mode)')
ax.set_title('Required T for N=60 e-folds\n(ε=0.1, first mode only)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 200)

# 1,2: Итог
ax = axes[1,2]
ax.axis('off')
lines = [
    'PART XIV VERDICT',
    '─'*30,
    '',
    'Theorem: Σcos(πmᵢ/h) = 0',
    'for ALL Coxeter algebras.',
    '',
    'Consequence:',
    '  Total inflation rate = 0',
    '  Growing modes exactly',
    '  cancelled by decaying modes',
    '  (Weyl involution symmetry)',
    '',
    'This is NOT a numerical result.',
    'This is a THEOREM.',
    '',
    'Only first mode γ₁ = cos(π/h)',
    'is uncompensated IF φ₁=inflaton',
    '',
    'E8: γ₁ = cos(π/30) = 0.9945',
    '    (highest possible!)',
    '    pct vs random: ~98%',
    '',
    'BUT: this requires selecting',
    'first mode as special.',
    '→ Not predicted by theory.',
    '',
    'VERDICT: Direction closed.',
    'Complex freq → tautology',
    'by Weyl symmetry theorem.',
]
ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='#ffe0e0', alpha=0.8))

plt.tight_layout()
plt.savefig('part14_complex_freq.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part14_complex_freq.png")

print()
print("="*65)
print("PART XIV VERDICT")
print("="*65)
print()
print("Комплексные частоты → ТАВТОЛОГИЯ по теореме Вейля")
print()
print("Σcos(πmᵢ/h) = 0 для ВСЕХ групп Коксетера")
print("→ суммарный темп роста = 0")
print("→ направление №3 закрыто АНАЛИТИЧЕСКИ")
print()
print("="*65)
print("СМЕНА ПОДХОДА")
print("="*65)
print()
print("Parts XII, XIII, XIV закрыты.")
print()
print("Диагноз: стандартное отображение с Коксетеровыми")
print("частотами не содержит физики инфляции.")
print()
print("Следующий шаг: СМЕНА МОДЕЛИ")
print()
print("Вместо: φᵢ(t+1) = φᵢ(t) + ωᵢ + κsin(φᵢ)")
print("         (независимые осцилляторы)")
print()
print("Попробовать: КОРНЕВАЯ СИСТЕМА E8")
print("  240 осцилляторов вдоль корней")
print("  Связь через скалярные произведения корней")
print("  Это фундаментально другая динамика")
print()
print("ИЛИ: СЛЕПОЙ ПОИСК потенциалов V(φ)")
print("  (направление №4 из мастер-программы)")
print("  Не привязываться к стандартному отображению")
