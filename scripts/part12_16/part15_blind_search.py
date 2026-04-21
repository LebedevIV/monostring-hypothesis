"""
Part XV: Blind Search — V(φ) потенциалы
Самый быстрый оставшийся тест Пути А.

Вопрос: существует ли комбинация весов w при которой
потенциал на базе Коксетеровых показателей даёт
n_s ∈ [0.960, 0.969] И r < 0.036?

Это не подгонка — это поиск существования.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEED     = 42
N_TRIALS = 50000   # перебор потенциалов
np.random.seed(SEED)

PLANCK_NS    = 0.9649
PLANCK_NS_1S = 0.0042
PLANCK_R_MAX = 0.036

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7},
}

def make_potential(m_arr, h, weights, phi_arr):
    """
    V(φ) = Σᵢ wᵢ · φ^(2mᵢ/h)
    Коксетеровы показатели как степени потенциала.
    """
    V = np.zeros_like(phi_arr)
    for i, mi in enumerate(m_arr):
        exp = 2.0 * mi / h
        V += weights[i] * np.power(np.abs(phi_arr), exp)
    return V

def slow_roll_observables(weights, m_arr, h,
                           phi_range=(0.1, 15.0),
                           n_points=500):
    """
    Вычисляет n_s и r для потенциала V(φ).

    Использует стандартные формулы slow-roll:
      ε = (1/2)(V'/V)²   [M_pl = 1]
      η = V''/V
      n_s = 1 - 6ε + 2η
      r = 16ε

    Возвращает (n_s, r) в точке φ* где ε минимально.
    """
    phi = np.linspace(phi_range[0], phi_range[1], n_points)
    V   = make_potential(m_arr, h, weights, phi)

    if np.any(V <= 0):
        V = V - V.min() + 1e-10

    # Численные производные
    dphi = phi[1] - phi[0]
    dV   = np.gradient(V, dphi)
    d2V  = np.gradient(dV, dphi)

    V_safe = np.where(V > 1e-10, V, 1e-10)

    eps = 0.5 * (dV / V_safe)**2
    eta = d2V / V_safe

    # Точка медленного скатывания: минимум ε
    # (исключаем края)
    interior = slice(10, -10)
    eps_int  = eps[interior]
    eta_int  = eta[interior]

    if len(eps_int) == 0 or eps_int.min() > 1.0:
        return np.nan, np.nan

    idx_min = np.argmin(eps_int)
    eps_sr  = eps_int[idx_min]
    eta_sr  = eta_int[idx_min]

    ns = 1.0 - 6.0*eps_sr + 2.0*eta_sr
    r  = 16.0 * eps_sr

    return float(ns), float(r)

# ── ОСНОВНОЙ ПЕРЕБОР ─────────────────────────────────────────

print("="*65)
print("Part XV: Blind Search — Coxeter Inflation Potentials")
print("="*65)
print(f"N_trials={N_TRIALS} per algebra + random control")
print(f"Target: n_s ∈ [{PLANCK_NS-PLANCK_NS_1S:.4f}, "
      f"{PLANCK_NS+PLANCK_NS_1S:.4f}], r < {PLANCK_R_MAX}")
print()

results_all  = {}
winners_all  = {}

for alg_name, alg in ALGEBRAS.items():
    m_arr = np.array(alg['m'], dtype=float)
    h     = alg['h']
    r_alg = len(m_arr)

    ns_list = []
    r_list  = []
    winners = []

    for trial in range(N_TRIALS):
        # Случайные веса (Dirichlet → сумма=1)
        w = np.random.dirichlet(np.ones(r_alg))

        ns, r = slow_roll_observables(w, m_arr, h)

        if np.isnan(ns):
            continue

        ns_list.append(ns)
        r_list.append(r)

        # Критерий победителя
        in_planck = (abs(ns - PLANCK_NS) < PLANCK_NS_1S
                     and r < PLANCK_R_MAX)
        if in_planck:
            winners.append({'w': w, 'ns': ns, 'r': r})

    ns_arr = np.array(ns_list)
    r_arr  = np.array(r_list)

    win_rate = len(winners) / max(len(ns_list), 1)

    results_all[alg_name] = {
        'ns': ns_arr, 'r': r_arr,
        'winners': winners,
        'win_rate': win_rate
    }
    winners_all[alg_name] = winners

    print(f"{alg_name} (rank={r_alg}, h={h}):")
    print(f"  n_s: mean={ns_arr.mean():.4f} ± {ns_arr.std():.4f}, "
          f"range=[{ns_arr.min():.3f}, {ns_arr.max():.3f}]")
    print(f"  r:   mean={r_arr.mean():.4f} ± {r_arr.std():.4f}")
    print(f"  Winners (Planck 1σ): {len(winners)} / {len(ns_list)} "
          f"= {win_rate*100:.2f}%")
    if winners:
        best = min(winners, key=lambda x: abs(x['ns']-PLANCK_NS))
        print(f"  Best: n_s={best['ns']:.4f}, r={best['r']:.4f}")
    print()

# ── СЛУЧАЙНЫЙ КОНТРОЛЬ ───────────────────────────────────────

print("── Random control (same structure, random exponents) ──")
print()

# Используем те же rank что у E8, но случайные показатели
rank_ctrl = 8
ns_rand   = []
r_rand    = []
winners_rand = []

for trial in range(N_TRIALS):
    # Случайные показатели (не из Коксетера)
    m_rand = np.random.uniform(0.5, 15.0, rank_ctrl)
    h_rand = 30.0  # тот же h что у E8
    w_rand = np.random.dirichlet(np.ones(rank_ctrl))

    ns, r = slow_roll_observables(w_rand, m_rand, h_rand)
    if np.isnan(ns):
        continue
    ns_rand.append(ns)
    r_rand.append(r)
    if (abs(ns - PLANCK_NS) < PLANCK_NS_1S
            and r < PLANCK_R_MAX):
        winners_rand.append({'ns': ns, 'r': r})

ns_rand = np.array(ns_rand)
r_rand  = np.array(r_rand)
win_rate_rand = len(winners_rand) / max(len(ns_rand), 1)

print(f"Random (rank=8, random exponents):")
print(f"  n_s: mean={ns_rand.mean():.4f} ± {ns_rand.std():.4f}")
print(f"  Winners: {len(winners_rand)}/{len(ns_rand)} "
      f"= {win_rate_rand*100:.2f}%")
print()

# ── СТАТИСТИКА ───────────────────────────────────────────────

print("── Comparison: Coxeter vs Random ──")
print()
print(f"{'Algebra':>8} {'win_rate':>10} {'vs Random':>10} "
      f"{'ratio':>8} {'Signal?':>10}")
print("-"*55)

for name in ALGEBRAS:
    wr = results_all[name]['win_rate']
    ratio = wr / max(win_rate_rand, 1e-6)
    delta = wr - win_rate_rand
    sig = ('*** YES' if ratio > 3 and wr > 0.01
           else '* weak' if ratio > 1.5
           else 'no')
    print(f"{name:>8} {wr*100:>9.2f}% {delta*100:>+9.2f}pp "
          f"{ratio:>8.2f}x {sig:>10}")

print(f"{'Random':>8} {win_rate_rand*100:>9.2f}%  "
      f"{'(baseline)':>20}")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Part XV: Blind Search — Coxeter Inflation Potentials\n'
             f'N={N_TRIALS} trials per algebra', fontsize=12)

cox_colors = {'E8':'#e74c3c','E6':'#3498db',
              'E7':'#2ecc71','A6':'#9b59b6'}

# 0,0: n_s distributions
ax = axes[0,0]
for name in ALGEBRAS:
    ns = results_all[name]['ns']
    ns_clean = ns[np.isfinite(ns)]
    ax.hist(ns_clean, bins=60, alpha=0.4,
            color=cox_colors[name],
            label=name, density=True)
ax.hist(ns_rand[np.isfinite(ns_rand)], bins=60,
        alpha=0.3, color='gray',
        label='Random', density=True)
ax.axvline(PLANCK_NS, color='k', lw=2, ls='--')
ax.axvspan(PLANCK_NS-PLANCK_NS_1S,
           PLANCK_NS+PLANCK_NS_1S,
           alpha=0.2, color='blue', label='Planck 1σ')
ax.set_xlabel('n_s')
ax.set_ylabel('Density')
ax.set_title('n_s distribution')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 3)

# 0,1: n_s vs r scatter (winners)
ax = axes[0,1]
for name in ALGEBRAS:
    w_list = results_all[name]['winners']
    if w_list:
        ns_w = [x['ns'] for x in w_list]
        r_w  = [x['r']  for x in w_list]
        ax.scatter(ns_w, r_w, color=cox_colors[name],
                   alpha=0.6, s=20, label=f'{name} ({len(w_list)})')
if winners_rand:
    ns_wr = [x['ns'] for x in winners_rand]
    r_wr  = [x['r']  for x in winners_rand]
    ax.scatter(ns_wr, r_wr, color='gray',
               alpha=0.3, s=10, label=f'Random ({len(winners_rand)})')

ax.axvline(PLANCK_NS, color='k', lw=1.5, ls='--')
ax.axvspan(PLANCK_NS-PLANCK_NS_1S,
           PLANCK_NS+PLANCK_NS_1S,
           alpha=0.15, color='blue')
ax.axhline(PLANCK_R_MAX, color='red',
           lw=1.5, ls=':', label='r<0.036')
ax.set_xlabel('n_s')
ax.set_ylabel('r')
ax.set_title('Winners: n_s vs r')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 0,2: Win rates bar chart
ax = axes[0,2]
names_bar  = list(ALGEBRAS.keys()) + ['Random']
rates_bar  = [results_all[n]['win_rate']*100
              for n in ALGEBRAS] + [win_rate_rand*100]
colors_bar = [cox_colors[n] for n in ALGEBRAS] + ['gray']
bars = ax.bar(names_bar, rates_bar,
              color=colors_bar, alpha=0.8)
ax.set_ylabel('Win rate (%)')
ax.set_title('Planck-compatible fraction')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, rates_bar):
    ax.text(bar.get_x()+bar.get_width()/2,
            val+0.01, f'{val:.2f}%',
            ha='center', fontsize=8)

# 1,0-1,1: V(φ) для лучшего победителя каждой алгебры
for idx, name in enumerate(list(ALGEBRAS.keys())[:4]):
    ax = axes[1, idx % 3] if idx < 3 else axes[1, 2]
    alg   = ALGEBRAS[name]
    m_arr = np.array(alg['m'], dtype=float)
    h     = alg['h']
    w_list = results_all[name]['winners']

    phi = np.linspace(0.1, 12, 300)

    if w_list:
        best = min(w_list,
                   key=lambda x: abs(x['ns']-PLANCK_NS))
        V = make_potential(m_arr, h, best['w'], phi)
        V = V / V.max()
        ax.plot(phi, V, color=cox_colors[name],
                lw=2.5,
                label=f"n_s={best['ns']:.4f}, r={best['r']:.4f}")
        ax.set_title(f'{name}: Best winner potential')
    else:
        ax.text(0.5, 0.5, 'No winners found',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{name}: No winners')

    ax.set_xlabel('φ')
    ax.set_ylabel('V(φ) / V_max')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 1,2: Summary
ax = axes[1,2]
ax.axis('off')
lines = [
    'PART XV SUMMARY',
    '─'*28,
    '',
    f'N = {N_TRIALS} trials/algebra',
    f'Planck target:',
    f'  n_s = {PLANCK_NS} ± {PLANCK_NS_1S}',
    f'  r < {PLANCK_R_MAX}',
    '',
    'Win rates:',
]
for name in ALGEBRAS:
    wr = results_all[name]['win_rate']
    lines.append(f'  {name}: {wr*100:.2f}%')
lines.append(f'  Random: {win_rate_rand*100:.2f}%')
lines += [
    '',
    'VERDICT:',
]
best_alg = max(ALGEBRAS,
               key=lambda n: results_all[n]['win_rate'])
best_wr  = results_all[best_alg]['win_rate']
ratio    = best_wr / max(win_rate_rand, 1e-6)
if ratio > 3:
    lines += [f'  {best_alg} wins {ratio:.1f}x more',
              '  than random → SIGNAL']
else:
    lines += ['  No algebra wins significantly',
              '  more than random.',
              '  → H0 holds.',
              '  → Change model.']

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part15_blind_search.png',
            dpi=120, bbox_inches='tight')
print("✓ Saved: part15_blind_search.png")

# ── ФИНАЛ ────────────────────────────────────────────────────

print()
print("="*65)
print("PART XV FINAL VERDICT")
print("="*65)
print()

best_alg = max(ALGEBRAS,
               key=lambda n: results_all[n]['win_rate'])
best_wr  = results_all[best_alg]['win_rate']
ratio    = best_wr / max(win_rate_rand, 1e-6)

if ratio > 3.0 and best_wr > 0.005:
    print(f"СИГНАЛ: {best_alg} даёт Planck-совместимые")
    print(f"потенциалы в {ratio:.1f}x чаще чем random")
    print()
    print("→ Публикуем как Part XV")
    print("→ Исследуем структуру потенциала-победителя")
else:
    print("H₀ не отвергается.")
    print()
    print(f"Лучшая алгебра ({best_alg}):")
    print(f"  win_rate = {best_wr*100:.2f}%")
    print(f"  ratio vs random = {ratio:.2f}x")
    print()
    print("ИТОГ Parts XII-XV:")
    print("  Стандартное отображение + Коксетер")
    print("  не содержит физики инфляции.")
    print()
    print("СЛЕДУЮЩИЙ ШАГ — СМЕНА МОДЕЛИ:")
    print()
    print("  Вариант 1: Корневая система E8")
    print("    240 осцилляторов, связь через ⟨α,β⟩")
    print()
    print("  Вариант 2: Квантовая модель")
    print("    Гамильтониан на решётке корней E8")
    print("    Квантовые флуктуации → инфляция")
    print()
    print("  Вариант 3: Случайные матрицы")
    print("    GUE/GOE спектры vs Коксетер")
    print("    Хорошо изучены в ядерной физике")
