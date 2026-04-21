"""
Part XV Step 2: Modified potential forms
Три варианта преодоления смещения n_s:

A) Starobinsky-like: V = V₀(1 - e^{-√(2/3)φ})²
   с Коксетеровыми поправками

B) Hilltop: V = V₀(1 - (φ/φ₀)^p)
   p из Коксетеровых показателей

C) Осцилляторный: V = V₀(1 + Σᵢ αᵢcos(mᵢφ/φ₀))
   Коксетеровы числа как частоты осцилляций
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED  = 42
N_TRIALS = 30000
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

def slow_roll_ns_r(V_arr, phi_arr):
    """n_s и r из массива V(φ)"""
    dphi  = phi_arr[1] - phi_arr[0]
    dV    = np.gradient(V_arr, dphi)
    d2V   = np.gradient(dV, dphi)
    V_s   = np.where(V_arr > 1e-10, V_arr, 1e-10)
    eps   = 0.5*(dV/V_s)**2
    eta   = d2V/V_s

    # Точка медленного скатывания
    interior = slice(5, -5)
    eps_i    = eps[interior]
    eta_i    = eta[interior]

    if len(eps_i)==0 or eps_i.min() > 1.0:
        return np.nan, np.nan

    idx = np.argmin(eps_i)
    return float(1 - 6*eps_i[idx] + 2*eta_i[idx]), \
           float(16*eps_i[idx])

phi = np.linspace(0.01, 20.0, 1000)

print("="*65)
print("Part XV Step 2: Modified Potential Forms")
print("="*65)
print()

all_winners = {}

# ════════════════════════════════════════════════════════════
# ФОРМА A: Starobinsky + Коксетеровы поправки
# V = (1 - e^{-√(2/3)φ})² · (1 + Σᵢ αᵢ · e^{-mᵢφ/h})
# ════════════════════════════════════════════════════════════

print("── Form A: Starobinsky × Coxeter corrections ──")
print()

# Базовый Старобинский (предсказывает n_s=0.965 точно!)
V_star_base = (1 - np.exp(-np.sqrt(2/3)*phi))**2
ns_star, r_star = slow_roll_ns_r(V_star_base, phi)
print(f"  Pure Starobinsky: n_s={ns_star:.4f}, r={r_star:.4f}")
print(f"  (Planck: n_s={PLANCK_NS:.4f})")
print()

winners_A = {}
for name, alg in ALGEBRAS.items():
    m_arr = np.array(alg['m'], dtype=float)
    h     = alg['h']
    r_alg = len(m_arr)
    wins  = []

    for trial in range(N_TRIALS):
        # Случайные коэффициенты поправок
        alpha = np.random.normal(0, 0.1, r_alg)

        # Коксетеровы поправки
        correction = 1.0 + sum(
            alpha[i] * np.exp(-m_arr[i]*phi/h)
            for i in range(r_alg)
        )

        V = V_star_base * np.abs(correction)
        V = np.where(V > 0, V, 1e-10)

        ns, r = slow_roll_ns_r(V, phi)
        if np.isnan(ns):
            continue

        if (abs(ns - PLANCK_NS) < PLANCK_NS_1S
                and r < PLANCK_R_MAX):
            wins.append({'ns': ns, 'r': r, 'alpha': alpha})

    winners_A[name] = wins
    print(f"  {name}: {len(wins)}/{N_TRIALS} = "
          f"{len(wins)/N_TRIALS*100:.2f}% winners")

print()

# ════════════════════════════════════════════════════════════
# ФОРМА B: Hilltop с Коксетеровыми степенями
# V = V₀(1 - (φ/μ)^p + ...)
# p = 2mᵢ/h (Коксетеровы показатели)
# ════════════════════════════════════════════════════════════

print("── Form B: Hilltop with Coxeter exponents ──")
print()

winners_B = {}
for name, alg in ALGEBRAS.items():
    m_arr = np.array(alg['m'], dtype=float)
    h     = alg['h']
    r_alg = len(m_arr)
    wins  = []

    exps = 2*m_arr/h  # Коксетеровы показатели

    for trial in range(N_TRIALS):
        w   = np.random.dirichlet(np.ones(r_alg))
        mu  = np.random.uniform(1.0, 15.0)

        # Hilltop: V = 1 - Σwᵢ(φ/μ)^pᵢ
        ratio = phi[:, None] / mu  # (N_phi, 1)
        V = 1.0 - np.sum(
            w * ratio**exps, axis=1
        )

        # Только там где V > 0
        mask = V > 0.01
        if mask.sum() < 50:
            continue

        phi_m = phi[mask]
        V_m   = V[mask]
        ns, r = slow_roll_ns_r(V_m, phi_m)

        if np.isnan(ns):
            continue
        if (abs(ns - PLANCK_NS) < PLANCK_NS_1S
                and r < PLANCK_R_MAX):
            wins.append({'ns': ns, 'r': r,
                         'w': w, 'mu': mu})

    winners_B[name] = wins
    print(f"  {name}: {len(wins)}/{N_TRIALS} = "
          f"{len(wins)/N_TRIALS*100:.2f}% winners")

print()

# ════════════════════════════════════════════════════════════
# ФОРМА C: Осцилляторный потенциал (аксионная инфляция)
# V = V₀(1 + Σᵢ αᵢ·cos(mᵢ·φ/f))
# Коксетеровы mᵢ как моды осцилляций
# ════════════════════════════════════════════════════════════

print("── Form C: Axion-like oscillatory potential ──")
print()

winners_C = {}
for name, alg in ALGEBRAS.items():
    m_arr = np.array(alg['m'], dtype=float)
    h     = alg['h']
    r_alg = len(m_arr)
    wins  = []

    for trial in range(N_TRIALS):
        alpha = np.random.uniform(-0.2, 0.2, r_alg)
        f     = np.random.uniform(1.0, 10.0)

        # Модуляция на базе Старобинского
        oscillation = 1.0 + sum(
            alpha[i] * np.cos(m_arr[i]*phi/f)
            for i in range(r_alg)
        )
        V = V_star_base * oscillation
        V = np.where(V > 1e-10, V, 1e-10)

        ns, r = slow_roll_ns_r(V, phi)
        if np.isnan(ns):
            continue
        if (abs(ns - PLANCK_NS) < PLANCK_NS_1S
                and r < PLANCK_R_MAX):
            wins.append({'ns': ns, 'r': r,
                         'alpha': alpha, 'f': f})

    winners_C[name] = wins
    print(f"  {name}: {len(wins)}/{N_TRIALS} = "
          f"{len(wins)/N_TRIALS*100:.2f}% winners")

print()

# ── КОНТРОЛЬ для каждой формы ───────────────────────────────

print("── Random controls ──")
print()

# Контроль A
wins_rA = []
for trial in range(N_TRIALS):
    alpha = np.random.normal(0, 0.1, 8)
    m_rand = np.random.uniform(1, 30, 8)
    correction = 1.0 + sum(
        alpha[i]*np.exp(-m_rand[i]*phi/30)
        for i in range(8))
    V = V_star_base * np.abs(correction)
    V = np.where(V>0, V, 1e-10)
    ns, r = slow_roll_ns_r(V, phi)
    if not np.isnan(ns):
        if (abs(ns-PLANCK_NS)<PLANCK_NS_1S
                and r<PLANCK_R_MAX):
            wins_rA.append(1)
print(f"  Form A Random: {len(wins_rA)}/{N_TRIALS} = "
      f"{len(wins_rA)/N_TRIALS*100:.2f}%")

# Контроль C
wins_rC = []
for trial in range(N_TRIALS):
    alpha = np.random.uniform(-0.2, 0.2, 8)
    f     = np.random.uniform(1.0, 10.0)
    m_rand = np.random.uniform(1, 30, 8)
    oscillation = 1.0 + sum(
        alpha[i]*np.cos(m_rand[i]*phi/f)
        for i in range(8))
    V = V_star_base * oscillation
    V = np.where(V>1e-10, V, 1e-10)
    ns, r = slow_roll_ns_r(V, phi)
    if not np.isnan(ns):
        if (abs(ns-PLANCK_NS)<PLANCK_NS_1S
                and r<PLANCK_R_MAX):
            wins_rC.append(1)
print(f"  Form C Random: {len(wins_rC)}/{N_TRIALS} = "
      f"{len(wins_rC)/N_TRIALS*100:.2f}%")

# ── ИТОГОВАЯ СВОДКА ─────────────────────────────────────────

print()
print("="*65)
print("SUMMARY")
print("="*65)
print()
print(f"{'Algebra':>6} {'Form A':>10} {'Form B':>10} "
      f"{'Form C':>10}")
print("-"*45)

for name in ALGEBRAS:
    wA = len(winners_A[name])/N_TRIALS*100
    wB = len(winners_B[name])/N_TRIALS*100
    wC = len(winners_C[name])/N_TRIALS*100
    best = max(wA, wB, wC)
    flag = ' ←BEST' if best > 1.0 else ''
    print(f"{name:>6} {wA:>9.2f}% {wB:>9.2f}% "
          f"{wC:>9.2f}%{flag}")

print(f"{'Rand':>6} {len(wins_rA)/N_TRIALS*100:>9.2f}%"
      f"{'n/a':>11} "
      f"{len(wins_rC)/N_TRIALS*100:>9.2f}%")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Part XV Step 2: Modified Potential Forms\n'
    'Searching Planck-compatible Coxeter inflation',
    fontsize=12)

cox_colors = {'E8':'#e74c3c','E6':'#3498db',
              'E7':'#2ecc71','A6':'#9b59b6'}

# 0,0: Базовый Старобинский
ax = axes[0,0]
ax.plot(phi, V_star_base/V_star_base.max(),
        'k-', lw=2.5, label=f'Starobinsky\nn_s={ns_star:.4f}')
ax.set_xlabel('φ')
ax.set_ylabel('V/V_max')
ax.set_title('Baseline: Starobinsky potential')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

# 0,1: Лучшие потенциалы Form A
ax = axes[0,1]
ax.plot(phi, V_star_base/V_star_base.max(),
        'k--', lw=1.5, alpha=0.5, label='Starobinsky')
for name, wins in winners_A.items():
    if wins:
        best = min(wins, key=lambda x: abs(x['ns']-PLANCK_NS))
        m_arr = np.array(ALGEBRAS[name]['m'], dtype=float)
        h     = ALGEBRAS[name]['h']
        correction = 1.0 + sum(
            best['alpha'][i]*np.exp(-m_arr[i]*phi/h)
            for i in range(len(m_arr)))
        V = V_star_base * np.abs(correction)
        V = V/V.max()
        ax.plot(phi, V, color=cox_colors[name],
                lw=2, label=f"{name} n_s={best['ns']:.4f}")
ax.set_xlabel('φ')
ax.set_ylabel('V/V_max')
ax.set_title('Form A: Best winners')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

# 0,2: Win rates comparison
ax = axes[0,2]
forms    = ['Form A\n(Star×Cox)', 'Form B\n(Hilltop)', 'Form C\n(Axion)']
x        = np.arange(len(forms))
width    = 0.18
offsets  = np.linspace(-0.27, 0.27, len(ALGEBRAS))

for j, (name, col) in enumerate(cox_colors.items()):
    rates = [
        len(winners_A[name])/N_TRIALS*100,
        len(winners_B[name])/N_TRIALS*100,
        len(winners_C[name])/N_TRIALS*100,
    ]
    ax.bar(x + offsets[j], rates, width,
           label=name, color=col, alpha=0.8)

rand_rates = [
    len(wins_rA)/N_TRIALS*100,
    0,
    len(wins_rC)/N_TRIALS*100,
]
ax.bar(x + offsets[-1]+width, rand_rates, width,
       label='Random', color='gray', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(forms)
ax.set_ylabel('Win rate (%)')
ax.set_title('Winners by form and algebra')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# 1,0: n_s vs r для Form A winners
ax = axes[1,0]
for name, wins in winners_A.items():
    if wins:
        ns_w = [w['ns'] for w in wins]
        r_w  = [w['r']  for w in wins]
        ax.scatter(ns_w, r_w, color=cox_colors[name],
                   s=20, alpha=0.6, label=name)
ax.axvline(PLANCK_NS, color='k', lw=1.5, ls='--')
ax.axvspan(PLANCK_NS-PLANCK_NS_1S,
           PLANCK_NS+PLANCK_NS_1S,
           alpha=0.15, color='blue', label='Planck 1σ')
ax.axhline(PLANCK_R_MAX, color='red', lw=1.5, ls=':')
ax.set_xlabel('n_s')
ax.set_ylabel('r')
ax.set_title('Form A: Winners scatter')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1,1: Form C winners
ax = axes[1,1]
for name, wins in winners_C.items():
    if wins:
        ns_w = [w['ns'] for w in wins]
        r_w  = [w['r']  for w in wins]
        ax.scatter(ns_w, r_w, color=cox_colors[name],
                   s=20, alpha=0.6, label=name)
ax.axvline(PLANCK_NS, color='k', lw=1.5, ls='--')
ax.axvspan(PLANCK_NS-PLANCK_NS_1S,
           PLANCK_NS+PLANCK_NS_1S,
           alpha=0.15, color='blue', label='Planck 1σ')
ax.axhline(PLANCK_R_MAX, color='red', lw=1.5, ls=':')
ax.set_xlabel('n_s')
ax.set_ylabel('r')
ax.set_title('Form C: Winners scatter')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1,2: Вердикт
ax = axes[1,2]
ax.axis('off')

total_wins = {
    name: (len(winners_A[name]) +
           len(winners_B[name]) +
           len(winners_C[name]))
    for name in ALGEBRAS
}
best_name = max(total_wins, key=total_wins.get)

lines = [
    'PART XV STEP 2 VERDICT',
    '─'*28,
    '',
    'Form A (Starobinsky+Cox):',
]
for name in ALGEBRAS:
    w = len(winners_A[name])
    lines.append(f'  {name}: {w} winners')

lines += ['', 'Form B (Hilltop):']
for name in ALGEBRAS:
    w = len(winners_B[name])
    lines.append(f'  {name}: {w} winners')

lines += ['', 'Form C (Axion-like):']
for name in ALGEBRAS:
    w = len(winners_C[name])
    lines.append(f'  {name}: {w} winners')

lines += ['', '─'*28]
if total_wins[best_name] > 100:
    lines += [
        f'SIGNAL: {best_name} has',
        f'{total_wins[best_name]} total winners!',
        '→ Investigate structure',
    ]
else:
    lines += [
        'No strong signal found.',
        '',
        'Key insight from Step 1:',
        'Pure Coxeter potentials',
        'give n_s > Planck.',
        'Need negative-curvature',
        'region in V(φ).',
        '',
        '→ Next: Root system E8',
        '  (Part XVI)',
    ]

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part15_step2_modified.png',
            dpi=120, bbox_inches='tight')
print()
print("✓ Saved: part15_step2_modified.png")

# ── ФИНАЛ ───────────────────────────────────────────────────

print()
print("="*65)
print("FINAL DECISION")
print("="*65)
print()
total_all = sum(total_wins.values())

if total_all > 500:
    print("СИГНАЛ НАЙДЕН — продолжаем исследование")
else:
    print("Сигнал не найден в модифицированных формах.")
    print()
    print("КЛЮЧЕВОЙ ВЫВОД из Parts XII-XV:")
    print()
    print("  Стандартное отображение Коксетера")
    print("  → физически нейтрально (как random)")
    print()
    print("  Коксетеровы потенциалы V~φ^(2m/h)")
    print("  → систематически смещают n_s вверх")
    print("  → не попадают в Planck без fine-tuning")
    print()
    print("СЛЕДУЮЩИЙ ШАГ: Part XVI — Root System E8")
    print()
    print("  Принципиально другая физика:")
    print("  Не ωᵢ из mᵢ/h, а сами корни α ∈ E8")
    print("  240 корней, связь ⟨α,β⟩")
    print("  Матрица Картана как гамильтониан")
