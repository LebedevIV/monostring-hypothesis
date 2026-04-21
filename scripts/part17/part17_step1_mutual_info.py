"""
Part XVII: Mutual Information between Coxeter modes

H₀: I(φᵢ:φⱼ) for Coxeter algebras ≤ random controls

Method:
  1. Run standard map for E8, E6, E7, A6
  2. Compute pairwise MI for all mode pairs
  3. Compare with N=200 rank-matched random controls
  4. Also test: total correlation C = ΣI - I_joint

Physical interpretation:
  High MI → modes are statistically dependent
  → monostring modes "communicate"
  → potential mechanism for non-local correlations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from itertools import combinations

SEED     = 42
T        = 100000
WARMUP   = 10000
KAPPA    = 0.05
N_BINS   = 50     # bins per dimension for MI estimation
N_CTRL   = 200
np.random.seed(SEED)

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7},
}

# ── БАЗОВЫЕ ФУНКЦИИ ──────────────────────────────────────────

def coxeter_freqs(name):
    alg = ALGEBRAS[name]
    m = np.array(alg['m'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])

def random_freqs(rank, seed):
    return np.random.RandomState(seed).uniform(0.1, 2.0, rank)

def run_monostring(omegas, kappa=KAPPA, T=T,
                   warmup=WARMUP, seed=SEED):
    rng = np.random.RandomState(seed)
    r   = len(omegas)
    phi = rng.uniform(0, 2*np.pi, r)
    orbit = np.zeros((T - warmup, r))
    for t in range(T):
        phi = (phi + omegas + kappa*np.sin(phi)) % (2*np.pi)
        if t >= warmup:
            orbit[t - warmup] = phi
    return orbit

# ── ВЗАИМНАЯ ИНФОРМАЦИЯ ──────────────────────────────────────

def entropy_1d(x, n_bins=N_BINS):
    """Дифференциальная энтропия через гистограмму."""
    counts, _ = np.histogram(x, bins=n_bins,
                              range=(0, 2*np.pi))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def entropy_2d(x, y, n_bins=N_BINS):
    """Совместная энтропия H(X,Y)."""
    counts, _, _ = np.histogram2d(
        x, y, bins=n_bins,
        range=[[0, 2*np.pi], [0, 2*np.pi]])
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def mutual_information(x, y, n_bins=N_BINS):
    """I(X:Y) = H(X) + H(Y) - H(X,Y)"""
    Hx  = entropy_1d(x, n_bins)
    Hy  = entropy_1d(y, n_bins)
    Hxy = entropy_2d(x, y, n_bins)
    return Hx + Hy - Hxy

def total_correlation(orbit, n_bins=N_BINS):
    """
    Total correlation C = ΣᵢH(φᵢ) - H(φ₁,...,φᵣ)
    Мера многочастичной зависимости.

    Вычислить H(φ₁,...,φᵣ) для r>3 дорого.
    Используем аппроксимацию через сумму попарных MI:
    C_approx = Σᵢ<ⱼ I(φᵢ:φⱼ)
    """
    r = orbit.shape[1]
    pairs = list(combinations(range(r), 2))
    mi_sum = 0.0
    mi_vals = []
    for i, j in pairs:
        mi = mutual_information(orbit[:, i], orbit[:, j],
                                n_bins)
        mi_vals.append(mi)
        mi_sum += mi
    return mi_sum, np.array(mi_vals), pairs

def mi_matrix(orbit, n_bins=N_BINS):
    """Полная матрица попарных MI."""
    r = orbit.shape[1]
    M = np.zeros((r, r))
    for i in range(r):
        for j in range(i+1, r):
            mi = mutual_information(
                orbit[:, i], orbit[:, j], n_bins)
            M[i, j] = mi
            M[j, i] = mi
    return M

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("="*65)
print("Part XVII: Mutual Information between Coxeter modes")
print("="*65)
print(f"T={T}, warmup={WARMUP}, κ={KAPPA}, bins={N_BINS}")
print(f"N_ctrl={N_CTRL}")
print()

cox_results = {}

print("── Coxeter algebras ──")
for name in ALGEBRAS:
    ω     = coxeter_freqs(name)
    orbit = run_monostring(ω)
    tc, mi_vals, pairs = total_correlation(orbit)
    M     = mi_matrix(orbit)

    cox_results[name] = {
        'tc':      tc,
        'mi_vals': mi_vals,
        'mi_mean': mi_vals.mean(),
        'mi_max':  mi_vals.max(),
        'mi_min':  mi_vals.min(),
        'M':       M,
        'rank':    len(ω),
        'omegas':  ω,
    }

    print(f"  {name} (rank={len(ω)}, h={ALGEBRAS[name]['h']:2d}):")
    print(f"    MI mean={mi_vals.mean():.4f}, "
          f"max={mi_vals.max():.4f}, "
          f"min={mi_vals.min():.4f}")
    print(f"    Total correlation C={tc:.4f}")

print()

# ── СЛУЧАЙНЫЕ КОНТРОЛИ ───────────────────────────────────────

print(f"── Random controls (N={N_CTRL} per rank) ──")

rand_results = {}
for rank in [6, 7, 8]:
    print(f"  rank={rank}...")
    tc_list      = []
    mi_mean_list = []
    mi_max_list  = []

    for i in range(N_CTRL):
        ω     = random_freqs(rank, seed=1000*rank + i)
        orbit = run_monostring(ω)
        tc, mi_vals, _ = total_correlation(orbit)
        tc_list.append(tc)
        mi_mean_list.append(mi_vals.mean())
        mi_max_list.append(mi_vals.max())

        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N_CTRL}...")

    rand_results[rank] = {
        'tc':      np.array(tc_list),
        'mi_mean': np.array(mi_mean_list),
        'mi_max':  np.array(mi_max_list),
    }
    print(f"    TC:      {np.mean(tc_list):.4f} "
          f"± {np.std(tc_list):.4f}")
    print(f"    MI mean: {np.mean(mi_mean_list):.4f} "
          f"± {np.std(mi_mean_list):.4f}")
    print()

# ── СТАТИСТИКА ───────────────────────────────────────────────

print("── Statistical comparison (rank-matched) ──")
print()

for name in ALGEBRAS:
    rank  = cox_results[name]['rank']
    ref   = rand_results[rank]

    tc_cox = cox_results[name]['tc']
    mm_cox = cox_results[name]['mi_mean']
    mx_cox = cox_results[name]['mi_max']

    pct_tc = float(np.mean(ref['tc']      < tc_cox))*100
    pct_mm = float(np.mean(ref['mi_mean'] < mm_cox))*100
    pct_mx = float(np.mean(ref['mi_max']  < mx_cox))*100

    pval_tc = float(np.mean(ref['tc']      >= tc_cox))
    pval_mm = float(np.mean(ref['mi_mean'] >= mm_cox))
    pval_mx = float(np.mean(ref['mi_max']  >= mx_cox))

    def sig(p):
        return ('***' if p<0.001 else
                '**'  if p<0.01  else
                '*'   if p<0.05  else '')

    print(f"  {name} (rank={rank}):")
    print(f"    TC      = {tc_cox:.4f}  "
          f"pct={pct_tc:.0f}%  "
          f"p={pval_tc:.3f} {sig(pval_tc)}")
    print(f"    MI_mean = {mm_cox:.4f}  "
          f"pct={pct_mm:.0f}%  "
          f"p={pval_mm:.3f} {sig(pval_mm)}")
    print(f"    MI_max  = {mx_cox:.4f}  "
          f"pct={pct_mx:.0f}%  "
          f"p={pval_mx:.3f} {sig(pval_mx)}")
    print()

# ── СТРУКТУРА MI: какие пары связаны сильнее? ────────────────

print("── MI structure: which mode pairs are correlated? ──")
print()

for name in ['E8', 'E6']:
    M = cox_results[name]['M']
    r = M.shape[0]
    ω = cox_results[name]['omegas']

    print(f"  {name} MI matrix (rounded):")
    print("    ", end="")
    for i in range(r):
        print(f"  φ{i+1}", end="")
    print()

    for i in range(r):
        print(f"  φ{i+1}", end="")
        for j in range(r):
            if i == j:
                print(f"  ---", end="")
            else:
                print(f" {M[i,j]:.2f}", end="")
        print()
    print()

    # Пары с высокой MI
    pairs_sorted = sorted(
        [(M[i,j], i, j) for i in range(r)
         for j in range(i+1, r)],
        reverse=True)
    print(f"  Top-3 correlated pairs in {name}:")
    for mi_val, i, j in pairs_sorted[:3]:
        print(f"    φ{i+1}(ω={ω[i]:.3f}) ↔ "
              f"φ{j+1}(ω={ω[j]:.3f}): "
              f"MI={mi_val:.4f}")
    print()

# ── ПРОВЕРКА: MI vs |ωᵢ - ωⱼ| ───────────────────────────────

print("── Is MI driven by frequency proximity |ωᵢ-ωⱼ|? ──")
print()

for name in ['E8', 'E6']:
    M   = cox_results[name]['M']
    ω   = cox_results[name]['omegas']
    r   = M.shape[0]

    delta_omega = []
    mi_vals_flat = []
    for i in range(r):
        for j in range(i+1, r):
            delta_omega.append(abs(ω[i] - ω[j]))
            mi_vals_flat.append(M[i,j])

    delta_omega  = np.array(delta_omega)
    mi_vals_flat = np.array(mi_vals_flat)
    corr = np.corrcoef(delta_omega, mi_vals_flat)[0,1]
    print(f"  {name}: r(|Δω|, MI) = {corr:.3f}")
    if abs(corr) > 0.7:
        print(f"    → MI объясняется близостью частот!")
        print(f"    → Не специфика Коксетера")
    else:
        print(f"    → MI не объясняется |Δω|")

print()

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Part XVII: Mutual Information — Coxeter vs Random\n'
             'Do Coxeter modes "know" about each other?',
             fontsize=13)

cox_colors = {'E8':'#e74c3c', 'E6':'#3498db',
              'E7':'#2ecc71', 'A6':'#9b59b6'}

# ── Row 0: TC distributions
for col, rank in enumerate([6, 7, 8]):
    ax = fig.add_subplot(3, 4, col+1)
    ref = rand_results[rank]
    ax.hist(ref['tc'], bins=25, color='gray',
            alpha=0.6,
            label=f'Random r={rank}\n'
                  f'{ref["tc"].mean():.3f}±{ref["tc"].std():.3f}')

    for name in ALGEBRAS:
        if cox_results[name]['rank'] == rank:
            ax.axvline(cox_results[name]['tc'],
                      color=cox_colors[name], lw=2.5,
                      label=f'{name}={cox_results[name]["tc"]:.3f}')

    ax.set_xlabel('Total Correlation C')
    ax.set_ylabel('Count')
    ax.set_title(f'TC distribution (rank={rank})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# ── Summary panel (top right)
ax = fig.add_subplot(3, 4, 4)
ax.axis('off')
lines = ['MUTUAL INFORMATION', '─'*24, '']
for name in ALGEBRAS:
    rank = cox_results[name]['rank']
    ref  = rand_results[rank]
    tc   = cox_results[name]['tc']
    pv   = float(np.mean(ref['tc'] >= tc))
    s    = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else ''
    lines.append(f'{name}(r={rank}):')
    lines.append(f'  TC={tc:.3f} p={pv:.3f} {s}')
lines += ['', f'N_ctrl={N_CTRL} per rank',
          f'T={T}, κ={KAPPA}']
ax.text(0.05, 0.95, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

# ── Row 1: MI matrices для E8 и E6
for col, name in enumerate(['E8', 'E6']):
    ax = fig.add_subplot(3, 4, 5 + col)
    M  = cox_results[name]['M']
    im = ax.imshow(M, cmap='hot', vmin=0)
    plt.colorbar(im, ax=ax)
    r  = M.shape[0]
    ax.set_xticks(range(r))
    ax.set_yticks(range(r))
    ax.set_xticklabels([f'φ{i+1}' for i in range(r)],
                       fontsize=8)
    ax.set_yticklabels([f'φ{i+1}' for i in range(r)],
                       fontsize=8)
    ax.set_title(f'{name}: Pairwise MI matrix')
    for i in range(r):
        for j in range(r):
            if i != j:
                ax.text(j, i, f'{M[i,j]:.2f}',
                        ha='center', va='center',
                        fontsize=7, color='white'
                        if M[i,j] > M.max()*0.5
                        else 'black')

# Row 1, col 2-3: MI mean distributions
for col, rank in enumerate([6, 8]):
    ax = fig.add_subplot(3, 4, 7 + col)
    ref = rand_results[rank]
    ax.hist(ref['mi_mean'], bins=25, color='gray',
            alpha=0.6, label=f'Random r={rank}')
    for name in ALGEBRAS:
        if cox_results[name]['rank'] == rank:
            ax.axvline(cox_results[name]['mi_mean'],
                      color=cox_colors[name], lw=2.5,
                      label=f'{name}')
    ax.set_xlabel('Mean MI')
    ax.set_title(f'MI_mean distribution (rank={rank})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Row 2: MI vs |Δω| scatter
for col, name in enumerate(['E8', 'E6', 'A6']):
    ax = fig.add_subplot(3, 4, 9 + col)
    M   = cox_results[name]['M']
    ω   = cox_results[name]['omegas']
    r   = M.shape[0]

    dw, mi = [], []
    for i in range(r):
        for j in range(i+1, r):
            dw.append(abs(ω[i]-ω[j]))
            mi.append(M[i,j])
    dw = np.array(dw)
    mi = np.array(mi)
    corr = np.corrcoef(dw, mi)[0,1]

    ax.scatter(dw, mi, color=cox_colors[name],
               s=60, alpha=0.7,
               edgecolors='k', linewidths=0.5)
    # Линия тренда
    z = np.polyfit(dw, mi, 1)
    x_line = np.linspace(dw.min(), dw.max(), 50)
    ax.plot(x_line, np.polyval(z, x_line),
            'k--', lw=1.5, alpha=0.7)
    ax.set_xlabel('|ωᵢ - ωⱼ|')
    ax.set_ylabel('MI(φᵢ, φⱼ)')
    ax.set_title(f'{name}: MI vs |Δω|\nr={corr:.2f}')
    ax.grid(True, alpha=0.3)

# Row 2, col 3: Итог
ax = fig.add_subplot(3, 4, 12)
ax.axis('off')

# Финальный вердикт
any_signal = False
verdict_lines = ['VERDICT', '─'*24, '']
for name in ALGEBRAS:
    rank = cox_results[name]['rank']
    ref  = rand_results[rank]
    tc   = cox_results[name]['tc']
    pv   = float(np.mean(ref['tc'] >= tc))
    if pv < 0.05:
        any_signal = True
        verdict_lines.append(f'{name}: SIGNAL p={pv:.3f}')
    else:
        verdict_lines.append(f'{name}: H0 p={pv:.3f}')

verdict_lines += ['']
if any_signal:
    verdict_lines += [
        '→ Some Coxeter modes',
        '  show excess MI.',
        '→ Check if driven by',
        '  Weyl pairing ωᵢ=ωⱼ',
        '→ Part XVIII: mechanism'
    ]
else:
    verdict_lines += [
        '→ H₀ not rejected.',
        '→ Coxeter MI ≈ random.',
        '',
        'FINAL CONCLUSION:',
        'Classical monostring',
        'fully falsified across',
        'all 17 directions.',
    ]

color = '#ffe0e0' if not any_signal else '#e0ffe0'
ax.text(0.05, 0.95, '\n'.join(verdict_lines),
        transform=ax.transAxes, fontsize=9,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('part17_mutual_information.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part17_mutual_information.png")

# ── ФИНАЛЬНЫЙ ВЕРДИКТ ────────────────────────────────────────

print()
print("="*65)
print("PART XVII FINAL VERDICT")
print("="*65)
print()

signals = []
for name in ALGEBRAS:
    rank = cox_results[name]['rank']
    ref  = rand_results[rank]
    tc   = cox_results[name]['tc']
    pv   = float(np.mean(ref['tc'] >= tc))
    if pv < 0.05:
        signals.append((name, pv))

if signals:
    print("СИГНАЛ:")
    for name, pv in signals:
        print(f"  {name}: p={pv:.4f}")
    print()
    print("Проверяем механизм:")
    print("  1. Вейлевские пары ωᵢ=ωⱼ → ожидаемо высокий MI?")
    print("  2. Контроль n_unique?")
    print("  3. Если выживает → Part XVIII")
else:
    print("H₀ не отвергается ни для одной алгебры.")
    print()
    print("="*65)
    print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ: Parts I-XVII")
    print("="*65)
    print()
    print("Классическая гипотеза моноструны фальсифицирована")
    print("по 17 направлениям в 6 математических фреймворках.")
    print()
    print("Ни один физический сигнал не выжил строгий контроль.")
    print()
    print("Все выжившие результаты — математические теоремы:")
    print("  1. Σcos(πmᵢ/h) = 0  (теорема Вейля)")
    print("  2. det(C_E8) = 1     (унимодулярность)")
    print("  3. |Δ(h)| < 1 → критическая фаза (теор. Бете)")
    print("  4. Спектр Cay(W) = таблица характеров (Питер-Вейль)")
    print("  5. m_i+m_{r+1-i}=h  (инволюция Вейля)")
    print()
    print("Рекомендация:")
    print("  Опубликовать Parts I-XVII как")
    print("  полный отрицательный результат.")
    print("  Zenodo v12.0.0 (или v13.0.0 после этого Part).")
