"""
Part XIX Step 4: Exact diagonalization via Bogoliubov

Гамильтониан:
  H = Σᵢ ωᵢ aᵢ†aᵢ + κ Σᵢ<ⱼ Cᵢⱼ (aᵢ†aⱼ + aⱼ†aᵢ)

Это КВАДРАТИЧНЫЙ гамильтониан в операторах a, a†.
Запишем в матричной форме:

  H = Ψ† M Ψ

где Ψ = (a₁,...,aᵣ)ᵀ и M — эффективная матрица:
  M_ii = ωᵢ
  M_ij = κ·Cᵢⱼ/2  (для i≠j, т.к. Cᵢⱼ=Cⱼᵢ)

Собственные значения M = {Ω₁,...,Ωᵣ}
→ нормальные моды с частотами Ωₖ
→ точный спектр: E = Σₖ nₖ·Ωₖ + E_ZPF
где E_ZPF = Σₖ Ωₖ/2 (нулевые колебания, если квантовать)

Первый зазор: gap = min(Ωₖ) (наименьшая нормальная мода)
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ── МАТРИЦЫ КАРТАНА ──────────────────────────────────────────

def make_An(n):
    C = np.diag([2.0]*n)
    for i in range(n-1):
        C[i,i+1] = C[i+1,i] = -1.0
    return C

CARTAN = {
    'A2': make_An(2),
    'A3': make_An(3),
    'A6': make_An(6),
    'E6': np.array([
        [ 2,-1, 0, 0, 0, 0],
        [-1, 2,-1, 0, 0, 0],
        [ 0,-1, 2,-1, 0,-1],
        [ 0, 0,-1, 2,-1, 0],
        [ 0, 0, 0,-1, 2, 0],
        [ 0, 0,-1, 0, 0, 2]], dtype=float),
    'E8': np.array([
        [ 2,-1, 0, 0, 0, 0, 0, 0],
        [-1, 2,-1, 0, 0, 0, 0, 0],
        [ 0,-1, 2,-1, 0, 0, 0,-1],
        [ 0, 0,-1, 2,-1, 0, 0, 0],
        [ 0, 0, 0,-1, 2,-1, 0, 0],
        [ 0, 0, 0, 0,-1, 2,-1, 0],
        [ 0, 0, 0, 0, 0,-1, 2, 0],
        [ 0, 0,-1, 0, 0, 0, 0, 2]], dtype=float),
}

def get_cartan_frequencies(C):
    """
    Нормальные моды: ωᵢ из диагональных эл-тов,
    но правильно — из Карта-новой матрицы.
    Используем: ωᵢ = sqrt(λᵢ(C)) как в предыдущих тестах.
    """
    eigs = np.linalg.eigvalsh(C)
    return np.sqrt(np.maximum(eigs, 0))

# ── ТОЧНОЕ АНАЛИТИЧЕСКОЕ РЕШЕНИЕ ─────────────────────────────

def exact_spectrum_quadratic(C, kappa, mode='cartan_freq'):
    """
    Квадратичный гамильтониан:
      H = Σᵢ ωᵢ aᵢ†aᵢ + κ Σᵢ<ⱼ Cᵢⱼ (aᵢ†aⱼ + h.c.)

    Эффективная одночастичная матрица:
      M_ii = ωᵢ
      M_ij = κ·Cᵢⱼ   (для i≠j)

    Собственные значения M = нормальные частоты Ωₖ.

    Если все Ωₖ > 0: система стабильна.
    Если Ωₖ < 0: неустойчивость (конденсация Боголюбова).

    Спектр:
      E_{n₁,...,nᵣ} = Σₖ nₖ·Ωₖ
      gap = min(Ωₖ) при Ωₖ > 0
    """
    r      = C.shape[0]
    omegas = get_cartan_frequencies(C)

    # Одночастичная матрица
    M = np.diag(omegas.copy())
    for i in range(r):
        for j in range(r):
            if i != j:
                M[i,j] += kappa * C[i,j]

    # Собственные значения
    normal_freqs = np.linalg.eigvalsh(M)

    # Минимальный зазор (первое возбуждение)
    pos_freqs = normal_freqs[normal_freqs > 1e-10]
    gap       = pos_freqs.min() if len(pos_freqs) > 0 else 0.0

    # Нулевой уровень (нулевые колебания)
    ZPF = 0.5 * np.sum(np.abs(normal_freqs))

    n_negative = np.sum(normal_freqs < -1e-10)

    return {
        'normal_freqs': np.sort(normal_freqs),
        'gap':          gap,
        'ZPF':          ZPF,
        'n_negative':   n_negative,
        'stable':       (n_negative == 0),
        'omegas':       omegas,
        'M':            M,
    }

# ── СЛУЧАЙНЫЕ КОНТРОЛИ ───────────────────────────────────────

def random_cartan_like(rank, seed):
    """
    Случайная матрица с той же структурой, что Карта-нова:
    - диагональ = 2
    - внедиагональные: целые числа в [-1, 0, 1]
    - симметрична
    """
    rng = np.random.RandomState(seed)
    C   = 2 * np.eye(rank)
    for i in range(rank):
        for j in range(i+1, rank):
            v         = rng.choice([-1, 0, 1],
                                    p=[0.3, 0.4, 0.3])
            C[i,j]    = v
            C[j,i]    = v
    return C.astype(float)

def random_continuous(rank, seed):
    """
    Случайная симметричная матрица с диагональю=2,
    внедиагональные ~ U(-1.5, 0) (только притяжение,
    как в Картановых матрицах).
    """
    rng = np.random.RandomState(seed)
    A   = rng.uniform(-1.5, 0, (rank, rank))
    A   = (A + A.T) / 2
    np.fill_diagonal(A, 2.0)
    return A

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("="*65)
print("Part XIX Step 4: Exact Bogoliubov Diagonalization")
print("="*65)
print()
print("Квадратичный H диагонализуется точно.")
print("Нет проблемы усечения базиса Фока.")
print()

# ── 1. СКАНИРОВАНИЕ κ ───────────────────────────────────────

print("── 1. Normal mode frequencies vs κ ──")
print()

kappas = np.linspace(0.0, 3.0, 61)
algebras_scan = ['E6', 'A6', 'E8']

gap_vs_kappa = {a: [] for a in algebras_scan}
stable_vs_kappa = {a: [] for a in algebras_scan}

for alg in algebras_scan:
    C = CARTAN[alg]
    for kappa in kappas:
        res = exact_spectrum_quadratic(C, kappa)
        gap_vs_kappa[alg].append(res['gap'])
        stable_vs_kappa[alg].append(res['stable'])

# Найти критическое κ (gap → 0)
print(f"  {'Algebra':>8} {'κ_c (gap→0)':>14} "
      f"{'min gap':>10} {'at κ':>8}")
print("  " + "-"*44)

kc_data = {}
for alg in algebras_scan:
    gaps   = np.array(gap_vs_kappa[alg])
    # Критическое κ: где gap впервые < 0.01
    below  = np.where(gaps < 0.01)[0]
    kc     = kappas[below[0]] if len(below) > 0 else np.nan
    min_g  = gaps.min()
    at_k   = kappas[gaps.argmin()]
    kc_data[alg] = kc
    print(f"  {alg:>8} {kc:>14.3f} "
          f"{min_g:>10.5f} {at_k:>8.3f}")

print()

# ── 2. ПОЛНЫЙ СПЕКТР НОРМАЛЬНЫХ МОД ─────────────────────────

print("── 2. Normal mode spectrum at key κ values ──")
print()

key_kappas = [0.0, 0.5, 1.0, 2.0]
for alg in ['E6', 'E8']:
    C = CARTAN[alg]
    print(f"  {alg}:")
    print(f"  {'κ':>6} {'Ω₁':>8} {'Ω₂':>8} "
          f"{'Ω₃':>8} {'Ωₘᵢₙ':>8} "
          f"{'n_neg':>6} {'stable':>8}")
    print("  " + "-"*60)
    for kappa in key_kappas:
        res  = exact_spectrum_quadratic(C, kappa)
        freq = res['normal_freqs']
        stab = "✓" if res['stable'] else "✗ UNSTABLE"
        print(f"  {kappa:>6.1f} "
              f"{freq[0]:>8.4f} {freq[1]:>8.4f} "
              f"{freq[2]:>8.4f} "
              f"{res['gap']:>8.4f} "
              f"{res['n_negative']:>6} {stab:>8}")
    print()

# ── 3. СРАВНЕНИЕ С КОНТРОЛЯМИ (N=100) ───────────────────────

print("── 3. Statistical comparison (N=100 controls) ──")
print()

N_CTRL    = 100
kappa_cmp = 1.0

# Коксетерові алгебры
coxeter_results = {}
for alg in ['E6', 'A6', 'E8']:
    C   = CARTAN[alg]
    res = exact_spectrum_quadratic(C, kappa_cmp)
    coxeter_results[alg] = res
    print(f"  {alg}: gap={res['gap']:.5f}, "
          f"ZPF={res['ZPF']:.4f}, "
          f"stable={res['stable']}")

print()

# Случайные контроли
print(f"  Generating N={N_CTRL} random controls...")

# Тип A: Cartanlike (целочисленные)
rand_gaps_A = []
rand_gaps_B = []

for s in range(N_CTRL):
    # Тип A: целочисленный
    C_a  = random_cartan_like(6, seed=s)
    r_a  = exact_spectrum_quadratic(C_a, kappa_cmp)
    rand_gaps_A.append(r_a['gap'])

    # Тип B: непрерывный
    C_b  = random_continuous(6, seed=s)
    r_b  = exact_spectrum_quadratic(C_b, kappa_cmp)
    rand_gaps_B.append(r_b['gap'])

rand_gaps_A = np.array(rand_gaps_A)
rand_gaps_B = np.array(rand_gaps_B)

print(f"  Random-A (integer): "
      f"{rand_gaps_A.mean():.5f} ± {rand_gaps_A.std():.5f}")
print(f"  Random-B (contin.): "
      f"{rand_gaps_B.mean():.5f} ± {rand_gaps_B.std():.5f}")
print()

# Тест для каждой алгебры
print(f"  {'Algebra':>8} {'gap':>8} "
      f"{'vs Rnd-A':>10} {'p(A)':>8} "
      f"{'pct-A':>8} {'p(B)':>8}")
print("  " + "-"*58)

verdicts = {}
for alg in ['E6', 'A6', 'E8']:
    g = coxeter_results[alg]['gap']

    # Тест vs Random-A
    _, p_a = stats.mannwhitneyu(
        [g], rand_gaps_A, alternative='two-sided')
    pct_a  = np.mean(rand_gaps_A < g) * 100

    # Тест vs Random-B
    _, p_b = stats.mannwhitneyu(
        [g], rand_gaps_B, alternative='two-sided')

    sig_a = "***" if p_a < 0.05 else ""
    sig_b = "***" if p_b < 0.05 else ""

    print(f"  {alg:>8} {g:>8.5f} "
          f"{pct_a:>9.0f}% {p_a:>8.4f}{sig_a} "
          f"{p_b:>8.4f}{sig_b}")

    verdicts[alg] = {
        'gap': g, 'p_a': p_a,
        'p_b': p_b, 'pct_a': pct_a
    }

print()

# ── 4. ЗАВИСИМОСТЬ gap ОТ СТРУКТУРЫ МАТРИЦЫ ─────────────────

print("── 4. What determines the gap? ──")
print()
print("  Если gap = f(собств.знач. C), то это тавтология")
print()

for alg in ['E6', 'A6', 'E8']:
    C     = CARTAN[alg]
    res_0 = exact_spectrum_quadratic(C, kappa=0.0)
    res_1 = exact_spectrum_quadratic(C, kappa=1.0)
    eigs_C = np.linalg.eigvalsh(C)
    om_min = np.sqrt(np.maximum(eigs_C, 0)).min()

    print(f"  {alg}: min_eig(C)={eigs_C.min():.4f}, "
          f"ω_min={om_min:.4f}, "
          f"gap(κ=0)={res_0['gap']:.4f}, "
          f"gap(κ=1)={res_1['gap']:.4f}")

print()
print("  Если gap(κ=1) ≈ f(κ, eigenvalues(C)):")
print("  → это математическая тавтология, не физика")
print()

# Проверка: корреляция gap с min_eig(C) для рандомных
eig_mins_A = []
for s in range(N_CTRL):
    C_a = random_cartan_like(6, seed=s)
    eig_mins_A.append(np.linalg.eigvalsh(C_a).min())

eig_mins_A = np.array(eig_mins_A)
corr, p_corr = stats.pearsonr(eig_mins_A, rand_gaps_A)
print(f"  Корреляция min_eig(C) — gap: r={corr:.3f}, "
      f"p={p_corr:.4f}")
if abs(corr) > 0.7:
    print("  *** ТАВТОЛОГИЯ: gap определяется")
    print("      минимальным собственным значением C!")
    print("      Нет новой физики.")
else:
    print("  Слабая корреляция — gap не тавтологичен")
print()

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Part XIX Step 4: Exact Bogoliubov Diagonalization\n'
    'Quadratic Hamiltonian — Analytic Solution',
    fontsize=13)

colors_alg = {'E6': '#e74c3c',
               'A6': '#3498db',
               'E8': '#9b59b6'}

# 0,0: gap vs κ для алгебр
ax = axes[0,0]
for alg in algebras_scan:
    ax.plot(kappas, gap_vs_kappa[alg],
            lw=2.5, color=colors_alg.get(alg,'gray'),
            label=alg)
    kc = kc_data.get(alg, np.nan)
    if not np.isnan(kc):
        ax.axvline(kc, color=colors_alg.get(alg,'gray'),
                   ls=':', alpha=0.5)

ax.set_xlabel('κ')
ax.set_ylabel('gap = min(Ωₖ)')
ax.set_title('Normal mode gap vs κ\n(exact, no truncation)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 0,1: E6 нормальные моды vs κ
ax = axes[0,1]
C_e6 = CARTAN['E6']
kappas_fine = np.linspace(0, 3, 100)
all_freqs = []
for kappa in kappas_fine:
    res = exact_spectrum_quadratic(C_e6, kappa)
    all_freqs.append(res['normal_freqs'])
all_freqs = np.array(all_freqs)

for mode in range(all_freqs.shape[1]):
    ax.plot(kappas_fine, all_freqs[:, mode],
            lw=1.5, alpha=0.8)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('κ')
ax.set_ylabel('Ωₖ (normal mode freq.)')
ax.set_title('E6: all normal modes vs κ\n(instability at κ_c)')
ax.grid(True, alpha=0.3)

# 0,2: E8 нормальные моды vs κ
ax = axes[0,2]
C_e8 = CARTAN['E8']
all_freqs_e8 = []
for kappa in kappas_fine:
    res = exact_spectrum_quadratic(C_e8, kappa)
    all_freqs_e8.append(res['normal_freqs'])
all_freqs_e8 = np.array(all_freqs_e8)

for mode in range(all_freqs_e8.shape[1]):
    ax.plot(kappas_fine, all_freqs_e8[:, mode],
            lw=1.5, alpha=0.8)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('κ')
ax.set_ylabel('Ωₖ')
ax.set_title('E8: all normal modes vs κ')
ax.grid(True, alpha=0.3)

# 1,0: random vs Coxeter gap distribution
ax = axes[1,0]
ax.hist(rand_gaps_A, bins=20, color='gray',
        alpha=0.5, density=True, label='Random-A')
ax.hist(rand_gaps_B, bins=20, color='lightblue',
        alpha=0.5, density=True, label='Random-B')
for alg in ['E6', 'A6', 'E8']:
    g = verdicts[alg]['gap']
    ax.axvline(g, lw=2.5,
               color=colors_alg.get(alg,'k'),
               label=f'{alg}: {g:.4f}')
ax.set_xlabel('gap at κ=1.0')
ax.set_ylabel('Density')
ax.set_title('Exact gap: Coxeter vs Random\n(N=100 each)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,1: корреляция min_eig — gap
ax = axes[1,1]
ax.scatter(eig_mins_A, rand_gaps_A,
           color='gray', alpha=0.5, s=30,
           label='Random-A')
# Добавляем Коксетеровы точки
for alg in ['E6', 'A6']:
    C  = CARTAN[alg]
    em = np.linalg.eigvalsh(C).min()
    g  = verdicts[alg]['gap']
    ax.scatter(em, g, s=150,
               color=colors_alg[alg],
               zorder=5, label=alg)

# Линия регрессии
if len(eig_mins_A) > 3:
    coeffs = np.polyfit(eig_mins_A, rand_gaps_A, 1)
    x_fit  = np.linspace(eig_mins_A.min(),
                          eig_mins_A.max(), 50)
    ax.plot(x_fit, np.polyval(coeffs, x_fit),
            'r--', lw=2,
            label=f'r={corr:.2f}, p={p_corr:.3f}')

ax.set_xlabel('min eigenvalue of C')
ax.set_ylabel('gap at κ=1.0')
ax.set_title('Is gap a tautology?\n'
             '(gap ~ min_eig(C)?)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,2: Итог
ax = axes[1,2]
ax.axis('off')

# Финальный вердикт
any_signal = any(verdicts[a]['p_a'] < 0.05
                  or verdicts[a]['p_b'] < 0.05
                  for a in verdicts)
is_tautology = abs(corr) > 0.7

lines = [
    'PART XIX STEP 4 VERDICT',
    '─'*30,
    '',
    'Method: exact Bogoliubov diagonalization',
    '(no Fock space truncation)',
    '',
    f'κ_c (gap collapse):',
]
for alg in algebras_scan:
    kc = kc_data.get(alg, np.nan)
    lines.append(f'  {alg}: κ_c = {kc:.3f}')

lines += [
    '',
    f'Gap at κ=1.0:',
]
for alg in ['E6', 'A6', 'E8']:
    g  = verdicts[alg]['gap']
    pa = verdicts[alg]['p_a']
    lines.append(f'  {alg}: {g:.5f} '
                 f'(p={pa:.3f})')

lines += [
    '',
    f'Tautology check:',
    f'  r(min_eig, gap) = {corr:.3f}',
    f'  p = {p_corr:.4f}',
    '',
    'CONCLUSION:',
]

if is_tautology:
    color = '#ffe0e0'
    lines += [
        '  gap = f(eigenvalues of C)',
        '  → TAUTOLOGY confirmed',
        '  → No new physics',
        '',
        'Parts I-XIX: all falsified.',
        '→ Publish v14.0.0'
    ]
elif any_signal:
    color = '#e0ffe0'
    lines += [
        '  *** SIGNAL p<0.05 ***',
        '  Not a tautology!',
        '→ Part XX: mechanism',
        '→ N=1000 controls'
    ]
else:
    color = '#ffe0e0'
    lines += [
        '  p > 0.05: H₀ holds.',
        '  Not a tautology,',
        '  but no signal either.',
        '',
        'Parts I-XIX: all falsified.',
        '→ Publish v14.0.0'
    ]

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8.5,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('part19_step4_bogoliubov.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part19_step4_bogoliubov.png")

print()
print("="*65)
print("ИТОГ Part XIX — Квантовая моноструна")
print("="*65)
print()
print("Step 1 (временной спектр):  n_s = -0.84 (артефакт)")
print("Step 2 (пространственный):  n_s = +2.39 (артефакт)")
print("Step 3 (Фоковое усечение):  нет сходимости по n_max")
print("Step 4 (точное решение):    → смотри вывод выше")
print()
print("Квадратичный H точно диагонализуется.")
print("Если gap тавтологичен → нет новой физики.")
print("Если gap не тавтологичен, но p>0.05 → H₀.")
