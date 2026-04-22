"""
Part XIX Step 2: Quantum E8/E6 Spectrum — Proper Analysis

Исправления:
1. shift-invert mode для плотных спектров
2. n_max convergence test
3. Фокус на gap структуре (не на multiplet sizes)
4. E8 при n_max=2 (dim=6561, управляемо)
"""

import numpy as np
from scipy.sparse import kron, eye, diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh  # полная диагонализация
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ── ОПЕРАТОРЫ ────────────────────────────────────────────────

def a_dag(n_max):
    d = np.sqrt(np.arange(1, n_max+1))
    return diags(d, -1, shape=(n_max+1, n_max+1),
                 format='csr')

def n_op(n_max):
    return diags(np.arange(n_max+1, dtype=float),
                 0, format='csr')

def tensor_op(ops):
    result = ops[0]
    for op in ops[1:]:
        result = kron(result, op, format='csr')
    return result

def build_hamiltonian(r, C, omegas, kappa, n_max):
    I   = eye(n_max+1, format='csr')
    dim = (n_max+1)**r
    H   = csr_matrix((dim, dim), dtype=float)

    for i in range(r):
        ops    = [I]*r
        ops[i] = n_op(n_max)
        H      = H + omegas[i] * tensor_op(ops)

    AD = a_dag(n_max)
    AN = AD.T

    for i in range(r):
        for j in range(i+1, r):
            if abs(C[i,j]) < 1e-10:
                continue
            ops_ij    = [I]*r
            ops_ij[i] = AD
            ops_ij[j] = AN
            t_ij      = tensor_op(ops_ij)
            H = H + kappa * C[i,j] * (t_ij + t_ij.T)

    return H

def get_spectrum_safe(H, n_eigs=20, sigma=None):
    """
    Надёжное получение спектра:
    - При малых матрицах: полная диагонализация
    - При больших: ARPACK с shift-invert
    - При отказе ARPACK: возвращаем частичный результат
    """
    dim = H.shape[0]

    # Малые матрицы: точная диагонализация
    if dim <= 512:
        vals = eigh(H.toarray(),
                    eigvals_only=True,
                    subset_by_index=[0, min(n_eigs-1, dim-1)])
        return np.sort(vals.real)

    # Большие матрицы: ARPACK
    k = min(n_eigs, dim - 2)

    # Пробуем shift-invert (σ=0) — лучше для нижних уровней
    try:
        if sigma is not None:
            vals = eigsh(H, k=k, sigma=sigma,
                        which='LM',
                        return_eigenvectors=False,
                        maxiter=10000,
                        tol=1e-8)
        else:
            vals = eigsh(H, k=k, which='SM',
                        return_eigenvectors=False,
                        maxiter=10000,
                        tol=1e-8)
        return np.sort(vals.real)
    except Exception as e:
        # Пробуем с меньшим числом векторов
        try:
            k_small = k // 2
            vals = eigsh(H, k=k_small, sigma=0.0,
                        which='LM',
                        return_eigenvectors=False,
                        maxiter=20000,
                        tol=1e-6)
            print(f"    (fallback: got {k_small} eigs)")
            return np.sort(vals.real)
        except Exception as e2:
            print(f"    ARPACK failed: {e2}")
            return None

def get_frequencies(C):
    """Нормальные моды через собственные значения C."""
    eigs = np.linalg.eigvalsh(C)
    return np.sqrt(np.abs(eigs))

# ── МАТРИЦЫ КАРТАНА ──────────────────────────────────────────

CARTAN = {
    'A2': np.array([[ 2,-1],[-1, 2]], dtype=float),
    'A3': np.array([[ 2,-1, 0],
                    [-1, 2,-1],
                    [ 0,-1, 2]], dtype=float),
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

# ── ТЕСТ 1: СХОДИМОСТЬ ПО n_max ─────────────────────────────

print("="*65)
print("Part XIX Step 2: Convergence & Gap Analysis")
print("="*65)
print()

print("── Test 1: n_max convergence (E6, κ=1.0) ──")
print("  Проверяем: реален ли маленький gap_1?")
print()

C_e6     = CARTAN['E6']
omegas_e6 = get_frequencies(C_e6)
kappa_test = 1.0

print(f"  E6 frequencies: {np.round(omegas_e6, 3)}")
print()
print(f"  {'n_max':>6} {'dim':>8} {'gap_1':>10} "
      f"{'gap_2':>10} {'gap_3':>10} {'stable?':>8}")
print("  " + "-"*56)

prev_gap1 = None
for n_max in [1, 2, 3, 4]:
    dim = (n_max+1)**6
    if dim > 100000:
        print(f"  {n_max:>6} {dim:>8}  TOO LARGE")
        break

    H = build_hamiltonian(6, C_e6, omegas_e6,
                           kappa_test, n_max)

    # Полная диагонализация для малых, ARPACK для больших
    if dim <= 512:
        vals_all = eigh(H.toarray(), eigvals_only=True)
        vals = vals_all[:15]
    else:
        vals = get_spectrum_safe(H, n_eigs=15, sigma=0.0)

    if vals is None:
        print(f"  {n_max:>6} {dim:>8}  FAILED")
        continue

    E0   = vals[0]
    gaps = vals - E0
    g1   = gaps[1] if len(gaps)>1 else np.nan
    g2   = gaps[2] if len(gaps)>2 else np.nan
    g3   = gaps[3] if len(gaps)>3 else np.nan

    stable = ""
    if prev_gap1 is not None and not np.isnan(g1):
        rel_change = abs(g1 - prev_gap1) / (abs(prev_gap1) + 1e-10)
        stable = "✓" if rel_change < 0.05 else f"Δ={rel_change:.1%}"
    prev_gap1 = g1

    print(f"  {n_max:>6} {dim:>8} {g1:>10.5f} "
          f"{g2:>10.5f} {g3:>10.5f} {stable:>8}")

print()

# ── ТЕСТ 2: ЗАВИСИМОСТЬ gap ОТ κ ────────────────────────────

print("── Test 2: Gap vs κ (E6, n_max=2) ──")
print("  Ищем квантовый фазовый переход")
print()

n_max_main = 2
kappas_scan = np.array([0.0, 0.05, 0.1, 0.2, 0.3,
                          0.5, 0.7, 1.0, 1.5, 2.0])

gaps_e6  = []
gaps_a3  = []
gaps_rnd = []

print(f"  {'κ':>6} {'E6 gap_1':>10} {'A3 gap_1':>10} "
      f"{'Rnd gap_1':>10} {'E6/Rnd':>8}")
print("  " + "-"*52)

# Случайный контроль: random symmetric C, same rank
rng_ctrl = np.random.RandomState(42)
C_rnd_6  = rng_ctrl.randn(6,6)
C_rnd_6  = (C_rnd_6 + C_rnd_6.T) / 2
# Нормируем диагональ к 2
np.fill_diagonal(C_rnd_6, 2.0)
omegas_rnd = get_frequencies(C_rnd_6)

for kappa in kappas_scan:
    # E6
    H_e6 = build_hamiltonian(6, C_e6, omegas_e6,
                              kappa, n_max_main)
    v_e6 = get_spectrum_safe(H_e6, n_eigs=10, sigma=0.0)

    # A3
    C_a3 = CARTAN['A3']
    om_a3 = get_frequencies(C_a3)
    H_a3 = build_hamiltonian(3, C_a3, om_a3,
                              kappa, n_max_main)
    v_a3 = get_spectrum_safe(H_a3, n_eigs=10)

    # Random rank-6
    H_rnd = build_hamiltonian(6, C_rnd_6, omegas_rnd,
                               kappa, n_max_main)
    v_rnd = get_spectrum_safe(H_rnd, n_eigs=10, sigma=0.0)

    def gap1(v):
        if v is None or len(v) < 2:
            return np.nan
        return (v - v[0])[1]

    g_e6  = gap1(v_e6)
    g_a3  = gap1(v_a3)
    g_rnd = gap1(v_rnd)
    ratio = g_e6/g_rnd if (g_rnd > 1e-10 and
                            not np.isnan(g_e6)) else np.nan

    gaps_e6.append(g_e6)
    gaps_a3.append(g_a3)
    gaps_rnd.append(g_rnd)

    print(f"  {kappa:>6.2f} {g_e6:>10.5f} "
          f"{g_a3:>10.5f} {g_rnd:>10.5f} {ratio:>8.3f}")

print()

# ── ТЕСТ 3: E8 при n_max=1 ───────────────────────────────────

print("── Test 3: E8 quantum spectrum (n_max=1, dim=256) ──")
print()

C_e8    = CARTAN['E8']
omegas_e8 = get_frequencies(C_e8)
print(f"  E8 frequencies: {np.round(omegas_e8, 3)}")

e8_gaps_by_kappa = {}
kappas_e8 = [0.0, 0.5, 1.0, 2.0, 5.0]

print(f"  {'κ':>6} {'gap_1':>10} {'gap_2':>10} "
      f"{'gap_3':>10} {'gap_4':>10}")
print("  " + "-"*48)

for kappa in kappas_e8:
    H_e8 = build_hamiltonian(8, C_e8, omegas_e8,
                              kappa, n_max=1)
    # dim=256: точная диагонализация
    vals = eigh(H_e8.toarray(), eigvals_only=True)
    E0   = vals[0]
    gaps = vals - E0

    e8_gaps_by_kappa[kappa] = gaps[:20]

    g1 = gaps[1]; g2 = gaps[2]
    g3 = gaps[3]; g4 = gaps[4]
    print(f"  {kappa:>6.1f} {g1:>10.5f} {g2:>10.5f} "
          f"{g3:>10.5f} {g4:>10.5f}")

print()

# Дегенерация основного состояния
print("  Ground state degeneracy check (E8, κ=2.0):")
gaps_k2 = e8_gaps_by_kappa[2.0]
near_zero = np.sum(gaps_k2[:10] < 0.001)
print(f"  Levels with gap < 0.001: {near_zero}")
print()

# ── ТЕСТ 4: СРАВНЕНИЕ АЛГЕБР ПО МИНИМАЛЬНОМУ GAP ────────────

print("── Test 4: Minimum gap comparison at κ=1.0 ──")
print("  Ищем: E8 имеет наименьший gap → фазовый переход?")
print()

algebras_rank6 = {
    'E6':   CARTAN['E6'],
    'A6':   np.diag([2]*6) + np.diag([-1]*5, 1) + np.diag([-1]*5, -1),
    'D6':   None,  # строим отдельно
}

# D6 Карtan
D6 = np.zeros((6,6))
for i in range(5):
    D6[i,i]   =  2
    D6[i,i+1] = -1
    D6[i+1,i] = -1
D6[4,4] = 2; D6[5,5] = 2
D6[3,5] = -1; D6[5,3] = -1
algebras_rank6['D6'] = D6.astype(float)

# Случайные контроли
rng = np.random.RandomState(0)
for seed in [0, 1, 2, 3, 4]:
    rng2 = np.random.RandomState(seed)
    C_r  = rng2.randn(6,6)
    C_r  = (C_r + C_r.T)/2
    np.fill_diagonal(C_r, 2.0)
    algebras_rank6[f'Rnd{seed}'] = C_r

kappa_cmp = 1.0
n_max_cmp = 2

print(f"  {'Algebra':>8} {'dim':>6} {'min_gap':>10} "
      f"{'gap_2':>10} {'ratio':>8}")
print("  " + "-"*46)

gap_results = {}
for name, C_alg in algebras_rank6.items():
    om  = get_frequencies(C_alg)
    H   = build_hamiltonian(6, C_alg, om,
                             kappa_cmp, n_max_cmp)
    dim = H.shape[0]
    v   = get_spectrum_safe(H, n_eigs=10, sigma=0.0)
    if v is None:
        continue
    gaps = v - v[0]
    g1   = gaps[1]
    g2   = gaps[2] if len(gaps)>2 else np.nan
    ratio = g2/g1 if g1>1e-6 else np.nan

    gap_results[name] = g1
    print(f"  {name:>8} {dim:>6} {g1:>10.5f} "
          f"{g2:>10.5f} {ratio:>8.2f}")

print()

# Статистика
coxeter_gaps = [gap_results.get(n, np.nan)
                for n in ['E6','A6','D6']]
random_gaps  = [gap_results.get(f'Rnd{s}', np.nan)
                for s in range(5)]
coxeter_gaps = [g for g in coxeter_gaps
                if not np.isnan(g)]
random_gaps  = [g for g in random_gaps
                if not np.isnan(g)]

if coxeter_gaps and random_gaps:
    print(f"  Coxeter mean gap: {np.mean(coxeter_gaps):.5f} "
          f"± {np.std(coxeter_gaps):.5f}")
    print(f"  Random   mean gap: {np.mean(random_gaps):.5f} "
          f"± {np.std(random_gaps):.5f}")
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(coxeter_gaps,
                                     random_gaps)
    print(f"  t-test p = {p_val:.4f}")
    if p_val < 0.05:
        print("  *** SIGNIFICANT: Coxeter gaps ≠ Random! ***")
    else:
        print("  Not significant (H₀ holds)")
print()

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Part XIX Step 2: Quantum Spectrum Analysis\n'
    'Gap structure, n_max convergence, κ-dependence',
    fontsize=13)

# 0,0: E6 gap_1 vs κ
ax = axes[0,0]
ax.semilogy(kappas_scan, gaps_e6, 'o-',
            color='#e74c3c', lw=2, ms=8, label='E6')
ax.semilogy(kappas_scan, gaps_a3, 's--',
            color='#3498db', lw=2, ms=8, label='A3')
ax.semilogy(kappas_scan, gaps_rnd, '^:',
            color='gray', lw=2, ms=8, label='Random')
ax.set_xlabel('κ (coupling)')
ax.set_ylabel('gap_1 (log scale)')
ax.set_title('First gap vs κ\n(E6 vs A3 vs Random)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# 0,1: E8 spectrum at different κ (n_max=1)
ax = axes[0,1]
for kappa in kappas_e8:
    gaps = e8_gaps_by_kappa[kappa]
    ax.plot(range(min(12, len(gaps))),
            gaps[:12], 'o-',
            lw=1.5, ms=5,
            label=f'κ={kappa}')
ax.set_xlabel('Level n')
ax.set_ylabel('Gap (E_n - E_0)')
ax.set_title('E8 spectrum (n_max=1, dim=256)\nvs coupling κ')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 0,2: E8 первые 10 уровней при κ=2.0 (уровни энергии)
ax = axes[0,2]
gaps_k2 = e8_gaps_by_kappa[2.0]
levels   = np.sort(np.unique(np.round(gaps_k2[:20], 3)))
colors_lvl = plt.cm.viridis(np.linspace(0, 1, len(levels)))
for lvl, col in zip(levels, colors_lvl):
    count = np.sum(np.abs(gaps_k2[:20] - lvl) < 0.005)
    ax.axhline(lvl, lw=2, color=col,
               label=f'E={lvl:.3f} (×{count})')
ax.set_xlim(0, 1)
ax.set_ylabel('Gap (E - E_0)')
ax.set_title('E8 energy levels (κ=2.0)\ndegeneracy structure')
ax.legend(fontsize=7, loc='right')
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')

# 1,0: Сравнение алгебр при κ=1.0
ax = axes[1,0]
names  = list(gap_results.keys())
vals_g = list(gap_results.values())
colors_bar = ['#e74c3c' if n in ('E6','A6','D6')
               else 'gray' for n in names]
bars = ax.bar(range(len(names)), vals_g,
               color=colors_bar, alpha=0.7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('gap_1')
ax.set_title(f'Minimum gap comparison (κ={kappa_cmp})\nRed=Coxeter, Gray=Random')
ax.grid(True, alpha=0.3, axis='y')

# Добавляем значения
for bar, val in zip(bars, vals_g):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=7)

# 1,1: E6 gap collapse — log plot
ax = axes[1,1]
kv   = np.array(kappas_scan)
g_e6 = np.array(gaps_e6)
g_rnd = np.array(gaps_rnd)

valid = ~(np.isnan(g_e6) | np.isnan(g_rnd) |
          (g_e6 <= 0) | (g_rnd <= 0))
if valid.sum() > 3:
    ax.loglog(kv[valid], g_e6[valid], 'o-',
              color='#e74c3c', lw=2, ms=8,
              label='E6')
    ax.loglog(kv[valid], g_rnd[valid], 's--',
              color='gray', lw=2, ms=8,
              label='Random')
    # Power-law fit для E6
    if valid.sum() > 4:
        coeffs = np.polyfit(np.log(kv[valid]),
                            np.log(g_e6[valid]), 1)
        ax.loglog(kv[valid],
                  np.exp(np.polyval(coeffs,
                                    np.log(kv[valid]))),
                  'r:', lw=1.5,
                  label=f'fit: gap∝κ^{coeffs[0]:.2f}')

ax.set_xlabel('κ (log)')
ax.set_ylabel('gap_1 (log)')
ax.set_title('Gap scaling: gap ∝ κ^α?')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# 1,2: Сводка
ax = axes[1,2]
ax.axis('off')

lines = [
    'PART XIX STEP 2 SUMMARY',
    '─'*32,
    '',
    'Test 1 (n_max convergence):',
    '  → see convergence table above',
    '',
    'Test 2 (gap vs κ):',
    f'  E6 gap at κ=1.0: '
    f'{gaps_e6[kappas_scan.tolist().index(1.0)]:.5f}',
    f'  Rnd gap at κ=1.0: '
    f'{gaps_rnd[kappas_scan.tolist().index(1.0)]:.5f}',
    '',
    'Test 3 (E8, n_max=1):',
]

for kappa in kappas_e8:
    g = e8_gaps_by_kappa[kappa][1]
    lines.append(f'  κ={kappa}: gap_1={g:.5f}')

lines += [
    '',
    'Test 4 (algebra comparison):',
]
if coxeter_gaps and random_gaps:
    lines += [
        f'  Cox: {np.mean(coxeter_gaps):.5f}'
        f'±{np.std(coxeter_gaps):.5f}',
        f'  Rnd: {np.mean(random_gaps):.5f}'
        f'±{np.std(random_gaps):.5f}',
        f'  p = {p_val:.4f}',
    ]

# Вердикт
any_signal = (p_val < 0.05 if
              (coxeter_gaps and random_gaps)
              else False)

if any_signal:
    color = '#e0ffe0'
    lines += ['', '*** SIGNAL: p<0.05 ***',
              '→ Part XX: mechanism check']
else:
    color = '#ffe0e0'
    lines += ['', 'No signal (H₀ holds).',
              '→ Publish Parts I-XIX',
              '→ v14.0.0 complete']

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8.5,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('part19_step2_gaps.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part19_step2_gaps.png")

print()
print("="*65)
print("FINAL VERDICT Part XIX")
print("="*65)
print()

print("Три ключевых вопроса:")
print()
print("Q1: Стабилен ли маленький gap E6 при n_max→∞?")
print("    → Если gap растёт с n_max: артефакт усечения")
print("    → Если стабилен: реальный квазивырожденный уровень")
print()
print("Q2: Коллапсирует ли gap E6 быстрее, чем Random?")
print("    → gap(κ) ∝ κ^α: сравниваем α для E6 vs Random")
print("    → Если α_E6 >> α_Random: E6-специфичный переход")
print()
print("Q3: E8 при n_max=1 — есть ли вырождение ОС?")
print("    → Считаем уровни с gap < 0.001")
print()
print("Результаты выше отвечают на все три вопроса.")
