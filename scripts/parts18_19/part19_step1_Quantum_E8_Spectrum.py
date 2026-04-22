"""
Part XIX: Quantum E8 Spectrum — Proper Analysis

Три улучшения:
1. Варьируем κ в широком диапазоне (0.01 → 2.0)
2. Используем матрицу Картана (не частоты Коксетера)
3. Сравниваем с A1, A2, G2 (меньшие алгебры)
4. Ищем нетривиальную мультиплетную структуру
"""

import numpy as np
from scipy.sparse import kron, eye, diags, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ── ОПЕРАТОРЫ ────────────────────────────────────────────────

def a_dag(n_max):
    """Оператор рождения"""
    d = np.sqrt(np.arange(1, n_max+1))
    return diags(d, -1, shape=(n_max+1, n_max+1),
                 format='csr')

def a_ann(n_max):
    """Оператор уничтожения"""
    return a_dag(n_max).T

def n_op(n_max):
    """Оператор числа частиц"""
    return diags(np.arange(n_max+1, dtype=float),
                 0, format='csr')

def tensor_op(ops):
    """Тензорное произведение списка операторов"""
    result = ops[0]
    for op in ops[1:]:
        result = kron(result, op, format='csr')
    return result

# ── ГАМИЛЬТОНИАН ─────────────────────────────────────────────

def build_hamiltonian(r, C, omegas, kappa, n_max):
    """
    H = Σᵢ ωᵢ·n̂ᵢ + κ·Σᵢ<ⱼ Cᵢⱼ·(aᵢ†aⱼ + aᵢaⱼ†)

    r      = число мод
    C      = матрица связи r×r
    omegas = частоты мод
    kappa  = сила взаимодействия
    n_max  = макс. число квантов на моду
    """
    I   = eye(n_max+1, format='csr')
    dim = (n_max+1)**r
    H   = csr_matrix((dim, dim))

    # Кинетический член
    for i in range(r):
        ops    = [I]*r
        ops[i] = n_op(n_max)
        H      = H + omegas[i]*tensor_op(ops)

    # Взаимодействие
    AD = a_dag(n_max)
    AN = a_ann(n_max)

    for i in range(r):
        for j in range(i+1, r):
            if abs(C[i,j]) < 1e-10:
                continue

            # aᵢ†aⱼ
            ops_ij    = [I]*r
            ops_ij[i] = AD
            ops_ij[j] = AN
            t_ij      = tensor_op(ops_ij)

            H = H + kappa*C[i,j]*(t_ij + t_ij.T)

    return H

def get_spectrum(H, n_eigs=30):
    """Возвращает n_eigs нижних собственных значений."""
    k = min(n_eigs, H.shape[0]-2)
    vals = eigsh(H, k=k, which='SM',
                 return_eigenvectors=False)
    return np.sort(vals.real)

# ── АЛГЕБРЫ ──────────────────────────────────────────────────

# Матрицы Картана
CARTAN = {
    'A1': np.array([[2.]]),
    'A2': np.array([[ 2,-1],[-1, 2]], dtype=float),
    'G2': np.array([[ 2,-1],[-3, 2]], dtype=float),
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

# Частоты: диагональ матрицы Картана = 2 для всех
# Используем собственные значения C как физические частоты
def get_frequencies(C):
    """
    Нормальные моды: частоты = √(собственные значения C)
    При κ=0: ωᵢ = √λᵢ(C)
    """
    eigs = np.linalg.eigvalsh(C)
    return np.sqrt(np.abs(eigs))

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("="*65)
print("Part XIX Step 1: Quantum Spectrum Analysis")
print("="*65)
print()

# Параметры
N_MAX  = 3    # 4 уровня на моду (компромисс размер/точность)
N_EIGS = 25
KAPPAS = [0.0, 0.1, 0.5, 1.0, 2.0]

all_results = {}

for alg_name in ['A2', 'A3', 'G2', 'E6', 'E8']:
    C      = CARTAN[alg_name]
    r      = C.shape[0]
    omegas = get_frequencies(C)
    dim    = (N_MAX+1)**r

    # Пропускаем если матрица слишком большая
    if dim > 50000:
        print(f"{alg_name}: dim={dim} too large, skip")
        continue

    print(f"── {alg_name} (rank={r}, dim={dim}) ──")
    print(f"   omegas = {np.round(omegas,3)}")

    alg_results = {}

    for kappa in KAPPAS:
        H    = build_hamiltonian(r, C, omegas,
                                  kappa, N_MAX)
        vals = get_spectrum(H, n_eigs=N_EIGS)
        E0   = vals[0]
        gaps = vals - E0

        # Мультиплетная структура:
        # группируем уровни по близости
        threshold = 0.01 * (vals[-1] - vals[0] + 1e-10)
        multiplets = []
        current    = [gaps[0]]
        for g in gaps[1:]:
            if g - current[-1] < threshold:
                current.append(g)
            else:
                multiplets.append(current)
                current = [g]
        multiplets.append(current)

        # Размеры мультиплетов
        sizes = [len(m) for m in multiplets]

        alg_results[kappa] = {
            'E0': E0, 'gaps': gaps,
            'multiplets': multiplets,
            'sizes': sizes
        }

        # Выводим первые 8 уровней
        gaps_str = ', '.join(f'{g:.4f}' for g in gaps[:8])
        print(f"   κ={kappa:.1f}: gaps=[{gaps_str}]")
        print(f"          mult sizes: {sizes[:6]}")

    all_results[alg_name] = {
        'C': C, 'r': r,
        'omegas': omegas,
        'results': alg_results
    }
    print()

# ── СРАВНЕНИЕ МУЛЬТИПЛЕТОВ С СМ ─────────────────────────────

print("="*65)
print("Multiplet comparison with Standard Model")
print("="*65)
print()

# SM мультиплеты (размеры представлений):
# SU(3): 1, 3, 8
# SU(2): 1, 2, 3
# U(1):  1
# Частицы SM: 1(γ), 2(W±), 1(Z), 8(g), ...
SM_multiplets = [1, 1, 2, 3, 8]
print(f"SM key multiplet sizes: {SM_multiplets}")
print()

for alg_name in all_results:
    print(f"{alg_name} at κ=1.0:")
    res   = all_results[alg_name]['results'].get(1.0)
    if res is None:
        continue
    sizes = res['sizes']
    print(f"  Multiplet sizes: {sizes[:8]}")

    # Проверка пересечения с SM
    sm_set    = set(SM_multiplets)
    alg_set   = set(sizes[:8])
    overlap   = sm_set & alg_set
    print(f"  Overlap with SM: {overlap}")
    print()

# ── АНАЛИЗ κ-ЗАВИСИМОСТИ ────────────────────────────────────

print("="*65)
print("Gap structure vs κ (focus on E6, E8)")
print("="*65)
print()

for alg_name in ['E6', 'E8']:
    if alg_name not in all_results:
        print(f"{alg_name}: skipped (too large)")
        continue

    print(f"{alg_name}:")
    print(f"  {'κ':>6} {'gap_1':>8} {'gap_2':>8} "
          f"{'gap_3':>8} {'ratio_2/1':>12}")
    print("  " + "-"*50)

    for kappa in KAPPAS:
        res  = all_results[alg_name]['results'][kappa]
        gaps = res['gaps']
        g1   = gaps[1] if len(gaps)>1 else np.nan
        g2   = gaps[2] if len(gaps)>2 else np.nan
        g3   = gaps[3] if len(gaps)>3 else np.nan
        r21  = g2/g1 if g1>1e-6 else np.nan

        print(f"  {kappa:>6.1f} {g1:>8.4f} {g2:>8.4f} "
              f"{g3:>8.4f} {r21:>12.2f}")
    print()

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

n_algs = len(all_results)
fig, axes = plt.subplots(2, max(n_algs, 2),
                          figsize=(4*max(n_algs,2), 10))
fig.suptitle(
    'Part XIX: Quantum E8 Spectrum\n'
    'Energy levels vs coupling κ',
    fontsize=13)

for col, alg_name in enumerate(all_results):
    res_alg = all_results[alg_name]['results']

    # Row 0: уровни при κ=1.0
    ax = axes[0, col]
    res = res_alg.get(1.0, res_alg[KAPPAS[0]])
    gaps = res['gaps'][:20]
    ax.barh(range(len(gaps)), gaps,
            color='#3498db', alpha=0.7)
    ax.set_xlabel('Gap (E_n - E_0)')
    ax.set_ylabel('Level n')
    ax.set_title(f'{alg_name} spectrum (κ=1.0)')
    ax.grid(True, alpha=0.3, axis='x')

    # Отмечаем мультиплеты
    multiplets = res['multiplets']
    y = 0
    colors = ['#e74c3c','#2ecc71','#9b59b6',
               '#f39c12','#1abc9c']
    for m_idx, mult in enumerate(multiplets[:5]):
        for _ in mult:
            ax.axhline(y-0.5, color='k',
                       lw=0.5, alpha=0.3)
            y += 1
        ax.text(ax.get_xlim()[1]*0.7,
                y - len(mult)/2 - 0.5,
                f'd={len(mult)}',
                fontsize=8, color=colors[m_idx%5])

    # Row 1: gap_1 vs κ
    ax = axes[1, col]
    kv = KAPPAS
    g1_vals = []
    g2_vals = []
    for kappa in kv:
        res  = res_alg[kappa]
        gaps = res['gaps']
        g1_vals.append(gaps[1] if len(gaps)>1 else np.nan)
        g2_vals.append(gaps[2] if len(gaps)>2 else np.nan)

    ax.plot(kv, g1_vals, 'o-',
            color='#e74c3c', lw=2, ms=8,
            label='gap_1')
    ax.plot(kv, g2_vals, 's--',
            color='#3498db', lw=2, ms=8,
            label='gap_2')
    ax.set_xlabel('κ')
    ax.set_ylabel('Gap size')
    ax.set_title(f'{alg_name}: gaps vs κ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Скрываем лишние оси
for col in range(len(all_results), axes.shape[1]):
    axes[0, col].axis('off')
    axes[1, col].axis('off')

plt.tight_layout()
plt.savefig('part19_quantum_spectrum.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part19_quantum_spectrum.png")

# ── ФИНАЛЬНЫЙ ВЕРДИКТ ────────────────────────────────────────

print()
print("="*65)
print("PART XIX VERDICT")
print("="*65)
print()

# Проверяем: есть ли нетривиальная структура в E8?
if 'E8' in all_results:
    res_e8 = all_results['E8']['results']
    # При сильной связи κ=2.0
    kappa_max = max(KAPPAS)
    res = res_e8[kappa_max]
    sizes_e8 = res['sizes'][:6]

    print(f"E8 multiplet sizes at κ={kappa_max}: {sizes_e8}")
    print(f"SM key multiplets:                   {SM_multiplets}")

    # Совпадение
    overlap = set(sizes_e8) & set(SM_multiplets)
    if len(overlap) >= 3:
        print()
        print("ИНТЕРЕСНО: совпадение с SM размерностями!")
        print(f"  Overlap: {overlap}")
        print("  → Проверить: случайно ли это?")
        print("  → Сравнить с A_n, D_n алгебрами")
    else:
        print()
        print("Нет значимого совпадения с SM.")
        print()
        print("ИТОГОВОЕ РЕШЕНИЕ:")
        print("  Parts I-XIX: классическая и начало")
        print("  квантовой моноструны исчерпаны.")
        print()
        print("  Рекомендация: опубликовать v14.0.0")
        print("  с Parts I-XVIII (XIX как appendix).")
else:
    print("E8 не вычислен (dim слишком большая).")
    print("Результаты для меньших алгебр выше.")
    print()
    print("Для E8 нужен n_max=1 (dim=256, быстро)")
    print("или n_max=2 с усечённым базисом.")
