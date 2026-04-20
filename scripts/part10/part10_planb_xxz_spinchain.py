"""
Part X Plan B: XXZ Spin Chain with Coxeter/Temperley-Lieb structure

Физическая идея:
  Алгебра Темперли-Либа TL_n(q) порождается операторами eᵢ:
    eᵢ² = δ·eᵢ   (δ = q + q⁻¹)
    eᵢeᵢ±₁eᵢ = eᵢ
    eᵢeⱼ = eⱼeᵢ  |i-j| ≥ 2

  XXZ-гамильтониан:
    H_XXZ = -Σᵢ [σˣᵢσˣᵢ₊₁ + σʸᵢσʸᵢ₊₁ + Δ·σᶻᵢσᶻᵢ₊₁]

  Связь с TL: eᵢ = (δ/2)·I - Sᵢ·Sᵢ₊₁  (для S=1/2)
  При Δ = -δ/2 = -(q+q⁻¹)/2

  Специальные точки:
    q = exp(iπ/h):  Δ = -cos(π/h)
    E6: h=12 → Δ = -cos(π/12) ≈ -0.9659
    A6: h=7  → Δ = -cos(π/7)  ≈ -0.9009
    E8: h=30 → Δ = -cos(π/30) ≈ -0.9945

  Вопрос: отличается ли спектр/динамика XXZ при Δ=Δ(E6)
  от Δ=Δ(random) или Δ=Δ(A6)?

H0: Coxeter-специфические значения Δ не дают
    значимо отличного спектра от случайных Δ.
"""

import numpy as np
from scipy.linalg import eigvalsh, eigh
from scipy.stats import mannwhitneyu, spearmanr, ks_2samp
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh as sp_eigsh
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════
# Построение XXZ гамильтониана
# ════════════════════════════════════════════════════════

# Матрицы Паули
sx = np.array([[0,  0.5], [0.5,  0]], dtype=complex)
sy = np.array([[0, -0.5j],[0.5j, 0]], dtype=complex)
sz = np.array([[0.5, 0],  [0, -0.5]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kron_op(op, i, n):
    """Оператор op на сайте i в цепочке из n спинов."""
    ops = [I2]*n
    ops[i] = op
    result = ops[0]
    for k in range(1, n):
        result = np.kron(result, ops[k])
    return result

def xxz_hamiltonian(n, delta, J=1.0, pbc=True):
    """
    H = -J·Σᵢ [SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ + Δ·SᶻᵢSᶻᵢ₊₁]
    pbc: периодические граничные условия
    """
    dim = 2**n
    H   = np.zeros((dim, dim), dtype=complex)

    bonds = list(range(n-1))
    if pbc:
        bonds.append(n-1)  # связь n-1 → 0

    for i in bonds:
        j = (i+1) % n
        Sx_i = kron_op(sx, i, n)
        Sy_i = kron_op(sy, i, n)
        Sz_i = kron_op(sz, i, n)
        Sx_j = kron_op(sx, j, n)
        Sy_j = kron_op(sy, j, n)
        Sz_j = kron_op(sz, j, n)

        H -= J * (Sx_i@Sx_j + Sy_i@Sy_j + delta*Sz_i@Sz_j)

    return H.real  # XXZ вещественен

def heisenberg_hamiltonian(n, J_list, pbc=True):
    """
    Анизотропная цепочка с разными J на каждой связи.
    H = -Σᵢ Jᵢ·(SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ + SᶻᵢSᶻᵢ₊₁)
    Для тестирования случайных J.
    """
    dim = 2**n
    H   = np.zeros((dim, dim), dtype=complex)

    bonds = list(range(n-1))
    if pbc and len(J_list) == n:
        bonds.append(n-1)

    for k, i in enumerate(bonds):
        j = (i+1) % n
        Ji = J_list[k] if k < len(J_list) else J_list[-1]
        Sx_i = kron_op(sx, i, n)
        Sy_i = kron_op(sy, i, n)
        Sz_i = kron_op(sz, i, n)
        Sx_j = kron_op(sx, j, n)
        Sy_j = kron_op(sy, j, n)
        Sz_j = kron_op(sz, j, n)
        H -= Ji*(Sx_i@Sx_j + Sy_i@Sy_j + Sz_i@Sz_j)

    return H.real

def get_spectrum(H):
    """Собственные значения, отсортированные."""
    return np.sort(eigvalsh(H))

def spectral_gap(evals):
    """E₁ - E₀: щель над основным состоянием."""
    if len(evals) < 2:
        return 0.0
    return evals[1] - evals[0]

def level_spacing_stats(evals):
    """
    Статистика интервалов уровней.
    r̄ = ⟨min(s_n,s_{n+1})/max(s_n,s_{n+1})⟩
    GOE: r̄ ≈ 0.536 (хаос)
    Poisson: r̄ ≈ 0.386 (интегрируемость)
    """
    spacings = np.diff(evals)
    spacings = spacings[spacings > 1e-12]
    ratios = []
    for i in range(len(spacings)-1):
        s1, s2 = spacings[i], spacings[i+1]
        ratios.append(min(s1,s2)/max(s1,s2))
    return np.mean(ratios) if ratios else 0.0

def entanglement_entropy(psi, n, cut=None):
    """
    Энтропия запутанности для двухраздельной системы.
    psi: вектор состояния размерности 2^n
    cut: позиция разреза (по умолчанию n//2)
    """
    if cut is None:
        cut = n // 2
    dim_A = 2**cut
    dim_B = 2**(n-cut)
    psi_mat = psi.reshape(dim_A, dim_B)
    # SVD → singular values → энтропия
    sv = np.linalg.svd(psi_mat, compute_uv=False)
    sv = sv[sv > 1e-12]
    p  = sv**2
    p  = p / p.sum()
    return -np.sum(p * np.log(p))

def coxeter_delta(h):
    """Δ = -cos(π/h) для числа Коксетера h."""
    return -np.cos(np.pi / h)

# ════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Спектр при разных Δ
# ════════════════════════════════════════════════════════

rng = np.random.default_rng(42)

print("="*65)
print("Part X Plan B: XXZ Spin Chain with Coxeter parameters")
print("="*65)

print("\n[1] Coxeter-specific Δ values")
print(f"{'Algebra':8s} {'h':>4s} "
      f"{'Δ=−cos(π/h)':>14s} {'Δ (approx)':>12s}")
print("-"*44)

coxeter_algebras = {
    "A6":  7,   # rank 6, h=7
    "E6":  12,  # rank 6, h=12
    "E7":  18,  # rank 7, h=18
    "E8":  30,  # rank 8, h=30
    "G2":  6,   # rank 2, h=6
    "F4":  12,  # rank 4, h=12  ← same h as E6!
}

for name, h in coxeter_algebras.items():
    delta = coxeter_delta(h)
    print(f"  {name:6s}  h={h:2d}  "
          f"Δ = -cos(π/{h:2d}) = {delta:+.6f}")

print(f"\n  Special points:")
print(f"    Δ = -1.0:    Heisenberg (SU(2) symmetric)")
print(f"    Δ = 0.0:     XX model (free fermions)")
print(f"    Δ = +1.0:    antiferromagnetic Heisenberg")
print(f"    |Δ| > 1:     gapped (Ising-like)")
print(f"    |Δ| < 1:     gapless (critical, CFT)")

# ════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 2: Спектральная щель для n=8 спинов
# ════════════════════════════════════════════════════════

print("\n[2] Spectral gap E₁-E₀ for XXZ chain, n=8 spins")
print(f"{'Δ source':15s} {'Δ':>10s} "
      f"{'gap':>10s} {'r̄_ratio':>10s} {'S_ent':>8s}")
print("-"*58)

n_spins = 8

# Коксетеровы точки
cox_gaps = []
for name, h in coxeter_algebras.items():
    delta = coxeter_delta(h)
    H     = xxz_hamiltonian(n_spins, delta, pbc=True)
    evals = get_spectrum(H)
    gap   = spectral_gap(evals)
    r_bar = level_spacing_stats(evals)
    # Основное состояние
    _, evecs = eigh(H)
    S_ent = entanglement_entropy(evecs[:,0], n_spins)
    print(f"  {name:13s}  {delta:+10.6f}  "
          f"{gap:10.6f}  {r_bar:10.6f}  {S_ent:8.4f}")
    cox_gaps.append(gap)

print("  " + "-"*54)

# Случайные Δ (в том же диапазоне [-1, 0])
print("  [Random Δ values, same range]")
rnd_gaps_list = []
for seed in range(20):
    rs    = np.random.RandomState(seed)
    delta = rs.uniform(-1.0, -0.5)  # тот же диапазон
    H     = xxz_hamiltonian(n_spins, delta, pbc=True)
    evals = get_spectrum(H)
    gap   = spectral_gap(evals)
    r_bar = level_spacing_stats(evals)
    rnd_gaps_list.append(gap)
    if seed < 5:
        print(f"  Rnd_{seed:02d}        {delta:+10.6f}  "
              f"{gap:10.6f}  {r_bar:10.6f}")

rnd_gaps_arr = np.array(rnd_gaps_list)
print(f"\n  Random gaps: mean={rnd_gaps_arr.mean():.6f}, "
      f"std={rnd_gaps_arr.std():.6f}")

stat, p = mannwhitneyu(cox_gaps, rnd_gaps_arr,
                        alternative='two-sided')
print(f"  Mann-Whitney Coxeter vs Random: p={p:.4f}")

# ════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 3: Непрерывное сканирование Δ
# ════════════════════════════════════════════════════════

print("\n[3] Gap vs Δ continuous scan (n=8, pbc)")
print(f"{'Δ':>10s}  {'gap':>10s}  {'r̄':>8s}  {'phase':>12s}")
print("-"*48)

delta_values = np.linspace(-1.5, 0.5, 41)
gaps_scan    = []
r_scan       = []

for delta in delta_values:
    H     = xxz_hamiltonian(n_spins, delta, pbc=True)
    evals = get_spectrum(H)
    g     = spectral_gap(evals)
    r     = level_spacing_stats(evals)
    gaps_scan.append(g)
    r_scan.append(r)

gaps_scan = np.array(gaps_scan)
r_scan    = np.array(r_scan)

# Печатаем только каждую 4-ю точку
for i, (d, g, r) in enumerate(
        zip(delta_values, gaps_scan, r_scan)):
    if i % 4 == 0:
        phase = ("gapped" if abs(d) > 1.0
                 else "critical" if g < 0.05
                 else "gapped(small)")
        print(f"  {d:+10.4f}  {g:10.6f}  {r:8.4f}  {phase:>12s}")

# Отмечаем Коксетеровы точки
print(f"\n  Coxeter special points:")
for name, h in coxeter_algebras.items():
    d    = coxeter_delta(h)
    idx  = np.argmin(np.abs(delta_values - d))
    g    = gaps_scan[idx]
    r    = r_scan[idx]
    d_nn = delta_values[idx]
    print(f"    {name:4s} (h={h:2d}): "
          f"Δ={d:+.4f}→{d_nn:+.4f}, "
          f"gap={g:.6f}, r̄={r:.4f}")

# ════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 4: Entanglement entropy scaling
# ════════════════════════════════════════════════════════

print("\n[4] Entanglement entropy scaling with system size")
print("    (area law for gapped, log law for critical)")
print(f"{'n':>4s}  {'S_ent(E6)':>12s}  "
      f"{'S_ent(A6)':>12s}  {'S_ent(rnd)':>12s}")
print("-"*48)

delta_E6  = coxeter_delta(12)  # E6: h=12
delta_A6  = coxeter_delta(7)   # A6: h=7
delta_rnd = -0.7               # случайная точка

for n_s in [4, 6, 8, 10, 12]:
    # E6
    H_e6 = xxz_hamiltonian(n_s, delta_E6, pbc=False)
    _, ev_e6 = eigh(H_e6)
    S_e6 = entanglement_entropy(ev_e6[:,0], n_s)

    # A6
    H_a6 = xxz_hamiltonian(n_s, delta_A6, pbc=False)
    _, ev_a6 = eigh(H_a6)
    S_a6 = entanglement_entropy(ev_a6[:,0], n_s)

    # Random
    H_rnd = xxz_hamiltonian(n_s, delta_rnd, pbc=False)
    _, ev_rnd = eigh(H_rnd)
    S_rnd = entanglement_entropy(ev_rnd[:,0], n_s)

    print(f"  {n_s:4d}  {S_e6:12.6f}  {S_a6:12.6f}  {S_rnd:12.6f}")

# ════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 5: Dynkin-weighted chain
# ════════════════════════════════════════════════════════

print("\n[5] Dynkin-weighted XXZ chain")
print("    J_i from Dynkin diagram of E6")
print("    vs uniform J=1 vs random J")
print()

# Dynkin диаграмма E6:
#   1 - 3 - 4 - 5 - 6
#           |
#           2
# Связи: (1,3),(3,4),(4,5),(5,6),(4,2)
# J_i = метки Dynkin диаграммы (все = 1 для A-D-E)

# Для цепочки n=6: используем веса из матрицы Картана
# C_E6 diagonal = 2, off-diagonal = -1 для связанных

def cartan_matrix_E6():
    """Матрица Картана E6."""
    C = np.zeros((6,6))
    for i in range(6):
        C[i,i] = 2
    # Связи Dynkin E6:
    links = [(0,1),(1,2),(2,3),(3,4),(2,5)]
    for i,j in links:
        C[i,j] = -1; C[j,i] = -1
    return C

def cartan_matrix_A6():
    """Матрица Картана A6."""
    C = np.zeros((6,6))
    for i in range(6):
        C[i,i] = 2
    for i in range(5):
        C[i,i+1] = -1; C[i+1,i] = -1
    return C

n_s = 8  # спинов в цепочке

print(f"  Chain: n={n_s} spins, Δ=Δ(E6)={delta_E6:.4f}")
print()
print(f"  {'Config':20s}  {'gap':>10s}  {'r̄':>8s}  {'S_ent':>8s}")
print("  " + "-"*52)

configs = {
    "uniform_J=1":  np.ones(n_s-1),
    "E6_J∝|Cij|":  np.array([1,1,1,1,1,1,1]),  # все J=1 для ADE
    "random_J":     rng.uniform(0.5, 2.0, n_s-1),
    "Fibonacci_J":  np.array([(1+np.sqrt(5))/2]*7)[:n_s-1],
}

for cname, J_list in configs.items():
    H_c = heisenberg_hamiltonian(n_s, J_list[:n_s-1], pbc=False)
    evals_c = get_spectrum(H_c)
    _, evecs_c = eigh(H_c)
    gap_c = spectral_gap(evals_c)
    r_c   = level_spacing_stats(evals_c)
    S_c   = entanglement_entropy(evecs_c[:,0], n_s)
    print(f"  {cname:20s}  {gap_c:10.6f}  {r_c:8.4f}  {S_c:8.4f}")

# ════════════════════════════════════════════════════════
# ИТОГ
# ════════════════════════════════════════════════════════

print("\n" + "="*65)
print("SUMMARY: Part X Plan B preliminary results")
print("="*65)
print("""
  Key questions being tested:

  Q1: Does Δ(E6) give a special spectral gap?
      → Compare Coxeter vs random Δ

  Q2: Does entanglement entropy scale differently
      at Coxeter vs random Δ?
      → Area law (gapped) vs log law (critical CFT)

  Q3: Is there a phase transition at Δ(E6)?
      → Look for gap closing / level statistics change

  Q4: Does Dynkin-weighted chain give special spectrum?
      → E6 vs A6 vs random coupling pattern

  Physical motivation:
  At Δ = -cos(π/h), the XXZ chain has a quantum group
  symmetry U_q(SL(2)) with q = exp(iπ/h).
  This is the SAME q as in Coxeter representation theory.
  The connection is through Temperley-Lieb algebra.

  If Coxeter-specific Δ values give special entanglement
  scaling or phase transitions → physical content exists.
  If not → the monostring hypothesis has no realization
  in spin chain physics either.
""")
