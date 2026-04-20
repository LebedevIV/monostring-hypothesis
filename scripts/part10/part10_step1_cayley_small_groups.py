"""
Part X Step 1: Cayley graphs of Coxeter groups
Start small: A2(|W|=6), A3(24), B3(48), A4(120), B4(384)
Then: E6(51840) if feasible

H0: spectral gap λ₁(Cay(W_Coxeter)) is NOT special
    compared to random groups of same order and degree
"""

import numpy as np
from itertools import product
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════
# Представление групп Коксетера через перестановки
# ════════════════════════════════════════════════════════

def make_symmetric_group(n):
    """
    S_n как группа Коксетера A_{n-1}.
    Генераторы: транспозиции s_i = (i, i+1).
    |S_n| = n!
    """
    from itertools import permutations

    # Все перестановки
    elems = list(permutations(range(n)))
    idx   = {e: i for i, e in enumerate(elems)}

    def compose(p, q):
        return tuple(p[q[i]] for i in range(n))

    # Генераторы: транспозиции (i, i+1)
    generators = []
    for i in range(n-1):
        s = list(range(n))
        s[i], s[i+1] = s[i+1], s[i]
        generators.append(tuple(s))

    return elems, idx, generators, compose

def make_hyperoctahedral(n):
    """
    B_n = гипероктаэдральная группа (знаковые перестановки).
    |B_n| = 2^n · n!
    Генераторы: транспозиции + знаковое изменение первого элемента.
    """
    from itertools import permutations

    # Элементы: (перестановка, знаки)
    perms  = list(permutations(range(n)))
    signs_list = list(product([1,-1], repeat=n))
    elems  = [(p, s) for p in perms for s in signs_list]
    idx    = {e: i for i, e in enumerate(elems)}

    def compose(a, b):
        p_a, s_a = a
        p_b, s_b = b
        # (p_a, s_a) · (p_b, s_b): применяем b сначала
        p_c = tuple(p_a[p_b[i]] for i in range(n))
        s_c = tuple(s_a[p_b[i]] * s_b[i] for i in range(n))
        return (p_c, s_c)

    # Генераторы B_n:
    generators = []
    # s_0: меняем знак первого
    s0_perm  = tuple(range(n))
    s0_signs = tuple([-1 if i==0 else 1 for i in range(n)])
    generators.append((s0_perm, s0_signs))
    # s_i (i=1..n-1): транспозиция i-1, i
    for i in range(1, n):
        s_p = list(range(n)); s_p[i-1], s_p[i] = s_p[i], s_p[i-1]
        s_s = [1]*n
        generators.append((tuple(s_p), tuple(s_s)))

    return elems, idx, generators, compose

def cayley_graph(elems, idx, generators, compose):
    """
    Строит Кэли-граф как разреженную матрицу смежности.
    Генераторы и их инверсы — рёбра графа.
    """
    N    = len(elems)
    rows = []; cols = []

    for i, w in enumerate(elems):
        for s in generators:
            # найти ws
            ws = compose(w, s)
            j  = idx.get(ws)
            if j is not None:
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)

    # убрать дубли
    edges = set(zip(rows, cols))
    r, c  = zip(*edges) if edges else ([], [])
    data  = np.ones(len(r))
    A     = csr_matrix((data, (r, c)), shape=(N, N))
    return A

def normalized_laplacian_sparse(A):
    """Нормализованный лапласиан L = I - D^{-1/2} A D^{-1/2}."""
    deg  = np.array(A.sum(axis=1)).flatten()
    d    = np.where(deg > 0, deg, 1.0)
    D12  = 1.0 / np.sqrt(d)
    N    = A.shape[0]
    # L = I - D^{-1/2} A D^{-1/2}
    # Для разреженной: масштабируем строки и столбцы
    A_scaled = A.copy().astype(float)
    A_scaled = A_scaled.multiply(D12[:, None])
    A_scaled = A_scaled.multiply(D12[None, :])
    return A_scaled  # это D^{-1/2} A D^{-1/2}; λ Лапласа = 1 - λ_A

def spectral_gap(A, k_eigs=10):
    """
    Возвращает λ₁ (Fiedler), λ₂, spectral gap = λ₁,
    и отношение λ₁/λ_max.
    Для регулярного графа степени d:
      λ_Laplacian = 1 - λ_adjacency/d
      gap = λ₁_Laplacian = 1 - λ₂_adjacency/d
    """
    N   = A.shape[0]
    deg = np.array(A.sum(axis=1)).flatten()
    d   = deg[0]  # регулярный граф

    if N <= 500:
        # Точная диагонализация
        A_sc = normalized_laplacian_sparse(A)
        # L = I - A_sc
        L    = (csr_matrix(np.eye(N)) - A_sc).toarray()
        vals = np.sort(np.linalg.eigvalsh(L))
    else:
        # Частичная (только несколько малых собственных значений)
        A_sc = normalized_laplacian_sparse(A)
        # eigsh для наибольших собственных значений A_sc
        # (соответствуют наименьшим λ Лапласа)
        k    = min(k_eigs, N-2)
        vals_A, _ = eigsh(A_sc, k=k, which='LM')
        vals_L    = np.sort(1 - vals_A)[::-1]  # λ_L = 1 - λ_A
        vals      = np.sort(vals_L)

    nz   = np.where(vals > 1e-8)[0]
    if len(nz) == 0:
        return {"gap": 0, "lambda1": 0, "lambda2": 0,
                "lambda_max": vals[-1], "N": N, "degree": d}

    lam1 = vals[nz[0]]
    lam2 = vals[nz[1]] if len(nz) > 1 else lam1
    return {
        "gap":        lam1,
        "lambda1":    lam1,
        "lambda2":    lam2,
        "lambda_max": vals[-1],
        "ratio":      lam1 / vals[-1],
        "N":          N,
        "degree":     d,
    }

def random_regular_graph_spectrum(N, d, n_trials=20, rng=None):
    """
    Спектр случайного d-регулярного графа на N вершинах.
    Для сравнения с Кэли-графом.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gaps = []
    for _ in range(n_trials):
        # Конфигурационная модель для регулярного графа
        # (простой метод: случайное совершенное паросочетание)
        success = False
        for attempt in range(100):
            stubs = np.repeat(np.arange(N), d)
            rng.shuffle(stubs)
            rows = stubs[:len(stubs)//2*2:2]
            cols = stubs[1:len(stubs)//2*2:2]
            # проверить: нет петель, нет кратных рёбер
            edge_set = set()
            valid = True
            for r, c in zip(rows, cols):
                if r == c:
                    valid = False; break
                e = (min(r,c), max(r,c))
                if e in edge_set:
                    valid = False; break
                edge_set.add(e)
            if valid:
                success = True
                break
        if not success:
            continue

        r_arr = list(rows) + list(cols)
        c_arr = list(cols) + list(rows)
        data  = np.ones(len(r_arr))
        A_rnd = csr_matrix((data, (r_arr, c_arr)), shape=(N,N))
        sp    = spectral_gap(A_rnd)
        gaps.append(sp["gap"])

    return np.array(gaps)

# ════════════════════════════════════════════════════════
# ГЛАВНЫЙ ЭКСПЕРИМЕНТ
# ════════════════════════════════════════════════════════

print("="*65)
print("Part X Step 1: Cayley Graphs of Coxeter Groups")
print("="*65)

rng = np.random.default_rng(42)
results = {}

# ── A_n (Симметрические группы S_{n+1}) ───────────────
print("\n[1] Symmetric groups S_n = Coxeter A_{n-1}")
print(f"{'Group':10s} {'|W|':>7s} {'deg':>5s} "
      f"{'λ₁(gap)':>10s} {'λ_max':>8s} {'ratio':>8s}")
print("-"*52)

for n in [3, 4, 5, 6, 7]:
    elems, idx, gens, compose = make_symmetric_group(n)
    A = cayley_graph(elems, idx, gens, compose)
    sp = spectral_gap(A)
    name = f"S_{n}=A_{n-1}"
    results[name] = sp
    print(f"{name:10s} {sp['N']:7d} {int(sp['degree']):5d} "
          f"{sp['gap']:10.6f} {sp['lambda_max']:8.4f} "
          f"{sp['ratio']:8.6f}")

# ── B_n (Гипероктаэдральные группы) ───────────────────
print("\n[2] Hyperoctahedral groups B_n")
print(f"{'Group':10s} {'|W|':>7s} {'deg':>5s} "
      f"{'λ₁(gap)':>10s} {'λ_max':>8s} {'ratio':>8s}")
print("-"*52)

for n in [2, 3, 4]:
    elems, idx, gens, compose = make_hyperoctahedral(n)
    A = cayley_graph(elems, idx, gens, compose)
    sp = spectral_gap(A)
    name = f"B_{n}"
    results[name] = sp
    print(f"{name:10s} {sp['N']:7d} {int(sp['degree']):5d} "
          f"{sp['gap']:10.6f} {sp['lambda_max']:8.4f} "
          f"{sp['ratio']:8.6f}")

# ── Сравнение с random regular graph ──────────────────
print("\n[3] Random regular graphs (baseline)")
print(f"{'Config':15s} {'N':>7s} {'d':>4s} "
      f"{'gap_mean':>10s} {'gap_std':>9s} {'gap_p5':>8s}")
print("-"*56)

for N, d, label in [
    (6,   4,  "N=6,d=4"),
    (24,  6,  "N=24,d=6"),
    (120, 8,  "N=120,d=8"),
    (720, 10, "N=720,d=10"),
]:
    gaps = random_regular_graph_spectrum(N, d, n_trials=30, rng=rng)
    if len(gaps) > 0:
        print(f"{label:15s} {N:7d} {d:4d} "
              f"{gaps.mean():10.6f} {gaps.std():9.6f} "
              f"{np.percentile(gaps,5):8.6f}")

# ── Статистический тест ────────────────────────────────
print("\n[4] Spectral gap: Coxeter vs Random regular")
print("    (same N and degree)")
print()

# S_3 vs random N=6, d=4
elems3, idx3, gens3, compose3 = make_symmetric_group(3)
A3 = cayley_graph(elems3, idx3, gens3, compose3)
sp3 = spectral_gap(A3)
gaps3 = random_regular_graph_spectrum(6, int(sp3['degree']),
                                       n_trials=100, rng=rng)
pct3 = np.mean(gaps3 < sp3['gap']) * 100
print(f"  S_3 (A_2): gap={sp3['gap']:.6f}, "
      f"Random({int(sp3['degree'])}-reg): "
      f"mean={gaps3.mean():.6f}, "
      f"S_3 percentile={pct3:.1f}%")

# S_4 vs random N=24, d=6
elems4, idx4, gens4, compose4 = make_symmetric_group(4)
A4 = cayley_graph(elems4, idx4, gens4, compose4)
sp4 = spectral_gap(A4)
gaps4 = random_regular_graph_spectrum(24, int(sp4['degree']),
                                       n_trials=100, rng=rng)
pct4 = np.mean(gaps4 < sp4['gap']) * 100
print(f"  S_4 (A_3): gap={sp4['gap']:.6f}, "
      f"Random({int(sp4['degree'])}-reg): "
      f"mean={gaps4.mean():.6f}, "
      f"S_4 percentile={pct4:.1f}%")

# S_5 vs random N=120, d=8
elems5, idx5, gens5, compose5 = make_symmetric_group(5)
A5 = cayley_graph(elems5, idx5, gens5, compose5)
sp5 = spectral_gap(A5)
gaps5 = random_regular_graph_spectrum(120, int(sp5['degree']),
                                       n_trials=50, rng=rng)
pct5 = np.mean(gaps5 < sp5['gap']) * 100
print(f"  S_5 (A_4): gap={sp5['gap']:.6f}, "
      f"Random({int(sp5['degree'])}-reg): "
      f"mean={gaps5.mean():.6f}, "
      f"S_5 percentile={pct5:.1f}%")

# ── Теоретические предсказания ─────────────────────────
print("\n[5] Theoretical bounds")
print("""
  Alon–Boppana bound (random d-regular):
    λ₁ ≤ 1 - (2√(d-1))/d + o(1)  as N→∞

  Ramanujan graphs (optimal expanders):
    λ₁ ≥ 1 - (2√(d-1))/d  (Lubotzky-Phillips-Sarnak)

  For Cayley graphs of Coxeter groups:
    λ₁ determined by representation theory of W
    λ₁ = 1 - (max non-trivial representation eigenvalue)/d

  If Coxeter Cayley graphs are Ramanujan:
    → they are OPTIMAL expanders
    → deep connection to algebraic structure
""")

for d in [4, 6, 8, 10, 12]:
    alon = 1 - 2*np.sqrt(d-1)/d
    print(f"  d={d:2d}: Alon-Boppana λ₁ ≥ {alon:.6f}")

# ── Анализ собственного спектра S_4 ───────────────────
print("\n[6] Full spectrum analysis: S_4 = Cay(W(A_3), S)")
print("    (N=24, good size for full diagonalization)")

A4_arr = A4.toarray().astype(float)
deg4   = A4_arr.sum(axis=1)
d4     = deg4[0]
D12_4  = np.diag(1/np.sqrt(deg4))
L4     = np.eye(24) - D12_4 @ A4_arr @ D12_4
vals4  = np.sort(np.linalg.eigvalsh(L4))

print(f"\n  Degree d = {int(d4)}")
print(f"  Eigenvalues (Laplacian):")
unique_vals = np.unique(np.round(vals4, 6))
for v in unique_vals:
    mult = np.sum(np.abs(vals4 - v) < 1e-4)
    print(f"    λ = {v:.6f}  (multiplicity {mult})")

print(f"\n  Spectral gap λ₁ = {vals4[vals4 > 1e-8][0]:.6f}")
print(f"  Alon-Boppana bound: {1 - 2*np.sqrt(d4-1)/d4:.6f}")
is_ramanujan = vals4[vals4 > 1e-8][0] >= 1 - 2*np.sqrt(d4-1)/d4
print(f"  Is Ramanujan? {'YES ✓' if is_ramanujan else 'NO ✗'}")

# Мультиплетная структура ↔ представления группы
print(f"\n  Multiplicity structure encodes irreps of S_4:")
print(f"  Irreps of S_4: trivial(1), sign(1), std(3), std×sign(3), std²(2)")
print(f"  → expect eigenvalue multiplicities: 1, 2, 3, or 6")
