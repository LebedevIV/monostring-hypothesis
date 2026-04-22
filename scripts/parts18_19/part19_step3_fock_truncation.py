"""
Part XIX Step 3: Quantum Spectrum — ARPACK fix

Проблема: sigma=0 вырождена (E_0=0 точно).
Решение: всегда использовать полную диагонализацию
         для dim <= 15000, иначе sigma=-0.5.

Фокус:
  1. n_max convergence для E6 (полная диагонализация)
  2. Gap vs κ: E6 vs A6 vs Random (rank=6)
  3. E8 gap structure (n_max=1,2)
  4. Поиск нетривиального сигнала
"""

import numpy as np
from scipy.sparse import kron, eye, diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy import stats
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
    """
    H = Σᵢ ωᵢ·n̂ᵢ + κ·Σᵢ<ⱼ Cᵢⱼ·(aᵢ†aⱼ + aᵢaⱼ†)
    """
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

def get_spectrum(H, n_eigs=20):
    """
    Надёжный метод:
    - dim <= 15000: полная диагонализация (toarray + eigh)
    - dim > 15000: ARPACK с sigma = E_ground - 0.5

    НИКОГДА не используем sigma=0 (вырождение вакуума).
    """
    dim = H.shape[0]
    n_eigs = min(n_eigs, dim)

    if dim <= 15000:
        # Полная диагонализация — надёжно
        arr  = H.toarray()
        vals = eigh(arr, eigvals_only=True,
                    subset_by_index=[0, n_eigs-1])
        return np.sort(vals.real)
    else:
        # ARPACK с shift ниже вакуума
        # Сначала грубо оцениваем E_0
        try:
            v0 = eigsh(H, k=1, which='SM',
                       return_eigenvectors=False,
                       maxiter=5000, tol=1e-4)
            sigma = v0[0] - 0.5
        except Exception:
            sigma = -1.0

        try:
            vals = eigsh(H, k=min(n_eigs, dim-2),
                         sigma=sigma, which='LM',
                         return_eigenvectors=False,
                         maxiter=20000, tol=1e-8)
            return np.sort(vals.real)
        except Exception as e:
            print(f"    ARPACK failed: {e}")
            return None

def get_frequencies(C):
    eigs = np.linalg.eigvalsh(C)
    return np.sqrt(np.maximum(eigs, 0))

# ── МАТРИЦЫ КАРТАНА ──────────────────────────────────────────

def make_An(n):
    C = np.zeros((n, n))
    for i in range(n):
        C[i,i] = 2
        if i > 0:
            C[i,i-1] = C[i-1,i] = -1
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

# ── TEST 1: n_max СХОДИМОСТЬ E6 ──────────────────────────────

print("="*65)
print("Part XIX Step 3: Quantum Spectrum (ARPACK fixed)")
print("="*65)
print()
print("── Test 1: n_max convergence (E6, κ=1.0) ──")
print()

C_e6      = CARTAN['E6']
omegas_e6 = get_frequencies(C_e6)
kappa_t1  = 1.0

print(f"  E6 omegas: {np.round(omegas_e6, 4)}")
print()
print(f"  {'n_max':>6} {'dim':>8} {'gap_1':>10} "
      f"{'gap_2':>10} {'gap_3':>10} {'method':>8}")
print("  " + "-"*58)

prev_gap1 = None
conv_data = {}
for n_max in [1, 2, 3]:
    dim = (n_max+1)**6
    H   = build_hamiltonian(6, C_e6, omegas_e6,
                             kappa_t1, n_max)
    v   = get_spectrum(H, n_eigs=15)
    method = "full" if dim <= 15000 else "ARPACK"

    if v is None:
        print(f"  {n_max:>6} {dim:>8}  FAILED")
        continue

    E0    = v[0]
    gaps  = v - E0
    g1    = gaps[1] if len(gaps) > 1 else np.nan
    g2    = gaps[2] if len(gaps) > 2 else np.nan
    g3    = gaps[3] if len(gaps) > 3 else np.nan

    stable = ""
    if prev_gap1 is not None and not np.isnan(g1):
        rc = abs(g1 - prev_gap1)/(abs(prev_gap1)+1e-12)
        stable = f"Δ={rc:.1%}"
    prev_gap1 = g1
    conv_data[n_max] = {'dim': dim, 'gaps': gaps}

    print(f"  {n_max:>6} {dim:>8} {g1:>10.5f} "
          f"{g2:>10.5f} {g3:>10.5f} {method:>8}  {stable}")

print()

# ── TEST 2: gap vs κ (E6 vs A6 vs Random) ────────────────────

print("── Test 2: Gap vs κ — E6 vs A6 vs Random (n_max=2) ──")
print()

n_max_2  = 2
kappas_2 = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# Генерируем случайные матрицы ранга 6
rng = np.random.RandomState(SEED)
rand_Cs = []
for s in range(5):
    rng2  = np.random.RandomState(s*7+1)
    A     = rng2.randn(6,6)
    Cr    = (A + A.T)/2
    # Нормируем: диагональ=2, |offdiag|<=1
    D     = np.diag(np.diag(Cr))
    Cr    = Cr - D + 2*np.eye(6)
    scale = np.abs(Cr[~np.eye(6, dtype=bool)]).max()
    if scale > 0:
        Cr[~np.eye(6, dtype=bool)] /= scale
    rand_Cs.append(Cr)

results_gap = {'E6': [], 'A6': [], 'Random_mean': [],
               'Random_std': []}

print(f"  {'κ':>5} {'E6':>10} {'A6':>10} "
      f"{'Rnd mean':>10} {'Rnd std':>10} {'E6/Rnd':>8}")
print("  " + "-"*58)

for kappa in kappas_2:
    # E6
    H_e6 = build_hamiltonian(6, C_e6, omegas_e6,
                              kappa, n_max_2)
    v_e6 = get_spectrum(H_e6, n_eigs=8)
    g_e6 = (v_e6[1]-v_e6[0]) if (v_e6 is not None
                                   and len(v_e6)>1) else np.nan

    # A6
    C_a6     = CARTAN['A6']
    om_a6    = get_frequencies(C_a6)
    H_a6     = build_hamiltonian(6, C_a6, om_a6,
                                  kappa, n_max_2)
    v_a6     = get_spectrum(H_a6, n_eigs=8)
    g_a6     = (v_a6[1]-v_a6[0]) if (v_a6 is not None
                                       and len(v_a6)>1) else np.nan

    # Random (5 реализаций)
    g_rand = []
    for Cr in rand_Cs:
        om_r  = get_frequencies(Cr)
        H_r   = build_hamiltonian(6, Cr, om_r,
                                   kappa, n_max_2)
        v_r   = get_spectrum(H_r, n_eigs=8)
        if v_r is not None and len(v_r) > 1:
            g_rand.append(v_r[1]-v_r[0])
    g_rand = np.array(g_rand)
    gr_mean = g_rand.mean() if len(g_rand) else np.nan
    gr_std  = g_rand.std()  if len(g_rand) else np.nan

    ratio = g_e6/gr_mean if gr_mean > 1e-8 else np.nan

    results_gap['E6'].append(g_e6)
    results_gap['A6'].append(g_a6)
    results_gap['Random_mean'].append(gr_mean)
    results_gap['Random_std'].append(gr_std)

    print(f"  {kappa:>5.1f} {g_e6:>10.5f} {g_a6:>10.5f} "
          f"{gr_mean:>10.5f} {gr_std:>10.5f} {ratio:>8.3f}")

print()

# ── TEST 3: E8 при n_max=1,2 ────────────────────────────────

print("── Test 3: E8 spectrum (n_max=1 full, n_max=2 full) ──")
print()

C_e8      = CARTAN['E8']
omegas_e8 = get_frequencies(C_e8)
print(f"  E8 omegas: {np.round(omegas_e8, 4)}")
print()

e8_data = {}
kappas_e8 = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

print("  n_max=1 (dim=256):")
print(f"  {'κ':>6} {'gap_1':>10} {'gap_2':>10} "
      f"{'gap_3':>10} {'Δgap=g2-g1':>12}")
print("  " + "-"*52)

for kappa in kappas_e8:
    H   = build_hamiltonian(8, C_e8, omegas_e8,
                             kappa, n_max=1)
    v   = get_spectrum(H, n_eigs=12)
    if v is None:
        continue
    E0   = v[0]
    gaps = v - E0
    g1   = gaps[1]; g2 = gaps[2]; g3 = gaps[3]
    dg   = g2 - g1
    e8_data[kappa] = gaps[:12]
    print(f"  {kappa:>6.1f} {g1:>10.5f} "
          f"{g2:>10.5f} {g3:>10.5f} {dg:>12.5f}")

print()

# n_max=2 для E8 (dim=6561)
print("  n_max=2 (dim=6561) — проверка сходимости:")
kappas_e8_short = [0.0, 1.0, 5.0]
for kappa in kappas_e8_short:
    H   = build_hamiltonian(8, C_e8, omegas_e8,
                             kappa, n_max=2)
    v   = get_spectrum(H, n_eigs=10)
    if v is None:
        print(f"  κ={kappa}: FAILED")
        continue
    gaps = v - v[0]
    g1   = gaps[1]; g2 = gaps[2]
    print(f"  κ={kappa:.1f}: gap_1={g1:.5f}, "
          f"gap_2={g2:.5f}")

print()

# ── TEST 4: СТАТИСТИКА — СЛУЧАЙНЫЙ vs КОКСЕТЕР ───────────────

print("── Test 4: Statistical test at κ=1.0 (N=20 random) ──")
print()
print("  Генерируем N=20 случайных rank-6 матриц...")

kappa_stat = 1.0
n_max_stat = 2

# Коксетерові алгебры ранга 6
coxeter_names = ['E6', 'A6']
coxeter_gaps_stat = []
for name in coxeter_names:
    C   = CARTAN[name]
    om  = get_frequencies(C)
    H   = build_hamiltonian(6, C, om,
                             kappa_stat, n_max_stat)
    v   = get_spectrum(H, n_eigs=5)
    if v is not None and len(v) > 1:
        coxeter_gaps_stat.append(v[1]-v[0])
        print(f"  {name}: gap_1 = {v[1]-v[0]:.5f}")

# Случайные контроли (N=20)
random_gaps_stat = []
print()
print(f"  {'Seed':>6} {'gap_1':>10}  "
      f"{'Seed':>6} {'gap_1':>10}")
print("  " + "-"*36)

for s in range(20):
    rng2 = np.random.RandomState(s*13+7)
    A    = rng2.randn(6,6)
    Cr   = (A + A.T)/2
    Cr   = Cr - np.diag(np.diag(Cr)) + 2*np.eye(6)
    scale = np.abs(Cr[~np.eye(6,dtype=bool)]).max()
    if scale > 0:
        Cr[~np.eye(6,dtype=bool)] /= scale
    om_r = get_frequencies(Cr)
    H_r  = build_hamiltonian(6, Cr, om_r,
                              kappa_stat, n_max_stat)
    v_r  = get_spectrum(H_r, n_eigs=5)
    if v_r is not None and len(v_r) > 1:
        g = v_r[1]-v_r[0]
        random_gaps_stat.append(g)
        # Печатаем по два в строку
        if s % 2 == 0:
            print(f"  {s:>6} {g:>10.5f}", end='')
        else:
            print(f"  {s:>6} {g:>10.5f}")

if len(random_gaps_stat) % 2 == 1:
    print()

print()

coxeter_arr = np.array(coxeter_gaps_stat)
random_arr  = np.array(random_gaps_stat)

print(f"  Coxeter gaps: {np.round(coxeter_arr, 5)}")
print(f"  Coxeter mean: {coxeter_arr.mean():.5f} "
      f"± {coxeter_arr.std():.5f}")
print(f"  Random  mean: {random_arr.mean():.5f} "
      f"± {random_arr.std():.5f}")
print()

# Mann-Whitney (нет нормальности)
if len(coxeter_arr) >= 2 and len(random_arr) >= 5:
    stat, p_mw = stats.mannwhitneyu(
        coxeter_arr, random_arr,
        alternative='two-sided')
    print(f"  Mann-Whitney p = {p_mw:.4f}")

    # Percentile Coxeter в random distribution
    for name, g in zip(coxeter_names, coxeter_arr):
        pct = np.mean(random_arr < g) * 100
        print(f"  {name} at {pct:.0f}th percentile "
              f"of random (gap={g:.5f})")
    print()

    if p_mw < 0.05:
        print("  *** p<0.05: Coxeter gaps ≠ Random ***")
        verdict = "SIGNAL"
    else:
        verdict = "H0_HOLDS"
        print("  p>=0.05: H₀ holds")
else:
    verdict = "INSUFFICIENT_DATA"
    p_mw    = np.nan
    print("  Insufficient data for test")

print()

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Part XIX Step 3: Quantum Spectrum — '
    'Gap Analysis (ARPACK fixed)\n'
    'Fock space truncation, κ-dependence, '
    'Coxeter vs Random',
    fontsize=12)

kv = np.array(kappas_2)

# 0,0: gap_1 vs κ (E6, A6, Random)
ax = axes[0,0]
g_e6  = np.array(results_gap['E6'])
g_a6  = np.array(results_gap['A6'])
gr_m  = np.array(results_gap['Random_mean'])
gr_s  = np.array(results_gap['Random_std'])

valid_e6  = ~np.isnan(g_e6)
valid_a6  = ~np.isnan(g_a6)
valid_rnd = ~np.isnan(gr_m)

ax.plot(kv[valid_e6], g_e6[valid_e6], 'o-',
        color='#e74c3c', lw=2, ms=8, label='E6')
ax.plot(kv[valid_a6], g_a6[valid_a6], 's--',
        color='#3498db', lw=2, ms=8, label='A6')
ax.fill_between(kv[valid_rnd],
                (gr_m-gr_s)[valid_rnd],
                (gr_m+gr_s)[valid_rnd],
                alpha=0.2, color='gray')
ax.plot(kv[valid_rnd], gr_m[valid_rnd], '^:',
        color='gray', lw=2, ms=8,
        label='Random (mean±std)')
ax.set_xlabel('κ')
ax.set_ylabel('gap_1 = E_1 - E_0')
ax.set_title('First excitation gap vs κ\n(n_max=2, rank=6)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 0,1: E8 gap structure (n_max=1)
ax = axes[0,1]
kv_e8 = kappas_e8
for i, kappa in enumerate(kv_e8):
    if kappa not in e8_data:
        continue
    gaps = e8_data[kappa]
    ax.plot(range(min(10, len(gaps))),
            gaps[:10], 'o-',
            lw=1.5, ms=5,
            label=f'κ={kappa}',
            alpha=0.8)
ax.set_xlabel('Level n')
ax.set_ylabel('E_n - E_0')
ax.set_title('E8 spectrum (n_max=1, dim=256)\nGap structure vs κ')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 0,2: n_max convergence bar chart
ax = axes[0,2]
if conv_data:
    nmax_vals = sorted(conv_data.keys())
    gap1_vals = []
    for nm in nmax_vals:
        g = conv_data[nm]['gaps']
        gap1_vals.append(g[1] if len(g)>1 else np.nan)

    colors_nm = ['#3498db','#e74c3c','#2ecc71','#9b59b6']
    bars = ax.bar([f'n_max={nm}' for nm in nmax_vals],
                  gap1_vals,
                  color=colors_nm[:len(nmax_vals)],
                  alpha=0.8)
    for bar, val in zip(bars, gap1_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height(),
                    f'{val:.5f}', ha='center',
                    va='bottom', fontsize=9)

ax.set_ylabel('gap_1 (E6, κ=1.0)')
ax.set_title('n_max convergence (E6, κ=1.0)\n'
             'Artifact or real gap?')
ax.grid(True, alpha=0.3, axis='y')

# 1,0: Random vs Coxeter distribution
ax = axes[1,0]
ax.hist(random_arr, bins=12, color='gray',
        alpha=0.6, density=True,
        label=f'Random N=20\n'
              f'({random_arr.mean():.4f}'
              f'±{random_arr.std():.4f})')
for name, g in zip(coxeter_names, coxeter_arr):
    ax.axvline(g, lw=2.5,
               label=f'{name}: {g:.4f}')
ax.set_xlabel('gap_1')
ax.set_ylabel('Density')
ax.set_title(f'Coxeter vs Random distribution\n'
             f'(κ={kappa_stat}, n_max={n_max_stat})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1,1: E8 gap_1 vs κ — scaling
ax = axes[1,1]
kv_e8_arr  = np.array(kappas_e8)
gap1_e8    = np.array([e8_data[k][1]
                        for k in kappas_e8
                        if k in e8_data])
kv_e8_plot = np.array([k for k in kappas_e8
                         if k in e8_data])

ax.plot(kv_e8_plot, gap1_e8, 'o-',
        color='#9b59b6', lw=2, ms=10,
        label='E8 gap_1')

# Разметка режимов
if len(kv_e8_plot) > 2:
    ax.axvline(kv_e8_plot[len(kv_e8_plot)//2],
               color='k', ls='--', alpha=0.4,
               label='κ transition?')

ax.set_xlabel('κ')
ax.set_ylabel('gap_1')
ax.set_title('E8: gap_1 vs κ\n(n_max=1, dim=256)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1,2: Итоговый вердикт
ax = axes[1,2]
ax.axis('off')

lines = [
    'PART XIX STEP 3 SUMMARY',
    '─'*32,
    '',
    'Fix: полная диагонализация',
    '     вместо sigma=0 shift-invert',
    '',
    'Test 1 — n_max convergence (E6):',
]
for nm, d in conv_data.items():
    g = d['gaps']
    g1 = g[1] if len(g)>1 else np.nan
    lines.append(f'  n_max={nm}: gap_1={g1:.5f}')

lines += [
    '',
    'Test 2 — gap vs κ:',
    f'  E6  at κ=1.0: '
    f'{g_e6[kappas_2.index(1.0)]:.5f}',
    f'  Rnd at κ=1.0: '
    f'{gr_m[kappas_2.index(1.0)]:.5f}',
    '',
    'Test 3 — E8 (n_max=1):',
]
for kappa in [0.0, 1.0, 5.0]:
    if kappa in e8_data:
        g = e8_data[kappa][1]
        lines.append(f'  κ={kappa}: gap_1={g:.5f}')

lines += [
    '',
    'Test 4 — Statistics:',
    f'  Coxeter: '
    f'{coxeter_arr.mean():.4f}±'
    f'{coxeter_arr.std():.4f}',
    f'  Random:  '
    f'{random_arr.mean():.4f}±'
    f'{random_arr.std():.4f}',
    f'  p(MW) = {p_mw:.4f}',
    '',
    'VERDICT:',
]

if verdict == "SIGNAL":
    color = '#e0ffe0'
    lines += ['*** p<0.05: СИГНАЛ ***',
              '→ Проверить механизм (Part XX)',
              '→ N=100 контролей']
elif verdict == "H0_HOLDS":
    color = '#ffe0e0'
    lines += ['H₀ не отвергается.',
              '→ Квантовая версия тоже',
              '   не даёт сигнала.',
              '→ Публикуем v14.0.0']
else:
    color = '#fff3cd'
    lines += ['Недостаточно данных.',
              '→ Увеличить N']

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8.5,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('part19_step3_fixed.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part19_step3_fixed.png")

# ── ФИНАЛЬНЫЙ ВЕРДИКТ ────────────────────────────────────────

print()
print("="*65)
print("PART XIX STEP 3 — VERDICT")
print("="*65)
print()

if verdict == "SIGNAL":
    print("*** СИГНАЛ: Coxeter gaps статистически")
    print("    отличаются от Random (p<0.05) ***")
    print()
    print("Следующий шаг:")
    print("  1. N=100 случайных контролей")
    print("  2. Проверить механизм")
    print("  3. E8 при n_max=2 (dim=6561)")
elif verdict == "H0_HOLDS":
    print("H₀ не отвергается.")
    print()
    print("Квантовая версия гамильтониана")
    print("H = Σωᵢn̂ᵢ + κΣCᵢⱼ(aᵢ†aⱼ + h.c.)")
    print("не производит Coxeter-специфичных")
    print("сигналов в gap структуре.")
    print()
    print("ИТОГ Parts I-XIX:")
    print("  Классика (I-XVII):   фальсифицирована")
    print("  Квантовая (XVIII-XIX): фальсифицирована")
    print()
    print("→ Рекомендация: публиковать v14.0.0")
else:
    print("Результаты неопределённые.")
    print("Нужно больше контролей.")
