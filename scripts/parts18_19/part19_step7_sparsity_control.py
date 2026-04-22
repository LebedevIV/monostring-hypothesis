"""
Part XIX Step 7: Sparsity-controlled final test for E8

E8 имеет 7 ненулевых off-diagonal элементов из 28.
Ctrl-B имел ~14 в среднем → несправедливо.

Финальный тест:
  Ctrl-D: ровно 7 off-diagonal = -1, остальные = 0
          diagonal = 2, только PSD
          (точно соответствует структуре E8)

Если p(D) > 0.05 → E8 сигнал = артефакт спарсности
Если p(D) < 0.05 → реальный сигнал, требует Part XX
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ── КАРТАНОВА МАТРИЦА E8 ──────────────────────────────────────

C_E8 = np.array([
    [ 2,-1, 0, 0, 0, 0, 0, 0],
    [-1, 2,-1, 0, 0, 0, 0, 0],
    [ 0,-1, 2,-1, 0, 0, 0,-1],
    [ 0, 0,-1, 2,-1, 0, 0, 0],
    [ 0, 0, 0,-1, 2,-1, 0, 0],
    [ 0, 0, 0, 0,-1, 2,-1, 0],
    [ 0, 0, 0, 0, 0,-1, 2, 0],
    [ 0, 0,-1, 0, 0, 0, 0, 2]], dtype=float)

# Статистика E8
r8       = 8
pairs_8  = [(i,j) for i in range(r8)
             for j in range(i+1, r8)]
n_pairs  = len(pairs_8)  # = 28
nonzero_E8 = [(i,j) for i,j in pairs_8
               if C_E8[i,j] != 0]
n_nz_E8  = len(nonzero_E8)  # = 7

print("="*65)
print("Part XIX Step 7: Sparsity-Controlled E8 Test")
print("="*65)
print()
print(f"E8 structure:")
print(f"  rank = {r8}")
print(f"  off-diagonal pairs: {n_pairs}")
print(f"  nonzero off-diagonal: {n_nz_E8}")
print(f"  sparsity: {n_nz_E8}/{n_pairs} = "
      f"{n_nz_E8/n_pairs:.3f}")
print(f"  all off-diagonal values = -1 (no +1 in E8)")
print()

# ── УТИЛИТЫ ──────────────────────────────────────────────────

def safe_eigvalsh(M):
    try:
        if not np.all(np.isfinite(M)):
            return None
        v = np.linalg.eigvalsh(M)
        return v if np.all(np.isfinite(v)) else None
    except np.linalg.LinAlgError:
        return None

def normal_modes(C, kappa):
    eigs = safe_eigvalsh(C)
    if eigs is None:
        return None
    om = np.sqrt(np.maximum(eigs, 0))
    r  = C.shape[0]
    M  = np.diag(om)
    for i in range(r):
        for j in range(r):
            if i != j:
                M[i,j] += kappa * C[i,j]
    return safe_eigvalsh(M)

def find_kc(C, n_pts=500):
    for kappa in np.linspace(0, 5, n_pts):
        f = normal_modes(C, kappa)
        if f is not None and f[0] < -1e-8:
            return kappa
    return 5.0

def gap_frac(C, frac=0.8):
    kc    = find_kc(C)
    f     = normal_modes(C, frac*kc)
    if f is None:
        return None, kc
    pos   = f[f > 1e-8]
    return (pos.min() if len(pos) > 0
            else 0.0), kc

# ── E8 БАЗОВЫЕ ХАРАКТЕРИСТИКИ ────────────────────────────────

g_E8, kc_E8 = gap_frac(C_E8)
print(f"E8: κ_c={kc_E8:.4f}, "
      f"gap(0.8κ_c)={g_E8:.5f}")
print()

# ── ДИАГНОСТИКА: СПАРСНОСТЬ В CTRL-B ─────────────────────────

print("── Sparsity diagnosis in Ctrl-B ──")
print()

n_nz_B_list = []
rng_d = np.random.RandomState(42)
for s in range(500):
    C = 2.0 * np.eye(8)
    for i in range(8):
        for j in range(i+1, 8):
            v = rng_d.choice([-1, 0, 0, 1])
            C[i,j] = C[j,i] = float(v)
    eigs = safe_eigvalsh(C)
    if eigs is not None and eigs[0] >= -1e-10:
        nz = np.sum(np.abs(C[np.triu_indices(8,1)]) > 0.5)
        n_nz_B_list.append(nz)

n_nz_B = np.array(n_nz_B_list)
print(f"  Ctrl-B nonzero off-diag: "
      f"{n_nz_B.mean():.1f} ± {n_nz_B.std():.1f}")
print(f"  E8 nonzero off-diag:     {n_nz_E8}")
print(f"  Ratio: {n_nz_B.mean()/n_nz_E8:.2f}× "
      f"(Ctrl-B is denser)")
print()
if n_nz_B.mean() > n_nz_E8 * 1.2:
    print("  *** SPARSITY MISMATCH: Ctrl-B дenser than E8 ***")
    print("  → p_B=0.015 может быть артефактом спарсности")
print()

# ── ГЕНЕРАЦИЯ CTRL-D (ТОЧНОЕ СОВПАДЕНИЕ СПАРСНОСТИ) ──────────

def ctrl_D(seed, n_nonzero=7, rank=8,
           val=-1.0, max_tries=200):
    """
    Ctrl-D: точно n_nonzero off-diagonal элементов,
    все равны val (-1, как в E8).
    diagonal = 2 точно. Только PSD.
    """
    rng   = np.random.RandomState(seed)
    pairs = [(i,j) for i in range(rank)
              for j in range(i+1, rank)]

    for _ in range(max_tries):
        C      = 2.0 * np.eye(rank)
        chosen = rng.choice(len(pairs),
                             n_nonzero,
                             replace=False)
        for idx in chosen:
            i, j   = pairs[idx]
            C[i,j] = C[j,i] = val
        eigs = safe_eigvalsh(C)
        if eigs is not None and eigs[0] >= -1e-10:
            return C
    return None

# ── CTRL-E: СПАРСНОСТЬ + СЛУЧАЙНЫЕ ЗНАКИ ─────────────────────

def ctrl_E(seed, n_nonzero=7, rank=8,
           max_tries=200):
    """
    Ctrl-E: n_nonzero off-diagonal,
    значения ∈ {-1, +1} случайно.
    E8 имеет только -1, но проверим и +1.
    """
    rng   = np.random.RandomState(seed)
    pairs = [(i,j) for i in range(rank)
              for j in range(i+1, rank)]

    for _ in range(max_tries):
        C      = 2.0 * np.eye(rank)
        chosen = rng.choice(len(pairs),
                             n_nonzero,
                             replace=False)
        for idx in chosen:
            i, j   = pairs[idx]
            v      = rng.choice([-1.0, 1.0])
            C[i,j] = C[j,i] = v
        eigs = safe_eigvalsh(C)
        if eigs is not None and eigs[0] >= -1e-10:
            return C
    return None

# ── СБОР ДАННЫХ ───────────────────────────────────────────────

print("── Generating controls D & E (N=500) ──")
print()

N = 500
gaps_D = []
gaps_E = []
kcs_D  = []
kcs_E  = []

for s in range(N * 10):
    if len(gaps_D) >= N and len(gaps_E) >= N:
        break

    if len(gaps_D) < N:
        C = ctrl_D(seed=s, n_nonzero=n_nz_E8)
        if C is not None:
            g, kc = gap_frac(C)
            if g is not None:
                gaps_D.append(g)
                kcs_D.append(kc)

    if len(gaps_E) < N:
        C = ctrl_E(seed=s+100000, n_nonzero=n_nz_E8)
        if C is not None:
            g, kc = gap_frac(C)
            if g is not None:
                gaps_E.append(g)
                kcs_E.append(kc)

gaps_D = np.array(gaps_D[:N])
gaps_E = np.array(gaps_E[:N])
kcs_D  = np.array(kcs_D[:N])
kcs_E  = np.array(kcs_E[:N])

print(f"  Ctrl-D (nz={n_nz_E8}, val=-1): "
      f"N={len(gaps_D)}, "
      f"gap={gaps_D.mean():.5f}±{gaps_D.std():.5f}")
print(f"  Ctrl-E (nz={n_nz_E8}, val=±1): "
      f"N={len(gaps_E)}, "
      f"gap={gaps_E.mean():.5f}±{gaps_E.std():.5f}")
print(f"  E8:                             "
      f"gap={g_E8:.5f}")
print()

# ── СТАТИСТИЧЕСКИЕ ТЕСТЫ ─────────────────────────────────────

print("── Final statistical tests ──")
print()

test_results = {}
for cname, cgaps in [('D', gaps_D), ('E', gaps_E)]:
    if len(cgaps) < 20:
        print(f"  vs Ctrl-{cname}: insufficient data")
        test_results[cname] = {'p': np.nan,
                                'pct': np.nan}
        continue

    pct  = float(np.mean(cgaps < g_E8) * 100)
    _, p = stats.mannwhitneyu(
               [g_E8], cgaps,
               alternative='two-sided')
    p    = float(p)

    # Дополнительно: позиция E8 в распределении
    z = ((g_E8 - cgaps.mean()) /
          (cgaps.std() + 1e-12))

    sig = "*** SIGNAL" if p < 0.05 else "ns (H0)"

    print(f"  E8 vs Ctrl-{cname}: "
          f"pct={pct:.0f}%, "
          f"p={p:.4f}, "
          f"z={z:.2f} → {sig}")
    print(f"    E8 gap={g_E8:.5f} vs "
          f"ctrl mean={cgaps.mean():.5f}"
          f"±{cgaps.std():.5f}")

    test_results[cname] = {
        'p': p, 'pct': pct,
        'z': z, 'gaps': cgaps
    }

print()

# ── АНАЛИЗ: ЧТО ОСОБЕННОГО В E8? ─────────────────────────────

print("── What makes E8 special (if anything)? ──")
print()

# Топологическая структура графа связей
# E8: Dynkin diagram имеет ветвь (node 3 connects to 8)
# Случайные с той же спарсностью: разные топологии

def graph_props(C):
    """Базовые свойства графа связей."""
    r     = C.shape[0]
    A     = (np.abs(C) > 0.5).astype(float)
    np.fill_diagonal(A, 0)
    degrees   = A.sum(axis=1)
    # Связность
    # Простой BFS
    visited   = set([0])
    queue     = [0]
    while queue:
        node = queue.pop(0)
        for nb in range(r):
            if A[node,nb] > 0 and nb not in visited:
                visited.add(nb)
                queue.append(nb)
    connected = (len(visited) == r)
    # Диаметр (приближённо)
    max_deg   = degrees.max()
    min_deg   = degrees.min()
    return {
        'connected':  connected,
        'max_degree': max_deg,
        'min_degree': min_deg,
        'mean_degree': degrees.mean(),
        'has_branch':  max_deg >= 3,
    }

e8_props = graph_props(C_E8)
print(f"  E8 graph: {e8_props}")
print()

# Статистика по Ctrl-D
props_D = [graph_props(ctrl_D(s, n_nz_E8))
            for s in range(200)
            if ctrl_D(s, n_nz_E8) is not None]

connected_D = np.mean([p['connected']
                         for p in props_D])
branch_D    = np.mean([p['has_branch']
                         for p in props_D])
print(f"  Ctrl-D: connected={connected_D:.1%}, "
      f"has_branch={branch_D:.1%}")
print(f"  E8:     connected=True, has_branch=True")
print()

# Корреляция: связность → gap
gaps_conn    = []
gaps_disconn = []
for i, p in enumerate(props_D):
    if i < len(gaps_D):
        if p['connected']:
            gaps_conn.append(gaps_D[i])
        else:
            gaps_disconn.append(gaps_D[i])

if gaps_conn and gaps_disconn:
    print(f"  Connected Ctrl-D:    "
          f"gap={np.mean(gaps_conn):.5f}"
          f"±{np.std(gaps_conn):.5f} "
          f"(N={len(gaps_conn)})")
    print(f"  Disconnected Ctrl-D: "
          f"gap={np.mean(gaps_disconn):.5f}"
          f"±{np.std(gaps_disconn):.5f} "
          f"(N={len(gaps_disconn)})")
    print()
    print(f"  E8 gap={g_E8:.5f}")
    print()

    # Тест только против связных
    gaps_conn_arr = np.array(gaps_conn)
    if len(gaps_conn_arr) >= 20:
        pct_c = float(np.mean(gaps_conn_arr < g_E8)*100)
        _, p_c = stats.mannwhitneyu(
                     [g_E8], gaps_conn_arr,
                     alternative='two-sided')
        print(f"  E8 vs Connected Ctrl-D: "
              f"pct={pct_c:.0f}%, p={float(p_c):.4f}")
        test_results['D_connected'] = {
            'p': float(p_c), 'pct': pct_c,
            'gaps': gaps_conn_arr}

# ── ФИНАЛЬНЫЙ ВЕРДИКТ ─────────────────────────────────────────

print()
print("="*65)
print("PART XIX — DEFINITIVE VERDICT")
print("="*65)
print()

p_D = test_results.get('D', {}).get('p', np.nan)
p_E = test_results.get('E', {}).get('p', np.nan)
p_Dc = test_results.get('D_connected',
                          {}).get('p', np.nan)

print("Summary of all E8 tests:")
print(f"  Ctrl-A (diag>2):         p=0.0066 *** ARTIFACT")
print(f"  Ctrl-B (diag=2, dense):  p=0.0155 ***")
print(f"  Ctrl-C (sparsity≈7):     p=0.1034 ns")
print(f"  Ctrl-D (sparsity=7,v=-1):p={p_D:.4f} "
      f"{'***' if p_D<0.05 else 'ns'}")
print(f"  Ctrl-E (sparsity=7,v=±1):p={p_E:.4f} "
      f"{'***' if p_E<0.05 else 'ns'}")
if not np.isnan(p_Dc):
    print(f"  Ctrl-D connected only:   p={p_Dc:.4f} "
          f"{'***' if p_Dc<0.05 else 'ns'}")
print()

# Решение
p_vals_fair = [p for p in [p_D, p_E, p_Dc]
                if not np.isnan(p)]
n_sig_fair  = sum(1 for p in p_vals_fair if p < 0.05)

if n_sig_fair == 0:
    conclusion = "ARTIFACT"
    color_txt  = "красный"
    print("ВЫВОД: E8 сигнал — артефакт спарсности.")
    print()
    print("Объяснение:")
    print("  Ctrl-B: плотные матрицы (≈14 нз) → большой gap")
    print("  E8:     разреженная (7 нз) → малый gap")
    print("  При равной спарсности (Ctrl-D, C, E): p>0.05")
    print()
    print("  gap(E8) мал потому что E8 разреженна,")
    print("  а не потому что E8 — алгебра Ли.")
    print()
    print("  АРТЕФАКТ #11: Sparsity mismatch")
    print("  (добавить в каталог артефактов)")
    print()
    print("╔══════════════════════════════════════════╗")
    print("║  Parts I–XIX: COMPLETE FALSIFICATION     ║")
    print("║                                          ║")
    print("║  19 experiments → 0 physical signals     ║")
    print("║  11 artifacts documented                 ║")
    print("║  6 mathematical theorems                 ║")
    print("║                                          ║")
    print("║  Classical:  falsified (Parts I–XVII)    ║")
    print("║  Quantum:    falsified (Parts XVIII–XIX) ║")
    print("║                                          ║")
    print("║  → PUBLISH v14.0.0                      ║")
    print("╚══════════════════════════════════════════╝")
elif n_sig_fair >= 2:
    conclusion = "SIGNAL"
    print("*** СИГНАЛ ВЫЖИЛ в честных контролях! ***")
    print("→ Требует Part XX: механизм и N=1000 тест")
else:
    conclusion = "MARGINAL"
    print("Маргинальный результат (1 из 3 тестов).")
    print("→ Увеличить N до 1000 для решения")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Part XIX Step 7: Sparsity-Controlled E8 Test\n'
    'Final verdict on quantum monostring hypothesis',
    fontsize=13)

col_D = '#2ecc71'
col_E = '#f39c12'
col_B = '#3498db'
col_C = '#9b59b6'
col_E8 = '#e74c3c'

# 0,0: Спарсность в контролях
ax = axes[0,0]
ax.hist(n_nz_B_list, bins=15,
        color=col_B, alpha=0.7,
        label=f'Ctrl-B: {np.mean(n_nz_B_list):.1f}'
              f'±{np.std(n_nz_B_list):.1f}')
ax.axvline(n_nz_E8, color=col_E8, lw=3,
           label=f'E8: exactly {n_nz_E8}')
ax.set_xlabel('# nonzero off-diagonal')
ax.set_title('Sparsity mismatch:\nCtrl-B vs E8')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 0,1: Gap distributions (all controls)
ax = axes[0,1]
for gaps, col, label in [
        (gaps_D, col_D,
         f'Ctrl-D nz={n_nz_E8},v=-1\n'
         f'μ={gaps_D.mean():.4f}'),
        (gaps_E, col_E,
         f'Ctrl-E nz={n_nz_E8},v=±1\n'
         f'μ={gaps_E.mean():.4f}')]:
    if len(gaps) > 5:
        ax.hist(gaps, bins=30, color=col,
                alpha=0.5, density=True,
                label=label)
ax.axvline(g_E8, color=col_E8, lw=3,
           label=f'E8: {g_E8:.5f}')
ax.set_xlabel('gap at 0.8·κ_c')
ax.set_title('E8 vs sparsity-matched controls')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 0,2: p-value summary
ax = axes[0,2]
ctrl_labels = ['A\n(diag>2)', 'B\n(dense)',
                'C\n(sparse≈7)', 'D\n(nz=7,−1)',
                'E\n(nz=7,±1)']
p_vals = [0.0066, 0.0155, 0.1034, p_D, p_E]
colors_bar = []
for p in p_vals:
    if np.isnan(p):
        colors_bar.append('gray')
    elif p < 0.05:
        colors_bar.append('#e74c3c')
    else:
        colors_bar.append('#2ecc71')

valid = [(l, p, c) for l, p, c
          in zip(ctrl_labels, p_vals, colors_bar)
          if not np.isnan(p)]
if valid:
    ls, ps, cs = zip(*valid)
    bars = ax.bar(range(len(ls)), ps,
                   color=cs, alpha=0.8)
    ax.axhline(0.05, color='k', ls='--',
               lw=2, label='p=0.05')
    ax.set_xticks(range(len(ls)))
    ax.set_xticklabels(ls, fontsize=9)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{p:.3f}',
                ha='center', va='bottom',
                fontsize=8)
ax.set_ylabel('p-value (Mann-Whitney)')
ax.set_title('E8 signal: p-values across\n'
             'all control types')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 1,0: κ_c distributions
ax = axes[1,0]
ax.hist(kcs_D, bins=25, color=col_D,
        alpha=0.6, density=True,
        label=f'Ctrl-D: {kcs_D.mean():.3f}'
              f'±{kcs_D.std():.3f}')
ax.hist(kcs_E, bins=25, color=col_E,
        alpha=0.6, density=True,
        label=f'Ctrl-E: {kcs_E.mean():.3f}'
              f'±{kcs_E.std():.3f}')
ax.axvline(kc_E8, color=col_E8, lw=3,
           label=f'E8: κ_c={kc_E8:.3f}')
ax.set_xlabel('κ_c')
ax.set_title('κ_c distributions:\nE8 vs sparsity-matched')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,1: Connected vs disconnected gaps
ax = axes[1,1]
if gaps_conn and gaps_disconn:
    ax.hist(gaps_conn, bins=20,
            color='#2ecc71', alpha=0.6,
            density=True,
            label=f'Connected\n'
                  f'N={len(gaps_conn)}, '
                  f'μ={np.mean(gaps_conn):.4f}')
    ax.hist(gaps_disconn, bins=20,
            color='gray', alpha=0.6,
            density=True,
            label=f'Disconnected\n'
                  f'N={len(gaps_disconn)}, '
                  f'μ={np.mean(gaps_disconn):.4f}')
    ax.axvline(g_E8, color=col_E8, lw=3,
               label=f'E8: {g_E8:.5f}')
    ax.set_xlabel('gap')
    ax.set_title('Connectivity effect\n'
                 'in Ctrl-D')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 1,2: Финальный вердикт
ax = axes[1,2]
ax.axis('off')

if conclusion == "ARTIFACT":
    fc = '#ffe8e8'
elif conclusion == "SIGNAL":
    fc = '#e0ffe0'
else:
    fc = '#fff3cd'

lines = [
    'PART XIX — DEFINITIVE VERDICT',
    '═'*32,
    '',
    'E8 gap test results:',
    '  Ctrl-A (diag>2):   p=0.007 ARTIFACT',
    '  Ctrl-B (dense):    p=0.016 sparsity↑',
    '  Ctrl-C (sparse≈7): p=0.103 ns',
    f'  Ctrl-D (nz=7,-1):  p={p_D:.3f} '
    f'{"***" if p_D<0.05 else "ns"}',
    f'  Ctrl-E (nz=7,±1):  p={p_E:.3f} '
    f'{"***" if p_E<0.05 else "ns"}',
    '',
    f'Conclusion: {conclusion}',
    '',
]
if conclusion == "ARTIFACT":
    lines += [
        'E8 gap = f(sparsity),',
        'not f(Lie algebra structure).',
        '',
        'Artifact #11:',
        '  Sparsity mismatch',
        '  in Ctrl-B',
        '',
        '19 experiments → 0 signals',
        '11 artifacts documented',
        '6 theorems found',
        '',
        '→ PUBLISH v14.0.0',
    ]
elif conclusion == "SIGNAL":
    lines += [
        '*** Survived all fair controls ***',
        '→ Part XX: mechanism',
        '→ N=1000 replication',
    ]
else:
    lines += [
        'Increase N to 1000',
        'for definitive answer.',
    ]

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8.5,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=fc, alpha=0.9))

plt.tight_layout()
plt.savefig('part19_step7_final.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part19_step7_final.png")
