"""
Part XVII Step 2: Proper control for Weyl degeneracy

Ключевой вопрос: высокая MI в Коксетере объясняется
ТОЛЬКО наличием пар ωᵢ=ωⱼ, или есть что-то сверх?

Три контрольные группы:
  A) Random без повторений (как раньше) — нечестно
  B) Random с n_unique парами (честный контроль)
  C) Случайные значения, но структура пар как у Коксетера
     (те же |Δω|=0 пары, но другие значения)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED   = 42
T      = 100000
WARMUP = 10000
KAPPA  = 0.05
N_BINS = 50
N_CTRL = 200
np.random.seed(SEED)

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7},
}

def coxeter_freqs(name):
    alg = ALGEBRAS[name]
    m = np.array(alg['m'], dtype=float)
    return 2.0*np.sin(np.pi*m/alg['h'])

def run_monostring(omegas, kappa=KAPPA, T=T,
                   warmup=WARMUP, seed=SEED):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omegas))
    orbit = np.zeros((T-warmup, len(omegas)))
    for t in range(T):
        phi = (phi+omegas+kappa*np.sin(phi)) % (2*np.pi)
        if t >= warmup:
            orbit[t-warmup] = phi
    return orbit

def entropy_1d(x, n_bins=N_BINS):
    counts,_ = np.histogram(x, bins=n_bins,
                             range=(0,2*np.pi))
    p = counts/counts.sum()
    p = p[p>0]
    return -np.sum(p*np.log(p))

def entropy_2d(x, y, n_bins=N_BINS):
    counts,_,_ = np.histogram2d(x, y, bins=n_bins,
        range=[[0,2*np.pi],[0,2*np.pi]])
    p = counts/counts.sum()
    p = p[p>0]
    return -np.sum(p*np.log(p))

def MI(x, y):
    return entropy_1d(x)+entropy_1d(y)-entropy_2d(x,y)

def total_correlation(orbit):
    r = orbit.shape[1]
    tc, mi_list = 0.0, []
    for i in range(r):
        for j in range(i+1,r):
            m = MI(orbit[:,i], orbit[:,j])
            tc += m
            mi_list.append((i,j,m))
    return tc, mi_list

# ── ГЕНЕРАТОРЫ КОНТРОЛЬНЫХ ЧАСТОТ ────────────────────────────

def random_plain(rank, seed):
    """Контроль A: все уникальные"""
    return np.random.RandomState(seed).uniform(0.1,2.0,rank)

def random_weyl_paired(rank, seed):
    """
    Контроль B: n_unique пар как у Коксетера.
    rank=8 → 4 уникальных значения, каждое дважды.
    Симметричный массив: [a,b,c,d,d,c,b,a]
    """
    rng  = np.random.RandomState(seed)
    half = rank // 2
    base = rng.uniform(0.1, 2.0, half)
    return np.concatenate([base, base[::-1]])

def random_same_structure(name, seed):
    """
    Контроль C: та же СТРУКТУРА пар что у Коксетера
    (ωᵢ=ωⱼ для тех же позиций i,j),
    но значения случайные.
    """
    rng    = np.random.RandomState(seed)
    ω_cox  = coxeter_freqs(name)
    rank   = len(ω_cox)
    # Находим уникальные значения и их позиции
    unique, inverse = np.unique(
        np.round(ω_cox,4), return_inverse=True)
    n_u = len(unique)
    # Случайные значения для уникальных
    new_unique = rng.uniform(0.1, 2.0, n_u)
    # Заменяем
    return new_unique[inverse]

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("="*65)
print("Part XVII Step 2: Proper Weyl-degeneracy control")
print("="*65)
print()

# Сначала Коксетеровы алгебры
print("── Coxeter algebras ──")
cox_tc = {}
for name in ALGEBRAS:
    ω     = coxeter_freqs(name)
    orbit = run_monostring(ω)
    tc, mi_list = total_correlation(orbit)
    cox_tc[name] = tc

    # МИ только для Вейлевских пар (ωᵢ=ωⱼ)
    weyl_pairs = [(i,j,m) for i,j,m in mi_list
                  if abs(ω[i]-ω[j])<1e-6]
    nonweyl    = [(i,j,m) for i,j,m in mi_list
                  if abs(ω[i]-ω[j])>=1e-6]

    tc_weyl    = sum(m for _,_,m in weyl_pairs)
    tc_nonweyl = sum(m for _,_,m in nonweyl)

    n_unique = len(np.unique(np.round(ω,4)))

    print(f"  {name} (rank={len(ω)}, n_unique={n_unique}):")
    print(f"    TC_total   = {tc:.4f}")
    print(f"    TC_weyl    = {tc_weyl:.4f} "
          f"({len(weyl_pairs)} pairs, "
          f"{100*tc_weyl/tc:.1f}% of total)")
    print(f"    TC_nonweyl = {tc_nonweyl:.4f} "
          f"({len(nonweyl)} pairs, "
          f"{100*tc_nonweyl/tc:.1f}% of total)")
    cox_tc[name+'_weyl']    = tc_weyl
    cox_tc[name+'_nonweyl'] = tc_nonweyl
    cox_tc[name+'_n_unique'] = n_unique

print()

# Три контрольные группы для rank=8 (E8)
print("── Control groups for rank=8 ──")
print()

groups = {
    'A_plain':   [],
    'B_weyl':    [],
    'C_struct':  [],
}

for i in range(N_CTRL):
    # A: plain random
    ω = random_plain(8, seed=1000+i)
    orbit = run_monostring(ω)
    tc,_ = total_correlation(orbit)
    groups['A_plain'].append(tc)

    # B: Weyl-paired random
    ω = random_weyl_paired(8, seed=2000+i)
    orbit = run_monostring(ω)
    tc,_ = total_correlation(orbit)
    groups['B_weyl'].append(tc)

    # C: Same structure as E8
    ω = random_same_structure('E8', seed=3000+i)
    orbit = run_monostring(ω)
    tc,_ = total_correlation(orbit)
    groups['C_struct'].append(tc)

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_CTRL}...")

for gname, vals in groups.items():
    arr = np.array(vals)
    print(f"  {gname}: TC = {arr.mean():.4f} "
          f"± {arr.std():.4f}")

print()

# То же для rank=6 (E6, A6)
print("── Control groups for rank=6 ──")
print()

groups6 = {'A_plain':[], 'B_weyl':[], 'C_struct_E6':[]}

for i in range(N_CTRL):
    ω = random_plain(6, seed=4000+i)
    orbit = run_monostring(ω)
    tc,_ = total_correlation(orbit)
    groups6['A_plain'].append(tc)

    ω = random_weyl_paired(6, seed=5000+i)
    orbit = run_monostring(ω)
    tc,_ = total_correlation(orbit)
    groups6['B_weyl'].append(tc)

    ω = random_same_structure('E6', seed=6000+i)
    orbit = run_monostring(ω)
    tc,_ = total_correlation(orbit)
    groups6['C_struct_E6'].append(tc)

    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_CTRL}...")

for gname, vals in groups6.items():
    arr = np.array(vals)
    print(f"  {gname}: TC = {arr.mean():.4f} "
          f"± {arr.std():.4f}")

print()

# ── КЛЮЧЕВОЕ СРАВНЕНИЕ ────────────────────────────────────────

print("="*65)
print("KEY COMPARISON: E8 vs Weyl-matched controls")
print("="*65)
print()

e8_tc = cox_tc['E8']
B8    = np.array(groups['B_weyl'])
C8    = np.array(groups['C_struct'])

pval_vs_B = float(np.mean(B8 >= e8_tc))
pval_vs_C = float(np.mean(C8 >= e8_tc))
pct_vs_B  = float(np.mean(B8 < e8_tc))*100
pct_vs_C  = float(np.mean(C8 < e8_tc))*100

print(f"E8 TC = {e8_tc:.4f}")
print()
print(f"vs Control A (plain random):")
print(f"   TC_A = {np.mean(groups['A_plain']):.4f} "
      f"± {np.std(groups['A_plain']):.4f}")
print(f"   p = {float(np.mean(np.array(groups['A_plain'])>=e8_tc)):.4f}  "
      f"← нечестный контроль")
print()
print(f"vs Control B (Weyl-paired random):")
print(f"   TC_B = {B8.mean():.4f} ± {B8.std():.4f}")
print(f"   pct = {pct_vs_B:.0f}%, p = {pval_vs_B:.4f}  "
      f"← честный контроль")
print()
print(f"vs Control C (same structure as E8):")
print(f"   TC_C = {C8.mean():.4f} ± {C8.std():.4f}")
print(f"   pct = {pct_vs_C:.0f}%, p = {pval_vs_C:.4f}  "
      f"← строгий контроль")

print()
print("─"*65)

if pval_vs_C < 0.05:
    print("СИГНАЛ: E8 TC выше даже при контроле структуры!")
    print("→ Есть что-то специфически Коксетерово")
    print("→ Не просто паирование частот")
elif pval_vs_B < 0.05:
    print("ЧАСТИЧНЫЙ СИГНАЛ: E8 выше Weyl-paired,")
    print("но не выше same-structure.")
    print("→ Значит специфические ЗНАЧЕНИЯ частот важны,")
    print("  не только структура паирования")
else:
    print("АРТЕФАКТ ПОДТВЕРЖДЁН:")
    print("E8 TC объясняется полностью структурой Вейлевских пар.")
    print("Любые частоты с той же структурой дают тот же TC.")
    print("→ H₀ holds после правильного контроля")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Part XVII Step 2: MI Signal Decomposition\n'
    'Weyl-pairing vs genuine Coxeter structure',
    fontsize=13)

# 0,0: TC breakdown (Weyl vs Non-Weyl)
ax = axes[0,0]
names = list(ALGEBRAS.keys())
tc_w  = [cox_tc[n+'_weyl']    for n in names]
tc_nw = [cox_tc[n+'_nonweyl'] for n in names]
x = np.arange(len(names))
ax.bar(x, tc_w,  label='Weyl pairs (ωᵢ=ωⱼ)',
       color='#e74c3c', alpha=0.8)
ax.bar(x, tc_nw, bottom=tc_w,
       label='Non-Weyl pairs',
       color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('Total Correlation')
ax.set_title('TC = Weyl pairs dominate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 0,1: Control groups rank=8
ax = axes[0,1]
data  = [np.array(groups['A_plain']),
         B8, C8]
labels = ['A: Plain\nrandom',
          'B: Weyl-\npaired',
          'C: Same\nstructure']
colors = ['#95a5a6','#3498db','#e67e22']
bp = ax.boxplot(data, tick_labels=labels,
                patch_artist=True, notch=False)
for patch, col in zip(bp['boxes'], colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax.axhline(e8_tc, color='#e74c3c', lw=2.5,
           ls='--', label=f'E8={e8_tc:.2f}')
ax.set_ylabel('Total Correlation')
ax.set_title('E8 vs control groups (rank=8)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 0,2: p-values summary
ax = axes[0,2]
ax.axis('off')
lines = [
    'SIGNAL DECOMPOSITION',
    '─'*28, '',
    f'E8 TC = {e8_tc:.4f}',
    '',
    'Weyl pairs contribution:',
    f'  {cox_tc["E8_weyl"]:.4f} '
    f'({100*cox_tc["E8_weyl"]/e8_tc:.0f}%)',
    'Non-Weyl contribution:',
    f'  {cox_tc["E8_nonweyl"]:.4f} '
    f'({100*cox_tc["E8_nonweyl"]/e8_tc:.0f}%)',
    '',
    'vs Control B (Weyl-paired):',
    f'  p = {pval_vs_B:.4f}',
    '',
    'vs Control C (same struct):',
    f'  p = {pval_vs_C:.4f}',
    '',
    'VERDICT:',
]
if pval_vs_C < 0.05:
    verdict_color = '#e0ffe0'
    lines += ['GENUINE SIGNAL', 'beyond Weyl pairing']
elif pval_vs_B < 0.05:
    verdict_color = '#fffde0'
    lines += ['PARTIAL: freq values', 'matter, not just pairs']
else:
    verdict_color = '#ffe0e0'
    lines += ['ARTIFACT confirmed:', 'TC = Weyl pairing only']

ax.text(0.05, 0.95, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=verdict_color, alpha=0.8))

# 1,0: MI matrix E8 с подсветкой пар
ax = axes[1,0]
M_e8 = np.zeros((8,8))
ω_e8 = coxeter_freqs('E8')
for i in range(8):
    for j in range(8):
        if i!=j:
            M_e8[i,j] = MI(
                run_monostring(ω_e8)[:,i],
                run_monostring(ω_e8)[:,j])

# Используем сохранённые данные
ω_e8  = coxeter_freqs('E8')
orbit_e8 = run_monostring(ω_e8)
M_e8  = np.zeros((8,8))
for i in range(8):
    for j in range(i+1,8):
        v = MI(orbit_e8[:,i], orbit_e8[:,j])
        M_e8[i,j] = M_e8[j,i] = v

im = ax.imshow(M_e8, cmap='hot', vmin=0, vmax=3.5)
plt.colorbar(im, ax=ax)
# Подсветка Вейлевских пар
weyl_positions = [(0,7),(1,6),(2,5),(3,4)]
for i,j in weyl_positions:
    ax.add_patch(plt.Rectangle(
        (j-0.5,i-0.5), 1, 1,
        fill=False, edgecolor='cyan', lw=3))
    ax.add_patch(plt.Rectangle(
        (i-0.5,j-0.5), 1, 1,
        fill=False, edgecolor='cyan', lw=3))
ax.set_title('E8 MI matrix\n(cyan = Weyl pairs)')
ax.set_xlabel('Mode j')
ax.set_ylabel('Mode i')

# 1,1: Histogram TC для control C
ax = axes[1,1]
ax.hist(C8, bins=25, color='#e67e22',
        alpha=0.7, label=f'Control C\n({C8.mean():.3f}±{C8.std():.3f})')
ax.axvline(e8_tc, color='#e74c3c', lw=2.5,
           label=f'E8={e8_tc:.3f}')
ax.set_xlabel('Total Correlation')
ax.set_ylabel('Count')
ax.set_title('E8 vs same-structure random\n(most honest control)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,2: Weyl vs Non-Weyl для всех алгебр
ax = axes[1,2]
for name in ALGEBRAS:
    tc_total = cox_tc[name]
    tc_w2    = cox_tc[name+'_weyl']
    tc_nw2   = cox_tc[name+'_nonweyl']
    n_u      = cox_tc[name+'_n_unique']
    ax.scatter(n_u, tc_w2/tc_total,
               s=150, zorder=5,
               label=f'{name}(n_u={n_u})',
               edgecolors='k', linewidths=1)

ax.set_xlabel('n_unique frequencies')
ax.set_ylabel('Fraction of TC from Weyl pairs')
ax.set_title('Weyl contribution vs n_unique')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0.95, color='red', ls='--',
           alpha=0.5, label='95%')

plt.tight_layout()
plt.savefig('part17_step2_decomposition.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part17_step2_decomposition.png")

print()
print("="*65)
print("FINAL VERDICT Part XVII Steps 1-2")
print("="*65)
print()
print(f"E8 TC breakdown:")
print(f"  Weyl pairs:     {cox_tc['E8_weyl']:.4f} "
      f"({100*cox_tc['E8_weyl']/cox_tc['E8']:.1f}%)")
print(f"  Non-Weyl pairs: {cox_tc['E8_nonweyl']:.4f} "
      f"({100*cox_tc['E8_nonweyl']/cox_tc['E8']:.1f}%)")
print()
if pval_vs_C < 0.05:
    print("*** ПОДЛИННЫЙ СИГНАЛ ***")
    print("E8 имеет избыточную MI сверх структуры паирования")
    print("→ Продолжать исследование")
else:
    print("АРТЕФАКТ: MI сигнал = Вейлевское паирование")
    print()
    print("Механизм:")
    print("  ωᵢ = ω_{r+1-i} (теорема Вейля)")
    print("  → φᵢ(t) ≈ φ_{r+1-i}(t) + const")
    print("  → MI(φᵢ, φ_{r+1-i}) ≈ H_max")
    print("  → TC(Coxeter) >> TC(random без паирования)")
    print()
    print("  Случайный контроль НЕ имеет таких пар")
    print("  → нечестное сравнение в Step 1")
    print()
    print("ДОБАВИТЬ В КАТАЛОГ АРТЕФАКТОВ:")
    print("  'Weyl-pairing creates spurious MI signal'")
    print("  'Always use structure-matched controls for MI'")
    print()
    print("="*65)
    print("ИТОГ: Parts I-XVII полностью завершены.")
    print("Классическая гипотеза моноструны фальсифицирована.")
    print("="*65)
