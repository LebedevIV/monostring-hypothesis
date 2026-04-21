"""
Part XII Step 3: Symmetry vs Spread
Гипотеза: flat_frac определяется симметрией пар частот,
а не spread или алгеброй Коксетера как таковой.

Три группы контролей:
  A) Random без симметрии (как раньше)
  B) Random с forced Weyl symmetry: ω_{r+1-i} = ωᵢ
  C) Coxeter алгебры

Если flat_frac(B) ≈ flat_frac(Coxeter) → симметрия = механизм
Если flat_frac(B) ≈ flat_frac(A)       → симметрия не при чём
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kruskal

SEED     = 42
T        = 50000
WARMUP   = 5000
KAPPA    = 0.05
N_BINS   = 100
N_EACH   = 100   # контролей каждого типа
SMOOTH   = 3

np.random.seed(SEED)

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30, 'rank': 8},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12, 'rank': 6},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18, 'rank': 7},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7, 'rank': 6},
}

# ── ГЕНЕРАТОРЫ ЧАСТОТ ────────────────────────────────────────

def coxeter_freqs(name):
    alg = ALGEBRAS[name]
    m = np.array(alg['m'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])

def random_freqs_plain(rank, seed):
    """Обычные случайные частоты"""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 2.0, rank)

def random_freqs_symmetric(rank, seed):
    """
    Случайные частоты с Вейлевской симметрией:
    ω_{r+1-i} = ωᵢ  (как в Коксетере)
    Генерируем rank//2 уникальных, дублируем.
    """
    rng  = np.random.RandomState(seed)
    half = rank // 2
    base = rng.uniform(0.1, 2.0, half)
    if rank % 2 == 0:
        return np.concatenate([base, base[::-1]])
    else:
        mid = rng.uniform(0.1, 2.0, 1)
        return np.concatenate([base, mid, base[::-1]])

def random_freqs_spread_matched(rank, seed, target_spread):
    """
    Случайные частоты с тем же spread что у Coxeter
    Нет симметрии, но spread зафиксирован
    """
    rng   = np.random.RandomState(seed)
    raw   = rng.uniform(0, 1, rank)
    # Масштабируем до нужного spread
    raw   = (raw - raw.min())
    if raw.max() > 0:
        raw = raw / raw.max() * target_spread + 0.1
    return raw

# ── ДИНАМИКА ─────────────────────────────────────────────────

def run_monostring(omegas, kappa=KAPPA, T=T,
                   warmup=WARMUP, seed=SEED):
    rng = np.random.RandomState(seed)
    r   = len(omegas)
    phi = rng.uniform(0, 2*np.pi, r)
    orbit = np.zeros((T - warmup, r))
    for t in range(T):
        phi = (phi + omegas + kappa * np.sin(phi)) % (2*np.pi)
        if t >= warmup:
            orbit[t - warmup] = phi
    return orbit

# ── V_eff и slow-roll ────────────────────────────────────────

def compute_Veff(orbit, n_bins=N_BINS):
    phi1  = orbit[:, 0]
    phi_r = orbit[:, 1:]
    bins  = np.linspace(0, 2*np.pi, n_bins + 1)
    phi_m = 0.5 * (bins[:-1] + bins[1:])
    Veff  = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (phi1 >= bins[b]) & (phi1 < bins[b+1])
        if mask.sum() > 10:
            diff = phi1[mask, None] - phi_r[mask]
            Veff[b] = -KAPPA * np.cos(diff).sum(axis=1).mean()
    return phi_m, Veff

def flat_fraction(phi_mid, Veff, smooth=SMOOTH):
    valid = ~np.isnan(Veff)
    if valid.sum() < 20:
        return np.nan, np.nan
    phi_v = phi_mid[valid]
    V_v   = Veff[valid].copy()
    V_v   = V_v - V_v.min() + 0.01
    V_s   = gaussian_filter1d(V_v, smooth)
    dphi  = np.diff(phi_v).mean()
    dV    = np.gradient(V_s, dphi)
    V_safe = np.where(V_s > 1e-10, V_s, 1e-10)
    epsilon = 0.5 * (dV / V_safe)**2
    return float((epsilon < 0.01).mean()), float(np.median(epsilon))

def run_batch(freq_func, rank, n, seed_offset, **kwargs):
    """Запускает n симуляций, возвращает flat_fracs"""
    ffs = []
    for i in range(n):
        ω = freq_func(rank, seed=seed_offset + i, **kwargs)
        orbit = run_monostring(ω)
        phi_m, Veff = compute_Veff(orbit)
        ff, _ = flat_fraction(phi_m, Veff)
        if not np.isnan(ff):
            ffs.append(ff)
    return np.array(ffs)

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("=" * 65)
print("Part XII Step 3: Symmetry vs Spread Hypothesis")
print("=" * 65)
print(f"N={N_EACH} per group, T={T}, κ={KAPPA}")
print()

# Коксетеровы алгебры
print("── Coxeter algebras ──")
cox_results = {}
for name in ALGEBRAS:
    ω = coxeter_freqs(name)
    orbit = run_monostring(ω)
    phi_m, Veff = compute_Veff(orbit)
    ff, emd = flat_fraction(phi_m, Veff)
    cox_results[name] = {
        'ff': ff, 'eps_med': emd,
        'spread': ω.max()-ω.min(),
        'rank': len(ω),
        'symmetric': True,
        'n_unique': len(np.unique(np.round(ω, 4)))
    }
    print(f"  {name}: ff={ff:.3f}, spread={ω.max()-ω.min():.3f}, "
          f"n_unique={cox_results[name]['n_unique']}")

print()
E8_spread = cox_results['E8']['spread']
E8_rank   = cox_results['E8']['rank']

# Три группы контролей для rank=8 (E8)
print("── Control groups for rank=8 (comparing to E8) ──")
print()

# A) Plain random
print("  A) Plain random (rank=8)...")
ff_plain = run_batch(random_freqs_plain, rank=8,
                     n=N_EACH, seed_offset=1000)
print(f"     ff = {ff_plain.mean():.3f} ± {ff_plain.std():.3f}")

# B) Symmetric random (Weyl symmetry)
print("  B) Symmetric random (rank=8, Weyl pairs)...")
ff_sym = run_batch(random_freqs_symmetric, rank=8,
                   n=N_EACH, seed_offset=2000)
print(f"     ff = {ff_sym.mean():.3f} ± {ff_sym.std():.3f}")

# C) Spread-matched random
print(f"  C) Spread-matched random (rank=8, spread≈{E8_spread:.3f})...")
ff_spread = run_batch(random_freqs_spread_matched, rank=8,
                      n=N_EACH, seed_offset=3000,
                      target_spread=E8_spread)
print(f"     ff = {ff_spread.mean():.3f} ± {ff_spread.std():.3f}")

# D) Symmetric AND spread-matched
print(f"  D) Symmetric + spread-matched (rank=8)...")
ff_both = []
for i in range(N_EACH):
    rng  = np.random.RandomState(4000 + i)
    half = 4
    base = rng.uniform(0.1, 0.1 + E8_spread, half)
    # Нормируем spread
    base = (base - base.min())
    if base.max() > 0:
        base = base / base.max() * E8_spread + 0.1
    ω = np.concatenate([base, base[::-1]])
    orbit = run_monostring(ω)
    phi_m, Veff = compute_Veff(orbit)
    ff, _ = flat_fraction(phi_m, Veff)
    if not np.isnan(ff):
        ff_both.append(ff)
ff_both = np.array(ff_both)
print(f"     ff = {ff_both.mean():.3f} ± {ff_both.std():.3f}")

print()

# ── СТАТИСТИКА ───────────────────────────────────────────────

print("── Statistical analysis ──")
print()

E8_ff = cox_results['E8']['ff']

groups = {
    'Plain':        ff_plain,
    'Symmetric':    ff_sym,
    'Spread-match': ff_spread,
    'Sym+Spread':   ff_both,
}

print(f"E8 flat_frac = {E8_ff:.3f}")
print()
print(f"{'Group':<15} {'mean':>6} {'std':>6} "
      f"{'E8 pct':>8} {'p(E8>)':>8} {'Δ from plain':>13}")
print("-" * 65)

plain_mean = ff_plain.mean()
for gname, ffs in groups.items():
    pct  = float(np.mean(ffs < E8_ff)) * 100
    pval = float(np.mean(ffs >= E8_ff))
    delta = ffs.mean() - plain_mean
    print(f"{gname:<15} {ffs.mean():>6.3f} {ffs.std():>6.3f} "
          f"{pct:>7.0f}% {pval:>8.3f} {delta:>+13.3f}")

print()

# Kruskal-Wallis тест между группами
stat, p_kw = kruskal(ff_plain, ff_sym, ff_spread, ff_both)
print(f"Kruskal-Wallis (между группами): H={stat:.2f}, p={p_kw:.4f}")

# Ключевой вопрос: делает ли симметрия разницу?
print()
print("── Decomposition of E8 signal ──")
print()

# Насколько симметрия объясняет разницу?
explained_by_symmetry = ff_sym.mean() - ff_plain.mean()
explained_by_spread   = ff_spread.mean() - ff_plain.mean()
explained_by_both     = ff_both.mean() - ff_plain.mean()
E8_excess             = E8_ff - ff_plain.mean()

print(f"E8 excess над plain random:        {E8_excess:+.3f}")
print(f"Объяснено симметрией:              {explained_by_symmetry:+.3f} "
      f"({100*explained_by_symmetry/max(abs(E8_excess),1e-6):.0f}%)")
print(f"Объяснено spread:                  {explained_by_spread:+.3f} "
      f"({100*explained_by_spread/max(abs(E8_excess),1e-6):.0f}%)")
print(f"Объяснено обоими:                  {explained_by_both:+.3f} "
      f"({100*explained_by_both/max(abs(E8_excess),1e-6):.0f}%)")
residual = E8_excess - explained_by_both
print(f"Остаток (специфика E8):            {residual:+.3f} "
      f"({100*residual/max(abs(E8_excess),1e-6):.0f}%)")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Part XII Step 3: Symmetry vs Spread Analysis\n'
             'Decomposing E8 flat_frac signal', fontsize=13)

colors_ctrl = {
    'Plain':     '#95a5a6',
    'Symmetric': '#3498db',
    'Spread':    '#e67e22',
    'Sym+Spr':   '#9b59b6',
}
cox_colors = {'E8':'#e74c3c','E6':'#2ecc71',
              'E7':'#f39c12','A6':'#1abc9c'}

# 0,0: Распределения flat_frac
ax = axes[0, 0]
all_groups = [ff_plain, ff_sym, ff_spread, ff_both]
all_labels = ['Plain', 'Symmetric', 'Spread-match', 'Sym+Spread']
all_cols   = ['#95a5a6','#3498db','#e67e22','#9b59b6']

for ffs, lbl, col in zip(all_groups, all_labels, all_cols):
    ax.hist(ffs, bins=20, alpha=0.5, color=col,
            label=f'{lbl} ({ffs.mean():.3f})')

for name in ['E8','E6']:
    ax.axvline(cox_results[name]['ff'],
               color=cox_colors[name], lw=2.5,
               ls='--', label=name)

ax.set_xlabel('flat_frac (ε < 0.01)')
ax.set_ylabel('Count')
ax.set_title('flat_frac distributions')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 0,1: Boxplot сравнение
ax = axes[0, 1]
bp = ax.boxplot(all_groups, labels=all_labels,
                patch_artist=True, notch=False)
for patch, col in zip(bp['boxes'], all_cols):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)

for name in ALGEBRAS:
    ax.axhline(cox_results[name]['ff'],
               color=cox_colors[name], lw=2,
               ls='--', label=name, alpha=0.8)

ax.set_ylabel('flat_frac')
ax.set_title('Boxplot: control groups vs Coxeter')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=20)

# 0,2: Декомпозиция сигнала E8
ax = axes[0, 2]
components = ['E8\nexcess', 'Symmetry\nexplains',
              'Spread\nexplains', 'Both\nexplain', 'Residual\n(E8 specific)']
values = [E8_excess, explained_by_symmetry,
          explained_by_spread, explained_by_both, residual]
bar_colors = ['#e74c3c','#3498db','#e67e22','#9b59b6','#2ecc71']
bars = ax.bar(components, values, color=bar_colors, alpha=0.8)
ax.axhline(0, color='k', lw=1)
ax.set_ylabel('Δ flat_frac vs plain random')
ax.set_title('Signal decomposition for E8')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.002 * np.sign(val),
            f'{val:+.3f}', ha='center', va='bottom', fontsize=9)

# 1,0: Scatter: n_unique vs flat_frac
ax = axes[1, 0]
# Random plain
for i, ff in enumerate(ff_plain[:50]):
    ω = random_freqs_plain(8, seed=1000+i)
    n_u = len(np.unique(np.round(ω, 2)))
    ax.scatter(n_u + np.random.randn()*0.1, ff,
               color='lightgray', alpha=0.5, s=15)

for name in ALGEBRAS:
    ax.scatter(cox_results[name]['n_unique'],
               cox_results[name]['ff'],
               color=cox_colors[name], s=120,
               zorder=5, label=name,
               edgecolors='k', linewidths=1)

ax.set_xlabel('n_unique frequencies')
ax.set_ylabel('flat_frac')
ax.set_title('Unique frequencies vs Flatness')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,1: Scatter: spread vs flat_frac с группами
ax = axes[1, 1]
for i, ff in enumerate(ff_plain[:50]):
    ω = random_freqs_plain(8, seed=1000+i)
    ax.scatter(ω.max()-ω.min(), ff,
               color='lightgray', alpha=0.4, s=15)
for i, ff in enumerate(ff_sym[:50]):
    ω = random_freqs_symmetric(8, seed=2000+i)
    ax.scatter(ω.max()-ω.min(), ff,
               color='#3498db', alpha=0.4, s=15)

for name in ALGEBRAS:
    ax.scatter(cox_results[name]['spread'],
               cox_results[name]['ff'],
               color=cox_colors[name], s=120,
               zorder=5, label=name,
               edgecolors='k', linewidths=1)

ax.set_xlabel('Spread(ω)')
ax.set_ylabel('flat_frac')
ax.set_title('Spread vs Flatness (gray=plain, blue=symm)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,2: Итоговая таблица
ax = axes[1, 2]
ax.axis('off')

lines = [
    'STEP 3 VERDICT',
    '─' * 32,
    '',
    'E8 excess:        {:+.3f}'.format(E8_excess),
    'By symmetry:      {:+.3f} ({:.0f}%)'.format(
        explained_by_symmetry,
        100*explained_by_symmetry/max(abs(E8_excess),1e-6)),
    'By spread:        {:+.3f} ({:.0f}%)'.format(
        explained_by_spread,
        100*explained_by_spread/max(abs(E8_excess),1e-6)),
    'By both:          {:+.3f} ({:.0f}%)'.format(
        explained_by_both,
        100*explained_by_both/max(abs(E8_excess),1e-6)),
    'E8 residual:      {:+.3f} ({:.0f}%)'.format(
        residual,
        100*residual/max(abs(E8_excess),1e-6)),
    '',
    'Kruskal-Wallis:',
    '  H={:.2f}, p={:.4f}'.format(stat, p_kw),
    '',
    'INTERPRETATION:',
]

if abs(residual) < 0.02:
    lines += [
        '  E8 signal fully explained',
        '  by symmetry + spread.',
        '  → No algebra-specific effect.',
        '  H₀ holds.',
]
elif residual > 0.03:
    lines += [
        '  E8 has residual signal',
        '  NOT explained by symmetry',
        '  or spread alone!',
        '  → Genuine E8 effect?',
        '  → Proceed to Step 4.',
]
else:
    lines += [
        '  Ambiguous residual.',
        '  Increase N or T.',
]

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part12_step3_symmetry.png',
            dpi=150, bbox_inches='tight')
print('\n✓ Saved: part12_step3_symmetry.png')

# ── ФИНАЛ ────────────────────────────────────────────────────

print()
print("=" * 65)
print("DECISION TREE после Step 3:")
print("=" * 65)
print()
print("Residual > 0.03  → Step 4: vary κ, check n_s")
print("Residual 0-0.03  → E8 эффект = symmetry+spread артефакт")
print("                   → переходим к направлению №1 (КАМ)")
print("Residual < 0     → E8 хуже чем symmetric random")
print("                   → гипотеза отвергнута полностью")
