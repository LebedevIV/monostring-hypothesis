"""
Part XII Step 2: Verification and deeper analysis
Fixes:
  - ptp() → np.ptp() или ручной расчёт
  - matplotlib backend → Agg (без GUI)
  - N_random = 100 (больше контролей)
  - Диагностика ε_min = 0 (артефакт?)
  - Добавлен контроль: rank-matched random
  - Проверка: почему A6 > E8?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # без GUI — исправляет GL ошибку
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── ПАРАМЕТРЫ ────────────────────────────────────────────────

SEED     = 42
T        = 50000
WARMUP   = 5000
KAPPA    = 0.05
N_BINS   = 100
N_RANDOM = 100    # увеличили с 30 до 100
SMOOTH   = 3

np.random.seed(SEED)

# ── АЛГЕБРЫ ──────────────────────────────────────────────────

ALGEBRAS = {
    'E8': {'m': [1, 7, 11, 13, 17, 19, 23, 29], 'h': 30},
    'E6': {'m': [1,  4,  5,  7,  8, 11],         'h': 12},
    'E7': {'m': [1,  5,  7,  9, 11, 13, 17],      'h': 18},
    'A6': {'m': [1,  2,  3,  4,  5,  6],          'h':  7},
}

def coxeter_freqs(name):
    alg = ALGEBRAS[name]
    m = np.array(alg['m'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])

def random_freqs(rank, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 2.0, rank)

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

# ── V_eff ────────────────────────────────────────────────────

def compute_Veff(orbit, n_bins=N_BINS):
    phi1  = orbit[:, 0]
    phi_r = orbit[:, 1:]

    bins    = np.linspace(0, 2*np.pi, n_bins + 1)
    phi_mid = 0.5 * (bins[:-1] + bins[1:])
    Veff    = np.full(n_bins, np.nan)
    counts  = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (phi1 >= bins[b]) & (phi1 < bins[b+1])
        counts[b] = mask.sum()
        if counts[b] > 10:
            phi1_b  = phi1[mask]
            phi_r_b = phi_r[mask]
            diff     = phi1_b[:, None] - phi_r_b
            cos_sum  = np.cos(diff).sum(axis=1)
            Veff[b]  = -KAPPA * cos_sum.mean()

    return phi_mid, Veff, counts

# ── SLOW-ROLL ────────────────────────────────────────────────

def slow_roll_params(phi_mid, Veff, smooth=SMOOTH):
    valid = ~np.isnan(Veff)
    if valid.sum() < 20:
        return None, None, None, None, None

    phi_v = phi_mid[valid]
    V_v   = Veff[valid].copy()

    # Сдвиг чтобы V > 0
    V_v = V_v - V_v.min() + 0.01

    # Сглаживание
    V_s = gaussian_filter1d(V_v, smooth)

    dphi = np.diff(phi_v).mean()
    dV   = np.gradient(V_s, dphi)
    d2V  = np.gradient(dV,  dphi)

    # Защита от деления на 0
    V_safe = np.where(V_s > 1e-10, V_s, 1e-10)

    epsilon = 0.5 * (dV / V_safe)**2
    eta     = d2V / V_safe

    flat_fraction = float((epsilon < 0.01).mean())
    eps_min       = float(epsilon.min())
    eps_median    = float(np.median(epsilon))

    return epsilon, eta, flat_fraction, eps_min, eps_median

# ── ДИАГНОСТИКА ε_min = 0 ─────────────────────────────────

def diagnose_eps_zero(phi_mid, Veff, smooth=SMOOTH):
    """
    Проверяем: ε_min = 0 — артефакт или реально?
    Смотрим на минимальное |dV/dφ|
    """
    valid = ~np.isnan(Veff)
    phi_v = phi_mid[valid]
    V_v   = Veff[valid].copy()
    V_v   = V_v - V_v.min() + 0.01
    V_s   = gaussian_filter1d(V_v, smooth)

    dphi  = np.diff(phi_v).mean()
    dV    = np.gradient(V_s, dphi)

    return {
        'dV_min':    float(np.abs(dV).min()),
        'dV_max':    float(np.abs(dV).max()),
        'dV_mean':   float(np.abs(dV).mean()),
        'V_range':   float(V_s.max() - V_s.min()),
        'V_mean':    float(V_s.mean()),
    }

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("=" * 65)
print("Part XII Step 2: Verification with N=100 random controls")
print("=" * 65)
print(f"T={T}, warmup={WARMUP}, κ={KAPPA}, "
      f"bins={N_BINS}, N_random={N_RANDOM}")
print()

results = {}

# Коксетеровы алгебры
print("── Coxeter algebras ──")
for name in ALGEBRAS:
    omegas = coxeter_freqs(name)
    orbit  = run_monostring(omegas)
    phi_mid, Veff, counts = compute_Veff(orbit)
    eps, eta, ff, em, emd = slow_roll_params(phi_mid, Veff)
    diag = diagnose_eps_zero(phi_mid, Veff)

    results[name] = {
        'omegas': omegas, 'phi_mid': phi_mid, 'Veff': Veff,
        'epsilon': eps, 'eta': eta,
        'flat_frac': ff, 'eps_min': em, 'eps_median': emd,
        'diag': diag, 'type': 'coxeter',
        'rank': len(omegas)
    }

    print(f"{name:4s} (rank={len(omegas)}, h={ALGEBRAS[name]['h']:2d}): "
          f"flat_frac={ff:.3f}, "
          f"ε_med={emd:.4f}, "
          f"|dV|_min={diag['dV_min']:.2e}, "
          f"V_range={diag['V_range']:.4f}")

print()

# Случайный контроль — rank=8 (как E8)
print("── Random controls (rank=8, N=100) ──")
rand8_ff  = []
rand8_em  = []
rand8_emd = []

for i in range(N_RANDOM):
    omegas = random_freqs(rank=8, seed=200 + i)
    orbit  = run_monostring(omegas)
    phi_mid_r, Veff_r, _ = compute_Veff(orbit)
    _, _, ff, em, emd = slow_roll_params(phi_mid_r, Veff_r)
    if ff is not None:
        rand8_ff.append(ff)
        rand8_em.append(em)
        rand8_emd.append(emd)

rand8_ff  = np.array(rand8_ff)
rand8_emd = np.array(rand8_emd)

print(f"  flat_frac: mean={rand8_ff.mean():.3f} "
      f"± {rand8_ff.std():.3f}, "
      f"max={rand8_ff.max():.3f}")
print(f"  ε_median:  mean={rand8_emd.mean():.4f} "
      f"± {rand8_emd.std():.4f}")
print()

# Случайный контроль — rank=6 (как E6, A6)
print("── Random controls (rank=6, N=100) ──")
rand6_ff  = []
rand6_emd = []

for i in range(N_RANDOM):
    omegas = random_freqs(rank=6, seed=300 + i)
    orbit  = run_monostring(omegas)
    phi_mid_r, Veff_r, _ = compute_Veff(orbit)
    _, _, ff, em, emd = slow_roll_params(phi_mid_r, Veff_r)
    if ff is not None:
        rand6_ff.append(ff)
        rand6_emd.append(emd)

rand6_ff  = np.array(rand6_ff)
rand6_emd = np.array(rand6_emd)

print(f"  flat_frac: mean={rand6_ff.mean():.3f} "
      f"± {rand6_ff.std():.3f}, "
      f"max={rand6_ff.max():.3f}")
print()

# ── СТАТИСТИКА ───────────────────────────────────────────────

print("── Statistical comparison (rank-matched) ──")
print()

for name in ALGEBRAS:
    ff  = results[name]['flat_frac']
    emd = results[name]['eps_median']
    rk  = results[name]['rank']

    # Выбираем matched контроль
    ref_ff = rand8_ff if rk == 8 else rand6_ff

    pct_ff = float(np.mean(ref_ff < ff)) * 100
    # p-value: доля random >= наблюдаемого
    pval   = float(np.mean(ref_ff >= ff))

    print(f"{name} (rank={rk}):")
    print(f"  flat_frac = {ff:.3f}  "
          f"| random mean = {ref_ff.mean():.3f} ± {ref_ff.std():.3f}")
    print(f"  percentile = {pct_ff:.0f}%  "
          f"| p-value ≈ {pval:.3f}  "
          f"| {'* SIGNAL' if pval < 0.05 else 'H0 holds'}")
    print(f"  ε_median   = {emd:.4f}")
    print()

# ── КЛЮЧЕВОЙ ВОПРОС: почему A6 лучше E8? ─────────────────

print("── Key question: Why A6 > E8? ──")
print()
print("A6 frequencies:", np.round(coxeter_freqs('A6'), 4))
print("E8 frequencies:", np.round(coxeter_freqs('E8'), 4))
print()

# Spread (разброс частот)
for name in ALGEBRAS:
    ω = results[name]['omegas']
    print(f"{name}: spread={ω.max()-ω.min():.3f}, "
          f"mean={ω.mean():.3f}, "
          f"std={ω.std():.3f}, "
          f"n_unique={len(np.unique(np.round(ω,4)))}")

print()
print("Гипотеза: flat_frac ∝ spread(ω)?")
spreads = []
ffs     = []
for name in ALGEBRAS:
    ω = results[name]['omegas']
    spreads.append(ω.max() - ω.min())
    ffs.append(results[name]['flat_frac'])

corr = np.corrcoef(spreads, ffs)[0, 1]
print(f"Корреляция spread vs flat_frac: r = {corr:.3f}")
print("Если |r| > 0.9 → flat_frac объясняется spread, не алгеброй!")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Part XII Step 2: Emergent V_eff — Verification\n'
             'N=100 rank-matched controls', fontsize=13)

colors = {'E8': '#e74c3c', 'E6': '#3498db',
          'E7': '#2ecc71', 'A6': '#9b59b6'}

# ── Row 0: V_eff для каждой алгебры
for idx, name in enumerate(ALGEBRAS):
    ax = fig.add_subplot(4, 4, idx + 1)
    r  = results[name]
    valid = ~np.isnan(r['Veff'])

    V_plot = r['Veff'][valid]
    phi_plot = r['phi_mid'][valid]

    ax.plot(phi_plot, V_plot, color=colors[name], lw=2)
    ax.set_title(f'{name}: V_eff(φ₁)', fontsize=10)
    ax.set_xlabel('φ₁')
    ax.set_ylabel('V_eff')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.grid(True, alpha=0.3)

# ── Row 1: ε(φ₁)
for idx, name in enumerate(ALGEBRAS):
    ax = fig.add_subplot(4, 4, idx + 5)
    r  = results[name]
    if r['epsilon'] is not None:
        valid   = ~np.isnan(r['Veff'])
        phi_v   = r['phi_mid'][valid]
        eps_plt = r['epsilon'][:len(phi_v)]
        n       = min(len(phi_v), len(eps_plt))

        ax.semilogy(phi_v[:n], eps_plt[:n],
                    color=colors[name], lw=2)
        ax.axhline(0.01, color='red', lw=1.5,
                   ls='--', label='ε=0.01')
        ax.fill_between(
            phi_v[:n], 1e-8, 0.01,
            where=eps_plt[:n] < 0.01,
            alpha=0.3, color='green',
            label=f'flat={r["flat_frac"]:.2f}')

    ax.set_title(f'{name}: ε(φ₁)', fontsize=10)
    ax.set_xlabel('φ₁')
    ax.set_ylabel('ε')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-8, 1e3)

# ── Row 2: Сравнение с контролем
# flat_frac distribution — rank 8
ax = fig.add_subplot(4, 4, 9)
ax.hist(rand8_ff, bins=20, color='gray',
        alpha=0.7, label='Random rank=8')
for name in ['E8', 'E7']:
    ax.axvline(results[name]['flat_frac'],
               color=colors[name], lw=2.5, label=name)
ax.set_xlabel('flat_frac')
ax.set_title('Rank-8 comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# flat_frac distribution — rank 6
ax = fig.add_subplot(4, 4, 10)
ax.hist(rand6_ff, bins=20, color='gray',
        alpha=0.7, label='Random rank=6')
for name in ['E6', 'A6']:
    ax.axvline(results[name]['flat_frac'],
               color=colors[name], lw=2.5, label=name)
ax.set_xlabel('flat_frac')
ax.set_title('Rank-6 comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Scatter: spread vs flat_frac
ax = fig.add_subplot(4, 4, 11)
all_spreads = []
all_ffs     = []
all_colors  = []

# random rank=8
for i, ff in enumerate(rand8_ff):
    ω = random_freqs(rank=8, seed=200+i)
    all_spreads.append(ω.max() - ω.min())
    all_ffs.append(ff)
    all_colors.append('lightgray')

# Coxeter
for name in ALGEBRAS:
    ω = results[name]['omegas']
    all_spreads.append(ω.max() - ω.min())
    all_ffs.append(results[name]['flat_frac'])
    all_colors.append(colors[name])

ax.scatter(all_spreads[:-4], all_ffs[:-4],
           c='lightgray', alpha=0.5, s=20, label='Random')
for idx, name in enumerate(ALGEBRAS):
    ax.scatter(all_spreads[-(4-idx)], all_ffs[-(4-idx)],
               c=colors[name], s=80, zorder=5,
               label=name, edgecolors='k', linewidths=0.5)

ax.set_xlabel('Spread(ω) = max-min')
ax.set_ylabel('flat_frac')
ax.set_title(f'Spread vs Flatness\nr={corr:.2f}')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Сводная таблица
ax = fig.add_subplot(4, 4, 12)
ax.axis('off')

lines = ['RESULTS SUMMARY', '─'*30]
for name in ALGEBRAS:
    rk  = results[name]['rank']
    ff  = results[name]['flat_frac']
    ref = rand8_ff if rk == 8 else rand6_ff
    pv  = float(np.mean(ref >= ff))
    sig = '* SIG' if pv < 0.05 else ''
    lines.append(f'{name}(r={rk}): ff={ff:.3f} '
                 f'p={pv:.3f} {sig}')

lines += [
    '',
    f'rand8: {rand8_ff.mean():.3f}±{rand8_ff.std():.3f}',
    f'rand6: {rand6_ff.mean():.3f}±{rand6_ff.std():.3f}',
    '',
    f'spread-ff corr: {corr:.3f}',
    '',
    'VERDICT:',
    'If corr > 0.9:',
    '  flat_frac = spread artifact',
    'If corr < 0.5 AND p<0.05:',
    '  genuine signal',
]

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

# V_eff все вместе нормированные
ax = fig.add_subplot(4, 1, 4)
for name in ALGEBRAS:
    r     = results[name]
    valid = ~np.isnan(r['Veff'])
    V_v   = r['Veff'][valid]
    phi_v = r['phi_mid'][valid]
    vrange = V_v.max() - V_v.min()
    if vrange > 0:
        V_n = (V_v - V_v.min()) / vrange
    else:
        V_n = V_v
    ax.plot(phi_v, V_n, color=colors[name],
            lw=2, label=name)

ax.set_xlabel('φ₁ (radians)')
ax.set_ylabel('V_eff (normalized 0→1)')
ax.set_title('Shape comparison: all algebras')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0.5, color='k', lw=0.5, ls='--')

plt.tight_layout()
plt.savefig('part12_step2_verification.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part12_step2_verification.png")

# ── ФИНАЛЬНЫЙ ВЕРДИКТ ────────────────────────────────────────

print()
print("=" * 65)
print("FINAL VERDICT Part XII Step 2")
print("=" * 65)
print()
print("Ключевые вопросы:")
print(f"  1. Корреляция spread-flatness: r = {corr:.3f}")
if abs(corr) > 0.85:
    print("     → АРТЕФАКТ: flat_frac определяется spread(ω)")
    print("       E8 плоская потому что у неё большой spread частот")
    print("       НЕ из-за геометрии E8 как таковой")
elif abs(corr) < 0.5:
    print("     → НЕ артефакт spread: сигнал может быть реальным")
    print("       Переходим к Step 3: проверка n_s")
else:
    print("     → НЕОДНОЗНАЧНО: нужен partial correlation test")

print()
print("  2. p-values (rank-matched):")
for name in ALGEBRAS:
    rk  = results[name]['rank']
    ff  = results[name]['flat_frac']
    ref = rand8_ff if rk == 8 else rand6_ff
    pv  = float(np.mean(ref >= ff))
    print(f"     {name}: p = {pv:.3f} "
          f"{'← significant' if pv < 0.05 else ''}")
