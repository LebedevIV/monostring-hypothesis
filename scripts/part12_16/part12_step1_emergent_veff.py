"""
Part XII: Emergent Inflaton Potential from Monostring Dynamics
Step 1: Compute V_eff(φ₁) from mode interactions

H₀: V_eff flatness for Coxeter algebras ≤ random controls
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter1d

# ── 1. ПАРАМЕТРЫ ─────────────────────────────────────────────────────────────

SEED      = 42
T         = 50000   # длинная траектория для хорошей статистики
WARMUP    = 5000    # выброс начала
KAPPA     = 0.05    # стандартный параметр связи
N_BINS    = 100     # разрешение по φ₁
N_RANDOM  = 30      # контрольных случайных наборов частот
SMOOTH    = 3       # сглаживание для производных

np.random.seed(SEED)

# ── 2. АЛГЕБРЫ ───────────────────────────────────────────────────────────────

ALGEBRAS = {
    'E8': {'m': [1, 7, 11, 13, 17, 19, 23, 29], 'h': 30},
    'E6': {'m': [1,  4,  5,  7,  8, 11],         'h': 12},
    'E7': {'m': [1,  5,  7,  9, 11, 13, 17],      'h': 18},
    'A6': {'m': [1,  2,  3,  4,  5,  6],          'h':  7},
}

def coxeter_freqs(algebra_name):
    alg = ALGEBRAS[algebra_name]
    m, h = np.array(alg['m']), alg['h']
    return 2 * np.sin(np.pi * m / h)

def random_freqs(rank, seed=None):
    """Случайные частоты с тем же rank и диапазоном"""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 2.0, rank)

# ── 3. ДИНАМИКА ──────────────────────────────────────────────────────────────

def run_monostring(omegas, kappa=KAPPA, T=T, warmup=WARMUP, seed=SEED):
    """
    Стандартное отображение для r мод.
    Возвращает орбиту ПОСЛЕ warmup.
    """
    rng = np.random.RandomState(seed)
    r = len(omegas)
    phi = rng.uniform(0, 2*np.pi, r)

    orbit = np.zeros((T - warmup, r))

    for t in range(T):
        phi = (phi + omegas + kappa * np.sin(phi)) % (2 * np.pi)
        if t >= warmup:
            orbit[t - warmup] = phi

    return orbit   # shape: (T-warmup, r)

# ── 4. ВЫЧИСЛЕНИЕ V_eff ───────────────────────────────────────────────────────

def compute_Veff(orbit, n_bins=N_BINS):
    """
    V_eff(φ₁) = -<Σᵢ₌₂ʳ κ·cos(φ₁ - φᵢ)> усреднённое по времени
    при фиксированном φ₁ ∈ bin.

    Алгоритм:
      1. Разбить φ₁ на n_bins корзин
      2. Для каждой корзины усреднить Σᵢ cos(φ₁ - φᵢ)
      3. V_eff = -κ · это среднее
    """
    phi1  = orbit[:, 0]       # инфлатон
    phi_r = orbit[:, 1:]      # среда (r-1 мод)

    bins    = np.linspace(0, 2*np.pi, n_bins + 1)
    phi_mid = 0.5 * (bins[:-1] + bins[1:])
    Veff    = np.full(n_bins, np.nan)
    counts  = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (phi1 >= bins[b]) & (phi1 < bins[b+1])
        counts[b] = mask.sum()
        if counts[b] > 10:       # минимум точек в корзине
            phi1_b  = phi1[mask]
            phi_r_b = phi_r[mask]   # shape: (N_b, r-1)

            # Взаимодействие: Σⱼ cos(φ₁ - φⱼ) для каждого момента времени
            diff      = phi1_b[:, None] - phi_r_b   # (N_b, r-1)
            cos_sum   = np.cos(diff).sum(axis=1)     # (N_b,)
            Veff[b]   = -KAPPA * cos_sum.mean()

    return phi_mid, Veff, counts

# ── 5. SLOW-ROLL ПАРАМЕТРЫ ────────────────────────────────────────────────────

def slow_roll_params(phi_mid, Veff, smooth=SMOOTH):
    """
    ε = ½(V'/V)²   (в единицах M_Pl = 1)
    η = V''/V

    Используем численное дифференцирование со сглаживанием.
    """
    # Убираем NaN
    valid = ~np.isnan(Veff)
    if valid.sum() < 20:
        return None, None, None, None

    phi_v = phi_mid[valid]
    V_v   = Veff[valid]

    # Нормировка: V должна быть > 0 для slow-roll
    # Сдвигаем так чтобы min = 0.01 (избегаем деления на 0)
    V_v = V_v - V_v.min() + 0.01

    # Сглаживание
    V_s = gaussian_filter1d(V_v, smooth)

    # Производные (конечные разности)
    dphi  = np.diff(phi_v).mean()
    dV    = np.gradient(V_s, dphi)
    d2V   = np.gradient(dV,  dphi)

    # Slow-roll
    epsilon = 0.5 * (dV / V_s)**2
    eta     = d2V / V_s

    # Метрика плоскости: доля точек где ε < 0.01
    flat_fraction = (epsilon < 0.01).mean()

    # Минимальное ε (самое плоское место)
    eps_min = epsilon.min()

    return epsilon, eta, flat_fraction, eps_min

# ── 6. ОСНОВНОЙ ЭКСПЕРИМЕНТ ───────────────────────────────────────────────────

print("=" * 60)
print("Part XII Step 1: Emergent Inflaton Potential")
print("=" * 60)
print(f"T={T}, warmup={WARMUP}, κ={KAPPA}, bins={N_BINS}")
print()

results = {}

# 6a. Коксетеровы алгебры
print("── Coxeter algebras ──")
for name, alg in ALGEBRAS.items():
    omegas = coxeter_freqs(name)
    orbit  = run_monostring(omegas)
    phi_mid, Veff, counts = compute_Veff(orbit)
    epsilon, eta, flat_frac, eps_min = slow_roll_params(phi_mid, Veff)

    results[name] = {
        'omegas': omegas, 'phi_mid': phi_mid, 'Veff': Veff,
        'epsilon': epsilon, 'flat_frac': flat_frac, 'eps_min': eps_min,
        'type': 'coxeter'
    }

    print(f"{name:4s} (rank={len(omegas):1d}, h={alg['h']:2d}): "
          f"flat_frac={flat_frac:.3f}, "
          f"ε_min={eps_min:.4f}, "
          f"valid_bins={np.sum(~np.isnan(Veff)):3d}")

print()

# 6b. Случайный контроль
print("── Random controls ──")
random_flat_fracs = []
random_eps_mins   = []

for i in range(N_RANDOM):
    # Используем rank E8 = 8 (честное сравнение)
    omegas = random_freqs(rank=8, seed=100 + i)
    orbit  = run_monostring(omegas)
    phi_mid, Veff, counts = compute_Veff(orbit)
    epsilon, eta, flat_frac, eps_min = slow_roll_params(phi_mid, Veff)

    if flat_frac is not None:
        random_flat_fracs.append(flat_frac)
        random_eps_mins.append(eps_min)

    if i < 5:  # показываем первые 5
        print(f"  rand_{i:02d}: flat_frac={flat_frac:.3f}, ε_min={eps_min:.4f}")

print(f"  ... ({N_RANDOM} total)")
print(f"  Random mean flat_frac = {np.mean(random_flat_fracs):.3f} "
      f"± {np.std(random_flat_fracs):.3f}")
print(f"  Random mean ε_min     = {np.mean(random_eps_mins):.4f} "
      f"± {np.std(random_eps_mins):.4f}")

# ── 7. СТАТИСТИКА ─────────────────────────────────────────────────────────────

print()
print("── Statistical comparison ──")
for name in ALGEBRAS:
    ff = results[name]['flat_frac']
    em = results[name]['eps_min']

    # Percentile в распределении random
    pct_ff = np.mean(np.array(random_flat_fracs) < ff) * 100
    pct_em = np.mean(np.array(random_eps_mins)   > em) * 100  # > потому что меньше = лучше

    print(f"{name}: flat_frac={ff:.3f} "
          f"(percentile={pct_ff:.0f}%), "
          f"ε_min={em:.4f} "
          f"(percentile={pct_em:.0f}%)")

# Mann-Whitney для E8 vs Random (flat_frac)
# У нас 1 значение E8 vs N random — используем percentile
e8_ff = results['E8']['flat_frac']
pval_approx = 1.0 - np.mean(np.array(random_flat_fracs) < e8_ff)
print(f"\nE8 flat_frac p-value (approx) = {pval_approx:.3f}")
print(f"H₀ {'НЕ ОТВЕРГАЕТСЯ' if pval_approx > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")

# ── 8. ВИЗУАЛИЗАЦИЯ ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('Part XII: Emergent Inflaton Potential V_eff(φ₁)\n'
             'Monostring Hypothesis — Coxeter vs Random', fontsize=13)

colors = {'E8': '#e74c3c', 'E6': '#3498db',
          'E7': '#2ecc71', 'A6': '#9b59b6'}

# Row 0: V_eff(φ₁)
for idx, name in enumerate(ALGEBRAS):
    ax = axes[0, idx]
    r  = results[name]
    valid = ~np.isnan(r['Veff'])
    ax.plot(r['phi_mid'][valid], r['Veff'][valid],
            color=colors[name], lw=2)
    ax.set_title(f'{name}: Emergent V_eff(φ₁)', fontsize=10)
    ax.set_xlabel('φ₁')
    ax.set_ylabel('V_eff')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.grid(True, alpha=0.3)

# Row 1: ε(φ₁) — slow-roll parameter
for idx, name in enumerate(ALGEBRAS):
    ax = axes[1, idx]
    r  = results[name]
    if r['epsilon'] is not None:
        valid = ~np.isnan(r['Veff'])
        phi_v = r['phi_mid'][valid]
        # epsilon имеет ту же длину что phi_v
        eps_plot = r['epsilon'][:len(phi_v)]
        ax.semilogy(phi_v[:len(eps_plot)], eps_plot,
                    color=colors[name], lw=2)
        ax.axhline(0.01, color='red', lw=1.5, ls='--', label='ε=0.01')
        ax.fill_between(phi_v[:len(eps_plot)],
                        0, 0.01,
                        where=eps_plot < 0.01,
                        alpha=0.3, color='green',
                        label=f'flat: {r["flat_frac"]:.2f}')
    ax.set_title(f'{name}: Slow-roll ε(φ₁)', fontsize=10)
    ax.set_xlabel('φ₁')
    ax.set_ylabel('ε (log scale)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-6, 1e3)

# Row 2: Сравнение flat_frac — Coxeter vs Random
ax = axes[2, 0]
ax.hist(random_flat_fracs, bins=15, color='gray',
        alpha=0.7, label='Random (N=30)')
for name in ALGEBRAS:
    ax.axvline(results[name]['flat_frac'],
               color=colors[name], lw=2.5, label=name)
ax.set_xlabel('flat_fraction (ε < 0.01)')
ax.set_ylabel('Count')
ax.set_title('Flatness: Coxeter vs Random')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Row 2, col 1: ε_min comparison
ax = axes[2, 1]
ax.hist(random_eps_mins, bins=15, color='gray',
        alpha=0.7, label='Random')
for name in ALGEBRAS:
    ax.axvline(results[name]['eps_min'],
               color=colors[name], lw=2.5, label=name)
ax.set_xlabel('ε_min (lower = flatter)')
ax.set_ylabel('Count')
ax.set_title('Min slow-roll ε: Coxeter vs Random')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Row 2, col 2: V_eff для всех алгебр вместе
ax = axes[2, 2]
for name in ALGEBRAS:
    r = results[name]
    valid = ~np.isnan(r['Veff'])
    V_norm = r['Veff'][valid]
    if V_norm.ptp() > 0:
        V_norm = (V_norm - V_norm.min()) / V_norm.ptp()
    ax.plot(r['phi_mid'][valid], V_norm,
            color=colors[name], lw=2, label=name)
ax.set_xlabel('φ₁')
ax.set_ylabel('V_eff (normalized)')
ax.set_title('V_eff shape comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2, col 3: Сводка результатов
ax = axes[2, 3]
ax.axis('off')
summary_lines = [
    'SUMMARY: Part XII Step 1',
    '─' * 28,
    f'T = {T}, κ = {KAPPA}',
    f'N_random = {N_RANDOM}',
    '',
    'flat_frac (ε < 0.01):',
]
for name in ALGEBRAS:
    ff = results[name]['flat_frac']
    pct = np.mean(np.array(random_flat_fracs) < ff) * 100
    summary_lines.append(f'  {name}: {ff:.3f} ({pct:.0f}%-ile)')
summary_lines += [
    f'  Random: {np.mean(random_flat_fracs):.3f}±{np.std(random_flat_fracs):.3f}',
    '',
    f'H₀ (E8 = random):',
    f'  p ≈ {pval_approx:.3f}',
    f'  {"NOT rejected" if pval_approx > 0.05 else "REJECTED"}',
]
ax.text(0.05, 0.95, '\n'.join(summary_lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part12_step1_emergent_veff.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Saved: part12_step1_emergent_veff.png")
print("\n" + "=" * 60)
print("INTERPRETATION GUIDE:")
print("=" * 60)
print("flat_frac > 0.5 AND > 90th percentile of random → SIGNAL")
print("flat_frac < 0.3 OR < 50th percentile           → H₀ holds")
print()
print("Next step: if signal found → vary κ, check n_s")
print("           if no signal   → try κ >> 0.05 (strong coupling)")
