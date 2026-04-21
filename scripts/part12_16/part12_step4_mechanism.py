"""
Part XII Step 4: Mechanism Search
Цель: понять ПОЧЕМУ E8 имеет residual=140%

Три гипотезы:
  H1: Медленная мода (ω_min мала → φ₁ медленная)
  H2: Резонанс частот (соотношения ωᵢ/ωⱼ рациональны?)
  H3: Эффективная размерность (n_unique определяет ff?)

Метод: систематический перебор одного параметра
при фиксации остальных
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

SEED   = 42
T      = 50000
WARMUP = 5000
KAPPA  = 0.05
N_BINS = 100
SMOOTH = 3
N_CTRL = 200   # большой контроль для надёжности

np.random.seed(SEED)

# ── БАЗОВЫЕ ФУНКЦИИ (те же что раньше) ──────────────────────

def run_monostring(omegas, kappa=KAPPA, T=T,
                   warmup=WARMUP, seed=SEED):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omegas))
    orbit = np.zeros((T - warmup, len(omegas)))
    for t in range(T):
        phi = (phi + omegas + kappa*np.sin(phi)) % (2*np.pi)
        if t >= warmup:
            orbit[t - warmup] = phi
    return orbit

def compute_Veff(orbit, n_bins=N_BINS):
    phi1  = orbit[:, 0]
    phi_r = orbit[:, 1:]
    bins  = np.linspace(0, 2*np.pi, n_bins+1)
    phi_m = 0.5*(bins[:-1]+bins[1:])
    Veff  = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (phi1 >= bins[b]) & (phi1 < bins[b+1])
        if mask.sum() > 10:
            diff = phi1[mask, None] - phi_r[mask]
            Veff[b] = -KAPPA * np.cos(diff).sum(axis=1).mean()
    return phi_m, Veff

def get_ff(omegas):
    """Одна симуляция → flat_fraction"""
    orbit = run_monostring(omegas)
    phi_m, Veff = compute_Veff(orbit)
    valid = ~np.isnan(Veff)
    if valid.sum() < 20:
        return np.nan
    V_v = Veff[valid].copy()
    V_v = V_v - V_v.min() + 0.01
    V_s = gaussian_filter1d(V_v, SMOOTH)
    dphi = np.diff(phi_m[valid]).mean()
    dV   = np.gradient(V_s, dphi)
    V_safe = np.where(V_s > 1e-10, V_s, 1e-10)
    eps = 0.5*(dV/V_safe)**2
    return float((eps < 0.01).mean())

# E8 базовые частоты
E8_omegas = 2*np.sin(np.pi*np.array(
    [1,7,11,13,17,19,23,29])/30)
E8_ff = get_ff(E8_omegas)

print("="*65)
print("Part XII Step 4: Mechanism Search")
print("="*65)
print(f"E8 baseline: ff = {E8_ff:.3f}")
print(f"E8 omegas: {np.round(E8_omegas, 3)}")
print()

# ════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Влияние ω_min
# Берём E8, заменяем ω_min на разные значения
# ════════════════════════════════════════════════════════════

print("── Exp 1: Effect of ω_min (vary slowest mode) ──")
print("Держим все частоты E8 кроме первой и последней")
print()

omega_min_vals = np.array([0.05, 0.10, 0.15, 0.209,
                            0.30, 0.50, 0.80, 1.20])
ff_vs_omin = []

for w_min in omega_min_vals:
    # E8 с заменённой медленной модой
    ω = E8_omegas.copy()
    ω[0] = w_min    # первая мода
    ω[-1] = w_min   # последняя (симметрия)
    ff = get_ff(ω)
    ff_vs_omin.append(ff)
    print(f"  ω_min={w_min:.3f}: ff={ff:.3f}")

ff_vs_omin = np.array(ff_vs_omin)
corr_omin = np.corrcoef(1/omega_min_vals, ff_vs_omin)[0,1]
print(f"  r(1/ω_min, ff) = {corr_omin:.3f}")
print()

# ════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 2: Влияние κ (параметр связи)
# ════════════════════════════════════════════════════════════

print("── Exp 2: Effect of κ (coupling strength) ──")
print()

kappa_vals = np.array([0.01, 0.02, 0.05, 0.10,
                        0.20, 0.50, 0.97, 1.50])
ff_e8_kappa    = []
ff_rand_kappa  = []

# Фиксируем один random контроль для сравнения
rng_ctrl = np.random.RandomState(999)
rand_omegas = rng_ctrl.uniform(0.1, 2.0, 8)

for kap in kappa_vals:
    # E8
    orbit_e8 = run_monostring(E8_omegas, kappa=kap)
    phi_m, Veff = compute_Veff(orbit_e8)
    valid = ~np.isnan(Veff)
    if valid.sum() > 20:
        V_v = Veff[valid].copy() - Veff[valid].min() + 0.01
        V_s = gaussian_filter1d(V_v, SMOOTH)
        dphi = np.diff(phi_m[valid]).mean()
        dV   = np.gradient(V_s, dphi)
        V_safe = np.where(V_s>1e-10, V_s, 1e-10)
        eps = 0.5*(dV/V_safe)**2
        ff_e8 = float((eps<0.01).mean())
    else:
        ff_e8 = np.nan
    ff_e8_kappa.append(ff_e8)

    # Random
    orbit_r = run_monostring(rand_omegas, kappa=kap)
    phi_m, Veff = compute_Veff(orbit_r)
    valid = ~np.isnan(Veff)
    if valid.sum() > 20:
        V_v = Veff[valid].copy() - Veff[valid].min() + 0.01
        V_s = gaussian_filter1d(V_v, SMOOTH)
        dphi = np.diff(phi_m[valid]).mean()
        dV   = np.gradient(V_s, dphi)
        V_safe = np.where(V_s>1e-10, V_s, 1e-10)
        eps = 0.5*(dV/V_safe)**2
        ff_r = float((eps<0.01).mean())
    else:
        ff_r = np.nan
    ff_rand_kappa.append(ff_r)

    print(f"  κ={kap:.3f}: E8={ff_e8:.3f}, rand={ff_r:.3f}, "
          f"Δ={ff_e8-ff_r:+.3f}")

print()

# ════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 3: n_unique vs ff
# Создаём частоты с контролируемым n_unique
# ════════════════════════════════════════════════════════════

print("── Exp 3: Effect of n_unique frequencies ──")
print("Создаём частоты rank=8 с разным числом уникальных значений")
print()

def make_freqs_with_nunique(n_unique, rank=8, seed=0):
    """rank=8 частот, ровно n_unique различных значений"""
    rng  = np.random.RandomState(seed)
    base = rng.uniform(0.1, 2.0, n_unique)
    # Повторяем base чтобы заполнить rank
    indices = rng.choice(n_unique, rank, replace=True)
    ω = base[indices]
    # Симметризуем (как Коксетер)
    half = rank // 2
    ω_sym = np.concatenate([ω[:half], ω[:half][::-1]])
    return ω_sym

ff_by_nunique = {}
for n_u in [1, 2, 3, 4, 5, 6, 7, 8]:
    ffs_u = []
    for seed in range(30):
        ω = make_freqs_with_nunique(n_u, rank=8, seed=seed)
        ff = get_ff(ω)
        if not np.isnan(ff):
            ffs_u.append(ff)
    ff_by_nunique[n_u] = np.array(ffs_u)
    print(f"  n_unique={n_u}: ff={np.mean(ffs_u):.3f} "
          f"± {np.std(ffs_u):.3f}")

# E8 имеет n_unique=4
print(f"\n  E8 (n_unique=4): ff={E8_ff:.3f}  "
      f"[контроль n_u=4: {ff_by_nunique[4].mean():.3f}]")
print()

# ════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 4: "Fingerprint" — что уникально в E8?
# Перебираем модификации E8 по одной частоте
# ════════════════════════════════════════════════════════════

print("── Exp 4: E8 frequency fingerprint ──")
print("Заменяем по одной паре частот E8 на случайные")
print("Если ff падает → эта пара важна для flatness")
print()

N_REPLACE = 50  # попыток для каждой позиции
rng_fp = np.random.RandomState(777)

# E8 имеет 4 уникальных пары: (0,7), (1,6), (2,5), (3,4)
pairs = [(0,7), (1,6), (2,5), (3,4)]
pair_names = ['ω₁=0.209', 'ω₂=1.338', 'ω₃=1.827', 'ω₄=1.956']

print(f"  E8 baseline: ff={E8_ff:.3f}")
for (i,j), pname in zip(pairs, pair_names):
    ffs_replaced = []
    for trial in range(N_REPLACE):
        ω = E8_omegas.copy()
        new_val = rng_fp.uniform(0.1, 2.0)
        ω[i] = new_val
        ω[j] = new_val
        ff = get_ff(ω)
        if not np.isnan(ff):
            ffs_replaced.append(ff)

    mean_rep = np.mean(ffs_replaced)
    drop = E8_ff - mean_rep
    print(f"  Replace {pname} pair: ff={mean_rep:.3f} "
          f"(drop={drop:+.3f}) "
          f"{'← CRITICAL' if drop > 0.05 else ''}")

print()

# ════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Part XII Step 4: Mechanism Search\n'
             'What makes E8 special for V_eff flatness?',
             fontsize=13)

# 0,0: ff vs ω_min
ax = axes[0,0]
ax.plot(omega_min_vals, ff_vs_omin,
        'o-', color='#e74c3c', lw=2, ms=8, label='E8 modified')
ax.axvline(0.209, color='#e74c3c', ls='--',
           alpha=0.5, label=f'E8 ω_min=0.209')
ax.axhline(E8_ff, color='gray', ls=':', label=f'E8 ff={E8_ff:.3f}')
ax.set_xlabel('ω_min (slowest mode)')
ax.set_ylabel('flat_frac')
ax.set_title(f'Effect of ω_min\nr={corr_omin:.2f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 0,1: ff vs κ
ax = axes[0,1]
kv = np.array(kappa_vals)
fe = np.array(ff_e8_kappa)
fr = np.array(ff_rand_kappa)
ax.plot(kv, fe, 'o-', color='#e74c3c', lw=2, ms=8, label='E8')
ax.plot(kv, fr, 's--', color='gray', lw=2, ms=8, label='Random')
ax.fill_between(kv, fe, fr,
                where=fe>fr, alpha=0.2, color='green',
                label='E8 advantage')
ax.fill_between(kv, fe, fr,
                where=fe<fr, alpha=0.2, color='red',
                label='E8 disadvantage')
ax.axvline(0.05, color='blue', ls=':', alpha=0.5,
           label='current κ')
ax.set_xscale('log')
ax.set_xlabel('κ (log scale)')
ax.set_ylabel('flat_frac')
ax.set_title('ff vs coupling κ: E8 vs Random')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 0,2: n_unique vs ff
ax = axes[0,2]
nu_vals = list(ff_by_nunique.keys())
nu_means = [ff_by_nunique[k].mean() for k in nu_vals]
nu_stds  = [ff_by_nunique[k].std()  for k in nu_vals]
ax.errorbar(nu_vals, nu_means, yerr=nu_stds,
            fmt='o-', color='#3498db', lw=2,
            ms=8, capsize=5, label='Sym random')
ax.scatter([4], [E8_ff], color='#e74c3c',
           s=150, zorder=5, label=f'E8 (n_u=4)',
           edgecolors='k')
ax.set_xlabel('n_unique frequencies')
ax.set_ylabel('flat_frac')
ax.set_title('n_unique vs Flatness (rank=8, symmetric)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1,0: Fingerprint
ax = axes[1,0]
# Перестраиваем данные из Exp 4
drops = []
for (i,j), pname in zip(pairs, pair_names):
    ffs_r = []
    rng2 = np.random.RandomState(888)
    for trial in range(N_REPLACE):
        ω = E8_omegas.copy()
        nv = rng2.uniform(0.1, 2.0)
        ω[i] = nv; ω[j] = nv
        ff = get_ff(ω)
        if not np.isnan(ff):
            ffs_r.append(ff)
    drops.append(E8_ff - np.mean(ffs_r))

bar_colors = ['#e74c3c' if d > 0.05 else '#95a5a6'
              for d in drops]
bars = ax.bar(pair_names, drops, color=bar_colors, alpha=0.8)
ax.axhline(0, color='k', lw=1)
ax.axhline(0.05, color='red', ls='--',
           alpha=0.5, label='Critical threshold')
ax.set_ylabel('Drop in ff when pair replaced')
ax.set_title('E8 Frequency Fingerprint\n(red = critical pair)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, drops):
    ax.text(bar.get_x()+bar.get_width()/2,
            d+0.002, f'{d:+.3f}',
            ha='center', va='bottom', fontsize=9)
ax.tick_params(axis='x', rotation=15)

# 1,1: κ-dependence ratio E8/Random
ax = axes[1,1]
ratio = np.array(ff_e8_kappa) / np.array(ff_rand_kappa)
ax.plot(kappa_vals, ratio, 'o-', color='#e74c3c', lw=2, ms=8)
ax.axhline(1.0, color='k', lw=1, ls='--', label='E8=Random')
ax.axhline(1.5, color='green', lw=1, ls=':', label='+50%')
ax.fill_between(kappa_vals, 1, ratio,
                where=ratio>1, alpha=0.3, color='green')
ax.fill_between(kappa_vals, 1, ratio,
                where=ratio<1, alpha=0.3, color='red')
ax.set_xscale('log')
ax.set_xlabel('κ (log scale)')
ax.set_ylabel('ff(E8) / ff(Random)')
ax.set_title('E8 advantage ratio vs κ')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,2: Сводка
ax = axes[1,2]
ax.axis('off')

# Находим оптимальный κ
best_kappa_idx = np.nanargmax(
    np.array(ff_e8_kappa)/np.array(ff_rand_kappa))

lines = [
    'STEP 4 FINDINGS',
    '─'*32,
    '',
    'Exp 1: ω_min effect',
    f'  r(1/ω_min, ff) = {corr_omin:.3f}',
    f'  {"Strong" if abs(corr_omin)>0.7 else "Weak"} correlation',
    '',
    'Exp 2: κ effect',
    f'  Best κ for E8: {kappa_vals[best_kappa_idx]:.3f}',
    f'  Current κ=0.05: Δff={ff_e8_kappa[2]-ff_rand_kappa[2]:+.3f}',
    '',
    'Exp 3: n_unique effect',
    f'  E8 ff={E8_ff:.3f} vs',
    f'  ctrl(n_u=4)={ff_by_nunique[4].mean():.3f}',
    f'  E8 residual={'above' if E8_ff>ff_by_nunique[4].mean() else "below"}',
    '',
    'MECHANISM:',
]

if abs(corr_omin) > 0.7:
    lines.append('  ω_min is key driver')
elif E8_ff > ff_by_nunique[4].mean() + 0.05:
    lines.append('  Beyond symmetry/spread/n_unique')
    lines.append('  → genuine E8 structure')
else:
    lines.append('  n_unique explains most')
    lines.append('  → E8 not special per se')

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part12_step4_mechanism.png',
            dpi=150, bbox_inches='tight')
print('✓ Saved: part12_step4_mechanism.png')

# ── ФИНАЛЬНЫЙ ВЫВОД ──────────────────────────────────────────

print()
print("="*65)
print("STEP 4 DECISION:")
print("="*65)
print()
print(f"E8 ff = {E8_ff:.3f}")
print(f"Symmetric random (n_u=4): {ff_by_nunique[4].mean():.3f} "
      f"± {ff_by_nunique[4].std():.3f}")
print()
e8_vs_matched = E8_ff - ff_by_nunique[4].mean()
p_vs_matched = float(np.mean(ff_by_nunique[4] >= E8_ff))
print(f"E8 excess over n_unique=4 matched: {e8_vs_matched:+.3f}")
print(f"p-value: {p_vs_matched:.3f}")
print()
if p_vs_matched < 0.05:
    print("→ E8 имеет специфический эффект СВЕРХ симметрии и n_unique")
    print("→ Исследуем κ-зависимость (оптимальный κ для E8)")
    print("→ Step 5: compute n_s at optimal κ")
else:
    print("→ E8 эффект объясняется n_unique=4 и симметрией")
    print("→ Нет специфики E8 как алгебры")
    print("→ Закрываем V_eff, переходим к КАМ-переходам")
