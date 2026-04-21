"""
Part XII Step 5: Critical test at κ=0.01
Исправлена ошибка f-string с условным форматированием
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

SEED   = 42
T      = 100000
WARMUP = 10000
N_BINS = 100
SMOOTH = 3
N_CTRL = 200

np.random.seed(SEED)

ALGEBRAS = {
    'E8': {'m': [1,7,11,13,17,19,23,29], 'h': 30},
    'E6': {'m': [1,4,5,7,8,11],          'h': 12},
    'E7': {'m': [1,5,7,9,11,13,17],      'h': 18},
    'A6': {'m': [1,2,3,4,5,6],           'h':  7},
}

def fmt(x, decimals=3):
    """Безопасное форматирование float/nan"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'nan'
    return f'{x:.{decimals}f}'

def coxeter_freqs(name):
    alg = ALGEBRAS[name]
    m = np.array(alg['m'], dtype=float)
    return 2.0 * np.sin(np.pi * m / alg['h'])

def run_monostring(omegas, kappa, T=T, warmup=WARMUP, seed=SEED):
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, len(omegas))
    orbit = np.zeros((T - warmup, len(omegas)))
    for t in range(T):
        phi = (phi + omegas + kappa*np.sin(phi)) % (2*np.pi)
        if t >= warmup:
            orbit[t-warmup] = phi
    return orbit

def compute_Veff(orbit):
    phi1  = orbit[:, 0]
    phi_r = orbit[:, 1:]
    bins  = np.linspace(0, 2*np.pi, N_BINS+1)
    phi_m = 0.5*(bins[:-1]+bins[1:])
    Veff  = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = (phi1 >= bins[b]) & (phi1 < bins[b+1])
        if mask.sum() > 20:
            diff = phi1[mask, None] - phi_r[mask]
            Veff[b] = -np.cos(diff).sum(axis=1).mean()
    return phi_m, Veff

def get_ff_and_ns(omegas, kappa):
    """
    Возвращает (flat_fraction, n_s, eps_median)
    Все три могут быть nan при проблемах
    """
    try:
        orbit = run_monostring(omegas, kappa)
        phi_m, Veff = compute_Veff(orbit)
        valid = ~np.isnan(Veff)

        if valid.sum() < 30:
            return np.nan, np.nan, np.nan

        phi_v  = phi_m[valid]
        V_v    = Veff[valid].copy()
        V_v    = V_v - V_v.min() + 0.01
        V_s    = gaussian_filter1d(V_v, SMOOTH)

        dphi   = np.diff(phi_v).mean()
        dV     = np.gradient(V_s, dphi)
        d2V    = np.gradient(dV,  dphi)
        V_safe = np.where(V_s > 1e-10, V_s, 1e-10)

        eps = 0.5*(dV/V_safe)**2
        eta = d2V/V_safe

        ff      = float((eps < 0.01).mean())
        eps_med = float(np.median(eps))

        # n_s только там где slow-roll выполняется
        sr_mask = eps < 0.01
        if sr_mask.sum() > 5:
            ns_arr = 1.0 - 6.0*eps[sr_mask] + 2.0*eta[sr_mask]
            ns_val = float(np.median(ns_arr))
        else:
            ns_val = np.nan

        return ff, ns_val, eps_med

    except Exception:
        return np.nan, np.nan, np.nan

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

kappa_test = [0.005, 0.010, 0.020, 0.050]

print("="*65)
print("Part XII Step 5: κ-scan with N=200 controls")
print("="*65)
print(f"T={T}, warmup={WARMUP}, N_ctrl={N_CTRL}")
print()

all_results = {}

for kappa in kappa_test:
    print(f"── κ = {kappa:.3f} ──")

    # Коксетеровы алгебры
    cox = {}
    for name in ALGEBRAS:
        ω = coxeter_freqs(name)
        ff, ns, em = get_ff_and_ns(ω, kappa)
        cox[name] = {'ff': ff, 'ns': ns, 'eps_med': em}
        print(f"  {name}: ff={fmt(ff)}, "
              f"n_s={fmt(ns)}, "
              f"eps_med={fmt(em)}")

    # Контроли rank=8
    print(f"  Running {N_CTRL} random controls...")
    ctrl_ff = []
    ctrl_ns = []

    for i in range(N_CTRL):
        rng = np.random.RandomState(500 + i)
        ω   = rng.uniform(0.1, 2.0, 8)
        ff, ns, _ = get_ff_and_ns(ω, kappa)
        if not np.isnan(ff):
            ctrl_ff.append(ff)
        if not np.isnan(ns):
            ctrl_ns.append(ns)

        # Прогресс каждые 50
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N_CTRL} done...")

    ctrl_ff = np.array(ctrl_ff)
    ctrl_ns = np.array(ctrl_ns) if ctrl_ns else np.array([np.nan])

    e8_ff = cox['E8']['ff']
    e8_ns = cox['E8']['ns']

    # Статистика
    if not np.isnan(e8_ff) and len(ctrl_ff) > 0:
        pct_ff = float(np.mean(ctrl_ff < e8_ff)) * 100
        pval   = float(np.mean(ctrl_ff >= e8_ff))
    else:
        pct_ff = np.nan
        pval   = np.nan

    print(f"  Random ff: {fmt(ctrl_ff.mean())} "
          f"± {fmt(ctrl_ff.std())}, "
          f"max={fmt(ctrl_ff.max())}")
    print(f"  E8 percentile: {fmt(pct_ff, 1)}%,  "
          f"p={fmt(pval, 3)}  "
          f"{'*** SIGNAL' if (not np.isnan(pval)) and pval < 0.01 else '* marginal' if (not np.isnan(pval)) and pval < 0.05 else 'H0 holds'}")

    # n_s vs Planck
    planck_ns  = 0.9649
    planck_err = 0.0042
    if not np.isnan(e8_ns):
        delta_ns = e8_ns - planck_ns
        in_1sig  = abs(delta_ns) < planck_err
        print(f"  E8 n_s={fmt(e8_ns, 4)}, "
              f"Planck={planck_ns}, "
              f"Δ={fmt(delta_ns, 4)}, "
              f"{'IN 1σ !!!' if in_1sig else 'outside 1σ'}")

        if len(ctrl_ns) > 10:
            ns_closer = float(np.mean(
                np.abs(ctrl_ns - planck_ns) > abs(delta_ns)
            )) * 100
            print(f"  E8 closer to Planck than "
                  f"{fmt(ns_closer, 1)}% of random")

    print()

    all_results[kappa] = {
        'cox':     cox,
        'ctrl_ff': ctrl_ff,
        'ctrl_ns': ctrl_ns,
        'pval':    pval,
        'pct':     pct_ff,
    }

# ── ИТОГОВАЯ ТАБЛИЦА ─────────────────────────────────────────

print("="*65)
print("SUMMARY TABLE")
print("="*65)
print()
print(f"{'κ':>7} {'E8_ff':>7} {'rand_ff':>14} "
      f"{'pct':>6} {'p':>7} {'E8_ns':>8} {'Planck?':>9}")
print("-"*65)

planck_ns  = 0.9649
planck_err = 0.0042

for kappa in kappa_test:
    res    = all_results[kappa]
    e8     = res['cox']['E8']
    cf     = res['ctrl_ff']
    pval   = res['pval']
    pct    = res['pct']

    rand_str   = f"{fmt(cf.mean())}±{fmt(cf.std())}"
    in_planck  = (not np.isnan(e8['ns'])) and \
                 (abs(e8['ns'] - planck_ns) < planck_err)
    planck_str = '*** YES' if in_planck else 'no'

    print(f"{kappa:>7.3f} {fmt(e8['ff']):>7} "
          f"{rand_str:>14} "
          f"{fmt(pct,1):>6}% "
          f"{fmt(pval,3):>7} "
          f"{fmt(e8['ns'],4):>8} "
          f"{planck_str:>9}")

print()
print(f"Planck: n_s = {planck_ns} ± {planck_err}")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle(
    'Part XII Step 5: κ-scan — E8 vs N=200 Random controls\n'
    'Searching for inflation signature in weak-coupling regime',
    fontsize=13)

cox_colors = {'E8':'#e74c3c','E6':'#3498db',
              'E7':'#2ecc71','A6':'#9b59b6'}

for col, kappa in enumerate(kappa_test):
    res     = all_results[kappa]
    ctrl_ff = res['ctrl_ff']
    ctrl_ns = res['ctrl_ns']
    cox     = res['cox']
    pval    = res['pval']
    pct     = res['pct']

    # ── Row 0: flat_frac distribution
    ax = axes[0, col]
    if len(ctrl_ff) > 5:
        ax.hist(ctrl_ff, bins=25, color='gray', alpha=0.6,
                label=f'Random\n{fmt(ctrl_ff.mean())}±{fmt(ctrl_ff.std())}')
    for name in ALGEBRAS:
        ff = cox[name]['ff']
        if not np.isnan(ff):
            ax.axvline(ff, color=cox_colors[name],
                      lw=2.5, label=f'{name}={fmt(ff)}')
    sig_str = ('***' if (not np.isnan(pval)) and pval < 0.01
               else '*' if (not np.isnan(pval)) and pval < 0.05
               else '')
    ax.set_title(f'κ={kappa}  p={fmt(pval,3)} {sig_str}',
                 fontsize=10)
    ax.set_xlabel('flat_frac')
    ax.set_ylabel('Count')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # ── Row 1: n_s distribution
    ax = axes[1, col]
    if len(ctrl_ns) > 10 and not np.all(np.isnan(ctrl_ns)):
        ns_clean = ctrl_ns[~np.isnan(ctrl_ns)]
        if len(ns_clean) > 5:
            ax.hist(ns_clean, bins=25, color='gray',
                    alpha=0.6, label='Random')
    ax.axvline(planck_ns, color='black', lw=2,
               ls='--', label='Planck 0.9649')
    ax.axvspan(planck_ns - planck_err,
               planck_ns + planck_err,
               alpha=0.15, color='blue', label='Planck 1σ')
    for name in ALGEBRAS:
        ns = cox[name]['ns']
        if not np.isnan(ns):
            ax.axvline(ns, color=cox_colors[name],
                      lw=2, label=f'{name}={fmt(ns,3)}')
    ax.set_title(f'n_s  κ={kappa}', fontsize=10)
    ax.set_xlabel('n_s')
    ax.set_ylabel('Count')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # ── Row 2: E8 vs Random scatter (ff vs n_s)
    ax = axes[2, col]
    ns_clean = ctrl_ns[~np.isnan(ctrl_ns)] \
        if len(ctrl_ns) > 0 else np.array([])
    ff_for_ns = ctrl_ff[:len(ns_clean)] \
        if len(ns_clean) > 0 else np.array([])

    if len(ns_clean) > 5:
        ax.scatter(ns_clean, ff_for_ns,
                   color='gray', alpha=0.3, s=15,
                   label='Random')

    for name in ALGEBRAS:
        ff = cox[name]['ff']
        ns = cox[name]['ns']
        if not np.isnan(ff) and not np.isnan(ns):
            ax.scatter(ns, ff, color=cox_colors[name],
                      s=150, zorder=5, label=name,
                      edgecolors='k', linewidths=1)

    ax.axvline(planck_ns, color='black', lw=1.5,
               ls='--', alpha=0.7)
    ax.axvspan(planck_ns - planck_err,
               planck_ns + planck_err,
               alpha=0.1, color='blue')
    ax.set_xlabel('n_s')
    ax.set_ylabel('flat_frac')
    ax.set_title(f'ff vs n_s  κ={kappa}', fontsize=10)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part12_step5_kappa_scan.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part12_step5_kappa_scan.png")

# ── ФИНАЛЬНЫЙ ВЕРДИКТ ────────────────────────────────────────

print()
print("="*65)
print("FINAL VERDICT Part XII Step 5")
print("="*65)
print()

# Лучший κ по p-value
valid_kappas = [(k, all_results[k]['pval'])
                for k in kappa_test
                if not np.isnan(all_results[k]['pval'])]

if valid_kappas:
    best_k, best_p = min(valid_kappas, key=lambda x: x[1])
    best_p_bonf    = best_p * len(kappa_test)

    print(f"Лучший κ: {best_k},  p={best_p:.4f}")
    print(f"Bonferroni (×{len(kappa_test)}): p*={best_p_bonf:.4f}")
    print()

    # Проверяем n_s при лучшем κ
    best_ns = all_results[best_k]['cox']['E8']['ns']
    if not np.isnan(best_ns):
        in_planck = abs(best_ns - planck_ns) < planck_err
        print(f"E8 n_s при κ={best_k}: {fmt(best_ns, 4)}")
        print(f"Попадает в Planck 1σ: {'ДА !!!' if in_planck else 'НЕТ'}")
        print()

    if best_p_bonf < 0.01:
        print("*** СИЛЬНЫЙ СИГНАЛ (Bonferroni-corrected)")
        print("    E8 значимо отличается от random при слабой связи")
        print("    → Проверяем механизм: Step 6 (КАМ + V_eff combined)")
    elif best_p_bonf < 0.05:
        print("*   СЛАБЫЙ СИГНАЛ (не выживает Bonferroni строго)")
        print("    → Увеличить N_CTRL=500, T=200000 для подтверждения")
    else:
        print("    H₀ не отвергается")
        print("    → Закрываем V_eff направление")
        print("    → Переходим к КАМ-переходам (направление №1)")

    # Проверка: n_s в Planck при любом κ?
    print()
    print("n_s в Planck 1σ:")
    for kappa in kappa_test:
        ns = all_results[kappa]['cox']['E8']['ns']
        if not np.isnan(ns):
            in_p = abs(ns - planck_ns) < planck_err
            print(f"  κ={kappa}: n_s={fmt(ns,4)}  "
                  f"{'<<< IN PLANCK 1σ' if in_p else ''}")