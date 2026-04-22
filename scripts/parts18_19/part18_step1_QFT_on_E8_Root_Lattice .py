"""
Part XVIII: Quantum Monostring — QFT on E8 Root Lattice
Исправлена ошибка: убран параметр warmup из вызова
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

SEED = 42
np.random.seed(SEED)

# ── ГЕНЕРАЦИЯ КОРНЕЙ E8 ──────────────────────────────────────

def generate_E8_roots():
    """
    240 корней E8 в R^8:
    - 112 корней вида (±1,±1,0,...,0) — все перестановки пар
    - 128 корней вида (±½,...,±½) — нечётное число минусов
    """
    roots = []

    # Тип 1: ±eᵢ ± eⱼ, i < j  → 2 × C(8,2) = 112 корней
    for i in range(8):
        for j in range(i+1, 8):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)

    # Тип 2: (±½)^8 с нечётным числом минусов → 128 корней
    for bits in range(256):
        signs = np.array([(( bits >> i) & 1)*2 - 1
                          for i in range(8)], dtype=float)
        n_minus = np.sum(signs < 0)
        if n_minus % 2 == 1:          # нечётное число −1
            roots.append(0.5 * signs)

    roots = np.array(roots)
    print(f"  Type-1 roots: 112, Type-2 roots: 128")
    print(f"  Total: {len(roots)}")
    print(f"  All norms²: {np.unique(np.round((roots**2).sum(1),4))}")
    return roots

# ── ПРОВЕРКА КОРНЕВОЙ СИСТЕМЫ ────────────────────────────────

def verify_E8(roots):
    """Базовая проверка: все корни должны иметь норму √2."""
    norms_sq = (roots**2).sum(axis=1)
    all_norm2 = np.allclose(norms_sq, 2.0, atol=1e-10)
    inner_products = np.dot(roots, roots.T)
    unique_ip = np.unique(np.round(inner_products, 4))
    print(f"  All |α|²=2: {all_norm2}")
    print(f"  Unique inner products: {unique_ip}")
    return all_norm2

# ── МАТРИЦА ВЗАИМОДЕЙСТВИЙ ───────────────────────────────────

def build_coupling_matrix(roots):
    """
    C[i,j] = ⟨αᵢ, αⱼ⟩ — скалярные произведения корней.
    Это НЕ матрица Картана (та только для простых корней).
    Это полная матрица связей: 240×240.
    """
    return roots @ roots.T   # (240, 240)

# ── ДИНАМИКА ─────────────────────────────────────────────────

def run_E8_dynamics(roots, C, kappa=1.0, T=50000,
                    warmup=5000, dt=0.01, seed=SEED):
    """
    Velocity-Verlet интегратор для гамильтониана:
      H = ½Σpᵢ² + κ·ΣCᵢⱼ·cos(φᵢ−φⱼ)

    Сила: Fᵢ = −∂H/∂φᵢ = κ·Σⱼ Cᵢⱼ·sin(φᵢ−φⱼ)
    """
    rng    = np.random.RandomState(seed)
    n      = len(roots)

    phi    = rng.uniform(0, 2*np.pi, n)
    p      = rng.normal(0, 0.1, n)

    def force(phi_):
        diff  = phi_[:, None] - phi_[None, :]   # (n,n)
        return kappa * (C * np.sin(diff)).sum(axis=1)

    orbit_phi = np.zeros((T - warmup, n))
    orbit_p   = np.zeros((T - warmup, n))

    E_list = []   # для контроля сохранения энергии

    for t in range(T):
        F   = force(phi)
        phi = phi + p*dt + 0.5*F*dt**2
        F2  = force(phi)
        p   = p + 0.5*(F + F2)*dt

        if t >= warmup:
            orbit_phi[t - warmup] = phi % (2*np.pi)
            orbit_p[t - warmup]   = p

            # Энергия (каждые 1000 шагов)
            if (t - warmup) % 1000 == 0:
                E_kin = 0.5*(p**2).sum()
                diff  = phi[:,None] - phi[None,:]
                E_pot = kappa*(C*np.cos(diff)).sum()
                E_list.append(E_kin + E_pot)

    return orbit_phi, orbit_p, np.array(E_list)

# ── СПЕКТР МОЩНОСТИ ──────────────────────────────────────────

def power_spectrum_and_ns(orbit, fs=1.0):
    """
    Вычисляет усреднённый спектр мощности по всем модам.
    Оценивает n_s через логарифмический наклон P(k).

    n_s - 1 = d ln P / d ln k
    """
    n_modes = orbit.shape[1]
    all_P   = []

    for i in range(n_modes):
        f, P = welch(orbit[:, i],
                     fs=fs,
                     nperseg=min(1024, len(orbit)//4),
                     return_onesided=True)
        all_P.append(P)

    P_avg = np.mean(all_P, axis=0)

    # Логарифмический наклон (исключаем f=0 и края)
    mask = (f > f[1]) & (f < f[-1]*0.9) & (P_avg > 0)
    if mask.sum() > 10:
        log_f   = np.log(f[mask])
        log_P   = np.log(P_avg[mask])
        coeff   = np.polyfit(log_f, log_P, 1)
        slope   = coeff[0]
        ns_est  = 1.0 + slope   # n_s - 1 = slope
    else:
        ns_est = np.nan

    return f, P_avg, ns_est

# ── СЛУЧАЙНЫЙ КОНТРОЛЬ ───────────────────────────────────────

def run_random_control(rank=8, kappa=1.0, T=20000,
                        warmup=2000, dt=0.01, seed=999):
    """
    Аналогичная динамика, но с random coupling matrix
    той же размерности rank×rank (не 240×240).
    Быстрый контроль.
    """
    rng  = np.random.RandomState(seed)
    n    = rank
    A    = rng.randn(n, n)
    C_r  = (A + A.T) / 2   # симметричная

    phi  = rng.uniform(0, 2*np.pi, n)
    p    = rng.normal(0, 0.1, n)

    def force(phi_):
        diff = phi_[:,None] - phi_[None,:]
        return kappa*(C_r*np.sin(diff)).sum(axis=1)

    orbit = np.zeros((T - warmup, n))
    for t in range(T):
        F    = force(phi)
        phi  = phi + p*dt + 0.5*F*dt**2
        F2   = force(phi)
        p    = p + 0.5*(F + F2)*dt
        if t >= warmup:
            orbit[t-warmup] = phi % (2*np.pi)

    f, P, ns = power_spectrum_and_ns(orbit)
    return ns

# ── ОСНОВНОЙ ЭКСПЕРИМЕНТ ─────────────────────────────────────

print("="*65)
print("Part XVIII: QFT on E8 Root Lattice")
print("="*65)
print()

print("── Step 1: Generate and verify E8 roots ──")
roots = generate_E8_roots()
print()
print("── Step 2: Verify root system ──")
verify_E8(roots)
print()

print("── Step 3: Build coupling matrix 240×240 ──")
C = build_coupling_matrix(roots)
print(f"  C shape: {C.shape}")
print(f"  C diagonal (all = |α|²=2): "
      f"{np.unique(np.round(np.diag(C),4))}")
print(f"  C off-diag unique values: "
      f"{np.unique(np.round(C[~np.eye(240,dtype=bool)],2))}")
print()

# Короткий тест для проверки кода (T=5000)
print("── Step 4: Short test run κ=0.5 (T=5000) ──")
orbit_test, _, E_test = run_E8_dynamics(
    roots, C, kappa=0.5,
    T=5000, warmup=500, dt=0.01)
print(f"  Orbit shape: {orbit_test.shape}")
print(f"  Energy drift: "
      f"{(E_test[-1]-E_test[0])/abs(E_test[0])*100:.2f}%")
print(f"  φ range: [{orbit_test.min():.3f}, "
      f"{orbit_test.max():.3f}]")
print()

# Основные запуски (T=30000 — компромисс скорость/точность)
print("── Step 5: Main experiment (T=30000) ──")
print()

kappa_vals = [0.05, 0.10, 0.50, 1.00]
results    = {}

PLANCK_NS    = 0.9649
PLANCK_NS_1S = 0.0042

for kappa in kappa_vals:
    print(f"  κ={kappa:.2f}...", end='', flush=True)
    orbit, _, E_hist = run_E8_dynamics(
        roots, C, kappa=kappa,
        T=30000, warmup=3000, dt=0.01)

    f, P_avg, ns = power_spectrum_and_ns(orbit)

    E_drift = ((E_hist[-1]-E_hist[0]) /
               max(abs(E_hist[0]),1e-10)) * 100

    results[kappa] = {
        'orbit': orbit, 'f': f,
        'P': P_avg, 'ns': ns,
        'E_drift': E_drift
    }

    in_planck = (not np.isnan(ns) and
                 abs(ns - PLANCK_NS) < PLANCK_NS_1S)
    flag = ' ← IN PLANCK 1σ!!!' if in_planck else ''

    print(f" n_s={ns:.4f}, "
          f"E_drift={E_drift:.2f}%{flag}")

print()

# Random контроли (быстрые, rank=8)
print("── Step 6: Random controls (rank=8, N=20) ──")
rand_ns = []
for i in range(20):
    ns_r = run_random_control(
        rank=8, kappa=0.5,
        T=10000, warmup=1000, seed=100+i)
    rand_ns.append(ns_r)
rand_ns = np.array([x for x in rand_ns
                    if not np.isnan(x)])
print(f"  Random n_s: {rand_ns.mean():.4f} "
      f"± {rand_ns.std():.4f}")
print()

# ── СТАТИСТИКА ───────────────────────────────────────────────

print("── Statistical comparison ──")
print()
print(f"{'κ':>6} {'E8 n_s':>8} {'Δ Planck':>10} "
      f"{'pct vs rand':>13} {'Signal?':>10}")
print("-"*55)

for kappa in kappa_vals:
    ns = results[kappa]['ns']
    if np.isnan(ns):
        print(f"{kappa:>6.2f}    nan")
        continue
    delta   = ns - PLANCK_NS
    pct     = float(np.mean(rand_ns < ns))*100
    in_1sig = abs(delta) < PLANCK_NS_1S
    flag    = '*** YES' if in_1sig else ''
    print(f"{kappa:>6.2f} {ns:>8.4f} {delta:>+10.4f} "
          f"{pct:>12.0f}% {flag:>10}")

print()
print(f"Planck: n_s = {PLANCK_NS} ± {PLANCK_NS_1S}")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    'Part XVIII: Quantum Monostring — E8 Root Lattice\n'
    'Power spectrum and spectral index n_s',
    fontsize=13)

colors = {0.05:'#1abc9c', 0.10:'#3498db',
          0.50:'#e74c3c', 1.00:'#9b59b6'}

# 0,0: Корни E8 (проекция на первые 2 оси)
ax = axes[0,0]
type1 = roots[np.abs(roots).max(axis=1) > 0.6]
type2 = roots[np.abs(roots).max(axis=1) <= 0.6]
ax.scatter(type1[:,0], type1[:,1],
           s=20, color='blue', alpha=0.6,
           label=f'Type-1 (±1): {len(type1)}')
ax.scatter(type2[:,0], type2[:,1],
           s=20, color='red', alpha=0.6,
           label=f'Type-2 (±½): {len(type2)}')
ax.set_xlabel('α₁')
ax.set_ylabel('α₂')
ax.set_title('E8 roots (projection α₁-α₂)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 0,1: Coupling matrix C
ax = axes[0,1]
# Показываем первые 40×40 для читаемости
C_sub = C[:40, :40]
im = ax.imshow(C_sub, cmap='RdBu',
               vmin=-2, vmax=2)
plt.colorbar(im, ax=ax)
ax.set_title('Coupling matrix C[i,j]=⟨αᵢ,αⱼ⟩\n(first 40×40)')
ax.set_xlabel('Root j')
ax.set_ylabel('Root i')

# 0,2: Спектры мощности
ax = axes[0,2]
for kappa in kappa_vals:
    f_arr = results[kappa]['f']
    P_arr = results[kappa]['P']
    ns    = results[kappa]['ns']
    mask  = f_arr > 0
    ax.loglog(f_arr[mask], P_arr[mask],
              color=colors[kappa], lw=2,
              label=f'κ={kappa}, n_s={ns:.3f}')
ax.set_xlabel('Frequency k')
ax.set_ylabel('Power P(k)')
ax.set_title('Power spectra — all κ')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, which='both')

# 1,0: n_s vs κ
ax = axes[1,0]
kv  = kappa_vals
nsv = [results[k]['ns'] for k in kv]
ax.plot(kv, nsv, 'o-', color='#e74c3c',
        lw=2, ms=10, label='E8 n_s')
ax.axhline(PLANCK_NS, color='green', lw=2,
           ls='--', label=f'Planck {PLANCK_NS}')
ax.axhspan(PLANCK_NS - PLANCK_NS_1S,
           PLANCK_NS + PLANCK_NS_1S,
           alpha=0.2, color='green',
           label='Planck 1σ')
ax.set_xlabel('κ (coupling)')
ax.set_ylabel('n_s')
ax.set_title('Spectral index vs coupling')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,1: Random control histogram
ax = axes[1,1]
ax.hist(rand_ns, bins=15, color='gray',
        alpha=0.7,
        label=f'Random rank=8\n'
              f'({rand_ns.mean():.3f}±{rand_ns.std():.3f})')
for kappa in kappa_vals:
    ns = results[kappa]['ns']
    if not np.isnan(ns):
        ax.axvline(ns, color=colors[kappa],
                   lw=2.5,
                   label=f'E8 κ={kappa}: {ns:.3f}')
ax.axvline(PLANCK_NS, color='black', lw=2,
           ls='--', label='Planck')
ax.axvspan(PLANCK_NS - PLANCK_NS_1S,
           PLANCK_NS + PLANCK_NS_1S,
           alpha=0.15, color='green')
ax.set_xlabel('n_s')
ax.set_ylabel('Count')
ax.set_title('E8 vs Random n_s distribution')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 1,2: Сводка
ax = axes[1,2]
ax.axis('off')

lines = [
    'PART XVIII SUMMARY',
    '─'*30,
    '',
    'H = ½Σpᵢ² + κΣCᵢⱼcos(φᵢ−φⱼ)',
    f'N_roots = {len(roots)}',
    f'C[i,j] = ⟨αᵢ,αⱼ⟩ ∈ {{-2,-1,0,1,2}}',
    '',
    'n_s results:',
]
for kappa in kappa_vals:
    ns    = results[kappa]['ns']
    delta = ns - PLANCK_NS if not np.isnan(ns) else np.nan
    sig   = '***' if abs(delta) < PLANCK_NS_1S else ''
    lines.append(f'  κ={kappa:.2f}: {ns:.4f} '
                 f'(Δ={delta:+.4f}) {sig}')

lines += [
    '',
    f'Random: {rand_ns.mean():.4f}±{rand_ns.std():.4f}',
    '',
    'VERDICT:',
]

# Находим лучший κ
valid_ns = {k: results[k]['ns'] for k in kappa_vals
            if not np.isnan(results[k]['ns'])}
if valid_ns:
    best_k  = min(valid_ns,
                  key=lambda k: abs(valid_ns[k]-PLANCK_NS))
    best_ns = valid_ns[best_k]
    if abs(best_ns - PLANCK_NS) < PLANCK_NS_1S:
        color = '#e0ffe0'
        lines += ['*** IN PLANCK 1σ ***',
                  f'κ={best_k}: n_s={best_ns:.4f}',
                  '→ Need r computation',
                  '→ Part XIX: full analysis']
    elif abs(best_ns - PLANCK_NS) < 0.05:
        color = '#fffde0'
        lines += [f'Close: κ={best_k}',
                  f'n_s={best_ns:.4f}',
                  f'Δ={best_ns-PLANCK_NS:+.4f}',
                  '→ Fine-tune κ',
                  '→ Longer T for precision']
    else:
        color = '#ffe0e0'
        lines += ['Not close to Planck.',
                  '→ Try different dt or T',
                  '→ Or different H form']
else:
    color = '#ffe0e0'
    lines += ['No convergence.']

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.8))

plt.tight_layout()
plt.savefig('part18_E8_quantum.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: part18_E8_quantum.png")

# ── ФИНАЛ ────────────────────────────────────────────────────

print()
print("="*65)
print("PART XVIII VERDICT")
print("="*65)
print()

valid = {k: v['ns'] for k, v in results.items()
         if not np.isnan(v['ns'])}

if not valid:
    print("Нет сходимости ни при одном κ.")
    print("Попробовать: меньший dt, другие нач. условия.")
else:
    best_k  = min(valid, key=lambda k: abs(valid[k]-PLANCK_NS))
    best_ns = valid[best_k]
    delta   = best_ns - PLANCK_NS

    print(f"Лучший результат: κ={best_k}, n_s={best_ns:.4f}")
    print(f"Расстояние от Planck: Δ={delta:+.4f} "
          f"({abs(delta)/PLANCK_NS_1S:.1f}σ)")
    print()

    if abs(delta) < PLANCK_NS_1S:
        print("*** ПОПАДАНИЕ В PLANCK 1σ! ***")
        print("Это потенциально значимый результат.")
        print()
        print("ОБЯЗАТЕЛЬНО проверить:")
        print("  1. Не артефакт ли n_s оценки?")
        print("  2. Стабильно ли при большем T?")
        print("  3. Зависит ли от seed?")
        print("  4. Какой контроль (random) даёт?")
        print()
        print("→ Part XIX: детальная проверка")
    elif abs(delta) < 0.05:
        print("Близко к Planck, но вне 1σ.")
        print(f"Нужно улучшить: |Δ|={abs(delta):.4f} > {PLANCK_NS_1S:.4f}")
        print()
        print("Варианты:")
        print("  → Тонкая настройка κ (бинарный поиск)")
        print("  → Более длинная траектория (T=100000)")
        print("  → Другая форма гамильтониана")
    else:
        print("Далеко от Planck. H₀ не отвергается.")
        print()
        print("Варианты:")
        print("  → Квантовая (матричная) версия")
        print("  → Матрица Картана вместо полной C")
        print("  → Публиковать как есть (Parts I-XVII)")