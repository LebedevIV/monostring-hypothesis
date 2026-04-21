"""
Part XVI: Cartan Matrix Dynamics
Гамильтониан определяется матрицей Картана.

Физика:
  Матрица Картана C_ij кодирует углы между
  простыми корнями. Это НЕ произвольно.

  E8 матрица Картана — уникальна в том смысле,
  что det(C_E8) = 1 (унимодулярна).

Вопрос: даёт ли динамика с C_E8 инфляцию?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

SEED   = 42
np.random.seed(SEED)

# ── МАТРИЦЫ КАРТАНА ──────────────────────────────────────────

# E8 матрица Картана (точная, из таблиц)
C_E8 = np.array([
    [ 2, -1,  0,  0,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0,  0,  0],
    [ 0, -1,  2, -1,  0,  0,  0, -1],
    [ 0,  0, -1,  2, -1,  0,  0,  0],
    [ 0,  0,  0, -1,  2, -1,  0,  0],
    [ 0,  0,  0,  0, -1,  2, -1,  0],
    [ 0,  0,  0,  0,  0, -1,  2,  0],
    [ 0,  0, -1,  0,  0,  0,  0,  2]
], dtype=float)

C_E6 = np.array([
    [ 2, -1,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0, -1],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2,  0],
    [ 0,  0, -1,  0,  0,  2]
], dtype=float)

C_E7 = np.array([
    [ 2, -1,  0,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0,  0],
    [ 0, -1,  2, -1,  0,  0, -1],
    [ 0,  0, -1,  2, -1,  0,  0],
    [ 0,  0,  0, -1,  2, -1,  0],
    [ 0,  0,  0,  0, -1,  2,  0],
    [ 0,  0, -1,  0,  0,  0,  2]
], dtype=float)

C_A6 = np.array([
    [ 2, -1,  0,  0,  0,  0],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0,  0],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2, -1],
    [ 0,  0,  0,  0, -1,  2]
], dtype=float)

CARTAN = {
    'E8': C_E8, 'E6': C_E6,
    'E7': C_E7, 'A6': C_A6
}

# ── СВОЙСТВА МАТРИЦ КАРТАНА ──────────────────────────────────

print("="*65)
print("Part XVI: Cartan Matrix Dynamics")
print("="*65)
print()
print("── Properties of Cartan matrices ──")
print()

for name, C in CARTAN.items():
    det   = np.linalg.det(C)
    eigs  = np.linalg.eigvalsh(C)
    rank  = C.shape[0]
    trace = np.trace(C)
    print(f"{name} (rank={rank}):")
    print(f"  det={det:.1f}, trace={trace:.0f}")
    print(f"  eigenvalues: {np.round(eigs,3)}")
    print(f"  min_eig={eigs.min():.4f} "
          f"(>0 → positive definite)")
    print()

# ── ДИНАМИКА С МАТРИЦЕЙ КАРТАНА ──────────────────────────────

def run_cartan_dynamics(C, T=50000, warmup=5000,
                         kappa=0.5, dt=0.01, seed=SEED):
    """
    Непрерывная динамика:
      dφᵢ/dt = pᵢ
      dpᵢ/dt = -Σⱼ Cᵢⱼ · sin(φᵢ - φⱼ)

    Интегрирование методом Рунге-Кутта 4-го порядка.
    kappa масштабирует силу взаимодействия.
    """
    rng  = np.random.RandomState(seed)
    r    = C.shape[0]
    phi  = rng.uniform(0, 2*np.pi, r)
    p    = rng.normal(0, 0.1, r)

    orbit_phi = np.zeros((T - warmup, r))
    orbit_p   = np.zeros((T - warmup, r))

    def force(phi_):
        # F_i = -Σ_j C_ij · sin(φ_i - φ_j)
        diff  = phi_[:, None] - phi_[None, :]
        return -kappa * (C * np.sin(diff)).sum(axis=1)

    for t in range(T):
        # Velocity Verlet (симплектический)
        F    = force(phi)
        phi  = phi + p*dt + 0.5*F*dt**2
        F2   = force(phi)
        p    = p + 0.5*(F + F2)*dt

        if t >= warmup:
            orbit_phi[t - warmup] = phi
            orbit_p[t - warmup]   = p

    return orbit_phi, orbit_p

print("── Running Cartan dynamics ──")
print()

kappa_vals = [0.1, 0.5, 1.0, 2.0]
results    = {}

for name in CARTAN:
    C    = CARTAN[name]
    r    = C.shape[0]
    results[name] = {}

    for kappa in kappa_vals:
        orbit_phi, orbit_p = run_cartan_dynamics(
            C, kappa=kappa, T=20000, warmup=2000)

        # Энергия: H = Σpᵢ²/2 + Σᵢⱼ Cᵢⱼcos(φᵢ-φⱼ)
        E_kin = 0.5 * (orbit_p**2).sum(axis=1)
        phi   = orbit_phi
        diff  = phi[:, :, None] - phi[:, None, :]
        E_pot = -(CARTAN[name][None,:,:]*np.cos(diff)).sum(axis=(1,2))
        E_tot = E_kin + E_pot

        # Ляпунов через разброс траекторий
        phi_std = orbit_phi.std(axis=1)  # std по модам

        # Темп диффузии: d<σ²>/dt
        t_arr  = np.arange(len(phi_std))
        if len(t_arr) > 100:
            coeff  = np.polyfit(t_arr, phi_std, 1)
            spread_rate = coeff[0]
        else:
            spread_rate = 0.0

        # Энергия монотонна? (консервативность)
        E_drift = (E_tot[-100:].mean() - E_tot[:100].mean())

        results[name][kappa] = {
            'E_drift':     E_drift,
            'spread_rate': spread_rate,
            'E_std':       E_tot.std(),
            'phi_range':   np.ptp(orbit_phi, axis=0).mean()
        }

        print(f"  {name} κ={kappa:.1f}: "
              f"E_drift={E_drift:.4f}, "
              f"spread={spread_rate:.4f}/step, "
              f"φ_range={results[name][kappa]['phi_range']:.2f}")

    print()

# ── СРАВНЕНИЕ С RANDOM МАТРИЦАМИ ─────────────────────────────

print("── Random positive-definite matrices (control) ──")
print()

N_RAND = 20
rand_results = {kappa: [] for kappa in kappa_vals}

for i in range(N_RAND):
    rng = np.random.RandomState(1000+i)
    # Случайная SPD матрица rank=8
    A = rng.randn(8, 8)
    C_rand = A @ A.T / 8 + np.eye(8)*0.5
    # Нормировка: диагональ = 2 (как у Картана)
    D = np.diag(C_rand).copy()
    C_rand = C_rand / D[:, None] * 2

    for kappa in [0.5, 1.0]:
        orbit_phi, orbit_p = run_cartan_dynamics(
            C_rand, kappa=kappa,
            T=20000, warmup=2000, seed=2000+i)

        phi_std = orbit_phi.std(axis=1)
        t_arr   = np.arange(len(phi_std))
        coeff   = np.polyfit(t_arr, phi_std, 1)
        rand_results[kappa].append(coeff[0])

for kappa in [0.5, 1.0]:
    arr = np.array(rand_results[kappa])
    print(f"  κ={kappa}: spread_rate = "
          f"{arr.mean():.4f} ± {arr.std():.4f}")

print()

# ── КЛЮЧЕВОЕ СРАВНЕНИЕ ────────────────────────────────────────

print("── E8 vs Random: spread_rate comparison ──")
print()

for kappa in [0.5, 1.0]:
    rand_arr = np.array(rand_results[kappa])
    print(f"κ={kappa}:")
    for name in CARTAN:
        sr  = results[name][kappa]['spread_rate']
        pct = float(np.mean(rand_arr < sr))*100
        pv  = float(np.mean(rand_arr >= sr))
        print(f"  {name}: spread={sr:.4f}  "
              f"pct={pct:.0f}%  p={pv:.3f}  "
              f"{'* SIG' if pv<0.05 else ''}")
    print()

# ── ПОИСК ИНФЛЯЦИОННОГО РЕЖИМА ───────────────────────────────

print("── Inflation search: growing energy mode? ──")
print()

for name in ['E8']:
    C = CARTAN[name]
    for kappa in [0.5, 1.0, 2.0]:
        # Долгая траектория
        orbit_phi, orbit_p = run_cartan_dynamics(
            C, kappa=kappa, T=50000, warmup=5000)

        # PCA: ищем растущую коллективную моду
        from numpy.linalg import svd
        U, S, Vt = svd(orbit_phi - orbit_phi.mean(axis=0),
                       full_matrices=False)

        # Проекция на первую главную компоненту
        pc1 = orbit_phi @ Vt[0]

        # Растёт ли pc1?
        t = np.arange(len(pc1))
        # Логарифм RMS
        window = 500
        rms = np.array([
            np.std(pc1[max(0,i-window):i+window])
            for i in range(0, len(pc1), 100)
        ])
        t_rms = np.arange(len(rms))
        if len(t_rms) > 10:
            coeff = np.polyfit(t_rms, np.log(rms+1e-10), 1)
            growth = coeff[0]
        else:
            growth = 0.0

        print(f"  E8 κ={kappa}: "
              f"PC1 growth = {growth:.6f}/window, "
              f"e-folds per 10^5 = {growth*1000:.2f}")

# ── ВИЗУАЛИЗАЦИЯ ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Part XVI: Cartan Matrix Dynamics\n'
             'E8 vs Random positive-definite matrices',
             fontsize=13)

cox_colors = {'E8':'#e74c3c','E6':'#3498db',
              'E7':'#2ecc71','A6':'#9b59b6'}

# 0,0: Матрица Картана E8
ax = axes[0,0]
im = ax.imshow(C_E8, cmap='RdBu', vmin=-2, vmax=2)
plt.colorbar(im, ax=ax)
ax.set_title('E8 Cartan Matrix\ndet=1, all eigenvalues>0')
ax.set_xlabel('Node i')
ax.set_ylabel('Node j')
for i in range(8):
    for j in range(8):
        ax.text(j, i, f'{int(C_E8[i,j])}',
                ha='center', va='center', fontsize=9)

# 0,1: Спектр матриц Картана
ax = axes[0,1]
for name, C in CARTAN.items():
    eigs = np.sort(np.linalg.eigvalsh(C))
    ax.plot(range(len(eigs)), eigs,
            'o-', color=cox_colors[name],
            lw=2, ms=8, label=name)
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Eigenvalue')
ax.set_title('Cartan matrix spectrum\n(all positive → stable)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=1, ls='--')

# 0,2: spread_rate vs κ
ax = axes[0,2]
for name in CARTAN:
    kv = kappa_vals
    sv = [results[name][k]['spread_rate'] for k in kv]
    ax.plot(kv, sv, 'o-', color=cox_colors[name],
            lw=2, ms=8, label=name)

# Random band
for kappa in [0.5, 1.0]:
    arr = np.array(rand_results[kappa])
    ax.errorbar([kappa], [arr.mean()],
                yerr=[arr.std()],
                fmt='s', color='gray',
                ms=10, capsize=5, zorder=5)

ax.set_xlabel('κ')
ax.set_ylabel('Spread rate (per step)')
ax.set_title('Phase spread rate vs κ')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,0: Фазовая траектория E8 при оптимальном κ
ax = axes[1,0]
best_kappa = max(kappa_vals,
    key=lambda k: results['E8'][k]['spread_rate'])
orbit_vis, _ = run_cartan_dynamics(
    C_E8, kappa=best_kappa, T=10000, warmup=1000)

for i in range(min(4, orbit_vis.shape[1])):
    ax.plot(orbit_vis[:500, i], alpha=0.7,
            label=f'φ_{i+1}')
ax.set_xlabel('Time step')
ax.set_ylabel('φᵢ')
ax.set_title(f'E8 phase trajectories (κ={best_kappa})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,1: Energy E8
ax = axes[1,1]
orbit_phi2, orbit_p2 = run_cartan_dynamics(
    C_E8, kappa=1.0, T=20000, warmup=2000)
E_kin = 0.5*(orbit_p2**2).sum(axis=1)
diff  = orbit_phi2[:,:,None] - orbit_phi2[:,None,:]
E_pot = -(C_E8[None,:,:]*np.cos(diff)).sum(axis=(1,2))
E_tot = E_kin + E_pot
ax.plot(E_tot, 'r-', lw=1, alpha=0.7)
ax.set_xlabel('Time step')
ax.set_ylabel('Total energy H')
ax.set_title('E8 energy conservation (κ=1.0)')
ax.grid(True, alpha=0.3)

# 1,2: Сводка
ax = axes[1,2]
ax.axis('off')

lines = [
    'PART XVI RESULTS',
    '─'*30,
    '',
    'Cartan matrix properties:',
]
for name, C in CARTAN.items():
    det  = np.linalg.det(C)
    emin = np.linalg.eigvalsh(C).min()
    lines.append(f'  {name}: det={det:.1f}, '
                 f'λ_min={emin:.3f}')

lines += ['', 'Spread rates at κ=1.0:']
rand10 = np.array(rand_results[1.0])
for name in CARTAN:
    sr  = results[name][1.0]['spread_rate']
    pv  = float(np.mean(rand10 >= sr))
    lines.append(f'  {name}: {sr:.4f} p={pv:.3f}')

lines += [
    f'  Random: {rand10.mean():.4f}±{rand10.std():.4f}',
    '',
    'VERDICT:',
]

e8_sr   = results['E8'][1.0]['spread_rate']
e8_pval = float(np.mean(rand10 >= e8_sr))
if e8_pval < 0.05:
    lines += ['  E8 SIGNAL: p<0.05',
              '  E8 spreads differently',
              '  from random matrices!']
else:
    lines += ['  H0: E8 Cartan ≈ random',
              '  matrix dynamics.',
              '',
              '  → Change equation of motion',
              '  → Or change model entirely']

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('part16_cartan.png',
            dpi=120, bbox_inches='tight')
print()
print("✓ Saved: part16_cartan.png")

# ── ФИНАЛ ────────────────────────────────────────────────────

print()
print("="*65)
print("PART XVI VERDICT")
print("="*65)
print()

e8_sr   = results['E8'][1.0]['spread_rate']
e8_pval = float(np.mean(rand10 >= e8_sr))

if e8_pval < 0.01:
    print("*** СИЛЬНЫЙ СИГНАЛ")
    print(f"    E8 p={e8_pval:.4f}")
    print("    → Part XVII: вычислить n_s")
elif e8_pval < 0.05:
    print("*   СЛАБЫЙ СИГНАЛ")
    print(f"    E8 p={e8_pval:.4f}")
    print("    → Повторить с N=100 контролей")
else:
    print("    H0 не отвергается")
    print(f"    E8 p={e8_pval:.4f}")
    print()
    print("ИТОГОВОЕ РЕШЕНИЕ:")
    print("  Parts I-XVI исчерпали стандартный")
    print("  подход к гипотезе моноструны.")
    print()
    print("  Рекомендация:")
    print("  1. Опубликовать отрицательный результат")
    print("     (Parts I-XI как статья)")
    print("  2. Сделать паузу")
    print("  3. Если продолжать — нужна принципиально")
    print("     новая идея, не модификация старой")