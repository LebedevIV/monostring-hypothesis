"""
Part IX Step 4: Механизм PCA_ratio
Гипотеза: PCA_ratio измеряет "медленность" орбиты —
насколько она заполняет тор медленно/неравномерно.
НЕ является алгебраическим свойством.
"""

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# ── функции ────────────────────────────────────────────

def coxeter_omegas(m_list, h):
    return np.array([2*np.sin(np.pi*mi/h) for mi in m_list])

def build_orbit(omegas, T=2000, kappa=0.05, warmup=500, seed=42):
    r = np.random.RandomState(seed)
    rank = len(omegas)
    phi  = r.uniform(0, 2*np.pi, rank)
    for _ in range(warmup):
        phi = (phi + omegas + kappa*np.sin(phi)) % (2*np.pi)
    orbit = np.zeros((T, rank))
    for t in range(T):
        phi = (phi + omegas + kappa*np.sin(phi)) % (2*np.pi)
        orbit[t] = phi
    return orbit

def toric_embed(orbit):
    return np.concatenate([np.cos(orbit), np.sin(orbit)], axis=1)

def pca_ratio(X):
    pca = PCA().fit(X)
    ev  = pca.explained_variance_
    return ev[0] / (ev[-1] + 1e-12), ev

def orbit_speed(omegas, kappa=0.05):
    """
    Характеристики скорости орбиты:
    - mean_omega: средняя частота
    - min_omega: минимальная частота (самое медленное измерение)
    - spread: max/min отношение частот
    - near_resonance: сколько пар близко к рациональному отношению
    """
    n = len(omegas)
    min_om = np.min(omegas)
    max_om = np.max(omegas)
    mean_om = np.mean(omegas)

    # Насколько близко к резонансу (простые рациональные отношения)
    near_res = 0
    for i in range(n):
        for j in range(i+1, n):
            r = omegas[i] / omegas[j]
            for p in range(1, 6):
                for q in range(1, 6):
                    if abs(r - p/q) < 0.02:
                        near_res += 1

    return {
        "min_omega":   min_om,
        "max_omega":   max_om,
        "mean_omega":  mean_om,
        "spread":      max_om / (min_om + 1e-10),
        "near_resonance": near_res,
        "sum_omega":   np.sum(omegas),
    }

def torus_filling(orbit, n_bins=20):
    """
    Насколько равномерно орбита заполняет тор.
    Используем проекцию на первые 2 измерения.
    Равномерное заполнение → PCA_ratio ≈ 1
    Неравномерное (1D кривая) → PCA_ratio >> 1
    """
    rank = orbit.shape[1]
    results = {}

    # Для каждой пары измерений: насколько заполнен 2D квадрат?
    entropies = []
    for i in range(min(rank, 4)):
        for j in range(i+1, min(rank, 4)):
            phi_i = orbit[:, i] % (2*np.pi)
            phi_j = orbit[:, j] % (2*np.pi)
            H, _, _ = np.histogram2d(
                phi_i, phi_j,
                bins=n_bins,
                range=[[0, 2*np.pi], [0, 2*np.pi]]
            )
            H = H / H.sum()
            H_flat = H[H > 0]
            entropy = -np.sum(H_flat * np.log(H_flat))
            max_entropy = np.log(n_bins**2)
            entropies.append(entropy / max_entropy)

    results["mean_fill"] = np.mean(entropies)  # 1=равномерно, 0=нет
    results["min_fill"]  = np.min(entropies)
    return results

# ════════════════════════════════════════════════════════
# ТЕСТ: что предсказывает PCA_ratio?
# ════════════════════════════════════════════════════════

print("="*65)
print("Part IX Step 4: Understanding PCA_ratio mechanism")
print("="*65)

T    = 2000
rng  = np.random.default_rng(42)

# ── Тест A: Контролируемый эксперимент с min_omega ─────
print("\n[A] PCA_ratio vs min(omega) — controlled experiment")
print("    Fixed: rank=6, max(omega)=2.0, others random")
print(f"{'min_om':>8s}  {'PCA_ratio':>12s}  {'fill':>7s}  {'spread':>8s}")
print("-"*42)

for min_om in [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9]:
    rs = np.random.RandomState(0)
    # фиксируем min_omega, остальные случайные в [min_om+0.1, 2.0]
    if min_om + 0.1 < 2.0:
        others = rs.uniform(min_om + 0.1, 2.0, 5)
    else:
        others = np.ones(5) * 2.0
    omegas = np.concatenate([[min_om], others])
    orbit  = build_orbit(omegas, T, seed=0)
    X      = toric_embed(orbit)
    pr, _  = pca_ratio(X)
    tf     = torus_filling(orbit)
    sp     = orbit_speed(omegas)
    print(f"{min_om:8.2f}  {pr:12.1f}  {tf['mean_fill']:7.4f}  "
          f"{sp['spread']:8.2f}")

# ── Тест B: spread (max/min omega) ─────────────────────
print("\n[B] PCA_ratio vs frequency spread (max/min)")
print("    Fixed: rank=6, min=0.5, varying max")
print(f"{'max_om':>8s}  {'spread':>8s}  {'PCA_ratio':>12s}  {'fill':>7s}")
print("-"*45)

for max_om in [0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
    rs = np.random.RandomState(1)
    omegas = np.concatenate([[0.5],
                              rs.uniform(0.5, max_om, 5)])
    orbit  = build_orbit(omegas, T, seed=1)
    X      = toric_embed(orbit)
    pr, _  = pca_ratio(X)
    tf     = torus_filling(orbit)
    spread = max_om / 0.5
    print(f"{max_om:8.2f}  {spread:8.2f}  {pr:12.1f}  "
          f"{tf['mean_fill']:7.4f}")

# ── Тест C: Coxeter vs matched-spread Random ──────────
print("\n[C] Fair comparison: matched frequency spread")
print("    Coxeter algebras vs Random with SAME min/max omega")

omegas_E6 = coxeter_omegas([1,4,5,7,8,11], 12)
min_e6 = omegas_E6.min()  # ≈ 0.518
max_e6 = omegas_E6.max()  # ≈ 1.932

print(f"\n  E6 omega range: [{min_e6:.3f}, {max_e6:.3f}], "
      f"spread={max_e6/min_e6:.2f}")

orbit_e6 = build_orbit(omegas_E6, T, seed=42)
X_e6     = toric_embed(orbit_e6)
pr_e6, _ = pca_ratio(X_e6)
tf_e6    = torus_filling(orbit_e6)
print(f"  E6: PCA_ratio={pr_e6:.1f}, fill={tf_e6['mean_fill']:.4f}")

print(f"\n  Random rank=6, omega ∈ [{min_e6:.3f}, {max_e6:.3f}]:")
matched_pca = []
for seed in range(30):
    rs = np.random.RandomState(seed)
    om = rs.uniform(min_e6, max_e6, 6)
    orbit = build_orbit(om, T, seed=seed)
    X     = toric_embed(orbit)
    pr, _ = pca_ratio(X)
    matched_pca.append(pr)

matched_pca = np.array(matched_pca)
print(f"  mean={matched_pca.mean():.1f}, "
      f"std={matched_pca.std():.1f}, "
      f"max={matched_pca.max():.1f}, "
      f"p95={np.percentile(matched_pca, 95):.1f}")

stat, p = mannwhitneyu([pr_e6], matched_pca, alternative='greater')
print(f"  E6 > matched Random: p={p:.4f}")

# Перцентиль E6 в распределении Random
pct = np.mean(matched_pca < pr_e6) * 100
print(f"  E6 percentile in matched Random: {pct:.1f}%")

# ── Тест D: Число независимых частот ──────────────────
print("\n[D] Role of near-degenerate frequencies")
print("    E6 has TWO pairs of equal frequencies:")
print(f"    {np.round(omegas_E6, 4)}")
print("    → effectively rank_independent = 4, not 6")
print()

# Degeneracy test:
unique_om = np.unique(np.round(omegas_E6, 6))
print(f"  Unique frequencies in E6: {len(unique_om)} "
      f"out of {len(omegas_E6)}")
print(f"  Values: {np.round(unique_om, 4)}")

# Сравним с 4 независимых частот в том же диапазоне
print("\n  rank=4 independent vs rank=6 with 2 pairs:")
for test_name, om_test in [
    ("E6_exact",    omegas_E6),
    ("4_unique",    np.array([0.518, 1.0, 1.5, 1.932])),
    ("6_equal_pairs", np.array([0.518, 0.518, 1.2, 1.2, 1.932, 1.932])),
    ("6_all_equal",   np.ones(6) * 1.0),
    ("6_spread",      np.array([0.3, 0.6, 1.0, 1.4, 1.8, 2.0])),
]:
    orbit = build_orbit(om_test, T, seed=42)
    X     = toric_embed(orbit)
    pr, ev = pca_ratio(X)
    tf    = torus_filling(orbit)
    print(f"  {test_name:20s}: PCA={pr:12.1f}, "
          f"fill={tf['mean_fill']:.4f}")

# ── Тест E: Главный вопрос ─────────────────────────────
print("\n" + "="*65)
print("CRITICAL TEST E: Is PCA_ratio a physics signal or")
print("                 just a property of frequency degeneracy?")
print("="*65)

print("\n  E6 has EXACT degenerate pairs: ω₁=ω₆, ω₂=ω₅")
print("  This means the torus orbit lies on a 4D subtorus")
print("  → PCA_ratio captures dimensionality reduction, not algebra")
print()

# Контроль: случайные частоты с принудительными парами
print("  Random with forced pairs vs E6 vs truly random:")
print(f"  {'Config':25s}  {'PCA_ratio':>12s}  {'eff_dim':>8s}")
print("  " + "-"*50)

orbit = build_orbit(omegas_E6, T, seed=42)
X     = toric_embed(orbit)
pr, _ = pca_ratio(X); pca_c = PCA().fit(X)
print(f"  {'E6_exact':25s}  {pr:12.1f}  "
      f"{(np.sum(pca_c.explained_variance_)**2/np.sum(pca_c.explained_variance_**2)):8.3f}")

for seed in range(5):
    rs = np.random.RandomState(seed * 10)
    base = rs.uniform(0.5, 2.0, 3)
    om_paired = np.concatenate([base, base])  # 3 пары
    om_paired += rs.normal(0, 0.001, 6)       # малый шум
    orbit = build_orbit(om_paired, T, seed=seed)
    X     = toric_embed(orbit)
    pr, _ = pca_ratio(X); pca_c = PCA().fit(X)
    effd  = (np.sum(pca_c.explained_variance_)**2 /
             np.sum(pca_c.explained_variance_**2))
    print(f"  {'forced_pairs_'+str(seed):25s}  {pr:12.1f}  {effd:8.3f}")

for seed in range(5):
    rs = np.random.RandomState(seed * 7)
    om = rs.uniform(0.5, 2.0, 6)  # все разные
    orbit = build_orbit(om, T, seed=seed)
    X     = toric_embed(orbit)
    pr, _ = pca_ratio(X); pca_c = PCA().fit(X)
    effd  = (np.sum(pca_c.explained_variance_)**2 /
             np.sum(pca_c.explained_variance_**2))
    print(f"  {'truly_random_'+str(seed):25s}  {pr:12.1f}  {effd:8.3f}")

print("\n" + "="*65)
print("SUMMARY: Physical interpretation")
print("="*65)
print("""
  Based on the experiments:

  PCA_ratio measures: how much the orbit concentrates
  along a few principal directions in embedding space.

  HIGH PCA_ratio occurs when:
    - Frequencies are near-degenerate (pairs)
    - min(omega) is very small (slow filling)
    - kappa ≈ 0 (near-integrable)

  E6 has EXACT degenerate pairs by Weyl symmetry:
    omega_1 = omega_6, omega_2 = omega_5
  This forces the orbit onto a lower-dim submanifold.

  CONCLUSION (to be verified by numbers):
    PCA_ratio is HIGH for Coxeter algebras because
    their Weyl symmetry forces frequency degeneracies,
    NOT because of any deep physical structure.
""")
