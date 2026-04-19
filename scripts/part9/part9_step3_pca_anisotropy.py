"""
Part IX Step 3: PCA_ratio как главная наблюдаемая
Гипотеза: Coxeter частоты → экстремальная анизотропия орбиты
Тест: N=50 случайных конфигураций vs Coxeter алгебры
"""

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# ── все функции в одном файле ──────────────────────────

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

def pca_anisotropy(X):
    """Отношение наибольшего к наименьшему собственному значению PCA."""
    pca = PCA()
    pca.fit(X)
    ev = pca.explained_variance_
    return {
        "ratio":    ev[0] / (ev[-1] + 1e-12),
        "ratio_31": ev[0] / (ev[2]  + 1e-12),
        "eff_dim":  np.sum(ev)**2 / np.sum(ev**2),
        "top1_frac": ev[0] / ev.sum(),
        "top3_frac": ev[:3].sum() / ev.sum(),
        "eigenvalues": ev,
    }

def build_graph(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, idxs = nbrs.kneighbors(X)
    N = len(X)
    sigma = np.median(dists[:,1:k+1])
    if sigma < 1e-10: sigma = 1.0
    rows, cols, wts, csts = [], [], [], []
    for i in range(N):
        for j_idx in range(1, k+1):
            j = idxs[i, j_idx]
            d = dists[i, j_idx]
            w = np.exp(-d**2 / (2*sigma**2))
            rows += [i,j]; cols += [j,i]
            wts  += [w,w]; csts += [d,d]
    A   = csr_matrix((wts,  (rows,cols)), shape=(N,N))
    C   = csr_matrix((csts, (rows,cols)), shape=(N,N))
    deg = np.array(A.sum(axis=1)).flatten()
    return A, C, deg, sigma

def graph_metrics(C, n_sample=200, rng=None):
    """Метрики графа из матрицы кратчайших путей."""
    if rng is None:
        rng = np.random.default_rng(42)
    N   = C.shape[0]
    D   = shortest_path(C, directed=False)
    idx = rng.choice(N, min(n_sample, N), replace=False)
    D_s = D[np.ix_(idx, idx)]
    fin = D_s[np.isfinite(D_s) & (D_s > 0)]
    return {
        "mean":  np.mean(fin),
        "std":   np.std(fin),
        "cv":    np.std(fin) / np.mean(fin),
        "max":   np.max(fin),
        "sigma_graph": float(C.data.mean()) if len(C.data) > 0 else 0,
    }

def frequency_irrationality(omegas):
    """
    Мера иррациональности набора частот.
    Для Coxeter: ωᵢ = 2sin(π·m/h) — иррациональные числа
    Ключевое свойство: попарные отношения ωᵢ/ωⱼ
    """
    n = len(omegas)
    ratios = []
    for i in range(n):
        for j in range(i+1, n):
            r = omegas[i] / omegas[j]
            # Как близко к рациональному числу p/q с q<=20?
            best_err = 1.0
            for q in range(1, 21):
                p = round(r * q)
                err = abs(r - p/q)
                if err < best_err:
                    best_err = err
            ratios.append(best_err)
    return np.mean(ratios), np.std(ratios)

# ════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Большая выборка Random vs Coxeter
# ════════════════════════════════════════════════════════

print("="*65)
print("Part IX Step 3: PCA anisotropy — systematic test")
print("="*65)

rng  = np.random.default_rng(42)
T    = 2000
K    = 10
SEED = 42

# Коксетеровы алгебры
COXETER = {
    "E6": coxeter_omegas([1,4,5,7,8,11], 12),
    "A6": coxeter_omegas([1,2,3,4,5,6],  7),
    "E7": coxeter_omegas([1,5,7,9,11,13,17], 18),
    "E8": coxeter_omegas([1,7,11,13,17,19,23,29], 30),
    "G2": coxeter_omegas([1,5], 6),
    "F4": coxeter_omegas([1,5,7,11], 12),
    "B4": coxeter_omegas([1,3,5,7], 8),
}

# Случайные конфигурации (N=50)
N_RANDOM = 50

print("\n[1] Coxeter algebras:")
print(f"{'Name':8s} {'PCA_r':>10s} {'eff_d':>7s} "
      f"{'top3':>7s} {'irrat':>8s} {'sigma':>7s}")
print("-"*52)

cox_data = {}
for name, omegas in COXETER.items():
    orbit = build_orbit(omegas, T, seed=SEED)
    X     = toric_embed(orbit)
    pa    = pca_anisotropy(X)
    A,C,deg,sig = build_graph(X, K)
    gm    = graph_metrics(C, rng=rng)
    irr_mean, irr_std = frequency_irrationality(omegas)

    cox_data[name] = {
        "pca_ratio": pa["ratio"],
        "eff_dim":   pa["eff_dim"],
        "top3":      pa["top3_frac"],
        "irrat":     irr_mean,
        "sigma":     sig,
        "path_cv":   gm["cv"],
        "omegas":    omegas,
    }
    print(f"{name:8s} {pa['ratio']:10.1f} {pa['eff_dim']:7.2f} "
          f"{pa['top3_frac']:7.4f} {irr_mean:8.5f} {sig:7.4f}")

print("\n[2] Random configurations (N=50):")

rnd_pca   = []
rnd_effd  = []
rnd_top3  = []
rnd_irrat = []
rnd_sigma = []

# Выборка: разные ранги и диапазоны частот
for i in range(N_RANDOM):
    rs    = np.random.RandomState(i)
    rank  = rs.choice([2, 4, 6, 8])          # разные ранги
    lo, hi = rs.uniform(0.1, 0.5), rs.uniform(1.5, 2.5)
    omegas = rs.uniform(lo, hi, rank)

    orbit = build_orbit(omegas, T, seed=i)
    X     = toric_embed(orbit)
    pa    = pca_anisotropy(X)
    A,C,deg,sig = build_graph(X, K)
    irr_mean, _ = frequency_irrationality(omegas)

    rnd_pca.append(pa["ratio"])
    rnd_effd.append(pa["eff_dim"])
    rnd_top3.append(pa["top3_frac"])
    rnd_irrat.append(irr_mean)
    rnd_sigma.append(sig)

rnd_pca   = np.array(rnd_pca)
rnd_effd  = np.array(rnd_effd)
rnd_top3  = np.array(rnd_top3)
rnd_irrat = np.array(rnd_irrat)

print(f"  PCA_ratio:  mean={rnd_pca.mean():.1f}, "
      f"std={rnd_pca.std():.1f}, "
      f"max={rnd_pca.max():.1f}, "
      f"p95={np.percentile(rnd_pca, 95):.1f}")
print(f"  eff_dim:    mean={rnd_effd.mean():.2f}, "
      f"std={rnd_effd.std():.2f}")
print(f"  top3_frac:  mean={rnd_top3.mean():.4f}, "
      f"std={rnd_top3.std():.4f}")
print(f"  irrat:      mean={rnd_irrat.mean():.5f}, "
      f"std={rnd_irrat.std():.5f}")

# ── Статистика ─────────────────────────────────────────
print("\n[3] Statistical tests:")

cox_pca  = [cox_data[n]["pca_ratio"] for n in cox_data]
cox_effd = [cox_data[n]["eff_dim"]   for n in cox_data]
cox_top3 = [cox_data[n]["top3"]      for n in cox_data]
cox_irr  = [cox_data[n]["irrat"]     for n in cox_data]

for metric, c_vals, r_vals in [
    ("PCA_ratio", cox_pca,  rnd_pca),
    ("eff_dim",   cox_effd, rnd_effd),
    ("top3_frac", cox_top3, rnd_top3),
    ("irrat",     cox_irr,  rnd_irrat),
]:
    stat, p = mannwhitneyu(c_vals, r_vals, alternative='two-sided')
    c_m = np.mean(c_vals); r_m = np.mean(r_vals)
    direction = ">" if c_m > r_m else "<"
    print(f"  {metric:12s}: Cox={c_m:.2f} {direction} "
          f"Rnd={r_m:.2f},  p={p:.4f}")

# ── Откуда PCA_ratio у Rnd_42? ────────────────────────
print("\n[4] Why was Rnd_42 anomalous?")
rs42 = np.random.RandomState(42)
om42 = rs42.uniform(0.5, 2.0, 6)
print(f"  Rnd_42 omegas: {np.round(om42, 4)}")
irr42, _ = frequency_irrationality(om42)
print(f"  irrationality: {irr42:.5f}")
print(f"  PCA_ratio:     {rnd_pca[0]:.1f}  "
      f"(note: Rnd_42 used rank=6, others vary)")

# ── ЭКСПЕРИМЕНТ 2: Что КОНКРЕТНО создаёт высокий PCA_ratio? ──
print("\n" + "="*65)
print("EXPERIMENT 2: What drives high PCA_ratio?")
print("="*65)

# 2a. Зависимость от κ
print("\n[2a] E6: PCA_ratio vs κ")
omegas_E6 = coxeter_omegas([1,4,5,7,8,11], 12)
print(f"{'κ':>8s}  {'PCA_ratio':>12s}  {'eff_dim':>8s}")
print("-"*35)
for kappa in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
    orbit = build_orbit(omegas_E6, T, kappa=kappa, seed=42)
    X     = toric_embed(orbit)
    pa    = pca_anisotropy(X)
    print(f"{kappa:8.2f}  {pa['ratio']:12.1f}  {pa['eff_dim']:8.3f}")

# 2b. Зависимость от T
print("\n[2b] E6: PCA_ratio vs T")
print(f"{'T':>7s}  {'PCA_ratio':>12s}  {'top3':>8s}")
print("-"*33)
for T_test in [200, 500, 1000, 2000, 5000]:
    orbit = build_orbit(omegas_E6, T_test, kappa=0.05, seed=42)
    X     = toric_embed(orbit)
    pa    = pca_anisotropy(X)
    print(f"{T_test:7d}  {pa['ratio']:12.1f}  {pa['top3_frac']:8.4f}")

# 2c. Что произойдёт с перемешанными частотами E6?
print("\n[2c] Shuffled E6 frequencies: does PCA_ratio survive?")
omegas_E6_sorted = np.sort(omegas_E6)
for trial in range(5):
    rng_s = np.random.default_rng(trial)
    om_sh = rng_s.permutation(omegas_E6)
    orbit = build_orbit(om_sh, T, kappa=0.05, seed=42)
    X     = toric_embed(orbit)
    pa    = pca_anisotropy(X)
    print(f"  shuffle_{trial}: PCA_ratio={pa['ratio']:.1f}, "
          f"eff_dim={pa['eff_dim']:.3f}")

# 2d. Рациональные приближения Coxeter частот
print("\n[2d] Rational approximations of E6 frequencies:")
print("  (does rationality destroy high PCA_ratio?)")
omegas_E6_exact = coxeter_omegas([1,4,5,7,8,11], 12)
print(f"  E6 exact: {np.round(omegas_E6_exact, 6)}")

for n_denom in [4, 8, 16, 64]:
    om_rat = np.round(omegas_E6_exact * n_denom) / n_denom
    orbit  = build_orbit(om_rat, T, kappa=0.05, seed=42)
    X      = toric_embed(orbit)
    pa     = pca_anisotropy(X)
    print(f"  rounded to 1/{n_denom}: "
          f"PCA_ratio={pa['ratio']:.1f}, "
          f"eff_dim={pa['eff_dim']:.3f}")

# 2e. Uniform [0,2π] random frequencies
print("\n[2e] Various frequency types, rank=6:")
freq_types = {
    "E6_exact":  coxeter_omegas([1,4,5,7,8,11], 12),
    "uniform_01": np.random.RandomState(0).uniform(0.0, 0.1, 6),
    "uniform_12": np.random.RandomState(0).uniform(1.0, 2.0, 6),
    "equal":      np.ones(6) * 1.0,
    "arithmetic": np.linspace(0.5, 2.0, 6),
    "geometric":  np.geomspace(0.5, 2.0, 6),
}
print(f"{'type':15s}  {'PCA_ratio':>12s}  {'eff_dim':>8s}  {'top3':>7s}")
print("-"*48)
for fname, om in freq_types.items():
    orbit = build_orbit(om, T, kappa=0.05, seed=42)
    X     = toric_embed(orbit)
    pa    = pca_anisotropy(X)
    print(f"{fname:15s}  {pa['ratio']:12.1f}  "
          f"{pa['eff_dim']:8.3f}  {pa['top3_frac']:7.4f}")

# ── Итог ───────────────────────────────────────────────
print("\n" + "="*65)
print("CONCLUSIONS")
print("="*65)
print("""
  From this analysis we determine:
  1. Whether PCA_ratio distinguishes Coxeter from Random (p-value)
  2. Whether κ drives the effect (or orbit itself)
  3. Whether shuffled E6 preserves PCA_ratio (set vs order)
  4. Whether rational approximations destroy it (irrationality test)
  5. What frequency structure maximizes PCA_ratio
""")
