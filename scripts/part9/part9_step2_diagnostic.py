"""
Part IX Step 2: Правильная постановка вопроса
Вместо z_geo — измеряем анизотропию графа напрямую
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp
import warnings
warnings.filterwarnings('ignore')

# ── функции (полный файл, без импортов из других модулей) ──

def coxeter_omegas(m_list, h):
    return np.array([2*np.sin(np.pi*mi/h) for mi in m_list])

def build_orbit(omegas, T, kappa=0.05, warmup=500, seed=42):
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

def build_graph(X, k):
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

def laplacian_eigs(A, deg):
    N   = len(deg)
    d   = np.where(deg>0, deg, 1.0)
    D12 = np.diag(1/np.sqrt(d))
    L   = np.eye(N) - D12 @ A.toarray() @ D12
    return eigh(L)

def band_idx(vals, q_frac=0.20):
    nz = np.where(vals > 1e-10)[0]
    q  = max(8, int(q_frac * len(nz)))
    mc = nz[len(nz)//2]
    return {
        "low":  nz[:q],
        "mid":  nz[max(0,mc-q//2):mc+q//2],
        "high": nz[-q:],
    }

# ════════════════════════════════════════════════════════
# НОВЫЕ НАБЛЮДАЕМЫЕ — не z_geo
# ════════════════════════════════════════════════════════

def graph_anisotropy(C, n_sample=100, rng=None):
    """
    Анизотропия графа: CV коэффициента вариации
    расстояний от случайных узлов.

    Изотропный граф: все узлы "одинаково далеко" от источника
    Анизотропный: некоторые направления короче

    Возвращает: mean CV, std CV
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N   = C.shape[0]
    D   = shortest_path(C, directed=False)

    cvs = []
    sources = rng.choice(N, min(n_sample, N), replace=False)
    for s in sources:
        row = D[s]
        fin = row[np.isfinite(row) & (row > 0)]
        if len(fin) < 10:
            continue
        cvs.append(np.std(fin) / np.mean(fin))
    return np.mean(cvs), np.std(cvs)

def spectral_anisotropy(vecs, bands):
    """
    Анизотропия спектральных мод:
    насколько неравномерно распределена амплитуда мод
    по пространству для разных полос.

    Возвращает IPR и его отношение high/low
    """
    results = {}
    N = vecs.shape[0]
    for name, idx in bands.items():
        if len(idx) == 0:
            continue
        # IPR для каждой моды
        iprs = []
        for i in idx:
            v = vecs[:, i]
            v2 = v**2
            ipr = np.sum(v2**2) / np.sum(v2)**2
            iprs.append(ipr * N)  # нормировано: 1=делок, N=полная лок
        results[name] = np.mean(iprs)
    return results

def path_length_distribution(C, n_pairs=500, rng=None):
    """
    Распределение длин кратчайших путей.
    Анизотропный граф: бимодальное или широкое распределение.
    Возвращает: mean, std, CV, skewness
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N   = C.shape[0]
    D   = shortest_path(C, directed=False)

    pairs = []
    idxs  = rng.choice(N, (n_pairs*2, 2), replace=True)
    for s, t in idxs:
        if s != t and np.isfinite(D[s,t]) and D[s,t] > 0:
            pairs.append(D[s,t])
        if len(pairs) >= n_pairs:
            break

    pairs = np.array(pairs)
    from scipy.stats import skew
    return {
        "mean": np.mean(pairs),
        "std":  np.std(pairs),
        "cv":   np.std(pairs)/np.mean(pairs),
        "skew": skew(pairs),
    }

def orbit_geometry(X):
    """
    Прямая геометрия орбиты без графа.
    PCA анизотропия: λ₁/λ_last (отношение главных компонент)
    """
    pca = PCA()
    pca.fit(X)
    ev  = pca.explained_variance_
    return {
        "pca_ratio":   ev[0] / ev[-1],
        "pca_ratio32": ev[0] / ev[2],  # первые vs третья
        "effective_dim": (np.sum(ev)**2 / np.sum(ev**2)),
        "top3_frac":  ev[:3].sum() / ev.sum(),
    }

def mode_spatial_correlation(vecs, C, band, n_sample=50, rng=None):
    """
    Корреляция между амплитудой мод и близостью к случайному узлу.
    Если моды пространственно структурированы:
    узлы близко к "центру" имеют другую амплитуду.

    Возвращает: mean |ρ| по модам в полосе
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N   = vecs.shape[0]
    D   = shortest_path(C, directed=False)

    sources = rng.choice(N, min(n_sample, N), replace=False)
    rhos = []
    for i in band[:20]:  # первые 20 мод полосы
        v2 = vecs[:, i]**2
        for s in sources[:10]:
            d_s = D[s]
            fin = np.isfinite(d_s)
            if fin.sum() < 10:
                continue
            r, p = spearmanr(d_s[fin], v2[fin])
            if np.isfinite(r):
                rhos.append(abs(r))
    return np.mean(rhos) if rhos else 0.0

# ════════════════════════════════════════════════════════
# ГЛАВНЫЙ ЭКСПЕРИМЕНТ
# ════════════════════════════════════════════════════════

rng = np.random.default_rng(42)

CONFIGS = {
    "E6":  (coxeter_omegas([1,4,5,7,8,11], 12), 42),
    "A6":  (coxeter_omegas([1,2,3,4,5,6],   7), 42),
    "E8":  (coxeter_omegas([1,7,11,13,17,19,23,29], 30), 42),
}
# Random controls
for seed in [42, 137, 271]:
    rs = np.random.RandomState(seed)
    CONFIGS[f"Rnd_{seed}"] = (rs.uniform(0.5, 2.0, 6), seed)

T = 2000; K = 10

print("="*70)
print("Part IX Step 2: Direct geometric observables (no z_geo)")
print("="*70)

all_results = {}

for name, (omegas, seed) in CONFIGS.items():
    orbit = build_orbit(omegas, T, seed=seed)
    X     = toric_embed(orbit)
    A, C, deg, sigma = build_graph(X, K)
    vals, vecs = laplacian_eigs(A, deg)
    bands = band_idx(vals)

    # 1. Геометрия орбиты (прямая)
    og = orbit_geometry(X)

    # 2. Анизотропия графа
    g_aniso_mean, g_aniso_std = graph_anisotropy(C, n_sample=80, rng=rng)

    # 3. Спектральная анизотропия (IPR)
    ipr = spectral_anisotropy(vecs, bands)

    # 4. Распределение путей
    pld = path_length_distribution(C, n_pairs=300, rng=rng)

    # 5. Корреляция мод с расстоянием
    rho_high = mode_spatial_correlation(
        vecs, C, bands["high"], n_sample=30, rng=rng)
    rho_low  = mode_spatial_correlation(
        vecs, C, bands["low"],  n_sample=30, rng=rng)

    all_results[name] = {
        "pca_ratio": og["pca_ratio"],
        "eff_dim":   og["effective_dim"],
        "top3_frac": og["top3_frac"],
        "g_aniso":   g_aniso_mean,
        "ipr_high":  ipr.get("high", np.nan),
        "ipr_low":   ipr.get("low",  np.nan),
        "path_cv":   pld["cv"],
        "path_mean": pld["mean"],
        "rho_high":  rho_high,
        "rho_low":   rho_low,
        "sigma":     sigma,
    }

    print(f"\n{'─'*50}")
    print(f"  {name}  (rank={len(omegas)}, σ={sigma:.3f})")
    print(f"  Orbit:  PCA_ratio={og['pca_ratio']:.2f}, "
          f"eff_dim={og['effective_dim']:.2f}, "
          f"top3={og['top3_frac']:.3f}")
    print(f"  Graph:  aniso={g_aniso_mean:.4f}±{g_aniso_std:.4f}, "
          f"path_cv={pld['cv']:.4f}, path_mean={pld['mean']:.2f}")
    print(f"  IPR:    high={ipr.get('high',0):.3f}, "
          f"low={ipr.get('low',0):.3f}, "
          f"ratio={ipr.get('high',1)/ipr.get('low',1):.3f}")
    print(f"  ModCorr: ρ_high={rho_high:.4f}, ρ_low={rho_low:.4f}")

# ── Сводная таблица ────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Name':12s} {'PCA_r':>7s} {'eff_d':>6s} "
      f"{'g_aniso':>8s} {'path_cv':>8s} "
      f"{'IPR_h':>7s} {'ρ_high':>7s}")
print("-"*60)

cox_names = ["E6","A6","E8"]
rnd_names = [k for k in all_results if k.startswith("Rnd")]

for name in list(CONFIGS.keys()):
    r = all_results[name]
    print(f"{name:12s} {r['pca_ratio']:7.2f} {r['eff_dim']:6.2f} "
          f"{r['g_aniso']:8.4f} {r['path_cv']:8.4f} "
          f"{r['ipr_high']:7.3f} {r['rho_high']:7.4f}")

# ── Статистика: Coxeter vs Random ─────────────────────
print("\n" + "="*70)
print("STATISTICAL COMPARISON: Coxeter vs Random")
print("="*70)

metrics = ["pca_ratio","eff_dim","g_aniso","path_cv","ipr_high","rho_high"]
for m in metrics:
    cox_vals = [all_results[n][m] for n in cox_names
                if np.isfinite(all_results[n][m])]
    rnd_vals = [all_results[n][m] for n in rnd_names
                if np.isfinite(all_results[n][m])]
    if len(cox_vals) < 2 or len(rnd_vals) < 2:
        continue
    stat, p = mannwhitneyu(cox_vals, rnd_vals, alternative='two-sided')
    direction = "Cox>Rnd" if np.mean(cox_vals)>np.mean(rnd_vals) else "Cox<Rnd"
    print(f"  {m:12s}: Cox={np.mean(cox_vals):.4f}, "
          f"Rnd={np.mean(rnd_vals):.4f}, "
          f"p={p:.4f}  {direction}")

# ── Ключевой вопрос: что отличает E6 от Random? ───────
print("\n" + "="*70)
print("KEY FINDING: What distinguishes Coxeter from Random?")
print("="*70)

print("\n  sigma (graph bandwidth):")
for name in CONFIGS:
    r = all_results[name]
    tag = "← COX" if name in cox_names else ""
    print(f"    {name:12s}: σ={r['sigma']:.4f}  {tag}")

print("\n  path_mean (graph diameter proxy):")
for name in CONFIGS:
    r = all_results[name]
    tag = "← COX" if name in cox_names else ""
    print(f"    {name:12s}: mean_path={r['path_mean']:.2f}  {tag}")
