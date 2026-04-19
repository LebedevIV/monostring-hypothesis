"""
Part IX Step 5: Честный итог + единственный оставшийся вопрос
Что РЕАЛЬНО отличает Coxeter от Random после контроля артефактов?

Контролируем:
1. Вырождение частот (убираем пары → уникальные частоты)
2. Spread (min/max диапазон)
3. Связность графа (Fiedler)

Остаётся ли хоть что-то?
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

# ── все функции ────────────────────────────────────────

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

def build_graph(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dists, idxs = nbrs.kneighbors(X)
    N = len(X)
    sigma = np.median(dists[:,1:k+1])
    if sigma < 1e-10: sigma = 1.0
    rows, cols, wts, csts = [], [], [], []
    for i in range(N):
        for j_idx in range(1, k+1):
            j = idxs[i, j_idx]; d = dists[i, j_idx]
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

def fiedler_val(vals):
    nz = np.where(vals > 1e-10)[0]
    return vals[nz[0]] if len(nz) > 0 else 0.0

def all_observables(omegas, T=2000, k=10, seed=42, rng=None):
    """Вычислить ВСЕ наблюдаемые для данного набора частот."""
    if rng is None:
        rng = np.random.default_rng(42)

    orbit = build_orbit(omegas, T, seed=seed)
    X     = toric_embed(orbit)
    N, D  = X.shape

    # 1. Геометрия орбиты
    pca = PCA().fit(X)
    ev  = pca.explained_variance_
    pca_r   = ev[0] / (ev[-1] + 1e-12)
    eff_dim = np.sum(ev)**2 / np.sum(ev**2)
    n_unique = len(np.unique(np.round(omegas, 6)))

    # 2. Граф
    A, C, deg, sigma = build_graph(X, k)
    vals, vecs = laplacian_eigs(A, deg)
    fiedler = fiedler_val(vals)

    # 3. Заполнение тора (энтропия)
    rank = orbit.shape[1]
    entropies = []
    for i in range(min(rank, 3)):
        for j in range(i+1, min(rank, 4)):
            phi_i = orbit[:, i] % (2*np.pi)
            phi_j = orbit[:, j] % (2*np.pi)
            H, _, _ = np.histogram2d(phi_i, phi_j, bins=15,
                range=[[0,2*np.pi],[0,2*np.pi]])
            H = H / H.sum()
            H_f = H[H > 0]
            entropies.append(
                -np.sum(H_f * np.log(H_f)) / np.log(15**2))
    torus_fill = np.mean(entropies) if entropies else 0.0

    # 4. Lyapunov (максимальный показатель)
    # Метод: расхождение двух близких траекторий
    eps = 1e-6
    r1 = np.random.RandomState(seed)
    r2 = np.random.RandomState(seed)
    phi1 = r1.uniform(0, 2*np.pi, rank)
    phi2 = phi1 + r2.normal(0, eps, rank)
    lyap_sum = 0.0
    n_lyap   = 500
    for t in range(n_lyap):
        phi1 = (phi1 + omegas + 0.05*np.sin(phi1)) % (2*np.pi)
        phi2 = (phi2 + omegas + 0.05*np.sin(phi2)) % (2*np.pi)
        d = np.linalg.norm(phi1 - phi2)
        if d > 0:
            lyap_sum += np.log(d / eps)
            phi2 = phi1 + (phi2 - phi1) * eps / d
    lyapunov = lyap_sum / n_lyap

    # 5. Корреляционная размерность (быстрая оценка)
    idx  = rng.choice(len(X), min(300, len(X)), replace=False)
    Xs   = X[idx]
    from scipy.spatial.distance import pdist
    ds   = pdist(Xs)
    ds   = ds[ds > 0]
    r_lo = np.percentile(ds, 10)
    r_hi = np.percentile(ds, 40)
    r_v  = np.linspace(r_lo, r_hi, 10)
    C_r  = np.array([np.mean(ds < r) for r in r_v])
    ok   = (C_r > 0.01) & (C_r < 0.99)
    if ok.sum() >= 3:
        d_corr = np.polyfit(np.log(r_v[ok]),
                            np.log(C_r[ok]), 1)[0]
    else:
        d_corr = np.nan

    return {
        "pca_ratio":  pca_r,
        "eff_dim":    eff_dim,
        "n_unique":   n_unique,
        "fiedler":    fiedler,
        "torus_fill": torus_fill,
        "lyapunov":   lyapunov,
        "d_corr":     d_corr,
        "sigma":      sigma,
        "spread":     omegas.max()/(omegas.min()+1e-10),
    }

# ════════════════════════════════════════════════════════
# ГЛАВНЫЙ ЭКСПЕРИМЕНТ: контроль вырождения
# ════════════════════════════════════════════════════════

print("="*65)
print("Part IX Step 5: Controlled comparison")
print("After removing degeneracy artifact")
print("="*65)

rng = np.random.default_rng(42)
T   = 2000
K   = 10

# ── Набор 1: Coxeter с УНИКАЛЬНЫМИ частотами ──────────
# Берём только уникальные значения каждой алгебры
print("\n[1] Coxeter UNIQUE frequencies (no degenerate pairs):")
print(f"{'Name':12s} {'n_uniq':>7s} {'PCA_r':>10s} "
      f"{'eff_d':>7s} {'fill':>7s} {'lyap':>8s} {'D_c':>6s}")
print("-"*62)

cox_unique_data = {}
COXETER_FULL = {
    "E6": coxeter_omegas([1,4,5,7,8,11], 12),
    "A6": coxeter_omegas([1,2,3,4,5,6],  7),
    "E7": coxeter_omegas([1,5,7,9,11,13,17], 18),
    "E8": coxeter_omegas([1,7,11,13,17,19,23,29], 30),
    "G2": coxeter_omegas([1,5], 6),
    "F4": coxeter_omegas([1,5,7,11], 12),
}

for name, omegas_full in COXETER_FULL.items():
    # Только уникальные частоты
    om_uniq = np.unique(np.round(omegas_full, 8))
    if len(om_uniq) < 2:
        continue
    obs = all_observables(om_uniq, T=T, k=min(K, len(om_uniq)*2),
                           seed=42, rng=rng)
    cox_unique_data[name] = obs
    cox_unique_data[name]["omegas"] = om_uniq
    print(f"{name:12s} {len(om_uniq):7d} {obs['pca_ratio']:10.2f} "
          f"{obs['eff_dim']:7.3f} {obs['torus_fill']:7.4f} "
          f"{obs['lyapunov']:8.4f} {obs['d_corr']:6.3f}")

# ── Набор 2: Random с matched n_unique и spread ────────
print("\n[2] Random frequencies matched to E6 unique profile:")
print(f"    n_unique=3, spread=[{0.518:.3f}, {1.932:.3f}]")

om_e6_uniq = np.unique(np.round(COXETER_FULL["E6"], 8))
lo_e6 = om_e6_uniq.min(); hi_e6 = om_e6_uniq.max()
n_e6  = len(om_e6_uniq)  # = 3

matched_rnd = []
for seed in range(50):
    rs = np.random.RandomState(seed)
    om = np.sort(rs.uniform(lo_e6, hi_e6, n_e6))
    obs = all_observables(om, T=T, k=min(K, n_e6*2),
                           seed=seed, rng=rng)
    matched_rnd.append(obs)

print(f"\n  {'Metric':12s}  {'E6_unique':>12s}  "
      f"{'Rnd_mean':>10s}  {'Rnd_p95':>10s}  {'p-val':>8s}")
print("  " + "-"*58)

e6u = cox_unique_data.get("E6", {})
for metric in ["pca_ratio","eff_dim","torus_fill","lyapunov","d_corr"]:
    e6_val  = e6u.get(metric, np.nan)
    rnd_arr = np.array([r[metric] for r in matched_rnd
                        if np.isfinite(r[metric])])
    if len(rnd_arr) == 0:
        continue
    pct = np.mean(rnd_arr < e6_val) * 100
    # p-value: Mann-Whitney (одна точка vs распределение)
    # используем перцентиль как меру
    p_approx = 1 - pct/100
    print(f"  {metric:12s}  {e6_val:12.4f}  "
          f"{rnd_arr.mean():10.4f}  "
          f"{np.percentile(rnd_arr,95):10.4f}  "
          f"p≈{p_approx:.3f}")

# ── Набор 3: Полный контроль — всё одинаково кроме алгебры
print("\n[3] STRICTEST CONTROL:")
print("    Same rank, same spread, same n_unique, different values")
print("    Only difference: Coxeter structure vs random")
print()

# E6 уникальные: [0.5176, 1.7321, 1.9319] — rank 3
# Сравним с 100 случайными наборами из 3 чисел в [0.5, 2.0]
om_e6u = np.array([0.517638, 1.732051, 1.931852])

all_3pt_rnd = []
for seed in range(100):
    rs = np.random.RandomState(seed)
    # matched: 3 точки в том же диапазоне
    om = np.sort(rs.uniform(0.5, 2.0, 3))
    orbit = build_orbit(om, T=T, seed=seed)
    X     = toric_embed(orbit)
    pca_c = PCA().fit(X)
    ev    = pca_c.explained_variance_
    pr    = ev[0] / (ev[-1] + 1e-12)
    ed    = np.sum(ev)**2 / np.sum(ev**2)
    all_3pt_rnd.append({"pca_ratio": pr, "eff_dim": ed,
                         "omegas": om})

# E6 уникальные
orbit_e6u = build_orbit(om_e6u, T=T, seed=42)
X_e6u     = toric_embed(orbit_e6u)
pca_e6u   = PCA().fit(X_e6u)
ev_e6u    = pca_e6u.explained_variance_
pr_e6u    = ev_e6u[0] / (ev_e6u[-1] + 1e-12)
ed_e6u    = np.sum(ev_e6u)**2 / np.sum(ev_e6u**2)

rnd_pca3 = np.array([r["pca_ratio"] for r in all_3pt_rnd])
rnd_ed3  = np.array([r["eff_dim"]   for r in all_3pt_rnd])

print(f"  E6 unique [0.518, 1.732, 1.932]:")
print(f"    PCA_ratio = {pr_e6u:.2f}")
print(f"    eff_dim   = {ed_e6u:.4f}")
print(f"    percentile in Random(n=100): "
      f"PCA={np.mean(rnd_pca3<pr_e6u)*100:.1f}%, "
      f"eff_dim={np.mean(rnd_ed3<ed_e6u)*100:.1f}%")
print(f"\n  Random 3-point distribution:")
print(f"    PCA_ratio: mean={rnd_pca3.mean():.2f}, "
      f"p95={np.percentile(rnd_pca3,95):.2f}, "
      f"max={rnd_pca3.max():.2f}")
print(f"    eff_dim:   mean={rnd_ed3.mean():.4f}, "
      f"std={rnd_ed3.std():.4f}")

# ── Финал: что реально специфично для Coxeter? ─────────
print("\n" + "="*65)
print("FINAL: Lyapunov spectrum — last candidate")
print("="*65)
print("""
  After controlling for degeneracy, spread, rank:
  Only Lyapunov exponent remains as candidate.
  Coxeter orbits are near-integrable (κ<<1, small Lyapunov).
  But this is just κ=0.05 being small — not algebra-specific.
""")

# Lyapunov vs Random (matched все параметры)
print("  Lyapunov exponent: E6_unique vs 3-point Random")
lyap_rnd = []
for seed in range(50):
    rs = np.random.RandomState(seed)
    om = rs.uniform(0.5, 2.0, 3)
    obs = all_observables(om, T=500, k=6, seed=seed, rng=rng)
    lyap_rnd.append(obs["lyapunov"])

obs_e6u = all_observables(om_e6u, T=500, k=6, seed=42, rng=rng)
lyap_rnd = np.array(lyap_rnd)

print(f"  E6_unique: lyap={obs_e6u['lyapunov']:.6f}")
print(f"  Random:    mean={lyap_rnd.mean():.6f}, "
      f"std={lyap_rnd.std():.6f}")
stat, p = mannwhitneyu([obs_e6u['lyapunov']], lyap_rnd,
                        alternative='two-sided')
print(f"  Mann-Whitney p={p:.4f}")

print("\n" + "="*65)
print("SCORECARD UPDATE")
print("="*65)
print("""
  FALSIFIED in Part IX:
  ✗ z_geo effect (Part VIII) — code artifact, z≈0 correctly
  ✗ PCA_ratio as Coxeter signal — degeneracy artifact (Weyl pairs)
  ✗ D_corr as unique E6 property — depends on frequency set
  ✗ tau ∝ h (Part VII) — coincidence
  ✗ All geometric field analogies (Steps 1-6, Part VIII)

  CONFIRMED (robust, but not unique to Coxeter):
  ✓ Weyl symmetry → degenerate frequency pairs → orbit on subtorus
  ✓ Near-integrable dynamics (small Lyapunov) for κ<<1
  ✓ Memory time τ≈237 (but not proportional to h)
  ✓ D_corr ≈ 3 for rank-6 Coxeter (but also for shuffled)

  OPEN:
  ? Is there ANY observable that is:
    (a) significantly different for Coxeter vs Random
    (b) not explained by: rank, spread, degeneracy, κ
    (c) reproducible across seeds
""")
