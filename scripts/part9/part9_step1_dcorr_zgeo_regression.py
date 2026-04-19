"""
Part IX Step 1: D_corr → z_geo regression
Тест: D_corr предсказывает геодезическую фокусировку?
H0: нет корреляции между D_corr и z_geo(highλ)
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ── параметры ──────────────────────────────────────────
ALGEBRAS = {
    "E6":  {"m": [1,4,5,7,8,11],       "h": 12},
    "A6":  {"m": [1,2,3,4,5,6],        "h": 7 },
    "E7":  {"m": [1,5,7,9,11,13,17],   "h": 18},
    "E8":  {"m": [1,7,11,13,17,19,23,29], "h": 30},
    "G2":  {"m": [1,5],                "h": 6 },
    "F4":  {"m": [1,5,7,11],           "h": 12},
    "B4":  {"m": [1,3,5,7],            "h": 8 },
    "D5":  {"m": [1,3,5,7,4],          "h": 8 },
}
RANDOM_SEEDS = [42, 137, 271, 314, 999]

T      = 1500
K_NN   = 9       # matched Fiedler из Step 6d
KAPPA  = 0.05
WARMUP = 500
BAND_Q = 0.20
N_PAIRS = 25

rng = np.random.default_rng(42)

# ── вспомогательные функции ────────────────────────────

def coxeter_omegas(m_list, h):
    return np.array([2*np.sin(np.pi*mi/h) for mi in m_list])

def build_orbit(omegas, T, kappa, warmup, seed=42):
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
            j = idxs[i,j_idx]; d = dists[i,j_idx]
            w = np.exp(-d**2/(2*sigma**2))
            rows+=[i,j]; cols+=[j,i]
            wts +=[w,w]; csts+=[d,d]
    A   = csr_matrix((wts,(rows,cols)), shape=(N,N))
    C   = csr_matrix((csts,(rows,cols)), shape=(N,N))
    deg = np.array(A.sum(axis=1)).flatten()
    return A, C, deg

def laplacian_eigs(A, deg, max_eigs=None):
    N   = len(deg)
    d   = np.where(deg>0, deg, 1.0)
    D12 = np.diag(1/np.sqrt(d))
    L   = np.eye(N) - D12 @ A.toarray() @ D12
    vals, vecs = eigh(L)
    return vals, vecs

def fiedler(vals):
    nz = np.where(vals > 1e-10)[0]
    return vals[nz[0]] if len(nz) > 0 else 0.0

def band_indices(vals, q_frac):
    nz  = np.where(vals > 1e-10)[0]
    q   = max(8, int(q_frac * len(nz)))
    mc  = nz[len(nz)//2]
    return {
        "lowλ":  nz[:q],
        "midλ":  nz[max(0,mc-q//2):mc+q//2],
        "highλ": nz[-q:],
    }

def z_geo_score(vecs, C, band_idx, n_pairs, rng):
    N = vecs.shape[0]
    D_sp = shortest_path(C, directed=False)
    scores = []
    for _ in range(n_pairs):
        s, t = rng.choice(N, 2, replace=False)
        path_len  = D_sp[s,t]
        if not np.isfinite(path_len) or path_len < 1e-10:
            continue
        # геодезический коридор: узлы с d(s,v)+d(v,t) < 1.5*d(s,t)
        ds = D_sp[s]; dt = D_sp[t]
        mask_geo  = (ds + dt) < 1.5 * path_len
        mask_off  = ~mask_geo
        if mask_geo.sum() < 3 or mask_off.sum() < 3:
            continue
        # time-averaged амплитуда мод в полосе
        amp2 = (vecs[:, band_idx]**2).mean(axis=1)
        mu_geo = amp2[mask_geo].mean()
        mu_off = amp2[mask_off].mean()
        sig    = amp2[mask_off].std() + 1e-10
        scores.append((mu_geo - mu_off) / sig)
    return np.mean(scores) if scores else 0.0

def corr_dim(X, n_pts=400, r_vals=None):
    """Корреляционная размерность через C(r) ~ r^D"""
    idx = rng.choice(len(X), min(n_pts, len(X)), replace=False)
    Xs  = X[idx]
    ds  = pdist(Xs)
    ds  = ds[ds > 0]
    if r_vals is None:
        r_vals = np.logspace(
            np.log10(np.percentile(ds, 5)),
            np.log10(np.percentile(ds, 50)), 20)
    C_r = np.array([np.mean(ds < r) for r in r_vals])
    ok  = (C_r > 0.01) & (C_r < 0.99)
    if ok.sum() < 4:
        return np.nan
    coeffs = np.polyfit(np.log(r_vals[ok]), np.log(C_r[ok]), 1)
    return coeffs[0]

# ── главный цикл ───────────────────────────────────────

results = []

print("="*62)
print("Part IX Step 1: D_corr → z_geo regression")
print("="*62)
print(f"\n{'Algebra':10s} {'D_corr':>8s} {'Fiedler':>8s} "
      f"{'z_geo_high':>11s} {'z_geo_low':>10s}")
print("-"*55)

for name, cfg in ALGEBRAS.items():
    omegas = coxeter_omegas(cfg["m"], cfg["h"])
    orbit  = build_orbit(omegas, T, KAPPA, WARMUP, seed=42)
    X      = toric_embed(orbit)

    A, C, deg = build_graph(X, K_NN)
    vals, vecs = laplacian_eigs(A, deg)
    f_val  = fiedler(vals)
    bands  = band_indices(vals, BAND_Q)

    z_high = z_geo_score(vecs, C, bands["highλ"], N_PAIRS, rng)
    z_low  = z_geo_score(vecs, C, bands["lowλ"],  N_PAIRS, rng)
    D_c    = corr_dim(X)

    print(f"{name:10s} {D_c:8.3f} {f_val:8.4f} "
          f"{z_high:11.3f} {z_low:10.3f}")

    results.append({
        "name": name, "D_corr": D_c, "fiedler": f_val,
        "z_high": z_high, "z_low": z_low, "rank": len(omegas)
    })

# Random controls
print("\n--- Random controls ---")
for seed in RANDOM_SEEDS:
    rs = np.random.RandomState(seed)
    omegas = rs.uniform(0.5, 2.0, 6)
    orbit  = build_orbit(omegas, T, KAPPA, WARMUP, seed=seed)
    X      = toric_embed(orbit)

    A, C, deg = build_graph(X, K_NN)
    vals, vecs = laplacian_eigs(A, deg)
    f_val  = fiedler(vals)
    bands  = band_indices(vals, BAND_Q)

    z_high = z_geo_score(vecs, C, bands["highλ"], N_PAIRS, rng)
    z_low  = z_geo_score(vecs, C, bands["lowλ"],  N_PAIRS, rng)
    D_c    = corr_dim(X)
    name   = f"Rnd_{seed}"

    print(f"{name:10s} {D_c:8.3f} {f_val:8.4f} "
          f"{z_high:11.3f} {z_low:10.3f}")

    results.append({
        "name": name, "D_corr": D_c, "fiedler": f_val,
        "z_high": z_high, "z_low": z_low, "rank": 6
    })

# ── статистика ─────────────────────────────────────────

import numpy as np
names   = [r["name"]   for r in results]
D_corrs = np.array([r["D_corr"]  for r in results])
z_highs = np.array([r["z_high"]  for r in results])
fiedlers= np.array([r["fiedler"] for r in results])

# Spearman: D_corr vs z_high
rho, p = spearmanr(D_corrs, z_highs)
print(f"\nSpearman(D_corr, z_high): ρ={rho:.3f}, p={p:.4f}")

# Partial: контроль Fiedler
# z_high ~ a*D_corr + b*Fiedler + c
from numpy.linalg import lstsq
X_reg = np.column_stack([D_corrs, fiedlers, np.ones(len(D_corrs))])
coef, _, _, _ = lstsq(X_reg, z_highs, rcond=None)
z_pred = X_reg @ coef
ss_res = np.sum((z_highs - z_pred)**2)
ss_tot = np.sum((z_highs - z_highs.mean())**2)
R2 = 1 - ss_res/ss_tot
print(f"OLS z_high ~ D_corr + Fiedler: R²={R2:.3f}")
print(f"  coef D_corr={coef[0]:.3f}, Fiedler={coef[1]:.3f}")

# Coxeter vs Random
cox_z = [r["z_high"] for r in results if not r["name"].startswith("Rnd")]
rnd_z = [r["z_high"] for r in results if     r["name"].startswith("Rnd")]
stat, p_mw = mannwhitneyu(cox_z, rnd_z, alternative='greater')
print(f"\nMann-Whitney Coxeter>Random: p={p_mw:.4f}")
print(f"  Coxeter z_high: {np.mean(cox_z):.2f}±{np.std(cox_z):.2f}")
print(f"  Random  z_high: {np.mean(rnd_z):.2f}±{np.std(rnd_z):.2f}")
