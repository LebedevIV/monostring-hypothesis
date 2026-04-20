"""
Part XI Step 2: Systematic quantum walk comparison
Тест: S(t) Cayley vs Random — статистически значимо?
H0: mean S(t) одинакова для Cayley и random regular graph
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import mannwhitneyu, ttest_ind
from itertools import permutations
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ── все функции в одном файле ──────────────────────────

def make_cayley_s_n(n, gen_type="adjacent"):
    """Кэли-граф S_n."""
    elems = list(permutations(range(n)))
    idx   = {e: i for i, e in enumerate(elems)}
    def compose(p, q):
        return tuple(p[q[i]] for i in range(n))

    if gen_type == "adjacent":
        gens = []
        for i in range(n-1):
            s = list(range(n)); s[i],s[i+1]=s[i+1],s[i]
            gens.append(tuple(s))
    else:  # all transpositions
        from itertools import combinations
        gens = []
        for i,j in combinations(range(n), 2):
            s = list(range(n)); s[i],s[j]=s[j],s[i]
            gens.append(tuple(s))

    N = len(elems)
    rows, cols = [], []
    for i, w in enumerate(elems):
        for s in gens:
            ws = compose(w, s)
            j  = idx[ws]
            rows.append(i); cols.append(j)
    edges = set(zip(rows, cols))
    r, c  = zip(*edges)
    A     = csr_matrix((np.ones(len(r)),(r,c)), shape=(N,N))
    A     = (A + A.T); A.data[:] = 1.0; A.eliminate_zeros()
    return A.toarray()

def random_regular_graph(N, d, seed=42):
    """Случайный d-регулярный граф."""
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        stubs = np.repeat(np.arange(N), d)
        rng.shuffle(stubs)
        s = len(stubs)//2*2
        r = stubs[:s:2]; c = stubs[1:s:2]
        edges = set()
        ok = True
        for ri, ci in zip(r, c):
            if ri == ci: ok=False; break
            e = (min(ri,ci), max(ri,ci))
            if e in edges: ok=False; break
            edges.add(e)
        if ok:
            A = np.zeros((N,N))
            for ri, ci in zip(r, c):
                A[ri,ci] = A[ci,ri] = 1.0
            return A
    return None

def norm_laplacian(A):
    deg = A.sum(axis=1)
    D12 = np.diag(1/np.sqrt(np.where(deg>0, deg, 1.0)))
    return np.eye(len(A)) - D12 @ A @ D12

def quantum_walk_entropy(L, t_vals, n_starts=5, rng=None):
    """
    Средняя энтропия квантового блуждания по времени.
    Усредняем по n_starts начальным вершинам.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(L)
    vals, vecs = eigh(L)

    entropies_by_t = []
    starts = rng.choice(N, min(n_starts, N), replace=False)

    for s in starts:
        psi0 = np.zeros(N); psi0[s] = 1.0
        c0   = vecs.T @ psi0

        ents = []
        for t in t_vals:
            phases = np.exp(-1j * vals * t)
            psi_t  = vecs @ (phases * c0)
            p      = np.abs(psi_t)**2
            p      = p[p > 1e-12]
            S      = -np.sum(p * np.log(p))
            ents.append(S)
        entropies_by_t.append(ents)

    # Среднее по стартам
    return np.mean(entropies_by_t, axis=0)

def return_prob_stats(L, t_vals, n_starts=10, rng=None):
    """Статистика возвратной вероятности."""
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(L)
    vals, vecs = eigh(L)

    all_returns = []
    starts = rng.choice(N, min(n_starts, N), replace=False)

    for s in starts:
        psi0 = np.zeros(N); psi0[s] = 1.0
        c0   = vecs.T @ psi0
        returns = []
        for t in t_vals:
            phases = np.exp(-1j * vals * t)
            psi_t  = vecs @ (phases * c0)
            p_s    = np.abs(psi_t[s])**2
            returns.append(p_s)
        all_returns.append(returns)

    return np.mean(all_returns, axis=0)

def spreading_alpha(L, t_vals, rng=None):
    """
    Степенной закон распространения:
    ⟨d²(t)⟩ ~ t^α
    α=1: диффузия, α=2: баллистика, α=0: локализация
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(L)
    vals, vecs = eigh(L)

    # Используем IPR как меру локализации
    # IPR(t) = Σ_v |ψ_v(t)|^4
    psi0 = np.zeros(N); psi0[0] = 1.0
    c0   = vecs.T @ psi0

    iprs = []
    for t in t_vals:
        phases = np.exp(-1j * vals * t)
        psi_t  = vecs @ (phases * c0)
        p      = np.abs(psi_t)**2
        iprs.append(np.sum(p**2))

    iprs = np.array(iprs)
    # ⟨d²⟩ ~ 1/IPR (обратная мера локализации)
    eff_spread = 1.0 / (iprs + 1e-12)

    # Фит степенного закона в логарифмическом масштабе
    t_arr = np.array(t_vals)
    ok    = (t_arr > 0.5) & (eff_spread > 0)
    if ok.sum() >= 4:
        alpha, _ = np.polyfit(
            np.log(t_arr[ok]),
            np.log(eff_spread[ok]), 1)
    else:
        alpha = np.nan
    return alpha, iprs

# ════════════════════════════════════════════════════════
# ГЛАВНЫЙ ТЕСТ
# ════════════════════════════════════════════════════════

rng  = np.random.default_rng(42)
N    = 24   # |S_4|
d    = 3    # степень
t_vals = np.linspace(0.1, 50, 300)

print("="*62)
print("Part XI Step 2: Systematic quantum walk comparison")
print("="*62)

# Кэли-граф S_4
A_cay = make_cayley_s_n(4, "adjacent")
L_cay = norm_laplacian(A_cay)

# N_TRIALS случайных графов
N_TRIALS = 30
print(f"\nGenerating {N_TRIALS} random {d}-regular graphs (N={N})...")

cay_S    = quantum_walk_entropy(L_cay, t_vals, n_starts=24, rng=rng)
cay_ret  = return_prob_stats(L_cay, t_vals, n_starts=24, rng=rng)
cay_alph, cay_ipr = spreading_alpha(L_cay, t_vals, rng=rng)

rnd_S_mean_list   = []
rnd_ret_mean_list = []
rnd_alpha_list    = []

for seed in range(N_TRIALS):
    A_r = random_regular_graph(N, d, seed=seed)
    if A_r is None:
        continue
    L_r = norm_laplacian(A_r)
    S_r   = quantum_walk_entropy(L_r, t_vals, n_starts=12, rng=rng)
    ret_r = return_prob_stats(L_r, t_vals, n_starts=12, rng=rng)
    alp_r, _ = spreading_alpha(L_r, t_vals, rng=rng)
    rnd_S_mean_list.append(np.mean(S_r))
    rnd_ret_mean_list.append(np.mean(ret_r))
    rnd_alpha_list.append(alp_r)

rnd_S_arr   = np.array(rnd_S_mean_list)
rnd_ret_arr = np.array(rnd_ret_mean_list)
rnd_alp_arr = np.array([a for a in rnd_alpha_list
                         if np.isfinite(a)])

# ── Результаты ─────────────────────────────────────────
print(f"\n[1] Mean entropy S(t) over t∈[0.1, 50]:")
print(f"  Cayley S_4:  {np.mean(cay_S):.4f}")
print(f"  Random mean: {rnd_S_arr.mean():.4f} "
      f"± {rnd_S_arr.std():.4f}")
print(f"  Max (uniform): {np.log(N):.4f}")
pct_S = np.mean(rnd_S_arr < np.mean(cay_S)) * 100
print(f"  Cayley percentile in Random: {pct_S:.1f}%")
_, p_S = mannwhitneyu([np.mean(cay_S)], rnd_S_arr,
                       alternative='two-sided')
print(f"  Mann-Whitney p = {p_S:.4f}")

print(f"\n[2] Mean return probability ⟨P(v→v,t)⟩:")
print(f"  Cayley S_4:  {np.mean(cay_ret):.6f}")
print(f"  Random mean: {rnd_ret_arr.mean():.6f} "
      f"± {rnd_ret_arr.std():.6f}")
print(f"  Uniform: {1/N:.6f}")
pct_ret = np.mean(rnd_ret_arr < np.mean(cay_ret))*100
print(f"  Cayley percentile in Random: {pct_ret:.1f}%")
_, p_ret = mannwhitneyu([np.mean(cay_ret)], rnd_ret_arr,
                         alternative='two-sided')
print(f"  Mann-Whitney p = {p_ret:.4f}")

print(f"\n[3] Spreading exponent α (⟨spread⟩ ~ t^α):")
print(f"  Cayley S_4:  α = {cay_alph:.4f}")
print(f"  Random mean: α = {rnd_alp_arr.mean():.4f} "
      f"± {rnd_alp_arr.std():.4f}")
print(f"  α≈1: diffusive, α≈2: ballistic, α≈0: localized")
pct_alph = np.mean(rnd_alp_arr < cay_alph)*100
print(f"  Cayley percentile in Random: {pct_alph:.1f}%")
_, p_alph = mannwhitneyu([cay_alph], rnd_alp_arr,
                          alternative='two-sided')
print(f"  Mann-Whitney p = {p_alph:.4f}")

# ── Временная эволюция ─────────────────────────────────
print(f"\n[4] Entropy S(t): Cayley vs Random (time slices)")
print(f"{'t':>8s}  {'S_Cayley':>10s}  "
      f"{'S_Rnd_mean':>12s}  {'z-score':>8s}")
print("-"*46)

# Нужны индивидуальные S_r(t), не только mean
rnd_S_by_t = []
for seed in range(min(10, N_TRIALS)):
    A_r = random_regular_graph(N, d, seed=seed)
    if A_r is None:
        continue
    L_r = norm_laplacian(A_r)
    S_r = quantum_walk_entropy(L_r, t_vals, n_starts=8, rng=rng)
    rnd_S_by_t.append(S_r)

rnd_S_by_t = np.array(rnd_S_by_t)  # (n_trials, n_t)
rnd_mean_t = rnd_S_by_t.mean(axis=0)
rnd_std_t  = rnd_S_by_t.std(axis=0) + 1e-10

for ti in range(0, len(t_vals), 30):
    t   = t_vals[ti]
    s_c = cay_S[ti]
    s_r = rnd_mean_t[ti]
    s_s = rnd_std_t[ti]
    z   = (s_c - s_r) / s_s
    print(f"  {t:6.2f}    {s_c:10.4f}    {s_r:12.4f}  {z:8.3f}")

# ── Сравнение алгебр ───────────────────────────────────
print(f"\n[5] Comparing Coxeter groups (quantum walk entropy)")
print(f"{'Group':10s}  {'N':>5s}  {'d':>3s}  "
      f"{'mean_S':>8s}  {'vs_Rnd%':>8s}  {'p':>8s}")
print("-"*50)

for n_sym in [3, 4, 5]:
    A_c = make_cayley_s_n(n_sym, "adjacent")
    L_c = norm_laplacian(A_c)
    Nc  = A_c.shape[0]
    dc  = int(A_c.sum(axis=1).mean())
    S_c = quantum_walk_entropy(L_c, t_vals,
                                n_starts=min(Nc, 15), rng=rng)
    ms_c = np.mean(S_c)

    # Случайные графы той же степени
    rnd_ms = []
    for seed in range(20):
        A_r = random_regular_graph(Nc, dc, seed=seed)
        if A_r is None:
            continue
        L_r = norm_laplacian(A_r)
        S_r = quantum_walk_entropy(L_r, t_vals,
                                    n_starts=8, rng=rng)
        rnd_ms.append(np.mean(S_r))

    rnd_ms = np.array(rnd_ms)
    pct    = np.mean(rnd_ms < ms_c) * 100
    if len(rnd_ms) > 0:
        _, p_val = mannwhitneyu([ms_c], rnd_ms,
                                 alternative='two-sided')
    else:
        p_val = np.nan

    label = f"S_{n_sym}=A_{n_sym-1}"
    print(f"  {label:10s}  {Nc:5d}  {dc:3d}  "
          f"{ms_c:8.4f}  {pct:8.1f}%  {p_val:8.4f}")

# ── Итог ───────────────────────────────────────────────
print("\n" + "="*62)
print("SUMMARY: Quantum walk on Cayley graphs")
print("="*62)
print(f"""
  Key observables:
    mean S(t):     entropy of probability distribution
    mean P(v→v):   return probability
    α (spreading): ⟨spread⟩ ~ t^α

  H0: Cayley graph quantum walk ≡ random regular graph

  If p < 0.05 for mean S(t):
    → Cayley graphs have LESS entropy = MORE coherence
    → Group structure preserves quantum coherence
    → This is a non-trivial dynamical property

  If p > 0.05 for all observables:
    → No distinction
    → Part XI also negative
    → Publish complete record as methodology paper
""")
