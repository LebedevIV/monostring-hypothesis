"""
Part XI: Quantum walk on Cayley graph of W(A_3) = S_4
Единственный нетривиальный нетестированный вопрос.

Непрерывное квантовое блуждание:
  |ψ(t)⟩ = exp(-i·L·t)|ψ(0)⟩

  где L = нормализованный лапласиан Cay(S_4)

Наблюдаемые:
  1. Hitting time: когда амплитуда достигает целевого узла?
  2. Mixing time: когда |ψ(t)|² ≈ uniform?
  3. Return probability: P(v→v, t) — специфична для группы?

H0: квантовое блуждание на Cay(W) не отличается
    от блуждания на случайном регулярном графе
    той же степени и размера.
"""

import numpy as np
from scipy.linalg import expm, eigvalsh, eigh
from itertools import permutations
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

def make_s4():
    """S_4 с соседними транспозициями."""
    elems = list(permutations(range(4)))
    idx   = {e: i for i, e in enumerate(elems)}
    def compose(p, q):
        return tuple(p[q[i]] for i in range(4))
    gens = []
    for i in range(3):
        s = list(range(4)); s[i],s[i+1] = s[i+1],s[i]
        gens.append(tuple(s))
    N = len(elems)
    rows, cols = [], []
    for i,w in enumerate(elems):
        for s in gens:
            ws = compose(w,s)
            j  = idx[ws]
            rows.append(i); cols.append(j)
    edges = set(zip(rows,cols))
    r,c   = zip(*edges)
    A     = csr_matrix((np.ones(len(r)),(r,c)),shape=(N,N))
    A     = (A+A.T); A.data[:]=1.0; A.eliminate_zeros()
    return A.toarray()

def random_regular_graph(N, d, seed=42):
    """Случайный d-регулярный граф на N вершинах."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        stubs = np.repeat(np.arange(N), d)
        rng.shuffle(stubs)
        s = len(stubs)//2*2
        r = stubs[:s:2]; c = stubs[1:s:2]
        edges = set()
        ok = True
        for ri,ci in zip(r,c):
            if ri==ci: ok=False; break
            e=(min(ri,ci),max(ri,ci))
            if e in edges: ok=False; break
            edges.add(e)
        if ok:
            A = np.zeros((N,N))
            for ri,ci in zip(r,c):
                A[ri,ci]=A[ci,ri]=1.0
            return A
    return None

def normalized_laplacian(A):
    deg  = A.sum(axis=1)
    D12  = np.diag(1/np.sqrt(np.where(deg>0,deg,1)))
    return np.eye(len(A)) - D12 @ A @ D12

def quantum_walk(L, psi0, t_max=50, n_steps=500):
    """
    Непрерывное квантовое блуждание |ψ(t)⟩ = exp(-iLt)|ψ(0)⟩
    Возвращает вероятности P(v,t) = |⟨v|ψ(t)⟩|²
    """
    vals, vecs = eigh(L)
    # ψ(t) = Σ_k exp(-i λ_k t) ⟨vk|ψ0⟩ |vk⟩
    c0    = vecs.T @ psi0  # коэффициенты
    times = np.linspace(0, t_max, n_steps)
    probs = np.zeros((n_steps, len(psi0)))
    for ti, t in enumerate(times):
        phases = np.exp(-1j * vals * t)
        psi_t  = vecs @ (phases * c0)
        probs[ti] = np.abs(psi_t)**2
    return times, probs

def hitting_time(probs, target, threshold=0.1):
    """Первое время когда P(target,t) > threshold."""
    for ti, p in enumerate(probs):
        if p[target] > threshold:
            return ti
    return len(probs)

def mixing_measure(probs):
    """
    ||P(t) - uniform||_1 как функция t.
    Для N вершин: uniform = 1/N.
    """
    N = probs.shape[1]
    uniform = np.ones(N) / N
    return np.array([np.sum(np.abs(p - uniform)) for p in probs])

print("="*60)
print("Part XI: Quantum Walk on Cayley Graph of S_4")
print("="*60)

N = 24  # |S_4|
d = 3   # степень (соседние транспозиции)

# Графы
A_cay = make_s4()
A_rnd = random_regular_graph(N, d, seed=42)

L_cay = normalized_laplacian(A_cay)
L_rnd = normalized_laplacian(A_rnd)

# Начальное состояние: δ на вершине 0 (единица группы)
psi0 = np.zeros(N); psi0[0] = 1.0

# Квантовое блуждание
t_max = 30.0
times, P_cay = quantum_walk(L_cay, psi0, t_max, 1000)
times, P_rnd = quantum_walk(L_rnd, psi0, t_max, 1000)

print(f"\n[1] Return probability P(e→e, t):")
print(f"{'t':>8s}  {'P_Cayley':>12s}  {'P_Random':>12s}")
print("-"*36)
for ti in [0,50,100,200,300,500,700,999]:
    if ti < len(times):
        print(f"  {times[ti]:6.2f}  "
              f"{P_cay[ti,0]:12.6f}  "
              f"{P_rnd[ti,0]:12.6f}")

print(f"\n[2] Mixing measure ||P(t)-uniform||₁:")
mix_cay = mixing_measure(P_cay)
mix_rnd = mixing_measure(P_rnd)
print(f"{'t':>8s}  {'mix_Cay':>10s}  {'mix_Rnd':>10s}")
print("-"*32)
for ti in [0,100,200,500,999]:
    if ti < len(times):
        print(f"  {times[ti]:6.2f}  "
              f"{mix_cay[ti]:10.6f}  "
              f"{mix_rnd[ti]:10.6f}")

print(f"\n[3] Hitting time (threshold=0.1):")
for target in [1, 12, 23]:
    ht_cay = hitting_time(P_cay, target) * t_max/1000
    ht_rnd = hitting_time(P_rnd, target) * t_max/1000
    print(f"  target={target:2d}: "
          f"Cayley={ht_cay:.2f}, Random={ht_rnd:.2f}")

print(f"\n[4] Time-averaged return probability:")
print(f"  Cayley: {P_cay[:,0].mean():.6f}")
print(f"  Random: {P_rnd[:,0].mean():.6f}")
print(f"  Uniform: {1/N:.6f}")

print(f"\n[5] Survival of quantum coherence:")
# Энтропия распределения вероятностей
def entropy(p):
    p = p[p>1e-12]
    return -np.sum(p*np.log(p))

ent_cay = [entropy(P_cay[ti]) for ti in range(0,1000,50)]
ent_rnd = [entropy(P_rnd[ti]) for ti in range(0,1000,50)]
print(f"  Mean S(t) Cayley: {np.mean(ent_cay):.4f}")
print(f"  Mean S(t) Random: {np.mean(ent_rnd):.4f}")
print(f"  Max entropy (uniform): {np.log(N):.4f}")
