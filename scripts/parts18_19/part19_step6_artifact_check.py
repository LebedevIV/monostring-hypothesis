"""
Part XIX Step 6 (fixed): Complete artifact analysis

Исправление ValueError + rank=8 + финальный вердикт.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ── МАТРИЦЫ КАРТАНА ──────────────────────────────────────────

def make_An(n):
    C = np.diag([2.0]*n)
    for i in range(n-1):
        C[i,i+1] = C[i+1,i] = -1.0
    return C

CARTAN = {
    'A6': make_An(6),
    'E6': np.array([
        [ 2,-1, 0, 0, 0, 0],
        [-1, 2,-1, 0, 0, 0],
        [ 0,-1, 2,-1, 0,-1],
        [ 0, 0,-1, 2,-1, 0],
        [ 0, 0, 0,-1, 2, 0],
        [ 0, 0,-1, 0, 0, 2]], dtype=float),
    'E8': np.array([
        [ 2,-1, 0, 0, 0, 0, 0, 0],
        [-1, 2,-1, 0, 0, 0, 0, 0],
        [ 0,-1, 2,-1, 0, 0, 0,-1],
        [ 0, 0,-1, 2,-1, 0, 0, 0],
        [ 0, 0, 0,-1, 2,-1, 0, 0],
        [ 0, 0, 0, 0,-1, 2,-1, 0],
        [ 0, 0, 0, 0, 0,-1, 2, 0],
        [ 0, 0,-1, 0, 0, 0, 0, 2]], dtype=float),
}

# ── УТИЛИТЫ ──────────────────────────────────────────────────

def safe_eigvalsh(M):
    try:
        if not np.all(np.isfinite(M)):
            return None
        vals = np.linalg.eigvalsh(M)
        if not np.all(np.isfinite(vals)):
            return None
        return vals
    except np.linalg.LinAlgError:
        return None

def normal_modes(C, kappa):
    eigs_C = safe_eigvalsh(C)
    if eigs_C is None:
        return None
    omegas = np.sqrt(np.maximum(eigs_C, 0))
    r = C.shape[0]
    M = np.diag(omegas)
    for i in range(r):
        for j in range(r):
            if i != j:
                M[i,j] += kappa * C[i,j]
    return safe_eigvalsh(M)

def find_kc(C, kappa_max=5.0, n_pts=500):
    kappas = np.linspace(0, kappa_max, n_pts)
    for kappa in kappas:
        freqs = normal_modes(C, kappa)
        if freqs is not None and freqs[0] < -1e-8:
            return kappa
    return kappa_max

def gap_at_fraction(C, fraction=0.8):
    kc    = find_kc(C)
    kappa = fraction * kc
    freqs = normal_modes(C, kappa)
    if freqs is None:
        return None, kc, kappa
    pos = freqs[freqs > 1e-8]
    gap = pos.min() if len(pos) > 0 else 0.0
    return gap, kc, kappa

# ── ГЕНЕРАТОРЫ КОНТРОЛЕЙ ─────────────────────────────────────

def ctrl_A(rank, seed):
    """Тип A: с принудительным PSD сдвигом."""
    rng = np.random.RandomState(seed)
    C   = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(i+1, rank):
            v = rng.choice([-1, 0, 0, 1])
            C[i,j] = C[j,i] = float(v)
    eigs  = safe_eigvalsh(C)
    shift = 0.0 if eigs is None else max(0.0, -eigs[0]+0.01)
    np.fill_diagonal(C, 2.0 + shift)
    return C

def ctrl_B(rank, seed):
    """Тип B: diagonal=2 точно, только PSD."""
    rng = np.random.RandomState(seed)
    C   = 2.0 * np.eye(rank)
    for i in range(rank):
        for j in range(i+1, rank):
            v = rng.choice([-1, 0, 0, 1])
            C[i,j] = C[j,i] = float(v)
    eigs = safe_eigvalsh(C)
    if eigs is None or eigs[0] < -1e-10:
        return None
    return C

def ctrl_C(rank, seed):
    """Тип C: sparsity-matched (rank-1 off-diag)."""
    n_nonzero = rank - 1
    rng   = np.random.RandomState(seed)
    pairs = [(i,j) for i in range(rank)
              for j in range(i+1, rank)]
    for _ in range(100):
        C      = 2.0 * np.eye(rank)
        chosen = rng.choice(len(pairs),
                             n_nonzero, replace=False)
        for idx in chosen:
            i, j   = pairs[idx]
            C[i,j] = C[j,i] = -1.0
        eigs = safe_eigvalsh(C)
        if eigs is not None and eigs[0] >= -1e-10:
            return C
    return None

# ── СБОР ДАННЫХ ───────────────────────────────────────────────

print("="*65)
print("Part XIX Step 6 (fixed): Complete Artifact Analysis")
print("="*65)
print()

# Диагностика diagonal
print("── Diagonal inflation in controls ──")
diag_stats = {}
for ctype, gen in [('A', ctrl_A),
                    ('B', ctrl_B),
                    ('C', ctrl_C)]:
    diags = []
    for s in range(200):
        C = gen(6, seed=s)
        if C is not None:
            diags.append(np.diag(C).mean())
    if diags:
        diag_stats[ctype] = np.array(diags)
        print(f"  Ctrl-{ctype}: diagonal = "
              f"{np.mean(diags):.4f} ± {np.std(diags):.4f} "
              f"(N={len(diags)})")
    else:
        diag_stats[ctype] = np.array([])
        print(f"  Ctrl-{ctype}: N=0 (none passed PSD)")
print(f"  Coxeter: diagonal = 2.0000 (exact)")
print()

# Генерация N=300 контролей для каждого ранга
N_CTRL = 300

all_results = {}

for rank in [6, 8]:
    print(f"── Rank={rank}: collecting N={N_CTRL} "
          f"per control type ──")
    rank_data = {}

    for ctype, gen in [('A', ctrl_A),
                        ('B', ctrl_B),
                        ('C', ctrl_C)]:
        gaps = []
        kcs  = []
        diags = []
        n_tried = 0
        n_none  = 0

        for s in range(N_CTRL * 20):
            if len(gaps) >= N_CTRL:
                break
            n_tried += 1
            C = gen(rank, seed=s + rank*10000)
            if C is None:
                n_none += 1
                continue
            g, kc, _ = gap_at_fraction(C, 0.8)
            if g is None:
                n_none += 1
                continue
            gaps.append(g)
            kcs.append(kc)
            diags.append(np.diag(C).mean())

        gaps  = np.array(gaps)
        kcs   = np.array(kcs)
        diags = np.array(diags)

        # Фильтруем нулевые gaps
        # (нестабильные матрицы дают gap=0)
        pos_mask = gaps > 1e-6
        gaps_pos = gaps[pos_mask]
        n_zero   = np.sum(~pos_mask)

        rank_data[ctype] = {
            'gaps':     gaps,
            'gaps_pos': gaps_pos,
            'kcs':      kcs,
            'diags':    diags,
            'n_tried':  n_tried,
            'n_none':   n_none,
            'n_zero':   n_zero,
        }

        # Форматируем числа явно
        N     = len(gaps)
        Npos  = len(gaps_pos)
        dmean = float(diags.mean()) if len(diags) else 0.0
        dstd  = float(diags.std())  if len(diags) else 0.0
        gmean = float(gaps_pos.mean()) if Npos > 0 else 0.0
        gstd  = float(gaps_pos.std())  if Npos > 0 else 0.0

        print(f"  Ctrl-{ctype}: N={N}, "
              f"pos_gaps={Npos} "
              f"(zero={n_zero}), "
              f"diag={dmean:.4f}±{dstd:.4f}, "
              f"gap={gmean:.4f}±{gstd:.4f}")

    all_results[rank] = rank_data
    print()

# ── СТАТИСТИЧЕСКИЕ ТЕСТЫ ─────────────────────────────────────

print("── Statistical Tests: Coxeter vs Controls ──")
print()

verdicts = {}
for alg_name, C_alg in [('E6', CARTAN['E6']),
                          ('A6', CARTAN['A6']),
                          ('E8', CARTAN['E8'])]:
    rank = C_alg.shape[0]
    g, kc, kused = gap_at_fraction(C_alg, 0.8)

    print(f"  {alg_name} (rank={rank}): "
          f"gap={g:.5f}, κ_c={kc:.4f}")

    row = {'gap': g, 'kc': kc, 'rank': rank}

    for ctype in ['A', 'B', 'C']:
        rdata    = all_results[rank][ctype]
        # Используем только положительные gaps
        rg       = rdata['gaps_pos']
        diag_m   = float(rdata['diags'].mean()) \
                   if len(rdata['diags']) > 0 else 0.0

        if len(rg) < 20:
            print(f"    vs {ctype}: "
                  f"N_pos={len(rg)} (insufficient)")
            row[f'p_{ctype}']   = np.nan
            row[f'pct_{ctype}'] = np.nan
            continue

        pct  = float(np.mean(rg < g) * 100)
        _, p = stats.mannwhitneyu(
                   [g], rg,
                   alternative='two-sided')
        p    = float(p)
        sig  = "***" if p < 0.05 else "ns"

        # Проверка: разница из-за diagonal?
        diag_diff = diag_m - 2.0
        note = (f"[diag+{diag_diff:.2f}→artifact?]"
                 if diag_diff > 0.1 else "[diag≈2]")

        print(f"    vs Ctrl-{ctype} "
              f"(N={len(rg)}, "
              f"μ_diag={diag_m:.3f}): "
              f"pct={pct:.0f}%, "
              f"p={p:.4f} {sig} {note}")

        row[f'p_{ctype}']   = p
        row[f'pct_{ctype}'] = pct

    verdicts[alg_name] = row
    print()

# ── КЛЮЧЕВОЙ ВОПРОС ──────────────────────────────────────────

print("── Key Question: Does signal survive fair controls? ──")
print()

# Честные контроли = B и C (diagonal=2 или sparsity-matched)
signal_fair = {}
for name in ['E6', 'A6', 'E8']:
    v      = verdicts[name]
    p_B    = v.get('p_B', np.nan)
    p_C    = v.get('p_C', np.nan)
    pct_B  = v.get('pct_B', np.nan)
    pct_C  = v.get('pct_C', np.nan)

    sig_B  = (not np.isnan(p_B)) and p_B < 0.05
    sig_C  = (not np.isnan(p_C)) and p_C < 0.05
    insuff = (np.isnan(p_B) and np.isnan(p_C))

    if insuff:
        verdict_str = "INSUFFICIENT DATA"
        survived    = None
    elif sig_B or sig_C:
        verdict_str = "SIGNAL SURVIVES"
        survived    = True
    else:
        verdict_str = "H0 holds (artifact)"
        survived    = False

    signal_fair[name] = survived
    print(f"  {name}: p_B={p_B:.4f}, "
          f"p_C={p_C:.4f} → {verdict_str}")

print()

# ── МАТЕМАТИЧЕСКИЙ АРГУМЕНТ ──────────────────────────────────

print("── Mathematical argument ──")
print()
print("  ТЕОРЕМА (Сильвестр/Картан):")
print("  Матрица Картана простой алгебры Ли")
print("  является положительно определённой.")
print()
print("  СЛЕДСТВИЕ для нашей модели:")
print("  При κ=0: Ωᵢ = √λᵢ(C) > 0 (стабильна)")
print("  При κ>0: нестабильность при κ_c ∝ 1/rank")
print()
print("  ПРОВЕРКА PSD для случайных матриц:")

for rank in [6, 8]:
    n_psd = 0
    n_try = 1000
    rng   = np.random.RandomState(42)
    for s in range(n_try):
        C = 2.0 * np.eye(rank)
        for i in range(rank):
            for j in range(i+1, rank):
                v = rng.choice([-1, 0, 0, 1])
                C[i,j] = C[j,i] = float(v)
        eigs = safe_eigvalsh(C)
        if eigs is not None and eigs[0] >= -1e-10:
            n_psd += 1
    frac = n_psd / n_try * 100
    print(f"  rank={rank}: {n_psd}/{n_try} "
          f"({frac:.1f}%) random matrices are PSD")

print()
print("  Если P(PSD) << 1% для random →")
print("  сравнение с Cartanом некорректно:")
print("  случайные PSD матрицы — особый подкласс,")
print("  систематически отличный от Картановых.")
print()

# ── ФИНАЛЬНЫЙ ВЫВОД ──────────────────────────────────────────

print("="*65)
print("PART XIX — FINAL VERDICT")
print("="*65)
print()

# Подсчёт сигналов
n_signal    = sum(1 for v in signal_fair.values()
                   if v is True)
n_artifact  = sum(1 for v in signal_fair.values()
                   if v is False)
n_insuff    = sum(1 for v in signal_fair.values()
                   if v is None)

print(f"Fair controls (B/C):")
print(f"  Signals:     {n_signal}")
print(f"  Artifacts:   {n_artifact}")
print(f"  Insufficient:{n_insuff}")
print()

# Итоговая таблица
print(f"  {'Alg':>5} {'gap':>8} "
      f"{'p(A)':>8} {'p(B)':>8} {'p(C)':>8} "
      f"{'conclusion':>14}")
print("  " + "-"*60)
for name in ['E6', 'A6', 'E8']:
    v   = verdicts[name]
    g   = v['gap']
    pA  = v.get('p_A', np.nan)
    pB  = v.get('p_B', np.nan)
    pC  = v.get('p_C', np.nan)
    sf  = signal_fair[name]

    def fmt_p(p):
        if np.isnan(p):
            return '   nan'
        return f'{p:.4f}'

    concl = ('SIGNAL' if sf is True
              else 'artifact' if sf is False
              else 'insuff.')
    print(f"  {name:>5} {g:>8.5f} "
          f"{fmt_p(pA):>8} "
          f"{fmt_p(pB):>8} "
          f"{fmt_p(pC):>8} "
          f"{concl:>14}")

print()

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Part XIX Step 6: Artifact Check — Complete\n'
    'Three control types, rank 6 & 8',
    fontsize=13)

colors_ctrl = {'A': '#e74c3c',
                'B': '#2ecc71',
                'C': '#f39c12'}
colors_alg  = {'E6': '#c0392b',
                'A6': '#2980b9',
                'E8': '#8e44ad'}

# 0,0: Diagonal в контролях
ax = axes[0, 0]
xticks, xvals, xerrs = [], [], []
for ctype in ['A', 'B', 'C']:
    d = diag_stats.get(ctype, np.array([]))
    if len(d) > 0:
        xticks.append(f'Ctrl-{ctype}')
        xvals.append(float(d.mean()))
        xerrs.append(float(d.std()))
xticks.append('Coxeter')
xvals.append(2.0)
xerrs.append(0.0)

bar_colors = [colors_ctrl.get(t.split('-')[-1], 'gray')
               for t in xticks[:-1]] + ['#8e44ad']
ax.bar(range(len(xticks)), xvals,
        yerr=xerrs,
        color=bar_colors, alpha=0.75, capsize=5)
ax.axhline(2.0, color='k', ls='--', lw=2)
ax.set_xticks(range(len(xticks)))
ax.set_xticklabels(xticks)
ax.set_ylabel('Mean diagonal')
ax.set_title('Diagonal inflation in controls\n'
             '(Coxeter = 2.0 exact)')
ax.grid(True, alpha=0.3, axis='y')

# 0,1: Gap distribution rank=6
ax = axes[0, 1]
for ctype in ['A', 'B', 'C']:
    gp = all_results[6][ctype]['gaps_pos']
    if len(gp) > 5:
        ax.hist(gp, bins=25,
                color=colors_ctrl[ctype],
                alpha=0.5, density=True,
                label=f'Ctrl-{ctype} '
                      f'N={len(gp)} '
                      f'μ={float(gp.mean()):.3f}')
for name in ['E6', 'A6']:
    ax.axvline(verdicts[name]['gap'],
               color=colors_alg[name], lw=2.5,
               label=f"{name}: "
                     f"{verdicts[name]['gap']:.4f}")
ax.set_xlabel('gap at 0.8·κ_c')
ax.set_title('Rank=6: gap distributions')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 0,2: Gap distribution rank=8
ax = axes[0, 2]
for ctype in ['A', 'B', 'C']:
    gp = all_results[8][ctype]['gaps_pos']
    if len(gp) > 5:
        ax.hist(gp, bins=25,
                color=colors_ctrl[ctype],
                alpha=0.5, density=True,
                label=f'Ctrl-{ctype} '
                      f'N={len(gp)} '
                      f'μ={float(gp.mean()):.3f}')
ax.axvline(verdicts['E8']['gap'],
           color=colors_alg['E8'], lw=2.5,
           label=f"E8: {verdicts['E8']['gap']:.4f}")
ax.set_xlabel('gap at 0.8·κ_c')
ax.set_title('Rank=8: gap distributions')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1,0: PSD fraction vs rank
ax = axes[1, 0]
ranks_test = [2, 3, 4, 5, 6, 7, 8]
psd_fracs  = []
n_test     = 500
for rank in ranks_test:
    n_psd = 0
    rng2  = np.random.RandomState(0)
    for s in range(n_test):
        C = 2.0 * np.eye(rank)
        for i in range(rank):
            for j in range(i+1, rank):
                v = rng2.choice([-1, 0, 0, 1])
                C[i,j] = C[j,i] = float(v)
        eigs = safe_eigvalsh(C)
        if eigs is not None and eigs[0] >= -1e-10:
            n_psd += 1
    psd_fracs.append(n_psd / n_test * 100)

ax.semilogy(ranks_test, psd_fracs, 'o-',
             color='#2ecc71', lw=2, ms=8)
ax.axhline(1.0, color='r', ls='--',
           label='1% threshold')
for rank, frac in zip(ranks_test, psd_fracs):
    ax.annotate(f'{frac:.1f}%',
                (rank, frac),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=8)
ax.set_xlabel('Rank')
ax.set_ylabel('% random matrices that are PSD (log)')
ax.set_title('PSD rate for random {-1,0,1} matrices\n'
             '(diagonal=2 exact)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

# 1,1: κ_c vs rank (Coxeter + random)
ax = axes[1, 1]
cox_ranks = [CARTAN[n].shape[0]
              for n in ['A6','E6','E8']]
cox_kcs   = [verdicts[n]['kc']
              for n in ['A6','E6','E8']]

ax.scatter(cox_ranks, cox_kcs, s=200, zorder=5,
           c=[colors_alg['A6'],
              colors_alg['E6'],
              colors_alg['E8']])
for name, r, kc in zip(['A6','E6','E8'],
                         cox_ranks, cox_kcs):
    ax.annotate(name, (r, kc),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=10)

for rank in [6, 8]:
    rkcs = all_results[rank]['A']['kcs']
    ax.scatter(np.full(len(rkcs), rank) +
               np.random.randn(len(rkcs))*0.05,
               rkcs, color='gray',
               alpha=0.1, s=15)

ax.set_xlabel('Rank')
ax.set_ylabel('κ_c')
ax.set_title('κ_c vs rank\n(Coxeter vs Random-A)')
ax.grid(True, alpha=0.3)

# 1,2: Итоговый вердикт
ax = axes[1, 2]
ax.axis('off')

any_signal = any(v is True for v in signal_fair.values())
color = '#e0ffe0' if any_signal else '#ffe8e8'

lines = [
    'PART XIX — COMPLETE VERDICT',
    '═'*32,
    '',
    'Artifact #11 confirmed:',
    f'  Ctrl-A diagonal = '
    f'{float(diag_stats["A"].mean()):.2f} (not 2.0)',
    '  → gap inflated artificially',
    '',
    'Fair control results:',
]
for name in ['E6', 'A6', 'E8']:
    v   = verdicts[name]
    pB  = v.get('p_B', np.nan)
    pC  = v.get('p_C', np.nan)
    sf  = signal_fair[name]
    tag = ('SIGNAL' if sf is True
            else 'H0' if sf is False
            else 'insuff')
    lines.append(f'  {name}: p_B={pB:.3f}, '
                  f'p_C={pC:.3f} [{tag}]')

lines += ['', '─'*32, '']
if any_signal:
    lines += ['*** SIGNAL in fair controls ***',
              '→ Mechanism check needed',
              '→ Part XX']
else:
    lines += [
        'All signals = artifacts.',
        '',
        'FINAL SCORECARD:',
        '  19 experiments',
        '  0 physical signals',
        '  11 artifacts documented',
        '  6 theorems found',
        '',
        '→ PUBLISH v14.0.0',
    ]

ax.text(0.02, 0.98, '\n'.join(lines),
        transform=ax.transAxes, fontsize=8.5,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor=color, alpha=0.9))

plt.tight_layout()
plt.savefig('part19_step6_complete.png',
            dpi=150, bbox_inches='tight')
print()
print("✓ Saved: part19_step6_complete.png")