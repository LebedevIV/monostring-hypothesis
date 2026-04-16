"""
Part VI Final Report
====================
Monostring Fragmentation: Summary of findings
"""

# ── Исправление 1: неинтерактивный бэкенд ──
import matplotlib
matplotlib.use('Agg')  # Без GUI — убирает ошибки GTK, GL, D-Bus

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Data from v7-v11 (hardcoded confirmed results)
dcorr_data = {
    'E6\n(Coxeter)':      (3.013, 0.125),
    'Shuf E6\n(same w)':  (3.057, 0.159),   # ω → w (ASCII)
    'Rand\n(uniform)':    (5.542, 0.072),
}

dS_profile = {
    20:  (+0.0003, 0.9944),
    30:  (-0.1741, 0.0000),
    40:  (-0.2387, 0.0000),
    60:  (-0.0294, 0.3694),
    80:  (-0.0173, 0.6257),
    120: (+0.1190, 0.0005),
    170: (-0.0230, 0.4376),
    250: (+0.0136, 0.6986),
    350: (+0.0744, 0.0481),
    500: (+0.0238, 0.3904),
    700: (-0.1499, 0.0000),
    1000:(-0.0753, 0.0113),
}

tau_sigma = {
    0.20: 228.0,
    0.35: 236.4,
    0.50: 236.9,
    0.70: 235.4,
    1.00: 237.1,
}

tau_freqs = {
    'E6':         297.1,
    'Shuf E6':    321.0,
    'Arithmetic': 100.4,
    'Uniform':    None,
    'Geometric':  None,
}

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    "Monostring Hypothesis -- Part VI: Fragmentation\n"
    "Summary of 11 computational experiments",
    fontsize=16, fontweight='bold', y=0.99,
    color='#1a1a2e')

gs = gridspec.GridSpec(
    3, 4, figure=fig,
    hspace=0.50, wspace=0.40)


# ── Panel 1: D_corr of orbits ─────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor('#f0f4f8')

labels = list(dcorr_data.keys())
means  = [dcorr_data[k][0] for k in labels]
stds   = [dcorr_data[k][1] for k in labels]
colors = ['#2196F3', '#FF9800', '#4CAF50']

bars = ax.bar(range(len(labels)), means,
               yerr=stds, capsize=8,
               color=colors, alpha=0.85,
               edgecolor='k', linewidth=1.2,
               error_kw=dict(elinewidth=2))

ax.axhline(3.02, c='red', ls='--', lw=2,
           label='Part V: 3.02', zorder=5)
ax.fill_between([-0.5, 2.5], [2.7, 2.7],
                [3.35, 3.35], alpha=0.08,
                color='red', label='+/-0.35 band')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('D_corr', fontsize=11)
ax.set_title('Finding 1: D_corr of orbits\n'
             'E6 ~ Shuf ~ 3.02  !=  Rand',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.set_ylim([0, 7])
ax.grid(True, alpha=0.3, axis='y')

ax.annotate('Same!', xy=(0.5, 3.2),
            fontsize=11, ha='center',
            color='darkblue', fontweight='bold')
ax.annotate('Different!', xy=(2, 5.8),
            fontsize=11, ha='center',
            color='darkgreen', fontweight='bold')


# ── Panel 2: dS(t) profile ────────────────────────────
ax = fig.add_subplot(gs[0, 1:3])
ax.set_facecolor('#f0f4f8')

T_arr  = sorted(dS_profile.keys())
dS_arr = [dS_profile[t][0] for t in T_arr]
pv_arr = [dS_profile[t][1] for t in T_arr]

for i in range(len(T_arr)-1):
    t1, t2 = T_arr[i], T_arr[i+1]
    d1, d2 = dS_arr[i], dS_arr[i+1]
    mid_d  = (d1+d2)/2
    c      = '#c8e6c9' if mid_d < 0 else '#ffcdd2'
    ax.fill_between([t1,t2], [d1,d2], 0,
                    alpha=0.4, color=c)

ax.plot(T_arr, dS_arr, 'o-',
        c='#1565C0', lw=2.5, ms=8,
        zorder=4, label='dS(t)')
ax.axhline(0, c='k', lw=2, zorder=3)

for t, d, p in zip(T_arr, dS_arr, pv_arr):
    if p < 0.01 and d < 0:
        ax.scatter(t, d, s=150, c='#2E7D32',
                   zorder=6, edgecolors='k',
                   linewidth=1.5)
    elif p < 0.01:
        ax.scatter(t, d, s=150, c='#C62828',
                   zorder=6, edgecolors='k',
                   linewidth=1.5)
    elif p < 0.05:
        ax.scatter(t, d, s=100, c='#F57F17',
                   zorder=6, edgecolors='k',
                   linewidth=1)

ax.annotate('E6 more\nordered',
            xy=(35, -0.23), fontsize=10,
            color='#2E7D32', ha='center',
            fontweight='bold')
ax.annotate('(oscillates)', xy=(200, 0.08),
            fontsize=9, color='gray',
            ha='center', style='italic')
ax.annotate('E6 more\nordered again',
            xy=(700, -0.15), fontsize=10,
            color='#2E7D32', ha='center',
            fontweight='bold')

ax.set_xlabel('t (evolution steps)', fontsize=11)
ax.set_ylabel('dS = S(E6) - S(Shuf)', fontsize=11)
ax.set_title(
    'Finding 2: Entropy difference dS(t)\n'
    '[green]=E6 ordered (p<0.01)  '
    '[red]=Shuf ordered (p<0.01)  '
    '[orange]=p<0.05',
    fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)


# ── Panel 3: tau(sigma) ───────────────────────────────
ax = fig.add_subplot(gs[0, 3])
ax.set_facecolor('#f0f4f8')

sigmas = sorted(tau_sigma.keys())
taus   = [tau_sigma[s] for s in sigmas]

ax.plot(sigmas, taus, 'o-', c='#1565C0',
        lw=2.5, ms=12,
        markerfacecolor='white',
        markeredgewidth=2.5)

ax.axhline(np.mean(taus[1:]), c='red',
           ls='--', lw=1.5,
           label='mean={:.0f}'.format(np.mean(taus[1:])))

ax.set_xlabel('sigma (fragmentation width)', fontsize=11)
ax.set_ylabel('tau_E6 (crossover steps)', fontsize=11)
ax.set_title('Finding 3: tau vs sigma\n'
             'tau ~ const (sigma-independent!)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim([0, 450])
ax.grid(True, alpha=0.3)

ax.annotate('tau ~ 236 steps\n(sigma-independent)',
            xy=(0.60, 236), xytext=(0.65, 320),
            fontsize=9, color='red',
            arrowprops=dict(arrowstyle='->',
                            color='red'))


# ── Panel 4: tau by frequency type ────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor('#f0f4f8')

freq_names = ['E6', 'Shuf\nE6',
              'Arith-\nmetic', 'Uniform',
              'Geo-\nmetric']
freq_taus  = [297.1, 321.0, 100.4, 0, 0]
freq_colors= ['#2196F3', '#FF9800',
               '#9C27B0', '#9E9E9E', '#9E9E9E']

bars2 = ax.bar(range(5), freq_taus,
               color=freq_colors, alpha=0.85,
               edgecolor='k', linewidth=1)

ax.set_xticks(range(5))
ax.set_xticklabels(freq_names, fontsize=8)
ax.set_ylabel('tau (steps)', fontsize=11)
ax.set_title('Finding 4: tau by freq. type\n'
             'Irrationality -> longer tau',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, freq_taus):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 5, '{:.0f}'.format(val),
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2,
                15, 'N/A',
                ha='center', va='bottom',
                fontsize=9, color='gray')


# ── Исправление 2: замена эмодзи на ASCII-символы ──

# ── Panel 5: Falsified claims ─────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor('#fff3e0')
ax.axis('off')

falsified = [
    "[X] D_corr(daughters) -> orbit",
    "    (T_conv = 700,000 steps)",
    "",
    "[X] E6 unique for D_corr ~ 3",
    "    (Shuf gives same 3.06)",
    "",
    "[X] Monotone arrow of time",
    "    (dS oscillates)",
    "",
    "[X] tau depends on sigma",
    "    (tau ~ const for sigma>=0.35)",
    "",
    "[X] Daughters follow orbit",
    "    (no convergence observed)",
    "",
    "[X] Kuramoto equalizes",
    "    (collapses to one point)",
]
ax.text(0.05, 0.97, "FALSIFIED\n" + "-"*28,
        transform=ax.transAxes,
        fontsize=11, fontweight='bold',
        va='top', color='#C62828',
        fontfamily='monospace')
ax.text(0.05, 0.82, "\n".join(falsified),
        transform=ax.transAxes,
        fontsize=8.5, fontfamily='monospace',
        va='top', color='#4a4a4a')


# ── Panel 6: Confirmed claims ─────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor('#e8f5e9')
ax.axis('off')

confirmed = [
    "[+] D_corr(E6 orbit) = 3.02",
    "    reproduces Part V exactly",
    "",
    "[+] D_corr = f(freq set, not order)",
    "    E6 ~ Shuf ~ 3.0",
    "    Rand ~ 5.5 (different!)",
    "",
    "[+] dS(t=30-40) < 0  p<0.0001",
    "    Initial ordering effect",
    "",
    "[+] tau(Arithmetic) << tau(E6)",
    "    Irrationality matters",
    "",
    "[+] Resonance -> T_rec(Rand) = inf",
    "    Structured w -> recurrent",
]
ax.text(0.05, 0.97, "CONFIRMED\n" + "-"*28,
        transform=ax.transAxes,
        fontsize=11, fontweight='bold',
        va='top', color='#2E7D32',
        fontfamily='monospace')
ax.text(0.05, 0.82, "\n".join(confirmed),
        transform=ax.transAxes,
        fontsize=8.5, fontfamily='monospace',
        va='top', color='#1a1a2e')


# ── Panel 7: New discoveries ──────────────────────────
ax = fig.add_subplot(gs[1, 3])
ax.set_facecolor('#e3f2fd')
ax.axis('off')

new_disc = [
    "[*] D_corr determined by SET",
    "    of frequencies, not E6",
    "    algebraic structure",
    "",
    "[*] Rand orbit: D_corr = 5.5",
    "    (not 6, not 3 -- why 5.5?)",
    "    -> uniform range but corr.",
    "",
    "[*] tau ~ 237 ~ Coxeter h x 20?",
    "    h(E6) = 12",
    "    12 x 20 = 240 ~ 237",
    "    Coincidence?",
    "",
    "[*] dS oscillation period",
    "    ~ 1/dw_min ~ 50 steps",
    "    -> beating frequencies",
]
ax.text(0.05, 0.97, "NEW DISCOVERIES\n" + "-"*28,
        transform=ax.transAxes,
        fontsize=11, fontweight='bold',
        va='top', color='#1565C0',
        fontfamily='monospace')
ax.text(0.05, 0.82, "\n".join(new_disc),
        transform=ax.transAxes,
        fontsize=8.5, fontfamily='monospace',
        va='top', color='#1a1a2e')


# ── Panel 8: Physical interpretation ─────────────────
ax = fig.add_subplot(gs[2, :])
ax.set_facecolor('#fafafa')
ax.axis('off')

ax.text(0.02, 0.92,
        "PHYSICAL PICTURE (Part VI conclusion):",
        fontsize=13, fontweight='bold',
        transform=ax.transAxes, color='#1a1a2e')

boxes = [
    (0.02, 0.55, 0.18, 0.28,
     "MONOSTRING\nE6 orbit on T^6\nD_corr = 3.02\n(quasi-3D)",
     '#BBDEFB', '#1565C0'),
    (0.27, 0.55, 0.18, 0.28,
     "BREAKUP\nResonance criterion\nN daughters born\nnear phi_break",
     '#FFE0B2', '#E65100'),
    (0.52, 0.55, 0.18, 0.28,
     "EARLY (t<40)\nS(E6) < S(Shuf)\nCommon origin\ndominator",
     '#C8E6C9', '#2E7D32'),
    (0.77, 0.55, 0.18, 0.28,
     "LATE (t>=300)\nS(E6) < S(Shuf)\nE6 attractor\n(3D orbit)",
     '#C8E6C9', '#2E7D32'),
]

for (x, y, w, h, txt, fc, ec) in boxes:
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec,
        linewidth=2,
        transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, txt,
            transform=ax.transAxes,
            fontsize=9, ha='center',
            va='center', fontweight='bold',
            color='#1a1a2e')

for x_start, x_end in [
        (0.20, 0.27), (0.45, 0.52),
        (0.70, 0.77)]:
    ax.annotate('', xy=(x_end, 0.69),
                xytext=(x_start, 0.69),
                xycoords='axes fraction',
                arrowprops=dict(
                    arrowstyle='->',
                    color='#555', lw=2.5))

rect_mid = mpatches.FancyBboxPatch(
    (0.27, 0.15), 0.43, 0.28,
    boxstyle="round,pad=0.02",
    facecolor='#FCE4EC',
    edgecolor='#880E4F',
    linewidth=2,
    transform=ax.transAxes)
ax.add_patch(rect_mid)
ax.text(0.485, 0.29,
        "TRANSITION REGION (t ~ 40-300)\n"
        "dS oscillates (quasi-periodic beating)\n"
        "Period ~ 50 steps ~ 1/dw_min\n"
        "tau ~ 237 steps (independent of sigma, N)",
        transform=ax.transAxes,
        fontsize=9.5, ha='center', va='center',
        color='#1a1a2e')

ax.text(0.50, 0.05,
        "The E6 orbit (D_corr=3.02) acts as an "
        "attractor for daughter strings.\n"
        "Daughters with w_E6 thermalize on this "
        "3D manifold -> lower entropy than\n"
        "null daughters exploring full T^6. "
        "This is consistent with Part V.",
        transform=ax.transAxes,
        fontsize=10, ha='center', va='bottom',
        style='italic', color='#37474F')

plt.savefig(
    'monostring_part6_final_summary.png',
    dpi=150, bbox_inches='tight',
    facecolor=fig.get_facecolor())
print("Saved: monostring_part6_final_summary.png")

# НЕ вызываем plt.show() — используем Agg бэкенд
# plt.show()  # убрано — Agg не поддерживает интерактивный показ
