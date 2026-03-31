import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
import time
import warnings
warnings.filterwarnings("ignore")

"""
CAUSAL SET APPROACH v2
======================
Fixes from v1:
1. N = 2000 (was 500)
2. Full look-ahead (no N//5 cutoff)
3. Better causal rules with auto-calibrated thresholds
4. Multiple independent runs for error bars
5. Careful control experiments
"""

# ================================================================
# MYRHEIM-MEYER DIMENSION ESTIMATOR
# ================================================================
def ordering_fraction_to_dim(f_val):
    """
    Invert f(d) = Gamma(d+1)*Gamma(d/2) / (4*Gamma(3d/2))
    to get dimension d from ordering fraction f.
    """
    if f_val <= 0 or f_val >= 1:
        return 0.0

    # Precompute f(d) for d from 1 to 10
    dims = np.arange(1.0, 10.01, 0.05)
    f_theory = np.array([
        gamma_func(d + 1) * gamma_func(d / 2) / (4 * gamma_func(3 * d / 2))
        for d in dims
    ])

    # Find where f_theory crosses f_val (f_theory is decreasing)
    if f_val > f_theory[0]:
        return dims[0]
    if f_val < f_theory[-1]:
        return dims[-1]

    idx = np.searchsorted(-f_theory, -f_val)  # Decreasing, so negate
    if idx == 0:
        return dims[0]
    if idx >= len(dims):
        return dims[-1]

    # Linear interpolation
    f_lo, f_hi = f_theory[idx], f_theory[idx - 1]
    d_lo, d_hi = dims[idx], dims[idx - 1]

    if abs(f_hi - f_lo) < 1e-15:
        return d_lo

    t = (f_val - f_lo) / (f_hi - f_lo)
    return d_lo + t * (d_hi - d_lo)


def estimate_dimension(order_matrix, N, n_samples=3000,
                        min_diamond_size=3, max_diamond_size=None):
    """
    Myrheim-Meyer dimension estimator.

    Sample causal diamonds (Alexandrov intervals) and compute
    the ordering fraction within each.
    """
    if max_diamond_size is None:
        max_diamond_size = N

    ordering_fractions = []
    diamond_sizes = []

    # Precompute: for each node, which nodes it precedes / is preceded by
    future = {i: set(np.where(order_matrix[i, :])[0]) for i in range(N)}
    past = {i: set(np.where(order_matrix[:, i])[0]) for i in range(N)}

    attempts = 0
    max_attempts = n_samples * 20

    while len(ordering_fractions) < n_samples and attempts < max_attempts:
        attempts += 1

        # Pick random p
        p = np.random.randint(0, N)
        if len(future[p]) == 0:
            continue

        # Pick random q in future of p
        q = np.random.choice(list(future[p]))

        # Alexandrov interval J(p,q) = future(p) ∩ past(q)
        J = list(future[p] & past[q])

        n_J = len(J)
        if n_J < min_diamond_size or n_J > max_diamond_size:
            continue

        # Count ordered pairs within J
        n_ordered = 0
        n_pairs = 0

        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                n_pairs += 1
                if order_matrix[J[i], J[j]] or order_matrix[J[j], J[i]]:
                    n_ordered += 1

        if n_pairs > 0:
            f = n_ordered / n_pairs
            ordering_fractions.append(f)
            diamond_sizes.append(n_J)

    if len(ordering_fractions) == 0:
        return 0.0, 0.0, 0.0, 0, []

    mean_f = np.mean(ordering_fractions)
    std_f = np.std(ordering_fractions) / np.sqrt(len(ordering_fractions))
    dim_est = ordering_fraction_to_dim(mean_f)

    # Error propagation: dim error from f error
    dim_plus = ordering_fraction_to_dim(mean_f - std_f)  # f decreasing -> dim increasing
    dim_minus = ordering_fraction_to_dim(mean_f + std_f)
    dim_err = abs(dim_plus - dim_minus) / 2

    return dim_est, dim_err, mean_f, len(ordering_fractions), diamond_sizes


# ================================================================
# CAUSAL SET GENERATORS
# ================================================================
def sprinkle_minkowski(N, dim):
    """Control: sprinkle N points into d-dim Minkowski diamond."""
    points = np.random.uniform(0, 1, (N, dim))
    t = points[:, 0]
    x = points[:, 1:] if dim > 1 else np.zeros((N, 1))

    order = np.zeros((N, N), dtype=bool)

    # Sort by time for efficiency
    t_order = np.argsort(t)
    t_sorted = t[t_order]
    x_sorted = x[t_order]

    for i in range(N):
        for j in range(i + 1, N):
            dt = t_sorted[j] - t_sorted[i]
            dx = np.sqrt(np.sum((x_sorted[j] - x_sorted[i])**2))
            if dx < dt:
                order[t_order[i], t_order[j]] = True

    return order


def monostring_causal_set(N, n_phases=6, rule='resonance',
                           density_target=0.05):
    """
    Generate causal set from Monostring ticks.

    Rules:
    - 'resonance': m ≺ n if phase distance < threshold (original SBE idea)
    - 'smooth': m ≺ n if phase change per tick is smooth (not jumpy)
    - 'directional': m ≺ n if phases evolve in a "forward" direction
    """
    omega = np.array([np.sqrt(p) for p in [2, 3, 5, 7, 11, 13]])[:n_phases]

    # Generate phase trajectory
    phases = np.zeros((N, n_phases))
    for n in range(N):
        phases[n] = (n * omega) % (2 * np.pi)

    order = np.zeros((N, N), dtype=bool)

    if rule == 'resonance':
        # m ≺ n if toroidal phase distance is small
        # Auto-calibrate threshold
        sample_dists = []
        for _ in range(50000):
            i = np.random.randint(0, N - 1)
            j = np.random.randint(i + 1, N)
            diff = np.abs(phases[j] - phases[i])
            diff = np.minimum(diff, 2 * np.pi - diff)
            sample_dists.append(np.sqrt(np.sum(diff**2)))

        threshold = np.percentile(sample_dists, density_target * 100)

        # Build order (FULL look-ahead, not truncated)
        for i in range(N):
            diff_all = np.abs(phases[i+1:] - phases[i])
            diff_all = np.minimum(diff_all, 2 * np.pi - diff_all)
            dists = np.sqrt(np.sum(diff_all**2, axis=1))
            close = np.where(dists < threshold)[0]
            for j_offset in close:
                order[i, i + 1 + j_offset] = True

    elif rule == 'smooth':
        # m ≺ n if average phase gradient between m and n is small
        # (smooth evolution = causal connection)
        for i in range(N):
            for j in range(i + 1, N):
                diff = np.abs(phases[j] - phases[i])
                diff = np.minimum(diff, 2 * np.pi - diff)
                grad = np.sqrt(np.sum(diff**2)) / (j - i)
                threshold = 0.5 / np.sqrt(j - i)  # Scale-dependent
                if grad < threshold:
                    order[i, j] = True

    elif rule == 'directional':
        # m ≺ n if phases at n are "ahead" of phases at m
        # in a specific sense (mimicking light-cone structure)
        #
        # Define "time direction" as the average phase velocity
        # "Spatial directions" as deviations from average
        mean_omega = np.mean(omega)

        for i in range(N):
            for j in range(i + 1, N):
                dt = j - i
                # Expected phase advance
                expected = (dt * omega) % (2 * np.pi)
                actual = phases[j] - phases[i]
                actual = (actual + np.pi) % (2 * np.pi) - np.pi

                # "Deviation" from expected = spatial distance
                deviation = np.sqrt(np.sum((actual - expected)**2))

                # "Time distance" = dt * mean_omega
                time_dist = dt * mean_omega

                # Causal if deviation < time_dist (inside light cone)
                if deviation < time_dist * 0.3:
                    order[i, j] = True

    # Compute actual density
    n_causal = np.sum(order)
    n_possible = N * (N - 1) // 2
    actual_density = n_causal / max(n_possible, 1)

    return order, phases, actual_density


# ================================================================
# TRANSITIVITY ENFORCEMENT
# ================================================================
def enforce_transitivity(order, N, max_iterations=3):
    """
    Ensure the causal order is transitive: if a≺b and b≺c then a≺c.
    This is required for a proper causal set.
    """
    for iteration in range(max_iterations):
        changed = False
        for i in range(N):
            future_i = set(np.where(order[i, :])[0])
            for j in future_i:
                future_j = set(np.where(order[j, :])[0])
                for k in future_j:
                    if not order[i, k]:
                        order[i, k] = True
                        changed = True
        if not changed:
            break
    return order


# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_causal_v2():
    print("=" * 70)
    print("  CAUSAL SET APPROACH v2 (CORRECTED)")
    print("  Larger N, full look-ahead, error bars, transitivity")
    print("=" * 70)

    # ============================================================
    # PART 1: CONTROL — Verify estimator accuracy
    # ============================================================
    print("\n[1/4] CONTROL: Minkowski sprinkle (verify estimator)")
    print("      {:>6s} {:>6s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "True D", "N", "Est D", "±err", "f", "#diam"))
    print("      " + "-" * 55)

    for true_dim in [2, 3, 4]:
        for N_ctrl in [300, 500, 1000, 1500]:
            t0 = time.time()
            order = sprinkle_minkowski(N_ctrl, true_dim)
            dim_est, dim_err, mean_f, n_diam, _ = estimate_dimension(
                order, N_ctrl, n_samples=2000)
            print("      {:>6d} {:>6d} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8d}  ({:.1f}s)".format(
                true_dim, N_ctrl, dim_est, dim_err, mean_f, n_diam, time.time() - t0))

    # ============================================================
    # PART 2: MONOSTRING CAUSAL SETS — Multiple rules
    # ============================================================
    print("\n[2/4] MONOSTRING: Three causal rules")

    N_mono = 1500  # Larger than v1

    rules_params = [
        ('resonance', {'density_target': 0.03}),
        ('resonance', {'density_target': 0.05}),
        ('resonance', {'density_target': 0.10}),
        ('resonance', {'density_target': 0.15}),
        ('smooth', {}),
        ('directional', {}),
    ]

    print("      {:>25s} | {:>6s} {:>6s} {:>8s} {:>8s} {:>8s}".format(
        "Rule", "Est D", "±err", "f", "#diam", "density"))
    print("      " + "-" * 65)

    mono_results = []

    for rule, params in rules_params:
        t0 = time.time()

        order, phases, density = monostring_causal_set(
            N_mono, n_phases=6, rule=rule, **params)

        # Enforce transitivity for small N
        if N_mono <= 2000:
            order = enforce_transitivity(order, N_mono, max_iterations=2)
            density = np.sum(order) / max(N_mono * (N_mono - 1) // 2, 1)

        dim_est, dim_err, mean_f, n_diam, sizes = estimate_dimension(
            order, N_mono, n_samples=2000, min_diamond_size=3)

        label = "{}(d={})".format(rule, params.get('density_target', '-'))
        mono_results.append({
            'label': label, 'dim': dim_est, 'err': dim_err,
            'f': mean_f, 'n_diam': n_diam, 'density': density,
            'rule': rule, 'params': params
        })

        print("      {:>25s} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8d} {:>8.4f}  ({:.1f}s)".format(
            label, dim_est, dim_err, mean_f, n_diam, density, time.time() - t0))

    # ============================================================
    # PART 3: REPRODUCIBILITY — Multiple runs for best rule
    # ============================================================
    print("\n[3/4] REPRODUCIBILITY: 5 runs of best rule")

    # Find best rule (closest to D=4)
    valid_results = [r for r in mono_results if r['dim'] > 0.5]
    if valid_results:
        best = min(valid_results, key=lambda r: abs(r['dim'] - 4.0))
        best_rule = best['rule']
        best_params = best['params']
        print("      Best rule: {} (D={:.2f})".format(best['label'], best['dim']))

        dims_multi = []
        for run in range(5):
            np.random.seed(run * 42 + 7)
            order, _, density = monostring_causal_set(
                N_mono, n_phases=6, rule=best_rule, **best_params)
            if N_mono <= 2000:
                order = enforce_transitivity(order, N_mono, max_iterations=2)
            dim_est, dim_err, mean_f, n_diam, _ = estimate_dimension(
                order, N_mono, n_samples=2000)
            dims_multi.append(dim_est)
            print("      Run {}: D = {:.3f} ± {:.3f}".format(
                run + 1, dim_est, dim_err))

        if dims_multi:
            print("      Mean: D = {:.3f} ± {:.3f}".format(
                np.mean(dims_multi), np.std(dims_multi)))

    # ============================================================
    # PART 4: PHASE COUNT SCAN — Does D depend on n_phases?
    # ============================================================
    print("\n[4/4] PHASE SCAN: D vs number of internal phases")
    print("      {:>8s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "n_phases", "Est D", "±err", "f", "#diam"))
    print("      " + "-" * 45)

    for n_ph in [2, 3, 4, 5, 6, 8, 10]:
        t0 = time.time()
        order, _, density = monostring_causal_set(
            N_mono, n_phases=n_ph, rule='resonance', density_target=0.05)
        if N_mono <= 2000:
            order = enforce_transitivity(order, N_mono, max_iterations=2)
        dim_est, dim_err, mean_f, n_diam, _ = estimate_dimension(
            order, N_mono, n_samples=2000)
        print("      {:>8d} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8d}  ({:.1f}s)".format(
            n_ph, dim_est, dim_err, mean_f, n_diam, time.time() - t0))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Monostring results vs D=4 target
    ax = axes[0]
    labels = [r['label'] for r in mono_results]
    dims = [r['dim'] for r in mono_results]
    errs = [r['err'] for r in mono_results]
    colors = ['green' if abs(d - 4) < 0.5 else 'orange' if abs(d - 4) < 1
              else 'red' for d in dims]
    ax.barh(range(len(labels)), dims, xerr=errs, color=colors,
            alpha=0.7, edgecolor='black', capsize=3)
    ax.axvline(4, ls='--', color='black', alpha=0.5, label='D=4')
    ax.axvline(2, ls=':', color='gray', alpha=0.5, label='D=2')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Estimated dimension')
    ax.set_title('Monostring Causal Dimension')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Control: estimator accuracy
    ax = axes[1]
    for true_dim, color in [(2, 'blue'), (3, 'green'), (4, 'red')]:
        Ns_ctrl = [300, 500, 1000, 1500]
        dims_ctrl = []
        for Nc in Ns_ctrl:
            order = sprinkle_minkowski(Nc, true_dim)
            d, _, _, _, _ = estimate_dimension(order, Nc, n_samples=1000)
            dims_ctrl.append(d)
        ax.plot(Ns_ctrl, dims_ctrl, 'o-', color=color, lw=2,
                label='True D={}'.format(true_dim))
        ax.axhline(true_dim, ls=':', color=color, alpha=0.3)
    ax.set_xlabel('N (number of points)')
    ax.set_ylabel('Estimated dimension')
    ax.set_title('Estimator Accuracy\n(should converge to true D)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Diamond size distribution for best result
    ax = axes[2]
    if valid_results:
        # Recompute best for sizes
        order, _, _ = monostring_causal_set(
            N_mono, rule=best_rule, **best_params)
        _, _, _, _, sizes = estimate_dimension(order, N_mono, n_samples=2000)
        if sizes:
            ax.hist(sizes, bins=30, color='teal', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Diamond size |J(p,q)|')
            ax.set_ylabel('Count')
            ax.set_title('Causal Diamond Sizes\n(best rule: {})'.format(
                best['label']))
    ax.grid(True, alpha=0.3)

    plt.suptitle('Monostring: Causal Set Approach v2',
                 fontsize=14, fontweight='bold')
    plt.savefig('causal_v2_results.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: causal_v2_results.png")
    plt.close()

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  CAUSAL SET VERDICT v2")
    print("=" * 70)

    if valid_results:
        closest = min(valid_results, key=lambda r: abs(r['dim'] - 4.0))
        print("  Closest to D=4: {} with D = {:.2f} ± {:.2f}".format(
            closest['label'], closest['dim'], closest['err']))

        if abs(closest['dim'] - 4.0) < 0.5:
            print("  *** D ≈ 4 ACHIEVED via causal set! ***")
            print("  This is a NON-TRIVIAL result if:")
            print("  1. It's robust across runs (check Part 3)")
            print("  2. It depends on n_phases (check Part 4)")
            print("  3. It's NOT just a function of density_target")
        elif abs(closest['dim'] - 2.0) < 0.5:
            print("  D ≈ 2 (not 4). The causal structure is too")
            print("  one-dimensional. Need richer causal rules.")
        else:
            print("  D = {:.1f} — neither 2 nor 4.".format(closest['dim']))
            print("  The causal rule needs refinement.")
    else:
        print("  No valid dimension estimates obtained.")
        print("  Causal rules produce too dense or too sparse orders.")

    print("\n  KEY INSIGHT: The causal rule IS the physics.")
    print("  Different rules = different 'laws of nature'.")
    print("  Finding the right rule = finding the right physics.")
    print("=" * 70)


if __name__ == "__main__":
    run_causal_v2()
