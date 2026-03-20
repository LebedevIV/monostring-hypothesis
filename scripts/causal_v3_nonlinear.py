import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
import time
import warnings
warnings.filterwarnings("ignore")

C_E6 = np.array([
    [ 2,-1, 0, 0, 0, 0],[-1, 2,-1, 0, 0, 0],[ 0,-1, 2,-1, 0,-1],
    [ 0, 0,-1, 2,-1, 0],[ 0, 0, 0,-1, 2, 0],[ 0, 0,-1, 0, 0, 2]
], dtype=np.float64)

# ================================================================
# DIMENSION ESTIMATOR
# ================================================================
def f_to_dim(f_val):
    if f_val <= 0 or f_val >= 1:
        return 0.0
    dims = np.arange(1.0, 10.01, 0.05)
    f_th = np.array([gamma_func(d+1) * gamma_func(d/2) / (4*gamma_func(3*d/2))
                      for d in dims])
    if f_val > f_th[0]: return dims[0]
    if f_val < f_th[-1]: return dims[-1]
    idx = np.searchsorted(-f_th, -f_val)
    if idx == 0: return dims[0]
    if idx >= len(dims): return dims[-1]
    t = (f_val - f_th[idx]) / (f_th[idx-1] - f_th[idx])
    return dims[idx] + t * (dims[idx-1] - dims[idx])

def estimate_dimension(order, N, n_samples=2000, min_size=3):
    future = {i: set(np.where(order[i])[0]) for i in range(N)}
    past = {i: set(np.where(order[:, i])[0]) for i in range(N)}

    fracs = []
    attempts = 0
    while len(fracs) < n_samples and attempts < n_samples * 30:
        attempts += 1
        p = np.random.randint(0, N)
        if not future[p]: continue
        q = np.random.choice(list(future[p]))
        J = list(future[p] & past[q])
        if len(J) < min_size: continue

        n_ord = 0
        n_pairs = 0
        for i in range(len(J)):
            for j in range(i+1, len(J)):
                n_pairs += 1
                if order[J[i], J[j]] or order[J[j], J[i]]:
                    n_ord += 1
        if n_pairs > 0:
            fracs.append(n_ord / n_pairs)

    if not fracs:
        return 0, 0, 0, 0
    mean_f = np.mean(fracs)
    sem_f = np.std(fracs) / np.sqrt(len(fracs))
    d = f_to_dim(mean_f)
    d_err = abs(f_to_dim(mean_f - sem_f) - f_to_dim(mean_f + sem_f)) / 2
    return d, d_err, mean_f, len(fracs)

# ================================================================
# CONTROL: MINKOWSKI SPRINKLE
# ================================================================
def sprinkle(N, dim):
    pts = np.random.uniform(0, 1, (N, dim))
    t = pts[:, 0]
    x = pts[:, 1:] if dim > 1 else np.zeros((N, 1))
    idx = np.argsort(t)
    t = t[idx]; x = x[idx]
    order = np.zeros((N, N), dtype=bool)
    for i in range(N):
        dt = t[i+1:] - t[i]
        dx = np.sqrt(np.sum((x[i+1:] - x[i])**2, axis=1))
        causal = dx < dt
        order[i, i+1:] = causal
    return order

# ================================================================
# MONOSTRING TRAJECTORY GENERATORS
# ================================================================
def linear_trajectory(N, n_phases=6):
    """Original linear winding — produces D ≈ 1 (known failure)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])
    phases = np.zeros((N, n_phases))
    for n in range(N):
        phases[n] = (n * omega) % (2 * np.pi)
    return phases

def e6_chaotic_trajectory(N, kappa=0.5, n_phases=6):
    """E6-coupled standard map — chaotic, breaks linear structure."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])

    # Use E6 Cartan for 6 phases, or truncate/extend for other sizes
    if n_phases == 6:
        C = C_E6.copy()
    elif n_phases < 6:
        C = C_E6[:n_phases, :n_phases].copy()
    else:
        C = np.zeros((n_phases, n_phases))
        C[:6, :6] = C_E6
        for i in range(6, n_phases):
            C[i, i] = 2
            if i > 0: C[i, i-1] = -1; C[i-1, i] = -1

    phases = np.zeros((N, n_phases))
    phases[0] = np.random.uniform(0, 2*np.pi, n_phases)
    for n in range(N - 1):
        phases[n+1] = (phases[n] + omega + kappa * C @ np.sin(phases[n])) % (2*np.pi)
    return phases

def e6_stochastic_trajectory(N, kappa=0.5, temperature=0.1, n_phases=6):
    """E6-coupled with thermal noise — stochastic quantization approach."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])

    if n_phases == 6:
        C = C_E6.copy()
    elif n_phases < 6:
        C = C_E6[:n_phases, :n_phases].copy()
    else:
        C = np.zeros((n_phases, n_phases))
        C[:6, :6] = C_E6
        for i in range(6, n_phases):
            C[i, i] = 2
            if i > 0: C[i, i-1] = -1; C[i-1, i] = -1

    phases = np.zeros((N, n_phases))
    phases[0] = np.random.uniform(0, 2*np.pi, n_phases)
    for n in range(N - 1):
        noise = np.random.normal(0, temperature, n_phases)
        phases[n+1] = (phases[n] + omega + kappa * C @ np.sin(phases[n]) + noise) % (2*np.pi)
    return phases

# ================================================================
# CAUSAL ORDER FROM PHASES (multiple rules)
# ================================================================
def build_causal_order(phases, rule='resonance', density_target=0.05,
                        max_connections_per_node=None):
    """
    Build causal partial order from phase trajectory.

    Key fix from v2: LIMIT connections per node to prevent
    density explosion from transitivity.
    """
    N, D = phases.shape

    if max_connections_per_node is None:
        max_connections_per_node = max(10, int(N * density_target * 2))

    # Compute all pairwise phase distances (for calibration)
    # Sample-based for large N
    sample_dists = []
    for _ in range(min(100000, N * 50)):
        i = np.random.randint(0, N - 1)
        j = np.random.randint(i + 1, N)
        diff = np.abs(phases[j] - phases[i])
        diff = np.minimum(diff, 2*np.pi - diff)
        sample_dists.append(np.sqrt(np.sum(diff**2)))

    threshold = np.percentile(sample_dists, density_target * 100)

    order = np.zeros((N, N), dtype=bool)

    if rule == 'resonance':
        # m ≺ n if phase distance < threshold
        for i in range(N):
            connections = 0
            diff_all = np.abs(phases[i+1:] - phases[i])
            diff_all = np.minimum(diff_all, 2*np.pi - diff_all)
            dists = np.sqrt(np.sum(diff_all**2, axis=1))
            close_indices = np.where(dists < threshold)[0]

            # Limit connections to prevent saturation
            if len(close_indices) > max_connections_per_node:
                close_indices = np.random.choice(
                    close_indices, max_connections_per_node, replace=False)

            for j_offset in close_indices:
                order[i, i + 1 + j_offset] = True

    elif rule == 'light_cone':
        # m ≺ n if phases are within a "cone":
        # temporal distance = |n - m|
        # phase distance = d(φ_m, φ_n)
        # causal if phase_dist < c * temporal_dist
        # where c is a "speed of light" parameter
        c_light = threshold / np.sqrt(N)  # Auto-scale

        for i in range(N):
            connections = 0
            for j in range(i + 1, N):
                if connections >= max_connections_per_node:
                    break
                dt = j - i
                diff = np.abs(phases[j] - phases[i])
                diff = np.minimum(diff, 2*np.pi - diff)
                dx = np.sqrt(np.sum(diff**2))
                if dx < c_light * dt:
                    order[i, j] = True
                    connections += 1

    elif rule == 'mixed':
        # Combine resonance (short-range) and light-cone (long-range)
        # Short range: direct phase match (like original SBE)
        # Long range: smooth evolution (like light cone)
        c_light = 0.5

        for i in range(N):
            connections = 0
            diff_all = np.abs(phases[i+1:] - phases[i])
            diff_all = np.minimum(diff_all, 2*np.pi - diff_all)
            dists = np.sqrt(np.sum(diff_all**2, axis=1))

            for j_offset in range(len(dists)):
                if connections >= max_connections_per_node:
                    break
                j = i + 1 + j_offset
                dt = j - i
                dx = dists[j_offset]

                # Resonance (short range) OR light cone (long range)
                if dx < threshold or dx < c_light * np.sqrt(dt):
                    order[i, j] = True
                    connections += 1

    actual_density = np.sum(order) / max(N * (N-1) // 2, 1)
    return order, actual_density

# ================================================================
# LIGHTWEIGHT TRANSITIVITY (only extend, don't saturate)
# ================================================================
def partial_transitivity(order, N, max_new_per_node=20):
    """
    Add transitive links, but limit how many new links per node.
    This prevents density explosion while maintaining partial transitivity.
    """
    for i in range(N):
        future_i = np.where(order[i])[0]
        new_count = 0
        for j in future_i:
            if new_count >= max_new_per_node:
                break
            future_j = np.where(order[j])[0]
            for k in future_j:
                if not order[i, k]:
                    order[i, k] = True
                    new_count += 1
                    if new_count >= max_new_per_node:
                        break
    return order

# ================================================================
# MAIN
# ================================================================
def run_causal_v3():
    print("=" * 70)
    print("  CAUSAL SET v3: NONLINEAR (CHAOTIC) TRAJECTORIES")
    print("  Fix: E6 coupling breaks linear chain structure")
    print("  Fix: Limited transitivity prevents saturation")
    print("=" * 70)

    N = 1000

    # ============================================================
    # PART 1: CONTROL
    # ============================================================
    print("\n[1/5] CONTROL: Minkowski sprinkle")
    print("      {:>6s} {:>5s} | {:>6s} {:>6s} {:>8s} {:>6s}".format(
        "TrueD", "N", "EstD", "±err", "f", "#dia"))
    print("      " + "-" * 50)

    for dim in [2, 3, 4]:
        t0 = time.time()
        order = sprinkle(N, dim)
        d, de, mf, nd = estimate_dimension(order, N)
        print("      {:>6d} {:>5d} | {:>6.2f} {:>6.2f} {:>8.4f} {:>6d}  ({:.1f}s)".format(
            dim, N, d, de, mf, nd, time.time()-t0))

    # ============================================================
    # PART 2: LINEAR vs CHAOTIC vs STOCHASTIC
    # ============================================================
    print("\n[2/5] TRAJECTORY COMPARISON (resonance rule)")
    print("      {:>25s} | {:>6s} {:>6s} {:>8s} {:>6s} {:>8s}".format(
        "Trajectory", "EstD", "±err", "f", "#dia", "density"))
    print("      " + "-" * 65)

    trajectories = {
        'Linear (original)': lambda: linear_trajectory(N),
        'E6 chaotic (k=0.3)': lambda: e6_chaotic_trajectory(N, kappa=0.3),
        'E6 chaotic (k=0.5)': lambda: e6_chaotic_trajectory(N, kappa=0.5),
        'E6 chaotic (k=1.0)': lambda: e6_chaotic_trajectory(N, kappa=1.0),
        'E6 stochastic (T=0.05)': lambda: e6_stochastic_trajectory(N, kappa=0.5, temperature=0.05),
        'E6 stochastic (T=0.1)': lambda: e6_stochastic_trajectory(N, kappa=0.5, temperature=0.1),
        'E6 stochastic (T=0.5)': lambda: e6_stochastic_trajectory(N, kappa=0.5, temperature=0.5),
        'Pure random': lambda: np.random.uniform(0, 2*np.pi, (N, 6)),
    }

    traj_results = {}

    for name, gen_func in trajectories.items():
        t0 = time.time()
        phases = gen_func()

        order, density = build_causal_order(
            phases, rule='resonance', density_target=0.05,
            max_connections_per_node=30)

        # Light transitivity
        order = partial_transitivity(order, N, max_new_per_node=10)
        density = np.sum(order) / max(N*(N-1)//2, 1)

        d, de, mf, nd = estimate_dimension(order, N, n_samples=2000)

        traj_results[name] = {'dim': d, 'err': de, 'f': mf,
                               'n_dia': nd, 'density': density}

        print("      {:>25s} | {:>6.2f} {:>6.2f} {:>8.4f} {:>6d} {:>8.4f}  ({:.1f}s)".format(
            name, d, de, mf, nd, density, time.time()-t0))

    # ============================================================
    # PART 3: RULE COMPARISON (for best trajectory)
    # ============================================================
    print("\n[3/5] CAUSAL RULE COMPARISON (E6 chaotic k=0.5)")
    print("      {:>15s} {:>8s} | {:>6s} {:>6s} {:>8s} {:>6s} {:>8s}".format(
        "Rule", "density_t", "EstD", "±err", "f", "#dia", "density"))
    print("      " + "-" * 65)

    phases_chaotic = e6_chaotic_trajectory(N, kappa=0.5)

    rule_configs = [
        ('resonance', 0.02),
        ('resonance', 0.05),
        ('resonance', 0.10),
        ('light_cone', 0.05),
        ('light_cone', 0.10),
        ('mixed', 0.05),
    ]

    rule_results = []

    for rule, dt in rule_configs:
        t0 = time.time()
        order, density = build_causal_order(
            phases_chaotic, rule=rule, density_target=dt,
            max_connections_per_node=30)
        order = partial_transitivity(order, N, max_new_per_node=10)
        density = np.sum(order) / max(N*(N-1)//2, 1)

        d, de, mf, nd = estimate_dimension(order, N, n_samples=2000)

        rule_results.append({
            'rule': rule, 'dt': dt, 'dim': d, 'err': de,
            'f': mf, 'n_dia': nd, 'density': density
        })

        print("      {:>15s} {:>8.3f} | {:>6.2f} {:>6.2f} {:>8.4f} {:>6d} {:>8.4f}  ({:.1f}s)".format(
            rule, dt, d, de, mf, nd, density, time.time()-t0))

    # ============================================================
    # PART 4: KAPPA SCAN (chaotic coupling strength)
    # ============================================================
    print("\n[4/5] KAPPA SCAN: D vs coupling strength")
    print("      {:>8s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "kappa", "EstD", "±err", "f", "density"))
    print("      " + "-" * 45)

    kappa_results = []
    for kappa in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        t0 = time.time()
        if kappa == 0:
            phases = linear_trajectory(N)
        else:
            phases = e6_chaotic_trajectory(N, kappa=kappa)

        order, density = build_causal_order(
            phases, rule='resonance', density_target=0.05,
            max_connections_per_node=30)
        order = partial_transitivity(order, N, max_new_per_node=10)
        density = np.sum(order) / max(N*(N-1)//2, 1)

        d, de, mf, nd = estimate_dimension(order, N, n_samples=2000)
        kappa_results.append({'kappa': kappa, 'dim': d, 'err': de,
                               'density': density})

        print("      {:>8.2f} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8.4f}  ({:.1f}s)".format(
            kappa, d, de, mf, density, time.time()-t0))

    # ============================================================
    # PART 5: N_PHASES SCAN (fixed, with extended primes)
    # ============================================================
    print("\n[5/5] PHASE SCAN: D vs number of phases")
    print("      {:>8s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "n_phases", "EstD", "±err", "f", "density"))
    print("      " + "-" * 45)

    for n_ph in [2, 3, 4, 5, 6, 8]:
        t0 = time.time()
        phases = e6_chaotic_trajectory(N, kappa=0.5, n_phases=n_ph)

        order, density = build_causal_order(
            phases, rule='resonance', density_target=0.05,
            max_connections_per_node=30)
        order = partial_transitivity(order, N, max_new_per_node=10)
        density = np.sum(order) / max(N*(N-1)//2, 1)

        d, de, mf, nd = estimate_dimension(order, N, n_samples=2000)

        print("      {:>8d} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8.4f}  ({:.1f}s)".format(
            n_ph, d, de, mf, density, time.time()-t0))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Trajectory comparison
    ax = axes[0]
    names = list(traj_results.keys())
    dims = [traj_results[n]['dim'] for n in names]
    errs = [traj_results[n]['err'] for n in names]
    colors = ['green' if abs(d-4) < 0.5 else 'orange' if abs(d-4) < 1.5
              else 'red' for d in dims]
    ax.barh(range(len(names)), dims, xerr=errs, color=colors,
            alpha=0.7, edgecolor='black', capsize=3)
    ax.axvline(4, ls='--', color='black', alpha=0.5, label='D=4')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Estimated D')
    ax.set_title('Trajectory type → Causal dimension')
    ax.legend(); ax.grid(True, alpha=0.3, axis='x')

    # 2. Kappa scan
    ax = axes[1]
    ks = [r['kappa'] for r in kappa_results]
    ds = [r['dim'] for r in kappa_results]
    es = [r['err'] for r in kappa_results]
    ax.errorbar(ks, ds, yerr=es, fmt='o-', color='navy', lw=2, capsize=3)
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='D=4')
    ax.axhline(1, ls=':', color='gray', alpha=0.5, label='D=1')
    ax.set_xlabel('Coupling kappa')
    ax.set_ylabel('Estimated D')
    ax.set_title('Causal dimension vs E6 coupling')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Rule comparison
    ax = axes[2]
    labels_r = ["{} d={}".format(r['rule'][:5], r['dt']) for r in rule_results]
    dims_r = [r['dim'] for r in rule_results]
    errs_r = [r['err'] for r in rule_results]
    ax.barh(range(len(labels_r)), dims_r, xerr=errs_r,
            color='teal', alpha=0.7, edgecolor='black', capsize=3)
    ax.axvline(4, ls='--', color='black', alpha=0.5)
    ax.set_yticks(range(len(labels_r)))
    ax.set_yticklabels(labels_r, fontsize=8)
    ax.set_xlabel('Estimated D')
    ax.set_title('Causal rule → Dimension')
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Causal Set v3: Nonlinear Trajectories',
                 fontsize=14, fontweight='bold')
    plt.savefig('causal_v3_results.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: causal_v3_results.png")
    plt.close()

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  CAUSAL SET v3 VERDICT")
    print("=" * 70)

    best_traj = max(traj_results.items(), key=lambda x: x[1]['dim'])
    best_rule = max(rule_results, key=lambda x: x['dim'])
    best_kappa = max(kappa_results, key=lambda x: x['dim'])

    print("  Best trajectory: {} (D = {:.2f})".format(
        best_traj[0], best_traj[1]['dim']))
    print("  Best rule: {} d={} (D = {:.2f})".format(
        best_rule['rule'], best_rule['dt'], best_rule['dim']))
    print("  Best kappa: {} (D = {:.2f})".format(
        best_kappa['kappa'], best_kappa['dim']))

    max_D = max(best_traj[1]['dim'], best_rule['dim'], best_kappa['dim'])

    if max_D > 3.5:
        print("\n  *** D ≈ 4 ACHIEVED ***")
    elif max_D > 2.5:
        print("\n  D ≈ 3 — progress but not yet 4D")
    elif max_D > 1.5:
        print("\n  D ≈ 2 — some multidimensionality emerging")
    else:
        print("\n  D ≈ 1 — still essentially one-dimensional")
        print("  The causal rule needs further refinement")

    print("\n  Key question: does chaotic E6 dynamics break")
    print("  the 1D chain structure and produce D > 2?")
    print("=" * 70)

if __name__ == "__main__":
    run_causal_v3()
