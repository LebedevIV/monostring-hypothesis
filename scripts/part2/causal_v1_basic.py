import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

"""
CAUSAL SET APPROACH TO THE MONOSTRING
=====================================

Instead of phases on a torus, the Monostring generates a
PARTIALLY ORDERED SET (poset) of events.

Key idea: tick m CAUSALLY PRECEDES tick n (m ≺ n) if:
1. m < n (chronological order)
2. Some condition on the "state" of the Monostring at m and n

The dimension of the resulting causal set is determined by
the Myrheim-Meyer estimator, which counts the fraction of
causally related pairs in a causal diamond (Alexandrov interval).

For a d-dimensional Minkowski spacetime:
<f> = d! / 2^d * (ordering fraction)

This gives D=2 for f=0.5, D=4 for f≈0.042, etc.

NO TORUS. NO PHASES. NO SYNCHRONIZATION. NO GRAPH.
Just a partial order on a set of points.
"""


def generate_causal_set_sprinkle(N, dim=4):
    """
    CONTROL: Standard causal set by sprinkling N points
    into a d-dimensional Minkowski spacetime diamond.

    This is the "correct answer" — we should get D ≈ dim.
    """
    # Sprinkle into [0,1]^dim with metric ds² = -dt² + dx²
    points = np.random.uniform(0, 1, (N, dim))

    # Time coordinate = first coordinate
    t = points[:, 0]
    x = points[:, 1:]

    # Causal order: m ≺ n if t_m < t_n AND spatial separation < time separation
    # (inside the light cone)
    order = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i + 1, N):
            dt = t[j] - t[i]
            if dt <= 0:
                continue
            dx = np.sqrt(np.sum((x[j] - x[i])**2))
            if dx < dt:  # Inside light cone
                order[i, j] = True

    return order, points


def generate_monostring_causal_set(N, rule='phase_distance',
                                     n_dims=6, causal_prob=0.1):
    """
    MONOSTRING: Generate a causal set from the Monostring's ticks.

    Each tick has an internal state (6 phases).
    Causal relation: m ≺ n if m < n AND some condition on phases.

    Rules:
    - 'phase_distance': m ≺ n if phases are "close enough"
    - 'phase_gradient': m ≺ n if the phase change is "smooth enough"
    - 'energy_threshold': m ≺ n if total phase energy below threshold
    """
    omega = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(5),
                       np.sqrt(7), np.sqrt(11), np.sqrt(13)])[:n_dims]

    # Generate phases (simple linear winding — no E6, no torus graph)
    phases = np.zeros((N, n_dims))
    for n in range(N):
        phases[n] = (n * omega) % (2 * np.pi)

    # Build causal order
    order = np.zeros((N, N), dtype=bool)

    if rule == 'phase_distance':
        # m ≺ n if phases at m and n are "close" on the torus
        # Threshold chosen to give ~causal_prob fraction of pairs
        threshold = 2.0  # Will be auto-tuned

        # Auto-tune threshold
        sample_dists = []
        for _ in range(10000):
            i, j = np.random.randint(0, N, 2)
            if i >= j:
                continue
            diff = np.abs(phases[j] - phases[i])
            diff = np.minimum(diff, 2*np.pi - diff)
            sample_dists.append(np.sqrt(np.sum(diff**2)))

        threshold = np.percentile(sample_dists, causal_prob * 100)

        # Build order (subsample for speed)
        max_pairs = min(N * 200, N * (N-1) // 2)
        pairs_checked = 0

        for i in range(N):
            for j in range(i + 1, min(i + N//5, N)):  # Look ahead
                diff = np.abs(phases[j] - phases[i])
                diff = np.minimum(diff, 2*np.pi - diff)
                dist = np.sqrt(np.sum(diff**2))
                if dist < threshold:
                    order[i, j] = True
                pairs_checked += 1
                if pairs_checked > max_pairs:
                    break
            if pairs_checked > max_pairs:
                break

    elif rule == 'phase_gradient':
        # m ≺ n if the total phase change is small
        # (smooth evolution = causal, jerky = spacelike)
        for i in range(N):
            for j in range(i + 1, min(i + N//5, N)):
                diff = np.abs(phases[j] - phases[i])
                diff = np.minimum(diff, 2*np.pi - diff)
                grad = np.sqrt(np.sum(diff**2)) / (j - i)
                if grad < 1.0:  # Smooth enough
                    order[i, j] = True

    elif rule == 'energy_threshold':
        # m ≺ n if "energy" at both points is below threshold
        energies = np.sum(np.sin(phases)**2, axis=1)
        median_E = np.median(energies)
        for i in range(N):
            if energies[i] > median_E:
                continue
            for j in range(i + 1, min(i + N//5, N)):
                if energies[j] < median_E:
                    order[i, j] = True

    return order, phases


def myrheim_meyer_dimension(order, N, n_samples=5000):
    """
    Myrheim-Meyer dimension estimator for a causal set.

    For a causal diamond (Alexandrov interval) J(p,q) = {r : p ≺ r ≺ q}:

    The ordering fraction f = (# of ordered pairs in J) / (|J| choose 2)

    relates to dimension via:
    f = Gamma(d+1) * Gamma(d/2) / (4 * Gamma(3d/2))

    Numerically:
    d=2: f = 0.5
    d=3: f ≈ 0.318
    d=4: f ≈ 0.208
    d=5: f ≈ 0.139
    d=6: f ≈ 0.094
    """
    from scipy.special import gamma

    def f_to_dim(f_val):
        """Invert the ordering fraction to get dimension."""
        if f_val <= 0 or f_val >= 1:
            return 0
        # Numerical inversion
        for d_test in np.arange(1.0, 10.0, 0.1):
            f_theory = (gamma(d_test + 1) * gamma(d_test / 2) /
                        (4 * gamma(3 * d_test / 2)))
            if f_theory < f_val:
                # Linear interpolation
                d_prev = d_test - 0.1
                f_prev = (gamma(d_prev + 1) * gamma(d_prev / 2) /
                          (4 * gamma(3 * d_prev / 2)))
                if abs(f_prev - f_theory) > 1e-10:
                    d_interp = d_prev + 0.1 * (f_val - f_prev) / (f_theory - f_prev)
                    return d_interp
                return d_test
        return 10.0

    # Sample causal diamonds
    ordering_fractions = []

    for _ in range(n_samples):
        # Pick two causally related points p ≺ q
        p = np.random.randint(0, N)
        # Find points q such that p ≺ q
        q_candidates = np.where(order[p, :])[0]
        if len(q_candidates) == 0:
            continue
        q = np.random.choice(q_candidates)

        # Find the Alexandrov interval J(p,q) = {r : p ≺ r ≺ q}
        J = []
        for r in range(p + 1, q):
            if order[p, r] and order[r, q]:
                J.append(r)

        n_J = len(J)
        if n_J < 3:
            continue

        # Count ordered pairs within J
        n_ordered = 0
        n_pairs = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                n_pairs += 1
                if order[J[i], J[j]]:
                    n_ordered += 1

        if n_pairs > 0:
            f = n_ordered / n_pairs
            ordering_fractions.append(f)

    if len(ordering_fractions) == 0:
        return 0, 0, []

    mean_f = np.mean(ordering_fractions)
    dim_estimate = f_to_dim(mean_f)

    return dim_estimate, mean_f, ordering_fractions


def run_causal_set_experiment():
    print("=" * 70)
    print("  MONOSTRING: CAUSAL SET APPROACH")
    print("  No torus, no graph, no synchronization")
    print("  Just a partial order → dimension from Myrheim-Meyer")
    print("=" * 70)

    N = 500  # Small N for causal sets (O(N²) memory)

    # ============================================================
    # CONTROL: Known-dimension sprinkle
    # ============================================================
    print("\n[1/3] Control: Minkowski sprinkle (known dimension)")
    print("      {:>8s} | {:>8s} {:>8s} {:>10s}".format(
        "True D", "Est. D", "f", "n_diamonds"))
    print("      " + "-" * 45)

    for true_dim in [2, 3, 4, 5]:
        t0 = time.time()
        order, _ = generate_causal_set_sprinkle(N, dim=true_dim)
        est_dim, mean_f, fracs = myrheim_meyer_dimension(order, N)
        print("      {:>8d} | {:>8.2f} {:>8.4f} {:>10d}  ({:.1f}s)".format(
            true_dim, est_dim, mean_f, len(fracs), time.time() - t0))

    # ============================================================
    # MONOSTRING CAUSAL SETS
    # ============================================================
    print("\n[2/3] Monostring causal sets (different rules)")
    print("      {:>20s} | {:>8s} {:>8s} {:>10s}".format(
        "Rule", "Est. D", "f", "n_diamonds"))
    print("      " + "-" * 55)

    rules = ['phase_distance', 'phase_gradient', 'energy_threshold']
    mono_results = {}

    for rule in rules:
        t0 = time.time()
        order, phases = generate_monostring_causal_set(
            N, rule=rule, n_dims=6, causal_prob=0.1)

        # Count causal relations
        n_causal = np.sum(order)
        n_possible = N * (N - 1) // 2

        est_dim, mean_f, fracs = myrheim_meyer_dimension(order, N)

        mono_results[rule] = {
            'dim': est_dim, 'f': mean_f, 'n_fracs': len(fracs),
            'n_causal': n_causal, 'density': n_causal / max(n_possible, 1)
        }

        print("      {:>20s} | {:>8.2f} {:>8.4f} {:>10d}  "
              "({:.1f}s, density={:.3f})".format(
            rule, est_dim, mean_f, len(fracs),
            time.time() - t0, n_causal / max(n_possible, 1)))

    # ============================================================
    # SCAN: Effect of causal probability
    # ============================================================
    print("\n[3/3] Scan: dimension vs causal density (phase_distance)")
    print("      {:>10s} | {:>8s} {:>8s} {:>10s}".format(
        "causal_p", "Est. D", "f", "n_diamonds"))
    print("      " + "-" * 45)

    for cp in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        t0 = time.time()
        order, _ = generate_monostring_causal_set(
            N, rule='phase_distance', causal_prob=cp)
        est_dim, mean_f, fracs = myrheim_meyer_dimension(order, N)
        print("      {:>10.3f} | {:>8.2f} {:>8.4f} {:>10d}  ({:.1f}s)".format(
            cp, est_dim, mean_f, len(fracs), time.time() - t0))

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  CAUSAL SET VERDICT")
    print("=" * 70)

    print("\n  This is a FIRST EXPLORATION — not a definitive test.")
    print("  Key question: can the Monostring's state evolution")
    print("  define a causal order that yields D = 4?")

    for rule, data in mono_results.items():
        status = "INTERESTING" if 3.5 < data['dim'] < 4.5 else "not D=4"
        print("    {}: D = {:.2f} ({})".format(rule, data['dim'], status))

    print("\n  If any rule gives D ≈ 4 → develop further")
    print("  If all give D ≠ 4 → need different causal rule")
    print("  If D depends on causal_prob → need physical principle to fix it")
    print("=" * 70)


if __name__ == "__main__":
    run_causal_set_experiment()
