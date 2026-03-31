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

def f_to_dim(f_val):
    if f_val <= 0 or f_val >= 1:
        return 0.0
    dims = np.arange(1.0, 10.01, 0.05)
    f_th = np.array([gamma_func(d+1)*gamma_func(d/2)/(4*gamma_func(3*d/2))
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
        n_ord = sum(1 for i in range(len(J)) for j in range(i+1, len(J))
                     if order[J[i], J[j]] or order[J[j], J[i]])
        n_pairs = len(J) * (len(J)-1) // 2
        if n_pairs > 0:
            fracs.append(n_ord / n_pairs)
    if not fracs:
        return 0, 0, 0, 0
    mf = np.mean(fracs)
    sf = np.std(fracs) / np.sqrt(len(fracs))
    d = f_to_dim(mf)
    de = abs(f_to_dim(mf-sf) - f_to_dim(mf+sf)) / 2
    return d, de, mf, len(fracs)

def sprinkle(N, dim):
    pts = np.random.uniform(0, 1, (N, dim))
    t = pts[:, 0]
    x = pts[:, 1:] if dim > 1 else np.zeros((N, 1))
    idx = np.argsort(t)
    t, x = t[idx], x[idx]
    order = np.zeros((N, N), dtype=bool)
    for i in range(N):
        dt = t[i+1:] - t[i]
        dx = np.sqrt(np.sum((x[i+1:] - x[i])**2, axis=1))
        order[i, i+1:] = dx < dt
    return order

def e6_trajectory(N, kappa=0.5, n_phases=6, temperature=0.0):
    primes = [2,3,5,7,11,13,17,19,23,29]
    omega = np.array([np.sqrt(p) for p in primes[:n_phases]])
    C = C_E6[:n_phases, :n_phases].copy() if n_phases <= 6 else np.eye(n_phases) * 2
    if n_phases <= 6:
        C = C_E6[:n_phases, :n_phases].copy()

    ph = np.zeros((N, n_phases))
    ph[0] = np.random.uniform(0, 2*np.pi, n_phases)
    for n in range(N-1):
        noise = np.random.normal(0, temperature, n_phases) if temperature > 0 else 0
        ph[n+1] = (ph[n] + omega + kappa * C @ np.sin(ph[n]) + noise) % (2*np.pi)
    return ph

def torus_dist(phi_a, phi_b):
    diff = np.abs(phi_a - phi_b)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.sqrt(np.sum(diff**2))

def build_light_cone_order(phases, c_speed, max_conn=50):
    """
    Light cone rule: m ≺ n if phase_distance(m,n) < c * (n - m)

    c_speed controls the opening angle of the light cone.
    Larger c → more connections → denser order → smaller f → larger D.
    Smaller c → fewer connections → sparser order → larger f → smaller D.
    """
    N = len(phases)
    order = np.zeros((N, N), dtype=bool)

    for i in range(N):
        conn = 0
        for j in range(i+1, N):
            if conn >= max_conn:
                break
            dt = j - i
            dx = torus_dist(phases[j], phases[i])
            if dx < c_speed * dt:
                order[i, j] = True
                conn += 1

    return order

def build_light_cone_sqrt(phases, c_speed, max_conn=50):
    """
    Modified light cone: m ≺ n if phase_distance < c * sqrt(n-m)

    Square root scaling gives different dimensional behavior.
    In Minkowski: dx < c*dt (linear)
    On a torus with quasi-periodic dynamics: dx scales as sqrt(dt)
    (because phases diffuse on the torus)
    """
    N = len(phases)
    order = np.zeros((N, N), dtype=bool)

    for i in range(N):
        conn = 0
        for j in range(i+1, N):
            if conn >= max_conn:
                break
            dt = j - i
            dx = torus_dist(phases[j], phases[i])
            if dx < c_speed * np.sqrt(dt):
                order[i, j] = True
                conn += 1

    return order

def build_mixed_cone(phases, c_short, c_long, crossover=50, max_conn=50):
    """
    Mixed rule:
    - Short range (dt < crossover): resonance-like (phase distance < threshold)
    - Long range (dt >= crossover): light-cone-like (dx < c * dt)

    This combines the original SBE resonance with causal structure.
    """
    N = len(phases)
    order = np.zeros((N, N), dtype=bool)

    for i in range(N):
        conn = 0
        for j in range(i+1, N):
            if conn >= max_conn:
                break
            dt = j - i
            dx = torus_dist(phases[j], phases[i])

            if dt < crossover:
                if dx < c_short:
                    order[i, j] = True
                    conn += 1
            else:
                if dx < c_long * dt:
                    order[i, j] = True
                    conn += 1

    return order

def run_causal_v4():
    print("=" * 70)
    print("  CAUSAL SET v4: LIGHT CONE PARAMETER SCAN")
    print("  Goal: find c_speed that gives D ≈ 4")
    print("=" * 70)

    N = 1000

    # ============================================================
    # PART 1: CONTROL
    # ============================================================
    print("\n[1/5] CONTROL: Minkowski sprinkle")
    for dim in [2, 3, 4, 5]:
        order = sprinkle(N, dim)
        d, de, mf, nd = estimate_dimension(order, N)
        print("      dim={}: D={:.2f}±{:.2f}, f={:.4f}, #dia={}".format(
            dim, d, de, mf, nd))

    # ============================================================
    # PART 2: LIGHT CONE c_speed SCAN
    # ============================================================
    print("\n[2/5] LINEAR LIGHT CONE: c_speed scan")
    print("      Target: find c where f ≈ 0.04 (gives D ≈ 4)")
    print("      {:>10s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "c_speed", "EstD", "±err", "f", "density"))
    print("      " + "-" * 45)

    phases = e6_trajectory(N, kappa=0.5)

    lc_results = []
    for c in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        t0 = time.time()
        order = build_light_cone_order(phases, c, max_conn=50)
        density = np.sum(order) / max(N*(N-1)//2, 1)
        d, de, mf, nd = estimate_dimension(order, N)
        lc_results.append({'c': c, 'dim': d, 'err': de, 'f': mf, 'density': density})
        print("      {:>10.3f} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8.4f}  ({:.1f}s)".format(
            c, d, de, mf, density, time.time()-t0))

    # ============================================================
    # PART 3: SQRT LIGHT CONE
    # ============================================================
    print("\n[3/5] SQRT LIGHT CONE: c * sqrt(dt) scaling")
    print("      {:>10s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "c_speed", "EstD", "±err", "f", "density"))
    print("      " + "-" * 45)

    sqrt_results = []
    for c in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        t0 = time.time()
        order = build_light_cone_sqrt(phases, c, max_conn=50)
        density = np.sum(order) / max(N*(N-1)//2, 1)
        d, de, mf, nd = estimate_dimension(order, N)
        sqrt_results.append({'c': c, 'dim': d, 'err': de, 'f': mf, 'density': density})
        print("      {:>10.2f} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8.4f}  ({:.1f}s)".format(
            c, d, de, mf, density, time.time()-t0))

    # ============================================================
    # PART 4: MIXED CONE (resonance + light cone)
    # ============================================================
    print("\n[4/5] MIXED CONE: short-range resonance + long-range light cone")
    print("      {:>8s} {:>8s} {:>8s} | {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "c_short", "c_long", "cross", "EstD", "±err", "f", "density"))
    print("      " + "-" * 65)

    mixed_results = []
    for c_short in [0.5, 1.0, 2.0]:
        for c_long in [0.02, 0.05, 0.1]:
            for crossover in [20, 50, 100]:
                t0 = time.time()
                order = build_mixed_cone(phases, c_short, c_long, crossover, max_conn=50)
                density = np.sum(order) / max(N*(N-1)//2, 1)
                d, de, mf, nd = estimate_dimension(order, N)
                mixed_results.append({
                    'c_short': c_short, 'c_long': c_long, 'cross': crossover,
                    'dim': d, 'err': de, 'f': mf, 'density': density
                })
                print("      {:>8.2f} {:>8.3f} {:>8d} | {:>6.2f} {:>6.2f} {:>8.4f} {:>8.4f}  ({:.1f}s)".format(
                    c_short, c_long, crossover, d, de, mf, density, time.time()-t0))

    # ============================================================
    # PART 5: TRAJECTORY DEPENDENCE (at best parameters)
    # ============================================================
    # Find best parameters from all results
    all_results = lc_results + sqrt_results + mixed_results
    valid = [r for r in all_results if r['dim'] > 0 and r.get('f', 0) > 0]

    if valid:
        closest_4 = min(valid, key=lambda r: abs(r['dim'] - 4.0))
        print("\n[5/5] BEST RESULT: D = {:.2f} (closest to 4)".format(closest_4['dim']))
        print("      Parameters: {}".format(
            {k: v for k, v in closest_4.items() if k not in ['dim', 'err', 'f', 'density']}))

        # Test this with different trajectories
        print("\n      Trajectory dependence at best parameters:")
        print("      {:>25s} | {:>6s} {:>6s} {:>8s}".format(
            "Trajectory", "EstD", "±err", "f"))
        print("      " + "-" * 50)

        for name, kappa, temp in [
            ("Linear", 0.0, 0.0),
            ("E6 k=0.3", 0.3, 0.0),
            ("E6 k=0.5", 0.5, 0.0),
            ("E6 k=1.0", 1.0, 0.0),
            ("Stochastic T=0.1", 0.5, 0.1),
            ("Pure random", -1, 0.0),
        ]:
            t0 = time.time()
            if kappa < 0:
                ph = np.random.uniform(0, 2*np.pi, (N, 6))
            elif kappa == 0:
                primes = [2,3,5,7,11,13]
                omega = np.array([np.sqrt(p) for p in primes])
                ph = np.zeros((N, 6))
                for n in range(N):
                    ph[n] = (n * omega) % (2*np.pi)
            else:
                ph = e6_trajectory(N, kappa=kappa, temperature=temp)

            # Use the best rule type
            if 'c_short' in closest_4:
                order = build_mixed_cone(ph, closest_4['c_short'],
                                          closest_4['c_long'],
                                          closest_4['cross'], max_conn=50)
            elif 'c' in closest_4:
                # Determine which type
                order = build_light_cone_order(ph, closest_4['c'], max_conn=50)
            else:
                order = build_light_cone_order(ph, 0.1, max_conn=50)

            d, de, mf, nd = estimate_dimension(order, N)
            print("      {:>25s} | {:>6.2f} {:>6.2f} {:>8.4f}  ({:.1f}s)".format(
                name, d, de, mf, time.time()-t0))

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Linear light cone
    ax = axes[0]
    cs = [r['c'] for r in lc_results]
    ds = [r['dim'] for r in lc_results]
    ax.semilogx(cs, ds, 'o-', color='navy', lw=2, markersize=8)
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='D=4')
    ax.axhline(1, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel('c_speed')
    ax.set_ylabel('Estimated D')
    ax.set_title('Linear light cone: dx < c*dt')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Sqrt light cone
    ax = axes[1]
    cs_s = [r['c'] for r in sqrt_results]
    ds_s = [r['dim'] for r in sqrt_results]
    ax.semilogx(cs_s, ds_s, 'o-', color='darkred', lw=2, markersize=8)
    ax.axhline(4, ls='--', color='red', alpha=0.5, label='D=4')
    ax.axhline(1, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel('c_speed')
    ax.set_ylabel('Estimated D')
    ax.set_title('Sqrt light cone: dx < c*sqrt(dt)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. f vs D (all results)
    ax = axes[2]
    all_f = [r.get('f', 0) for r in all_results if r.get('f', 0) > 0]
    all_d = [r['dim'] for r in all_results if r.get('f', 0) > 0]
    ax.scatter(all_f, all_d, c='teal', s=30, alpha=0.6)

    # Theoretical curve
    f_theory = np.linspace(0.001, 0.6, 200)
    d_theory = [f_to_dim(f) for f in f_theory]
    ax.plot(f_theory, d_theory, 'r-', lw=2, label='Theory: f(D)')

    ax.axhline(4, ls='--', color='black', alpha=0.3)
    ax.set_xlabel('Ordering fraction f')
    ax.set_ylabel('Estimated D')
    ax.set_title('All results: f vs D\n(should follow theoretical curve)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Causal Set v4: Light Cone Parameter Scan',
                 fontsize=14, fontweight='bold')
    plt.savefig('causal_v4_results.png', dpi=150, bbox_inches='tight')
    print("\n  Figure saved: causal_v4_results.png")
    plt.close()

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 70)
    print("  CAUSAL SET v4 VERDICT")
    print("=" * 70)

    if valid:
        print("  Closest to D=4: D = {:.2f} ± {:.2f}".format(
            closest_4['dim'], closest_4.get('err', 0)))

        if abs(closest_4['dim'] - 4.0) < 0.5:
            print("  *** D ≈ 4 FOUND ***")
            print("  But is it ROBUST? Check:")
            print("  1. Does D depend on trajectory type?")
            print("  2. Is D = 4 at a SPECIFIC c, or a range?")
            print("  3. Is f on the theoretical curve?")
        else:
            print("  D = 4 not achieved.")
            print("  Range of D found: {:.1f} to {:.1f}".format(
                min(r['dim'] for r in valid),
                max(r['dim'] for r in valid)))

    print("=" * 70)

if __name__ == "__main__":
    run_causal_v4()
