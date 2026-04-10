# Part V: Final Numerical Results

## Central result

D_corr(E6) = 3.021 ± 0.005 (N=1000, 15 seeds, torus metric)
D_corr(T³) = 2.996 ± 0.005 (reference: true dimension = 3.0)
D_corr(T⁴) = 3.930 ± 0.007 (reference: true dimension = 4.0)
D_corr(T⁶) = 5.453 ± 0.008 (reference: true dimension = 6.0)

E6 vs T³: Δ = 0.025, t=3.51, p=0.0015

Interpretation: E6 orbit is quasi-3D (D_corr ≈ 3.0).
All rank-6 algebras: D_corr ∈ [2.89, 3.02].

## d_s/D_corr ratio (k-NN, k=20, N=800)

| Manifold | D_corr | d_s_cal | ratio |
|----------|--------|---------|-------|
| T³       | 3.011  | 4.219   | 1.401 |
| S³       | 2.722  | 3.906   | 1.435 |
| E6       | 3.043  | 4.321   | 1.420 |
| T⁴       | 3.930  | 1.085   | 0.276 |
| T⁶       | 5.464  | 1.075   | 0.197 |

Conclusion: d_s ≈ 4 at k=20 identifies 3D structures.

## E6 uniqueness (k-NN, k=8, N=800, n=10)

| Algebra | d_s_cal | vs null |
|---------|---------|---------|
| E6      | 4.103   | t=92.65, p<0.0001 |
| B6      | 1.070   | t=2.46,  p=0.021  |
| D6      | 1.602   | t=3.74,  p=0.002  |
| A6      | 2.722   | t=4.45,  p=0.0004 |
| Null    | 0.930   | (reference)        |

E6 is unique by d_s at k=8, but d_s identifies it as 3D,
not 4D (see d_s/D_corr ratio above).

## Benchmark validation (Part V v3)

| Graph       | Expected | Measured | Error |
|-------------|----------|----------|-------|
| Path(300)   | 1.0      | 1.033    | 3.3%  |
| Cycle(200)  | 1.0      | 1.052    | 5.2%  |
| 2D Grid     | 2.0      | 1.941    | 3.0%  |
| 3D Grid     | 3.0      | 2.752    | 8.3%  |
| 4D Grid     | 4.0      | 3.616    | 9.6%  |

Calibration factor: 4.0/3.616 = 1.106
WARNING: this factor applies to grid graphs only,
not to k-NN graphs on arbitrary manifolds.

## Dark energy (20 MC × 35 epochs)

| Model      | ⟨d⟩_final | acc    | H      | DE?  |
|------------|-----------|--------|--------|------|
| A (null)   | 6.23      | -0.002 | +0.002 | No   |
| B (const)  | 14.8      | +0.037 | +0.023 | Yes  |
| C (feedbk) | 28.9      | +0.298 | +0.039 | Yes  |

C vs B: t=-0.083, p=0.934 → feedback irrelevant
C vs E (random phases): t=0.122, p=0.904 → E6 irrelevant
