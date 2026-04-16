# Part VII Scripts: τ ∝ h(Coxeter) Test

## Final result
**Hypothesis falsified.** No memory metric scales with h.
See `part7_v9_final_metrics.py`.

## Script inventory

| Script | Purpose | Outcome |
|--------|---------|---------|
| `part7_v5_diagnostic_scan.py` | Scan T, n, pct for D_corr | D_corr=4.09 everywhere |
| `part7_v7_reproduce_part6.py` | Find exact Part VI parameters | Validated pipeline |
| `part7_v8_crossover_tau.py` | Test τ_crossover metric | Metric broken (oscillations) |
| `part7_v9_final_metrics.py` | **MAIN: T_ord, f_neg, n_sig vs h** | r=-0.012, falsified |

## Failed attempts (archive/)

| Script | Why it failed |
|--------|--------------|
| `v1_wrong_frequencies.py` | Used raw exponents [1,4,5,7,8,11] instead of 2·sin(π·m/h) |
| `v2_linear_dynamics.py` | Linear dynamics: ΔS(shuffle)=0 by permutation invariance |
| `v3_phi_frequencies.py` | Used φ·m/h — different frequency scale, D_corr wrong |
| `v4_arithmetic_null.py` | Arithmetic null for A6 = identical to Coxeter → ΔS=0 |
| `v6_kappa_chaos.py` | κ=3.0: full chaos, D_corr=1.3, unphysical |

## Running the main experiment

```bash
pip install numpy scipy matplotlib
python scripts/part7/part7_v9_final_metrics.py
