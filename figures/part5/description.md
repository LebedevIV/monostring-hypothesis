figures/part5/
│
├── benchmarks_v3.png
│   # 5-panel figure: d_s(t) curves for Path, Cycle, 2D/3D/4D grid
│   # Demonstrates that the plateau is correctly detected
│   # Source: part5_v3_knn_benchmarks.py
│
├── knn_kscan.png
│   # E6 vs null: calibrated d_s as a function of k
│   # Shows that k*=20 gives d_s_cal≈4 for E6 while null stays below 3.5
│   # Source: part5_v4_knn_kscan.py
│
├── algebra_comparison_boxplot.png
│   # Box plot: E6/B6/D6/A6/Null at k=8
│   # Shows that E6 is unique by d_s
│   # Source: part5_v5_algebra_comparison.py
│
├── dcorr_precision.png
│   # Bar chart with confidence intervals: D_corr for all configurations
│   # Shows E6≈T³≈3, T⁴≈4, T⁶≈5.5
│   # Source: part5_v6_dcorr_precision.py
│
├── manifold_ds_curves.png
│   # 5-panel figure: d_s(t) curves for T³, T⁴, S³, E6, T⁶
│   # Shows where the plateau is located and why T⁴ gives only 1.09
│   # Source: part5_v7_manifold_comparison.py
│
├── dcorr_vs_ds_scatter.png       ← KEY FIGURE
│   # Scatter plot: D_corr on X-axis, d_s_cal on Y-axis
│   # Reference lines: d_s = D_corr and d_s = 1.40×D_corr
│   # E6, T³, and S³ lie on the 1.40×D_corr line
│   # T⁴ and T⁶ are far from both lines
│   # Source: part5_v7_manifold_comparison.py
│
├── algebra_dcorr_comparison.png
│   # D_corr for E6/A6/D6/B6/Uniform/Null
│   # Shows that all rank-6 algebras lie in [2.9, 3.0]
│   # Source: part5_v8_what_is_special.py
│
└── dark_energy_models.png
    # 4-panel figure: ⟨d⟩(t), velocity, acceleration, and H
    # Shows that models B and C produce dark energy, but E6 is irrelevant
    # Source: part5_v1_dark_energy_models.py
