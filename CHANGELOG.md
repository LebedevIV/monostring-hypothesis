# Changelog

## [7.0.0] — 2025-07-XX

### Added
- **Part VII:** Cross-algebra test of τ ∝ h(Coxeter)
  - 9 experimental scripts (v1–v9)
  - Algebras: A6 (h=7), E6 (h=12), E7 (h=18), E8 (h=30)
  - Paper: `paper/monostring_part7_coxeter.md`
  - Figure: `figures/part7/monostring_part7_final.png`

### Falsified
- τ ∝ h(Coxeter): r(T_ord, h) = −0.012, p = 0.988

### Fixed
- NumPy 2.0+ compatibility: np.trapz → np.trapezoid

### Confirmed
- D_corr(E6) ≈ 3-4 (scale-dependent, both values valid)
- Initial clustering decay τ ≈ O(100) steps (real effect)
- τ/h ≈ 20 for E6 is coincidental

## [6.0.0] — 2025-06-XX
- Part VI: Fragmentation & Memory Time
- Discovery: τ_E6 ≈ 237 steps (p < 0.0001)
- Discovery: D_corr depends on unordered set of frequencies

## [5.0.0] — 2025-05-XX  
- Part V: D_corr(E6) = 3.021 ± 0.005
- d_s ≈ 4 identified as 3D k-NN artifact

## [1.0.0–4.0.0] — Parts I–IV
- See paper/monostring_paper_en.md
