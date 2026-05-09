# Publication readiness (native BCD fit)

Use this as a **living checklist** for publication-quality photometry and modeling. Checking a box means the study is **documented** (methods text + numbers or a supplement appendix), not only that code exists.

**Convention:** Config keys live in [`src/config.py`](../src/config.py). Nominal runtime overrides may appear in [`main.py`](../main.py) (`nominal_overrides`); default solver behavior remains whatever is in `config.py` until you change it for a study.

---

## A. Scene model / GP

- [ ] **Dense GP prior validated** on real cutouts: stable hyperparameters (`GP_OPTIMIZE_HYPERPARAMS`, `INIT_LENGTH_SCALE`, `INIT_VARIANCE`), kernel order (`GP_MATERN_ORDER`), and behavior when `n_scene` forces diagonal fallback (`MAX_SCENE_PIXELS`; see [`src/gp_model.py`](../src/gp_model.py), [`src/solver.py`](../src/solver.py)).
- [ ] **GP vs independent-pixel baseline** compared (`USE_SCENE_GP_PRIOR`, `SCENE_INDEPENDENT_RIDGE`) with matched metrics (χ², nuclear annulus, template residuals). Nominal doc: [`docs/NOMINAL_NATIVE_SCIENCE_RUN.md`](NOMINAL_NATIVE_SCIENCE_RUN.md).
- [ ] **Central monotonic / host degeneracy tests** if using GP + optional host/nucleus (`ENFORCE_GP_CENTRAL_MONOTONICITY`, `GP_CENTRAL_MONOTONIC_*`, `USE_HOST_GAUSSIAN_CORE`, `HOST_GAUSSIAN_MIN_OFFSET_PX`, `USE_NUCLEAR_POINT_SOURCE`, `NUCLEAR_POINT_*`).
- [ ] **Campaign-tier workflow** exercised where relevant ([`scripts/iterative_native_fit.py`](../scripts/iterative_native_fit.py), [`src/native_fit_campaign.py`](../src/native_fit_campaign.py) `run_gp_tier_sequence` / `GP_TIER_GATE_*`).

## B. Nuclear / crowded core residuals & weights

- [ ] **Systematic treatment of bright core** vs cosmic-ray heuristic (`CR_BRIGHT_CORE_GUARD_PERCENTILE`, `CR_BRIGHT_CORE_GUARD_DILATION`, `CR_BRIGHT_CORE_GUARD_RADIUS_PX`, `CR_BRIGHT_CORE_GUARD_CENTER`; [`apply_native_cutout_cr_mask`](../src/native_fit_campaign.py)).
- [ ] **Pre-flagged bad pixels** (`sigma=inf`) policy (`UNMASK_SIGMA_INF_RADIUS_PX`, `UNMASK_SIGMA_INF_CENTER`) justified or replaced with instrument masks.
- [ ] **Importance weighting or extra variance** in the nucleus (beyond uniform pixel σ): hypothesis tested; if ad hoc, uncertainty on transient flux inflated accordingly.
- [ ] **QA annulus metrics** interpreted for “acceptably small disk residuals” (`GALAXY_*`, `SUPERRES_QA_*` in config; [`src/diagnostics.py`](../src/diagnostics.py) `write_fit_quality_report`).

## C. Transient: per epoch vs per BCD, then combine

- [ ] **Current solver behavior documented**: per-science-BCD transient fluxes + templates fixed at zero vs alternative parameterization; see results keys in solver output (`transient_epoch_fluxes`, `science_epoch_ids` in [`src/solver.py`](../src/solver.py)).
- [ ] **Controlled comparison**: one flux per `epoch_id` vs per exposure with **documented** inverse-variance (or full-covariance) epoch combination; light-curve CSV/plot conventions ([`main.py`](../main.py), [`src/diagnostics.py`](../src/diagnostics.py) `plot_lightcurve`).
- [ ] **Position**: floated shared transient position (`FLOAT_TRANSIENT_POSITION`, `TRANSIENT_POS_FD_STEP_ARCSEC`, `TRANSIENT_POS_RIDGE`) sensitivity summarized.

## D. Super-resolution (SR) adequacy

- [ ] **SR sweep** at fixed ROI (`SUPERSAMPLE_FACTOR`, `SCENE_WCS_STRICT_SUPERRES`, `MAX_SCENE_PIXELS`, `NATIVE_SCENE_PAD_PX`): convergence of χ², residual maps, and transient flux/position.
- [ ] **Full vs approximate scene Hessian** justified for the chosen grid (`PRF_GLS_LTWL_FULL_MAX_PIXELS`, `PRF_GLS_LTWL_DIAG_MAX_PIXELS`) — trade memory/time vs bias (dipoles, lattice artifacts).
- [ ] **Cutout extent** (`ANALYSIS_BOX_SIZE`, `NATIVE_CUTOUT_SIZE`) vs SR: scene grid size and edge effects (`PRF_CHI2_EXTRA_EDGE_EXCLUSION_PX` if used).

## E. Stars vs extended (GP) scene

- [ ] **Star list policy**: template catalog stars passed to joint solve ([`src/pipeline_fit.py`](../src/pipeline_fit.py)); distance cut / max stars documented.
- [ ] **Degeneracy tests**: template-only epochs — stars should not absorb static galaxy light; stress-test flux priors / fixing faint stars.
- [ ] **Diagnostics used**: [`plot_gp_vs_stars`](../src/diagnostics.py), template component stacks, PRF vs residual orientation plots.

## F. Channel 1 + cross-channel covariance

- [ ] **`CHANNEL` / pipeline** — Today the pipeline picks a band from data ([`src/pipeline_fit.py`](../src/pipeline_fit.py)); joint Ch1+Ch2 is **not** a single toggled config — requires design (see GitHub issues).
- [ ] **`PRF_DIR` PRFs** — Both band PRFs available and referenced per frame (`channel` in [`src/solver.py`](../src/solver.py) PRF loaders).
- [ ] **Cross-band covariance** — Specification: shared latent scene, block-diagonal per band, or correlated noise; publication methods must match implementation.

## G. Calibration & PRF forward model

- [ ] **MJy/sr → Jy / native pixel** correctness and independence of SR (`MJY_SR_TO_JY`, `PIXEL_SCALE` in [`src/config.py`](../src/config.py)).
- [ ] **P-map / intrapixel gain** policy for extended vs point sources (`PRF_APPLY_PMAP_GAIN`, `PMAP_DIR`).
- [ ] **PRF operator choices** documented (`PRF_ORDER_PROJECT_THEN_CONVOLVE`, `PRF_SPATIAL_ANCHORS_PER_AXIS`, `PRF_OPERATOR_MODE`, `PRF_OPERATOR_EXACT_MAX_PIXELS`, `PRF_APODIZATION_EDGE`, `PRF_APPLY_APODIZATION`).
- [ ] **External check** (subset): comparison to SSC aperture photometry or independent pipeline.

## H. Data quality, astrometry, reproducibility

- [ ] **Template vs science split** sensitivity (`SPLIT_DATE_MJD`, `EPOCH_WINDOW_DAYS`).
- [ ] **Alignment** statistics and exclusions (`ALIGN_MATCH_RADIUS_DEG`, `ALIGN_MIN_MATCHES`, `ALIGN_OFFSET_CLIP_SIGMA`; logs from [`src/preprocessing.py`](../src/preprocessing.py)).
- [ ] **Artifact masks**: beyond local CR mask — muxstripe, saturation, column pull-down if applicable to your fields.
- [ ] **Reproducibility package**: git commit, `config` snapshot with run, dependency versions, and input file manifest in supplement.
- [ ] **Held-out validation** (epoch or BCD jackknife / prediction test) where feasible.

**Native ↔ scene astrometry:** For each exposure, native detector pixels are mapped to sky using Astropy’s full `WCS` forward transform from the FITS header (including SIP when coefficients are present). The shared scene grid is built as a linear tangent-plane (`RA---TAN` / `DEC--TAN`) system without detector distortion. Runtime warnings that `CDELT` is ignored in favor of `CD` refer to redundant linear keywords, not to dropping SIP.

---

## Quick links

| Topic | Primary files |
|--------|----------------|
| Native cutouts & CR | [`src/pipeline_fit.py`](../src/pipeline_fit.py), [`src/native_fit_campaign.py`](../src/native_fit_campaign.py), [`src/preprocessing.py`](../src/preprocessing.py) |
| Joint solve / PRF / transient | [`src/solver.py`](../src/solver.py) |
| Diagnostics & LC | [`src/diagnostics.py`](../src/diagnostics.py), [`main.py`](../main.py) |
| Staged / ablation scripts | [`scripts/`](../scripts/) (e.g. `iterative_native_fit.py`, `step3_sr1_single_bcd_independent.py`, `n*_*.py`) |

See also: **[`docs/PUBLICATION_FIX_ORDER.md`](PUBLICATION_FIX_ORDER.md)** for a suggested **order to implement and close** related engineering work.
