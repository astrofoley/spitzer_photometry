# GitHub issues to create (copy-paste)

Use **New issue** on [astrofoley/spitzer_photometry](https://github.com/astrofoley/spitzer_photometry).  
Each block is one issue: **Title** line, then **Body** (GitHub issue editor accepts markdown).

Labels suggestion: `enhancement` for feature requests, `bug` for bugs, `documentation` for docs-only.

---

## FR-1 — Joint IRAC Ch1 + Ch2 native fit

**Title:** `[Feature] Joint Channel 1 and Channel 2 native photometry`

**Body:**
### Summary
The pipeline currently selects a single band from inputs (`src/pipeline_fit.py` / `CHANNEL` in `src/config.py`). For publication we need a defined strategy to include IRAC Ch1 and model cross-band structure (shared scene and/or correlated noise).

### Scope / ideas
- Option A: two sequential solves with shared astrometry and comparable reporting.
- Option B: single joint state vector with per-band PRFs (`src/solver.py` already takes `channel` for PRFs).
- Explicitly define whether cross-band covariance lives in the GP, in the data weighting, or in post-processing.

### Acceptance criteria
- Methods-level design doc + minimal prototype or flag-gated joint solve.
- Tests: at least one small-matrix sanity check.

### References
- `src/config.py`: `CHANNEL`, `PRF_DIR`
- `src/pipeline_fit.py`, `src/solver.py`

---

## FR-2 — Nuclear / core importance weighting / variance model

**Title:** `[Feature] Configurable per-pixel weights or variance inflation near nucleus`

**Body:**
### Summary
Residual structure near the galaxy nucleus may require higher effective weight or an explicit nuclear component rather than only CR guard and `UNMASK_SIGMA_INF_*`.

### Acceptance criteria
- Configurable weight map or annulus-based σ scaling (documented, default preserves current behavior).
- Diagnostics comparing χ² and transient flux bias with/without.

### References
- `CR_BRIGHT_CORE_GUARD_*`, `UNMASK_SIGMA_INF_*`, `NUCLEAR_POINT_*`, `src/native_fit_campaign.py`, `src/diagnostics.py` `write_fit_quality_report`

---

## FR-3 — Transient parameterization: per-epoch vs per-BCD + post-coadd

**Title:** `[Feature] Study tooling: epoch-level transient flux vs per-BCD with inverse-variance coadd`

**Body:**
### Summary
Solver product includes `transient_epoch_fluxes` / per-frame fluxes (`src/solver.py`). We need a repeatable comparison experiment: one amplitude per `epoch_id` vs per exposure, plus documented aggregation and uncertainty propagation for the light curve (`main.py` Table, `plot_lightcurve`).

### Acceptance criteria
- Script or documented driver that runs both modes on the same cutouts and writes comparison CSV + plots.
- Supplement-ready table of differences in peak flux and timing.

### References
- `main.py`, `src/diagnostics.py` `plot_lightcurve`, `src/solver.py` results dict

---

## FR-4 — SR / grid convergence automation

**Title:** `[Feature] SR (SUPERSAMPLE_FACTOR) sweep with standardized metrics`

**Body:**
### Summary
Publication needs evidence that `SUPERSAMPLE_FACTOR` (and `NATIVE_CUTOUT_SIZE`, `MAX_SCENE_PIXELS`) are adequate. Add a small driver that loops SR (or ROI) and writes `fit_quality_*.json` + residual PDFs for comparison.

### Acceptance criteria
- One command + output directory layout; metrics table (χ², nuclear RMSE, runtime).

### References
- `SUPERSAMPLE_FACTOR`, `PRF_GLS_LTWL_FULL_MAX_PIXELS`, `docs/NOMINAL_NATIVE_SCIENCE_RUN.md`, `scripts/step3_sr1_single_bcd_independent.py` (patterns)

---

## FR-5 — Stars vs GP: anti-leakage validation

**Title:** `[Feature] Template-epoch star/GP leakage tests and star priors`

**Body:**
### Summary
Field stars are included from the deep template catalog (`src/pipeline_fit.py`). Need tests that stars do not absorb extended host flux (template frames especially).

### Acceptance criteria
- Checklist or automated metric: template transient flux ~0 with sensible star priors or flux bounds.
- Optional: document max stars / distance threshold.

### References
- `src/diagnostics.py` `plot_gp_vs_stars`, `plot_template_component_stacks`, `src/pipeline_fit.py` star loop

---

## FR-6 — MAP uncertainty propagation (transient)

**Title:** `[Feature] Document and optionally export transient flux/position uncertainties from Hessian`

**Body:**
### Summary
Publication may require better than point estimates. Solver already exposes some `*_err` fields; we should verify they match the implemented linearization and optionally export full covariance blocks for key parameters.

### Acceptance criteria
- Doc section + validation on a tiny synthetic case (opt-in flag).

### References
- `src/solver.py` results keys, `src/fit_metrics.py`

---

## FR-7 — Reproducibility manifest per run

**Title:** `[Feature] Write run manifest (git SHA, config snapshot, file list) next to outputs`

**Body:**
### Summary
Output tree should include `run_manifest.json`: git commit, important `config` values, Python/dependency versions, list of input FITS paths.

### Acceptance criteria
- Written automatically by `main.py` or `pipeline_fit.run_pipeline_fit_core` to `OUTPUT_DIR` / `DIAGNOSTIC_DIR`.

### References
- `main.py`, `src/config.py`

---

## BUG-1 — Astropy WCS `cdelt` ignored when `cd` present (warning spam)

**Title:** `[Bug] Silence or fix repeated RuntimeWarning: cdelt ignored since cd is present (solver geometry)`

**Body:**
### Summary
Runs emit many `RuntimeWarning` from `src/solver.py` when reading WCS. Does not necessarily imply wrong geometry but clutters logs and makes real warnings hard to see.

### Acceptance criteria
- Either use APIs that avoid the warning, or filter once at pipeline start with a comment pointing to Astropy behavior.

### References
- `src/solver.py` (geometry / WCS cache), IRAC BCD headers

---

## BUG-2 — Full test suite duration / CI

**Title:** `[Bug/CI] Pytest suite too slow or hanging; add markers and smoke job`

**Body:**
### Summary
Full `pytest tests/` can exceed practical CI/local wait time. Need `@pytest.mark.slow`, a default fast subset, and optional GitHub Action.

### Acceptance criteria
- Document `pytest -m "not slow"` (or equivalent); CI runs fast subset on PR.

### References
- `tests/`, `pytest.ini`

---

## BUG-3 — Single effective channel in full pipeline

**Title:** `[Bug] Pipeline assumes one IRAC band per run — document prominently or error if mixed`

**Body:**
### Summary
If both Ch1 and Ch2 files appear in `DATA_DIR`, behavior may be ambiguous (first template picks channel string). Safer: explicit filter by `CHANNEL` or fail with clear message.

### Acceptance criteria
- Document in README + optional strict filter in `preprocessing.find_spitzer_files` or categorize step.

### References
- `src/pipeline_fit.py`, `src/config.py` `CHANNEL`

---
