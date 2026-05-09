"""src/config.py"""
import os
import numpy as np

# --- Pipeline Control ---
CHANNEL = 'ch2'

# --- Target ---
TRANSIENT_RA = 197.45037
TRANSIENT_DEC = -23.38148

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PRF_DIR = os.path.join(BASE_DIR, 'data', 'prfs')
PMAP_DIR = os.path.join(BASE_DIR, 'data', 'pmap_fits')
DIAGNOSTIC_DIR = os.path.join(OUTPUT_DIR, 'diagnostics')
SOURCE_CATALOG_PATH = os.path.join(OUTPUT_DIR, 'source_catalog.fits')

# --- Analysis Geometry ---
PIXEL_SCALE = 1.22
STAMP_SIZE_PX = 20
BORDER_PX = 10
ANALYSIS_BOX_SIZE = STAMP_SIZE_PX + 2 * BORDER_PX
PRF_OVERSAMPLE = 100 # Standard IRAC PRF oversampling

# Super Resolution / latent scene grid supersampling.
# The GP latent galaxy scene is represented on this North-up grid; increase this
# if reprojection/rotation artifacts become non-negligible versus noise.
SUPERSAMPLE_FACTOR = 2

# Buffer
SCENE_BUFFER_PX = 5
# Native-fit scene footprint padding (scene-grid pixels) around the union of BCD footprints.
# Keep this minimal to reduce unconstrained border regions that can leak through PRF convolution.
NATIVE_SCENE_PAD_PX = 0
# If True, enforce exact scene pixel scale PIXEL_SCALE/SUPERSAMPLE_FACTOR when building
# native-fit scene grids. If the footprint exceeds max_scene_pixels, raise an error
# instead of silently coarsening the scale.
SCENE_WCS_STRICT_SUPERRES = False
# Scene support threshold on reprojected native-valid mask when deciding if a scene pixel
# is constrained by data. Higher values shrink weakly constrained edge regions.
NATIVE_SCENE_SUPPORT_THRESHOLD = 0.85
# Native CR masker bright-core guard settings (used in native_fit_campaign._cr_mask_local).
# Pixels above this local-background percentile are protected from CR masking.
CR_BRIGHT_CORE_GUARD_PERCENTILE = 99.0
# Additional binary dilation (in pixels) applied to the bright-core guard.
CR_BRIGHT_CORE_GUARD_DILATION = 1
# Explicit transient-centered guard radius (pixels) where CR masking is disabled.
# 0 disables this guard.
CR_BRIGHT_CORE_GUARD_RADIUS_PX = 0.0
# Center used for CR_BRIGHT_CORE_GUARD_RADIUS_PX:
# "transient" => config.TRANSIENT_RA/DEC, "nuclear" => config.NUCLEAR_POINT_RA/DEC
CR_BRIGHT_CORE_GUARD_CENTER = "transient"

SPLIT_DATE_MJD = 58750.0
EPOCH_WINDOW_DAYS = 2.0

# --- Astrometric alignment robustness ---
# Match radius for BCD->deep-template source matching in world coordinates.
# 1e-4 deg ~= 0.36 arcsec (much tighter than prior 0.003 deg).
ALIGN_MATCH_RADIUS_DEG = 1.0e-4
# Minimum number of retained matches for applying a per-frame shift correction.
ALIGN_MIN_MATCHES = 8
# Robust clipping threshold (MAD-based, in sigma units) on per-source shift residuals.
ALIGN_OFFSET_CLIP_SIGMA = 3.5

# --- Flux Conversion (Robust) ---
# Calculate native detector pixel area in steradians manually to avoid UnitErrors.
# IMPORTANT: detector data are in native MJy/sr and this conversion must not depend on
# the scene-model supersampling factor.
pixel_scale_native_deg = PIXEL_SCALE / 3600.0
pixel_area_native_sr = (pixel_scale_native_deg * (np.pi / 180.0))**2
# MJy/sr -> Jy/native-pixel: (1e6 Jy/sr) * (sr/native-pixel)
MJY_SR_TO_JY = 1e6 * pixel_area_native_sr

# --- Solver ---
INIT_LENGTH_SCALE = 2.0
INIT_VARIANCE = 1.0
GAIN = 3.7
READ_NOISE = 15.0
# Shared sky offset for the transient PRF (linearized Newton step; science frames only).
FLOAT_TRANSIENT_POSITION = True
# Finite-difference step in arcseconds for ∂PRF/∂(RA, Dec) on the fixed scene grid.
TRANSIENT_POS_FD_STEP_ARCSEC = 0.05
# Ridge on fitted offset (in 1/deg²) to stabilize when signal is weak.
TRANSIENT_POS_RIDGE = 1e12
# Enforce f >= 0 for science-epoch transient amplitudes (bounded MAP, trust-constr).
TRANSIENT_NONNEGATIVE = True
# Optional fixed-position normalized Gaussian on the analysis stamp (shared flux across frames).
USE_HOST_GAUSSIAN_CORE = False
HOST_CORE_RA = None
HOST_CORE_DEC = None
HOST_CORE_SIGMA_PX = 1.5
# If set to a non-empty tuple/list, fit one nonnegative amplitude per listed sigma (same RA/Dec).
# If None, the solver uses a single Gaussian with HOST_CORE_SIGMA_PX.
HOST_GAUSSIAN_SIGMA_PX_LIST = None
HOST_CORE_NONNEGATIVE = True
# If host Gaussian center is closer than this (scene pixels) to the GP profile center,
# disable the host Gaussian to reduce GP/host-core degeneracy.
HOST_GAUSSIAN_MIN_OFFSET_PX = 1.0
# Optional explicit GP-profile center used for host-offset checks and central monotonic prior.
# If None, defaults to GALAXY_EXTENDED_CENTER_* when available, else TRANSIENT_*.
GP_PROFILE_CENTER_RA = None
GP_PROFILE_CENTER_DEC = None
# Soft central monotonic prior on GP annular means: discourage outer-annulus mean from
# exceeding inner-annulus mean by more than ALLOWED_DROP_JY.
ENFORCE_GP_CENTRAL_MONOTONICITY = True
GP_CENTRAL_MONOTONIC_ALLOWED_DROP_JY = 0.0
# Relative strength of annular monotonic prior vs median scene-diagonal Hessian.
GP_CENTRAL_MONOTONIC_STRENGTH_FRAC = 0.03
# Extra multiplier for one-sided post-solve enforcement on violated annulus pairs.
GP_CENTRAL_MONOTONIC_VIOLATION_BOOST = 8.0
# Number of one-sided enforcement re-solves (typically 1 is enough).
GP_CENTRAL_MONOTONIC_ENFORCEMENT_ITERS = 1
# Annulus edges in scene pixels for the central monotonic prior.
GP_CENTRAL_MONOTONIC_RADII_PX = (0.0, 1.5, 3.0, 4.5, 6.0)
# Optional unresolved nuclear point source (PRF-shaped; shared flux across all BCDs).
# Native campaign: if unset, HOST_CORE_* / GALAXY_EXTENDED_CENTER_* / GP_PROFILE_CENTER_* are used;
# transient coordinates are never used as the nuclear position.
USE_NUCLEAR_POINT_SOURCE = False
# Galaxy nucleus / first stab for optional unresolved nuclear PSF (native campaign & solver).
NUCLEAR_POINT_RA = 197.448762
NUCLEAR_POINT_DEC = -23.383962
NUCLEAR_POINT_NONNEGATIVE = True
# Nuclear point ΔRA/ΔDec subpixel solve (requires solver support for extra state).
FLOAT_NUCLEAR_POINT_POSITION = False
NUCLEAR_POINT_POS_RIDGE = 1e6
# Two-scale GP scene: clamp component amplitudes nonnegative (default True).
GP_COMPONENTS_NONNEGATIVE = True
# Optional sky center for extended galaxy QA (annulus masks in diagnostics). None = disabled.
GALAXY_EXTENDED_CENTER_RA = 197.448762
GALAXY_EXTENDED_CENTER_DEC = -23.383962
# Annulus around extended center for disk-focused metrics (analysis pixels).
GALAXY_QA_ANNULUS_INNER_PX = 4.0
GALAXY_QA_ANNULUS_OUTER_PX = 14.0
# Above this pixel count, the Matérn prior uses a diagonal fallback (scalability / memory).
# Native North-up scenes are often ~100–120 px per side (~1e4–1.5e4 pixels); keep above that
# so cutout/native fits use the same dense GP prior as smaller reprojected stamps.
MAX_SCENE_PIXELS = 25000
# Scene GP stationary kernel: "matern32" (smooth) or "matern12" (exponential / rougher).
GP_MATERN_ORDER = "matern32"
# Precision build: "matern" (default Matérn) or "diagonal" (sparse ε·I, diagnostics / tests).
GP_KERNEL_TYPE = "matern"
GP_DIAGONAL_EPS = 1e-10
# For large scenes that trigger GP fallback, optionally add nearest-neighbor smoothness
# in the precision matrix (Laplacian-style) to avoid unstable pixel-to-pixel artifacts.
# 0.0 keeps strictly diagonal fallback.
GP_FALLBACK_NEIGHBOR_SMOOTHNESS = 0.15
# If True, run GP hyperparameter optimization on template cutouts before solve.
GP_OPTIMIZE_HYPERPARAMS = True
# If False, bypass the GP prior entirely and fit an independent per-scene-pixel model
# with only a tiny diagonal ridge (SCENE_INDEPENDENT_RIDGE) for numerical stability.
USE_SCENE_GP_PRIOR = True
# Diagonal stabilizer used only when USE_SCENE_GP_PRIOR is False.
SCENE_INDEPENDENT_RIDGE = 1e-12
# Native GP tier sequence (--gp-tier-auto): stop after a tier if gate passes.
# Pass if center_reduced_chi2 <= max (when max > 0), OR improved vs tier-A baseline by at least improve.
GP_TIER_GATE_CENTER_CHI2_MAX = 0.0  # 0 or negative = disable absolute cap (use improvement only)
GP_TIER_GATE_IMPROVE_CENTER_CHI2 = 120.0  # require larger Tier-A gain before stopping escalation

# --- PRF forward model (plan: single response, optional anti-alias only) ---
# Edge cosine taper fraction (0–0.5): 0 disables apodization (preserves PRF wings).
PRF_APODIZATION_EDGE = 0.0
# Gaussian sigma in oversampled PRF pixels before reprojection; None or 0 = off (no stacked blur).
PRF_PREBLUR_IN_OVERSAMPLE_PIXELS = None
# Number of anchors per axis for spatially varying full-model PRF convolution on a frame.
# 1 => single PRF across frame, 3 => 3x3 smoothly blended local kernels.
PRF_SPATIAL_ANCHORS_PER_AXIS = 3
# PRF operator mode: "anchor" (fast approximation), "exact" (dense exact operator),
# or "auto" (exact for small scenes, anchor otherwise).
PRF_OPERATOR_MODE = "auto"
# Safety cap for exact PRF operator construction in auto mode.
PRF_OPERATOR_EXACT_MAX_PIXELS = 2500
# Keep SSC PRFs unmodified by default for scientific convolution.
PRF_APPLY_APODIZATION = False
# Intrapixel pmap gain is a point-source photometric correction; keep disabled by
# default for the linear extended-scene operator.
PRF_APPLY_PMAP_GAIN = False
# If True, apply order F = L_native P (project to BCD first, then PRF in BCD frame).
# If False, keep legacy order F = P L_scene.
PRF_ORDER_PROJECT_THEN_CONVOLVE = False
# Joint GLS: exact diag(Lᵀ W L) for the GP block needs O(n_pix) impulse responses per BCD
# (prohibitively slow for typical stamps). If n_scene <= this value, compute that diagonal;
# otherwise add only diag(W) for the data term on the GP block. rhs and L–B cross blocks
# still use L and Lᵀ. Default 0 = never use the exact impulse diagonal (always fast).
PRF_GLS_LTWL_DIAG_MAX_PIXELS = 0
# Joint GLS: build full scene-block L^T W L (including off-diagonals) when
# n_scene <= this cap. This is O(n_scene * n_native) operator applications and
# O(n_scene^2) memory/time in the dense normal matrix, so keep small.
# 0 disables full scene-block coupling (falls back to diagonal-only path above).
PRF_GLS_LTWL_FULL_MAX_PIXELS = 0
# Optional extra native-edge exclusion in chi^2 (scene pixels). This is disabled by default.
# Potential follow-up: set this to a multiple of the PRF support size once calibrated.
PRF_CHI2_EXTRA_EDGE_EXCLUSION_PX = 0

# Cosmetic nearest-neighbor replication for scene panels in native_fit_campaign PDFs only.
# This is not the physics super-resolution; use SUPERSAMPLE_FACTOR for the fitted scene grid.
DIAG_SUPERRES_DISPLAY_FACTOR = 40

# Legacy flag for alternate native stamp layout (see preprocessing). Prefer FIT_ON_NATIVE_PIXELS.
USE_NATIVE_CENTERED_CUTOUT = False
# If True, fit on detector-oriented native BCD cutouts; the joint forward map uses WCS + PRF per frame.
FIT_ON_NATIVE_PIXELS = True
# After `extract_native_analysis_cutouts`, crop each stamp to ``size`` x ``size`` pixels centered on
# ``(TRANSIENT_RA, TRANSIENT_DEC)`` in pixel space. 0 = keep full extraction box from preprocessing.
NATIVE_CUTOUT_SIZE = 0
# Some inputs mark bad pixels with sigma=inf before our CR pass. Optional: replace those in a disk
# around the nuclear or transient position with a local median sigma so bright cores stay in the fit.
UNMASK_SIGMA_INF_RADIUS_PX = 0.0
# "nuclear" (NUCLEAR_POINT_RA/DEC) or "transient" (TRANSIENT_RA/DEC).
UNMASK_SIGMA_INF_CENTER = "nuclear"

# Super-res QA: inner mask radius (analysis-stamp pixels) around target for annulus metric.
SUPERRES_QA_INNER_MASK_PX = 3.0
SUPERRES_QA_OUTER_RADIUS_PX = 12.0
SUPERRES_QA_POISSON_FRACTION_MAX = 0.10

# --- Diagnostics display (stretch / sigma-residual caps) ---
# Fraction of (p98−p2) used as AsinhNorm linear_width for flux-like panels.
DIAGNOSTIC_ASINH_WIDTH_FRAC = 0.12
# Clip |residual/sigma| display at this value (percentile first, then cap).
DIAGNOSTIC_RESID_SIGMA_DISPLAY_CAP = 6.0
# BCD flux panels: robust percentiles for AsinhNorm vmin/vmax (wider = more faint structure).
DIAGNOSTIC_BCD_ROBUST_PERCENTILES = (1.0, 95.0)
# If None, reuse DIAGNOSTIC_ASINH_WIDTH_FRAC for BCD rows.
DIAGNOSTIC_BCD_ASINH_WIDTH_FRAC = None
# Linear imshow vmin/vmax from percentiles of valid pixels (asymmetric clip).
DIAGNOSTIC_IMSHOW_PERCENTILES_LO = 1.0
DIAGNOSTIC_IMSHOW_PERCENTILES_HI = 95.0
# Residual panels (Jy or sigma): symmetric limits around 0 using p_lo/p_hi.
DIAGNOSTIC_RESIDUAL_PERCENTILES_LO = 1.0
DIAGNOSTIC_RESIDUAL_PERCENTILES_HI = 99.0
