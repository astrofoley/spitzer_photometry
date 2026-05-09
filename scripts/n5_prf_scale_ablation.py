#!/usr/bin/env python3
"""N=5, one iteration: PRF / super-resolution scale ablations on real template data.

Two runs (see plan):

1. **diagonal_gp_prf**: GP prior is diagonal (no Matérn coupling; no diagonal fallback neighbor
   smoothing). Full spatially varying PRF + bilinear scene→native projection.
2. **diagonal_gp_no_prf**: Same prior as (1), but PRF convolution **L** is replaced by identity
   on scene and native grids (**F ≈ P** when ``PRF_ORDER_PROJECT_THEN_CONVOLVE`` is False).

**Caveat:** Diagonal **Q** does not make the full normal equations diagonal in scene pixels:
per-BCD backgrounds introduce **H[scene, bg]** cross-terms (see solver ``run_gls_solve``).
That is expected for “GP + sky” fits.

Outputs under ``config.DIAGNOSTIC_DIR / n5_prf_scale_ablation /``.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _temporary_config,
    prepare_real_template_case,
    run_stage,
)
from src.prf_identity_context import identity_prf_operators_context  # noqa: E402

# Overrides applied around each ``run_stage`` (trial-local cfg does not include these).
_DIAG_BASE_CONFIG: Dict[str, Any] = {
    # Force diagonal GP prior (dense Matérn path skipped).
    "MAX_SCENE_PIXELS": 0,
    # Strict diagonal fallback: no Laplacian neighbor coupling on Q.
    "GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0,
    # Avoid annulus inequality constraints coupling GP pixels.
    "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
    "GP_OPTIMIZE_HYPERPARAMS": False,
    # Anchor PRF path unless extended identity context handles exact mode.
    "PRF_OPERATOR_MODE": "anchor",
}

# Merged into ``run_stage`` inner solver cfg after defaults so GP_FALLBACK ≠ 0.35 floor.
_SOLVER_CFG_MERGE = {"GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0}


def main() -> int:
    root_out = os.path.join(config.DIAGNOSTIC_DIR, "n5_prf_scale_ablation")
    os.makedirs(root_out, exist_ok=True)

    base_case = prepare_real_template_case()
    cut5 = list(base_case["template_cutouts"])[:5]
    real_case = {
        **dict(base_case),
        "template_cutouts": cut5,
    }

    runs_meta = []
    scenarios = [
        (
            "diagonal_gp_prf",
            False,
            "Diagonal Gaussian GP prior on scene pixels (no spatial correlation in Q); "
            "full SV-PRF and projection as in production.",
        ),
        (
            "diagonal_gp_no_prf",
            True,
            "Same diagonal prior as diagonal_gp_prf; PRF convolution replaced by identity "
            "(projection-only forward when PRF_ORDER_PROJECT_THEN_CONVOLVE is False).",
        ),
    ]

    with _temporary_config(_DIAG_BASE_CONFIG):
        for subdir, use_identity_prf, interpretation in scenarios:
            out_dir = os.path.join(root_out, subdir)
            os.makedirs(out_dir, exist_ok=True)
            ctx = identity_prf_operators_context() if use_identity_prf else nullcontext()
            with ctx:
                stage, iter_log = run_stage(
                    "N5",
                    5,
                    max_iterations=1,
                    center_radius_px=3.0,
                    output_dir=out_dir,
                    reduced_chi2_target=1.5,
                    real_case=real_case,
                    data_source="real",
                    allow_point_source=False,
                    require_nuclear_point=False,
                    aggressive_recovery=False,
                    solver_config_merge=_SOLVER_CFG_MERGE,
                    gp_amp_variance=float(getattr(config, "INIT_VARIANCE", 1.0)),
                )
            snap = {
                k: getattr(config, k, None)
                for k in (
                    "MAX_SCENE_PIXELS",
                    "GP_FALLBACK_NEIGHBOR_SMOOTHNESS",
                    "ENFORCE_GP_CENTRAL_MONOTONICITY",
                    "GP_OPTIMIZE_HYPERPARAMS",
                    "PRF_OPERATOR_MODE",
                    "PRF_ORDER_PROJECT_THEN_CONVOLVE",
                    "SUPERSAMPLE_FACTOR",
                    "PIXEL_SCALE",
                )
            }
            runs_meta.append(
                {
                    "subdir": subdir,
                    "identity_prf": use_identity_prf,
                    "interpretation": interpretation,
                    "config_snapshot": snap,
                    "stage": asdict(stage),
                    "iter_log": iter_log,
                }
            )

    manifest = {
        "output_root": root_out,
        "caveat": (
            "Diagonal Q does not imply a diagonal Hessian in scene pixels: per-BCD background "
            "parameters introduce cross-terms between scene amplitudes and sky levels."
        ),
        "runs": runs_meta,
    }
    man_path = os.path.join(root_out, "run_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
