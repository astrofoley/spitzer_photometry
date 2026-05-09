#!/usr/bin/env python3
"""N=10, iteration=1, super-res=5: independent scene pixels (diagonal GP) + PRF on vs off.

**Independent pixels** here means no Matérn coupling and no diagonal-fallback neighbor
smoothing: ``MAX_SCENE_PIXELS=0``, ``GP_FALLBACK_NEIGHBOR_SMOOTHNESS=0`` (also merged into
trial config), ``ENFORCE_GP_CENTRAL_MONOTONICITY=False``. Per-BCD backgrounds still couple
scene amplitudes in the Hessian.

**GP variance:** For diagonal Q, ``var`` is a per-pixel marginal variance. ``run_stage``'s
default ``1e-7`` would over-shrink the scene (sparse / stripy maps); this script passes
``gp_amp_variance=config.INIT_VARIANCE``.

Two runs under ``config.DIAGNOSTIC_DIR / n10_sr5_independent_gp_ablation/``:

- **with_prf**: full spatially varying PRF
- **no_prf**: ``identity_prf_operators_context`` (no convolution on scene/native)

Env:

- ``N10_SR_SCENE_PIXEL_CAP`` (default ``42000``): passed to ``_scene_wcs_budgeted`` to limit
  scene size / runtime (budget may coarsen effective pixel step slightly).

Outputs: PDFs, stacked residuals, iteration metrics, ``run_manifest.json``.
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
    build_template_real_case_from_cutouts,
    prepare_real_template_case,
    run_stage,
)
from src.prf_identity_context import identity_prf_operators_context  # noqa: E402

_SUPERSAMPLE = 5
_MAX_SCENE_PIXELS_BUDGET = int(os.environ.get("N10_SR_SCENE_PIXEL_CAP", "42000"))

_DIAG_CONFIG: Dict[str, Any] = {
    "SUPERSAMPLE_FACTOR": _SUPERSAMPLE,
    "MAX_SCENE_PIXELS": 0,
    "GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0,
    "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
    "GP_OPTIMIZE_HYPERPARAMS": False,
    "PRF_OPERATOR_MODE": "anchor",
}

_SOLVER_CFG_MERGE = {"GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0}


def main() -> int:
    out_root = os.path.join(config.DIAGNOSTIC_DIR, "n10_sr5_independent_gp_ablation")
    os.makedirs(out_root, exist_ok=True)

    print(f"SR={_SUPERSAMPLE}, scene pixel cap={_MAX_SCENE_PIXELS_BUDGET}")
    base = prepare_real_template_case()
    all_cut = list(base["template_cutouts"])
    if len(all_cut) < 10:
        raise RuntimeError(f"Need >= 10 template cutouts, got {len(all_cut)}")

    manifest: Dict[str, Any] = {
        "output_root": out_root,
        "supersample_factor": _SUPERSAMPLE,
        "max_scene_pixels_budget": _MAX_SCENE_PIXELS_BUDGET,
        "gp": "diagonal_independent_pixels_no_matern_no_neighbor_smooth",
        "runs": [],
    }

    with _temporary_config(_DIAG_CONFIG):
        real = build_template_real_case_from_cutouts(
            all_cut[:10],
            max_scene_pixels=_MAX_SCENE_PIXELS_BUDGET,
        )
        for sub, use_identity, label in (
            ("with_prf", False, "N10_SR5_IND_PRF"),
            ("no_prf", True, "N10_SR5_IND_NOPRF"),
        ):
            odir = os.path.join(out_root, sub)
            os.makedirs(odir, exist_ok=True)
            ctx = identity_prf_operators_context() if use_identity else nullcontext()
            with ctx:
                stage, ilog = run_stage(
                    label,
                    10,
                    max_iterations=1,
                    center_radius_px=3.0,
                    output_dir=odir,
                    reduced_chi2_target=1.5,
                    real_case=real,
                    data_source="real",
                    allow_point_source=False,
                    require_nuclear_point=False,
                    aggressive_recovery=False,
                    solver_config_merge=_SOLVER_CFG_MERGE,
                    # Diagonal GP: var is per-pixel marginal scale; default 1e-7 is far too small.
                    gp_amp_variance=float(getattr(config, "INIT_VARIANCE", 1.0)),
                )
            manifest["runs"].append(
                {
                    "subdir": sub,
                    "identity_prf": use_identity,
                    "stage_name": label,
                    "stage": asdict(stage),
                    "iter_log": ilog,
                }
            )

    mp = os.path.join(out_root, "run_manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
