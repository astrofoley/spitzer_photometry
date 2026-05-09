#!/usr/bin/env python3
"""N=1, iteration=1, super-res=5: same independence/diagonal-GP + PRF on vs off as N10 script.

Outputs under ``config.DIAGNOSTIC_DIR / n1_sr5_independent_gp_ablation/``.

Env ``N1_SR_SCENE_PIXEL_CAP`` (default ``42000``): scene footprint budget for
``build_template_real_case_from_cutouts``.

This diagnostic script can be slow when the scene footprint is large: the
solver builds dense normal equations whose runtime/memory scale steeply with
the number of fitted scene pixels. For quick iteration, set
``N1_SR_SCENE_PIXEL_CAP`` to something like ``2000``–``5000``.

The solver super-resolution is ``SUPERSAMPLE_FACTOR`` (5 here). Diagnostic PDFs
use a separate cosmetic zoom, ``config.DIAG_SUPERRES_DISPLAY_FACTOR``
(nearest-neighbor replication for imshow); do not set that equal to
``SUPERSAMPLE_FACTOR`` — they mean different things.

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
_MAX_SCENE_PIXELS_BUDGET = int(os.environ.get("N1_SR_SCENE_PIXEL_CAP", "42000"))
_SKIP_DIPOLE_REFINEMENT = os.environ.get("N1_SKIP_DIPOLE_REFINEMENT", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

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
    out_root = os.path.join(config.DIAGNOSTIC_DIR, "n1_sr5_independent_gp_ablation")
    os.makedirs(out_root, exist_ok=True)

    print(f"N=1 SR={_SUPERSAMPLE}, scene pixel cap={_MAX_SCENE_PIXELS_BUDGET}")
    base = prepare_real_template_case()
    all_cut = list(base["template_cutouts"])
    if len(all_cut) < 1:
        raise RuntimeError("No template cutouts")

    manifest: Dict[str, Any] = {
        "output_root": out_root,
        "n_bcd": 1,
        "supersample_factor": _SUPERSAMPLE,
        "max_scene_pixels_budget": _MAX_SCENE_PIXELS_BUDGET,
        "gp": "diagonal_independent_pixels_no_matern_no_neighbor_smooth",
        "runs": [],
    }

    with _temporary_config(_DIAG_CONFIG):
        real = build_template_real_case_from_cutouts(
            all_cut[:1],
            max_scene_pixels=_MAX_SCENE_PIXELS_BUDGET,
        )
        for sub, use_identity, label in (
            ("with_prf", False, "N1_SR5_IND_PRF"),
            ("no_prf", True, "N1_SR5_IND_NOPRF"),
        ):
            odir = os.path.join(out_root, sub)
            os.makedirs(odir, exist_ok=True)
            ctx = identity_prf_operators_context() if use_identity else nullcontext()
            with ctx:
                stage, ilog = run_stage(
                    label,
                    1,
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
                    gp_amp_variance=float(getattr(config, "INIT_VARIANCE", 1.0)),
                    skip_dipole_refinement=_SKIP_DIPOLE_REFINEMENT,
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
