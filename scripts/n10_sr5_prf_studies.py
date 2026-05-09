#!/usr/bin/env python3
"""N=10, elevated supersampling (default 5x), iteration=1: SR / PRF diagnostics on real templates.

Use **5x** by default (``N10_SUPERSAMPLE_FACTOR`` env, default ``5``) to test whether coarser
native scene grids contribute to structured residuals, with lower memory than 10x.

Runs (under ``config.DIAGNOSTIC_DIR / n10_sr{N}_runs /`` where *N* is the supersample factor):

1. **first10_prf** / **first10_noprf**: First 10 template BCDs, diagonal GP prior, full PRF vs
   identity PRF.

2. **diverse_prf**: Ten templates chosen for spread in native-grid orientation and CRPIX
   (greedy farthest-first), same diagonal GP, **PRF on**.

Override scene footprint cap with env ``N10_SR_SCENE_PIXEL_CAP`` (default ``60000``).

Outputs: ``NATIVE_FIT_DIAGNOSTIC_<stage>.pdf``, ``STACKED_RESIDUALS_<stage>.pdf``,
``ITER_METRICS_<stage>.png``, ``run_manifest.json``, ``diverse_selection.json``.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

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

_SUPERSAMPLE = int(os.environ.get("N10_SUPERSAMPLE_FACTOR", "5"))
_MAX_SCENE_PIXELS_BUDGET = int(os.environ.get("N10_SR_SCENE_PIXEL_CAP", "60000"))

_DIAG_CONFIG: Dict[str, Any] = {
    "SUPERSAMPLE_FACTOR": _SUPERSAMPLE,
    "MAX_SCENE_PIXELS": 0,
    "GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0,
    "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
    "GP_OPTIMIZE_HYPERPARAMS": False,
    "PRF_OPERATOR_MODE": "anchor",
}

_SOLVER_CFG_MERGE = {"GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0}


def _bcd_grid_features(raw_wcs) -> Tuple[float, float, float]:
    """Rotation-like angle (deg) of native +x on sky, and CRPIX."""
    cdmat = np.asarray(raw_wcs.pixel_scale_matrix, dtype=float)
    theta = float(np.degrees(np.arctan2(float(cdmat[0, 1]), float(cdmat[0, 0]))))
    crx, cry = float(raw_wcs.wcs.crpix[0]), float(raw_wcs.wcs.crpix[1])
    return theta, crx, cry


def select_diverse_cutout_indices(cutouts: Sequence[dict], n_select: int = 10) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Greedy farthest-first on (sin2θ, cos2θ, ΔCRPIXx, ΔCRPIXy) features."""
    n_all = len(cutouts)
    if n_all < n_select:
        raise RuntimeError(f"Need at least {n_select} cutouts, got {n_all}")
    feats = [_bcd_grid_features(c["raw_wcs"]) for c in cutouts]
    thetas = np.array([f[0] for f in feats], dtype=float)
    crxs = np.array([f[1] for f in feats], dtype=float)
    cys = np.array([f[2] for f in feats], dtype=float)
    med_x, med_y = float(np.median(crxs)), float(np.median(cys))
    tr = np.deg2rad(thetas)
    rot = np.stack([np.sin(2.0 * tr), np.cos(2.0 * tr)], axis=1)
    pos = np.stack([(crxs - med_x) / 25.0, (cys - med_y) / 25.0], axis=1)
    U = np.hstack([rot, pos])
    dist = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=2)
    i0, j0 = np.unravel_index(int(np.argmax(dist)), dist.shape)
    selected = [int(i0), int(j0)]
    while len(selected) < n_select:
        best_k = -1
        best_d = -1.0
        for k in range(n_all):
            if k in selected:
                continue
            md = min(float(dist[k, s]) for s in selected)
            if md > best_d:
                best_d = md
                best_k = k
        selected.append(best_k)
    rows = []
    for idx in selected:
        fn = str(cutouts[idx].get("filename", ""))
        th, cx, cy = feats[idx]
        rows.append(
            {
                "index": idx,
                "filename": fn,
                "theta_deg_native_x": th,
                "crpix1": cx,
                "crpix2": cy,
            }
        )
    return selected, rows


def main() -> int:
    sr = _SUPERSAMPLE
    tag = f"sr{sr}"
    out_root = os.path.join(config.DIAGNOSTIC_DIR, f"n10_{tag}_runs")
    os.makedirs(out_root, exist_ok=True)

    print(f"Supersample factor={sr}, scene pixel budget cap={_MAX_SCENE_PIXELS_BUDGET}")
    print("Loading full template case (native cutouts + default scene metadata)...")
    base = prepare_real_template_case()
    all_cut = list(base["template_cutouts"])
    if len(all_cut) < 10:
        raise RuntimeError(f"Expected >= 10 template cutouts, got {len(all_cut)}")

    div_idx, div_meta = select_diverse_cutout_indices(all_cut, 10)
    div_path = os.path.join(out_root, "diverse_selection.json")
    with open(div_path, "w", encoding="utf-8") as f:
        json.dump({"indices": div_idx, "frames": div_meta}, f, indent=2)
    print(f"Wrote diverse selection ({len(div_idx)} frames): {div_path}")

    lbl_prf = f"N10_{tag.upper()}_PRF"
    lbl_noprf = f"N10_{tag.upper()}_NOPRF"
    lbl_div = f"N10_{tag.upper()}_DIV_PRF"

    manifest: Dict[str, Any] = {
        "output_root": out_root,
        "supersample_factor": sr,
        "max_scene_pixels_budget": _MAX_SCENE_PIXELS_BUDGET,
        "note": (
            "Diagonal GP prior (MAX_SCENE_PIXELS=0). Per-BCD backgrounds still couple scene "
            "pixels in the Hessian. Diverse run uses full PRF only (no identity PRF)."
        ),
        "runs": [],
    }

    with _temporary_config(_DIAG_CONFIG):
        real_first = build_template_real_case_from_cutouts(
            all_cut[:10], max_scene_pixels=_MAX_SCENE_PIXELS_BUDGET,
        )
        for sub, use_id, label in (
            ("first10_prf", False, lbl_prf),
            ("first10_noprf", True, lbl_noprf),
        ):
            odir = os.path.join(out_root, sub)
            os.makedirs(odir, exist_ok=True)
            ctx = identity_prf_operators_context() if use_id else nullcontext()
            with ctx:
                stage, ilog = run_stage(
                    label,
                    10,
                    max_iterations=1,
                    center_radius_px=3.0,
                    output_dir=odir,
                    reduced_chi2_target=1.5,
                    real_case=real_first,
                    data_source="real",
                    allow_point_source=False,
                    require_nuclear_point=False,
                    aggressive_recovery=False,
                    solver_config_merge=_SOLVER_CFG_MERGE,
                    gp_amp_variance=float(getattr(config, "INIT_VARIANCE", 1.0)),
                )
            manifest["runs"].append(
                {
                    "subdir": sub,
                    "stage_name": label,
                    "identity_prf": use_id,
                    "selection": "first_10_templates",
                    "stage": asdict(stage),
                    "iter_log": ilog,
                }
            )

        real_div = build_template_real_case_from_cutouts(
            [all_cut[i] for i in div_idx], max_scene_pixels=_MAX_SCENE_PIXELS_BUDGET,
        )
        odir = os.path.join(out_root, "diverse_prf")
        os.makedirs(odir, exist_ok=True)
        stage, ilog = run_stage(
            lbl_div,
            10,
            max_iterations=1,
            center_radius_px=3.0,
            output_dir=odir,
            reduced_chi2_target=1.5,
            real_case=real_div,
            data_source="real",
            allow_point_source=False,
            require_nuclear_point=False,
            aggressive_recovery=False,
            solver_config_merge=_SOLVER_CFG_MERGE,
            gp_amp_variance=float(getattr(config, "INIT_VARIANCE", 1.0)),
        )
        manifest["runs"].append(
            {
                "subdir": "diverse_prf",
                "stage_name": lbl_div,
                "identity_prf": False,
                "selection": "diverse_greedy_farthest",
                "selected_indices": div_idx,
                "stage": asdict(stage),
                "iter_log": ilog,
            }
        )

    man_path = os.path.join(out_root, "run_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
