#!/usr/bin/env python3
"""Compare ell sensitivity for cropped N=1: normal vs exact scene coupling."""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, fit_metrics, gp_model, solver  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _reindex_epochs,
    _temporary_config,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
)


def _crop_cutout(cut, cy: int, cx: int, half: int):
    y0 = max(0, int(cy) - int(half))
    y1 = min(cut["data"].shape[0], int(cy) + int(half) + 1)
    x0 = max(0, int(cx) - int(half))
    x1 = min(cut["data"].shape[1], int(cx) + int(half) + 1)
    data = np.asarray(cut["data"], dtype=float)[y0:y1, x0:x1]
    sigma = np.asarray(cut["sigma"], dtype=float)[y0:y1, x0:x1]
    wcs = cut["wcs"].slice((slice(y0, y1), slice(x0, x1)))
    raw_wcs = cut["raw_wcs"].slice((slice(y0, y1), slice(x0, x1)))
    out = dict(cut)
    out["data"] = data
    out["sigma"] = sigma
    out["wcs"] = wcs
    out["raw_wcs"] = raw_wcs
    return out


def _run_scan(cutout, scene_wcs, scene_shape, stars, init_star_fluxes, ell_vals, var_fixed, exact: bool):
    rows = []
    models = []
    base_overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
        "PRF_GLS_LTWL_DIAG_MAX_PIXELS": int(scene_shape[0] * scene_shape[1] + 1) if exact else 0,
    }
    with _temporary_config(base_overrides):
        for ell in ell_vals:
            res = solver.run_gls_solve(
                [cutout],
                stars,
                init_star_fluxes,
                {"ell": float(ell), "var": float(var_fixed)},
                (float(ell), float(var_fixed)),
                np.zeros(scene_shape),
                scene_wcs,
                1,
            )
            pred = np.asarray(
                solver.predict_cutout_model(
                    res, [cutout], [], [], 0,
                    include_gp=True, include_transient=False, include_stars=False, include_host=False, include_nuclear_point=False,
                ),
                dtype=float,
            )
            models.append(pred.copy())
            m = fit_metrics.compute_fit_metrics(
                [cutout],
                res,
                stars,
                res.get("star_fluxes", init_star_fluxes),
                center_ra_deg=float(config.TRANSIENT_RA),
                center_dec_deg=float(config.TRANSIENT_DEC),
                center_radius_px=3.0,
            )
            rows.append(
                {
                    "ell": float(ell),
                    "total_reduced_chi2": float(m.get("total_reduced_chi2", np.nan)),
                    "center_reduced_chi2": float(m.get("center_reduced_chi2", np.nan)),
                },
            )
    drel = [None]
    for i in range(1, len(models)):
        d = models[i] - models[i - 1]
        num = float(np.sqrt(np.mean(d * d)))
        den = float(np.sqrt(np.mean(models[i - 1] * models[i - 1])))
        drel.append(num / max(den, 1e-30))
    for r, dr in zip(rows, drel):
        r["delta_rel_vs_prev_model"] = None if dr is None else float(dr)
    return rows


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_crop_exact_compare")
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cut = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])[0]
    apply_native_cutout_cr_mask(cut)

    nr = float(real_case["centers"]["nuc_ra"])
    nd = float(real_case["centers"]["nuc_dec"])
    cx, cy = cut["wcs"].world_to_pixel_values(nr, nd)
    cx = float(np.asarray(cx).ravel()[0])
    cy = float(np.asarray(cy).ravel()[0])
    crop_half = 28  # ~57x57 central crop
    cut_crop = _crop_cutout(cut, int(round(cy)), int(round(cx)), crop_half)

    scene_wcs = cut_crop["wcs"]
    scene_shape = cut_crop["data"].shape
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)

    _, var_opt = gp_model.optimize_hyperparameters([cut_crop])
    var_opt = float(max(float(var_opt), 1e-30))
    ell_vals = [20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

    rows_normal = _run_scan(cut_crop, scene_wcs, scene_shape, stars, init_star_fluxes, ell_vals, var_opt, exact=False)
    rows_exact = _run_scan(cut_crop, scene_wcs, scene_shape, stars, init_star_fluxes, ell_vals, var_opt, exact=True)

    summary = {
        "crop_shape": [int(scene_shape[0]), int(scene_shape[1])],
        "var_fixed": var_opt,
        "ell_values": ell_vals,
        "normal": rows_normal,
        "exact": rows_exact,
    }
    out_json = os.path.join(out_dir, "n1_crop_exact_compare_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_png = os.path.join(out_dir, "N1_CROP_EXACT_COMPARE.png")
    x = np.arange(len(ell_vals))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, [r["delta_rel_vs_prev_model"] or np.nan for r in rows_normal], marker="o", label="crop-normal")
    ax[0].plot(x, [r["delta_rel_vs_prev_model"] or np.nan for r in rows_exact], marker="o", label="crop-exact")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, [f"{e:g}" for e in ell_vals], rotation=45)
    ax[0].set_title("delta_rel model vs previous ell")
    ax[0].legend()
    ax[1].plot(x, [r["center_reduced_chi2"] for r in rows_normal], marker="o", label="crop-normal")
    ax[1].plot(x, [r["center_reduced_chi2"] for r in rows_exact], marker="o", label="crop-exact")
    ax[1].set_xticks(x, [f"{e:g}" for e in ell_vals], rotation=45)
    ax[1].set_title("center reduced chi2 vs ell")
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

