#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, fit_metrics, native_fit_campaign, preprocessing  # noqa: E402


def _prepare_rotated_template_case(n_bcd: int):
    all_files = preprocessing.find_spitzer_files(config.DATA_DIR)
    _, tpl_files = preprocessing.categorize_observations(all_files, config.SPLIT_DATE_MJD)
    if not tpl_files:
        raise RuntimeError("No template files available")

    tpl_files = [dict(f, is_template=True) for f in tpl_files[: int(n_bcd)]]
    target = SkyCoord(float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC), unit="deg")

    mosaic_wcs, mosaic_shape = preprocessing.define_mosaic_wcs(tpl_files, target)
    processed_tpl = preprocessing.reproject_to_grid(tpl_files, mosaic_wcs, mosaic_shape)
    tpl_cube = np.array([p["data"] for p in processed_tpl])
    med_stack, _ = preprocessing.create_median_stack(tpl_cube)
    source_cat = preprocessing.get_or_create_source_catalog(all_files)
    preprocessing.align_frames_to_template(tpl_files, med_stack, mosaic_wcs, source_cat)

    cutouts, cutout_wcs = preprocessing.extract_analysis_cutouts(tpl_files, target)
    cutouts = [dict(c) for c in cutouts]
    for i, c in enumerate(cutouts):
        c["epoch_id"] = i
        c["is_template"] = True

    # Legacy rotated path: scene grid is exactly the N-up analysis cutout grid.
    scene_wcs = cutout_wcs
    scene_shape = tuple(cutouts[0]["data"].shape)
    nuc_explicit = native_fit_campaign.resolve_explicit_nuclear_host_sky_deg()
    tr_ra, tr_dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
    centers = {
        "transient_ra": tr_ra,
        "transient_dec": tr_dec,
        "nuc_ra": float(nuc_explicit[0]) if nuc_explicit is not None else None,
        "nuc_dec": float(nuc_explicit[1]) if nuc_explicit is not None else None,
    }
    return {
        "template_cutouts": cutouts,
        "scene_wcs": scene_wcs,
        "scene_shape": scene_shape,
        "all_stars": [],
        "init_star_fluxes": np.zeros(0, dtype=float),
        "centers": centers,
    }


def _run_one(label: str, real_case: dict, out_dir: str):
    cutouts = [dict(c) for c in real_case["template_cutouts"][:10]]
    for i, c in enumerate(cutouts):
        c["epoch_id"] = i
        native_fit_campaign.apply_native_cutout_cr_mask(c)
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    centers = dict(real_case["centers"])

    native_fit_campaign._run_solver._stars = []
    native_fit_campaign._run_solver._star_fluxes = np.zeros(0, dtype=float)
    with native_fit_campaign._temporary_config(
        {
            "USE_HOST_GAUSSIAN_CORE": False,
            "USE_NUCLEAR_POINT_SOURCE": False,
        },
    ):
        results = native_fit_campaign._run_solver(cutouts, scene_wcs, scene_shape, 1.8, 1e-7)
    metrics = fit_metrics.compute_fit_metrics(
        cutouts,
        results,
        [],
        np.zeros(0, dtype=float),
        center_ra_deg=float(centers["nuc_ra"] or centers["transient_ra"]),
        center_dec_deg=float(centers["nuc_dec"] or centers["transient_dec"]),
        center_radius_px=3.0,
    )

    local_dir = os.path.join(out_dir, label)
    os.makedirs(local_dir, exist_ok=True)
    with native_fit_campaign._temporary_config(
        {
            "USE_HOST_GAUSSIAN_CORE": False,
            "USE_NUCLEAR_POINT_SOURCE": False,
        },
    ):
        diag_pdf = native_fit_campaign.write_native_fit_pdf("N10", cutouts, results, local_dir)
        stack_pdf = native_fit_campaign.write_stacked_residual_pdf("N10", cutouts, results, local_dir)
    return {
        "label": label,
        "n_bcd": 10,
        "ell": 1.8,
        "var": 1e-7,
        "metrics": metrics,
        "diagnostic_pdf": diag_pdf,
        "stacked_pdf": stack_pdf,
    }


def main():
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n10_one_iter_native_vs_rotated")
    os.makedirs(out_dir, exist_ok=True)

    native_case = native_fit_campaign.prepare_real_template_case()
    rotated_case = _prepare_rotated_template_case(10)

    native_res = _run_one("native", native_case, out_dir)
    rotated_res = _run_one("rotated_nup", rotated_case, out_dir)

    summary = {
        "native": native_res,
        "rotated_nup": rotated_res,
        "delta_center_reduced_chi2": (
            float(native_res["metrics"]["center_reduced_chi2"])
            - float(rotated_res["metrics"]["center_reduced_chi2"])
        ),
        "delta_total_reduced_chi2": (
            float(native_res["metrics"]["total_reduced_chi2"])
            - float(rotated_res["metrics"]["total_reduced_chi2"])
        ),
    }
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
