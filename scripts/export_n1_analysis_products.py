"""Export exact N=1 analysis products as FITS with pipeline WCS."""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
from astropy.io import fits

from src import config, native_fit_campaign, solver


def _write_fits(path: str, data: np.ndarray, wcs_obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = wcs_obj.to_header(relax=True)
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float64), header=hdr)
    hdu.writeto(path, overwrite=True)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export N=1 pipeline analysis products to FITS")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: output/analysis_fits_n1_export)",
    )
    p.add_argument(
        "--project-then-prf",
        action="store_true",
        help="Use project->PRF order (matches current diagnostic mode).",
    )
    return p


def main() -> int:
    args = _make_parser().parse_args()
    out_dir = args.out_dir or os.path.join(config.OUTPUT_DIR, "analysis_fits_n1_export")
    os.makedirs(out_dir, exist_ok=True)

    real_case = native_fit_campaign.prepare_real_template_case()
    cutout = dict(real_case["template_cutouts"][0])
    # Match run_stage() behavior for analysis masking.
    native_fit_campaign.apply_native_cutout_cr_mask(cutout)
    cutouts = [cutout]

    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    channel = "ch2" if "ch2" in str(cutout.get("filename", "")).lower() else "ch1"
    is_full = bool(cutout.get("is_full_array", False))

    # N=1 iteration=1 knobs from campaign path.
    ell = 1.8
    var = 1.0e-7

    with native_fit_campaign._temporary_config(
        {"PRF_ORDER_PROJECT_THEN_CONVOLVE": bool(args.project_then_prf)}
    ):
        results = native_fit_campaign._run_solver(cutouts, scene_wcs, scene_shape, ell, var)

    data_bcd = np.asarray(cutout["data"], dtype=float)
    bcd_wcs = cutout["wcs"]
    raw_wcs = cutout["raw_wcs"]
    model_scene = np.asarray(results["model_scene"], dtype=float)

    # (3) projected model in BCD frame (before PRF)
    proj_bcd = solver._project_scene_to_native(model_scene, scene_wcs, raw_wcs, data_bcd.shape)

    # (4) projected model convolved with PRF (no background term)
    prf_bcd = solver._apply_prf_operator_native(
        proj_bcd,
        raw_wcs,
        data_bcd.shape,
        channel,
        is_full_array=is_full,
    )

    # Full modeled BCD from the same prediction path used by diagnostics.
    pred_bcd = solver.predict_cutout_model(
        results,
        cutouts,
        [],
        [],
        0,
        include_gp=True,
        include_transient=True,
        include_stars=False,
        include_host=True,
        include_nuclear_point=True,
    )
    resid_bcd = data_bcd - np.asarray(pred_bcd, dtype=float)

    outputs: Dict[str, str] = {
        "bcd_cropped_data": os.path.join(out_dir, "N1_BCD_DATA_CROPPED.fits"),
        "model_superres_nup": os.path.join(out_dir, "N1_MODEL_SUPERRES_NUP.fits"),
        "model_projected_bcd": os.path.join(out_dir, "N1_MODEL_PROJECTED_TO_BCD.fits"),
        "model_projected_convolved_bcd": os.path.join(out_dir, "N1_MODEL_PROJECTED_CONVOLVED_BCD.fits"),
        "residual_bcd": os.path.join(out_dir, "N1_RESIDUAL_BCD_DATA_MINUS_MODEL.fits"),
    }

    _write_fits(outputs["bcd_cropped_data"], data_bcd, bcd_wcs)
    _write_fits(outputs["model_superres_nup"], model_scene, scene_wcs)
    _write_fits(outputs["model_projected_bcd"], proj_bcd, bcd_wcs)
    _write_fits(outputs["model_projected_convolved_bcd"], prf_bcd, bcd_wcs)
    _write_fits(outputs["residual_bcd"], resid_bcd, bcd_wcs)

    print("WROTE_FITS_PRODUCTS")
    for k, v in outputs.items():
        print(f"{k}={v}")
    print(f"prf_order_project_then_convolve={bool(args.project_then_prf)}")
    print(f"input_bcd_filename={cutout.get('filename', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
