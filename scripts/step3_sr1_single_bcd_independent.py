#!/usr/bin/env python3
"""Step 3: single-BCD, SR=1, independent scene pixels, no PRF convolution.

Model:
- constant background (one scalar for this BCD)
- one free amplitude per scene pixel on a North-up SR grid

The scene grid is fixed to (SR*N) x (SR*N), where N is the selected BCD side.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
from astropy.wcs import WCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, fit_metrics, solver  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _reindex_epochs,
    _scene_wcs_from_bcd_footprints,
    _temporary_config,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
    write_native_fit_pdf,
    write_stacked_residual_pdf,
)
from src.prf_identity_context import identity_prf_operators_context  # noqa: E402


def _north_up_fixed_scene_wcs_from_bcd(bcd_wcs: WCS, bcd_shape: tuple[int, int], sr: int) -> tuple[WCS, tuple[int, int]]:
    h_native, w_native = int(bcd_shape[0]), int(bcd_shape[1])
    if h_native != w_native:
        raise RuntimeError(f"Step 3 expects square BCD cutouts; got shape={bcd_shape}")
    n_side_min = int(sr) * h_native
    if n_side_min <= 0:
        raise RuntimeError(f"Invalid SR or native size: sr={sr}, shape={bcd_shape}")

    cx = 0.5 * (w_native - 1.0)
    cy = 0.5 * (h_native - 1.0)
    ra_c, dec_c = bcd_wcs.pixel_to_world_values(cx, cy)
    ra_c = float(np.asarray(ra_c).ravel()[0])
    dec_c = float(np.asarray(dec_c).ravel()[0])

    scene_wcs = WCS(naxis=2)
    # Start with minimum SR*N side length, then expand so all BCD corners are inside.
    scene_wcs.wcs.crpix = [0.5 * (n_side_min + 1.0), 0.5 * (n_side_min + 1.0)]
    scene_wcs.wcs.crval = [ra_c, dec_c]
    scene_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale_deg = float(config.PIXEL_SCALE) / float(sr) / 3600.0
    scene_wcs.wcs.cdelt = [-scale_deg, scale_deg]
    scene_wcs.wcs.pc = np.eye(2)

    corners = np.array(
        [
            [0.0, 0.0],
            [float(w_native - 1), 0.0],
            [0.0, float(h_native - 1)],
            [float(w_native - 1), float(h_native - 1)],
        ],
        dtype=np.float64,
    )
    ra_corn, dec_corn = bcd_wcs.pixel_to_world_values(corners[:, 0], corners[:, 1])
    xs, ys = scene_wcs.world_to_pixel_values(np.asarray(ra_corn, dtype=float), np.asarray(dec_corn, dtype=float))
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    x_span = float(np.max(xs) - np.min(xs))
    y_span = float(np.max(ys) - np.min(ys))
    # +3 gives a one-pixel guard on each side after ceil.
    n_side_cover = int(np.ceil(max(x_span, y_span) + 3.0))
    n_side = int(max(n_side_min, n_side_cover))

    # Recenter on the same sky point with the final side length.
    scene_wcs.wcs.crpix = [0.5 * (n_side + 1.0), 0.5 * (n_side + 1.0)]
    return scene_wcs, (n_side, n_side)


def _frame_rotation_deg(w: WCS) -> float:
    """Rotation of native +x pixel axis toward +RA/TAN tangent plane axis (degrees).

    Spitzer/IRAC pipelines often omit PC and encode scale+rotation in CD; pc alone then
    looks like identity while CD carries the real shear/rotation."""
    mat = getattr(w, "pixel_scale_matrix", None)
    if mat is not None:
        m = np.asarray(mat, dtype=float)
        if m.shape == (2, 2) and np.linalg.norm(m[:, 0]) > 1e-30:
            return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    ww = w.wcs
    cd = np.asarray(getattr(ww, "cd", [[0.0, 0.0], [0.0, 0.0]]), dtype=float)
    if cd.shape == (2, 2) and np.linalg.norm(cd) > 0 and np.linalg.norm(cd[:, 0]) > 1e-30:
        if not np.allclose(cd, 0.0):
            return float(np.degrees(np.arctan2(cd[1, 0], cd[0, 0])))
    if hasattr(ww, "pc") and ww.pc is not None:
        pc = np.asarray(ww.pc, dtype=float)
        cdelt = np.asarray(getattr(ww, "cdelt", [np.nan, np.nan]), dtype=float).ravel()
        if pc.shape == (2, 2) and cdelt.size >= 2 and np.all(np.isfinite(cdelt)):
            m = pc @ np.diag([float(cdelt[0]), float(cdelt[1])])
            if np.linalg.norm(m[:, 0]) > 1e-30:
                return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    return 0.0


def _pick_diverse_cutouts(cuts: list[dict], n_pick: int) -> list[dict]:
    if n_pick <= 0 or n_pick >= len(cuts):
        return list(cuts[: max(1, n_pick)])
    # Features: frame rotation + native pointing center.
    feats = []
    for c in cuts:
        ang = np.deg2rad(_frame_rotation_deg(c["raw_wcs"]))
        h, w = c["data"].shape
        cx = 0.5 * (w - 1.0)
        cy = 0.5 * (h - 1.0)
        ra, dec = c["raw_wcs"].pixel_to_world_values(cx, cy)
        feats.append([np.cos(ang), np.sin(ang), float(np.asarray(ra).ravel()[0]), float(np.asarray(dec).ravel()[0])])
    F = np.asarray(feats, dtype=np.float64)
    # Normalize RA/Dec to comparable scales.
    for j in (2, 3):
        mu = float(np.mean(F[:, j]))
        sd = float(np.std(F[:, j]))
        if sd > 0:
            F[:, j] = (F[:, j] - mu) / sd
        else:
            F[:, j] = 0.0
    picked = [0]
    while len(picked) < n_pick:
        rem = [i for i in range(len(cuts)) if i not in picked]
        if not rem:
            break
        dmin = []
        for i in rem:
            d = [float(np.linalg.norm(F[i] - F[j])) for j in picked]
            dmin.append(min(d) if d else 0.0)
        picked.append(rem[int(np.argmax(np.asarray(dmin, dtype=float)))])
    return [cuts[i] for i in picked]


def _north_up_fixed_scene_wcs_from_cutouts(cutouts: list[dict], sr: int) -> tuple[WCS, tuple[int, int]]:
    if not cutouts:
        raise RuntimeError("No cutouts provided")
    h_native, w_native = cutouts[0]["data"].shape
    if h_native != w_native:
        raise RuntimeError(f"Expected square BCD cutouts, got shape={(h_native, w_native)}")
    n_side_min = int(sr) * int(h_native)
    if n_side_min <= 0:
        raise RuntimeError(f"Invalid SR/native shape: sr={sr}, shape={(h_native, w_native)}")
    # Center on mean of selected BCD centers.
    ras, decs = [], []
    for c in cutouts:
        h, w = c["data"].shape
        ra, dec = c["raw_wcs"].pixel_to_world_values(0.5 * (w - 1.0), 0.5 * (h - 1.0))
        ras.append(float(np.asarray(ra).ravel()[0]))
        decs.append(float(np.asarray(dec).ravel()[0]))
    ra_c = float(np.mean(np.asarray(ras, dtype=float)))
    dec_c = float(np.mean(np.asarray(decs, dtype=float)))
    scene_wcs = WCS(naxis=2)
    scene_wcs.wcs.crpix = [0.5 * (n_side_min + 1.0), 0.5 * (n_side_min + 1.0)]
    scene_wcs.wcs.crval = [ra_c, dec_c]
    scene_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale_deg = float(config.PIXEL_SCALE) / float(sr) / 3600.0
    scene_wcs.wcs.cdelt = [-scale_deg, scale_deg]
    scene_wcs.wcs.pc = np.eye(2)
    bcd_wcs_list = [c["raw_wcs"] for c in cutouts]
    _, shape_cov = _scene_wcs_from_bcd_footprints(
        bcd_wcs_list,
        (h_native, w_native),
        float(config.PIXEL_SCALE) / float(sr),
        sky_points_deg=[(ra_c, dec_c)],
        pad_scene_px=1,
    )
    n_side = int(max(n_side_min, int(shape_cov[0]), int(shape_cov[1])))
    scene_wcs.wcs.crpix = [0.5 * (n_side + 1.0), 0.5 * (n_side + 1.0)]
    return scene_wcs, (n_side, n_side)


def _crop_cutout_to_size(cut: dict, size: int, ra_deg: float, dec_deg: float) -> dict:
    if size <= 0:
        return cut
    h, w = cut["data"].shape
    if size > h or size > w:
        raise RuntimeError(f"Requested cutout size={size} exceeds cutout shape={(h, w)}")
    # Center the requested sky point in *WCS pixel coordinates* after cropping.
    # Using float pixel coords avoids sub-pixel rounding drift that can move the
    # detector edge (where sigma is inf) into the "core" visualization radius.
    px, py = cut["wcs"].world_to_pixel_values(float(ra_deg), float(dec_deg))
    cx = float(np.asarray(px).ravel()[0])
    cy = float(np.asarray(py).ravel()[0])
    target_c = 0.5 * (float(size) - 1.0)  # center pixel index in cropped array
    x0 = int(np.round(cx - target_c))
    y0 = int(np.round(cy - target_c))
    x0 = int(np.clip(x0, 0, w - size))
    y0 = int(np.clip(y0, 0, h - size))
    x1 = x0 + int(size)
    y1 = y0 + int(size)
    out = dict(cut)
    out["data"] = np.asarray(cut["data"], dtype=float)[y0:y1, x0:x1]
    out["sigma"] = np.asarray(cut["sigma"], dtype=float)[y0:y1, x0:x1]
    out["wcs"] = cut["wcs"].slice((slice(y0, y1), slice(x0, x1)))
    out["raw_wcs"] = cut["raw_wcs"].slice((slice(y0, y1), slice(x0, x1)))
    return out


def main() -> int:
    step = int(os.environ.get("SR_STEP", "3"))
    sr = int(os.environ.get("SR_FACTOR", "1"))
    n_bcd = int(os.environ.get("N_BCD", "1"))
    use_prf = os.environ.get("USE_PRF_CONV", "0").strip().lower() in ("1", "true", "yes", "y")
    project_then_prf = os.environ.get("PRF_PROJECT_THEN_CONVOLVE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    scene_ridge = float(os.environ.get("SCENE_INDEPENDENT_RIDGE", "1e-12"))
    full_ltwl_cap = int(os.environ.get("PRF_GLS_LTWL_FULL_MAX_PIXELS", "0"))
    run_tag = os.environ.get("RUN_TAG", "").strip()
    cutout_size = int(os.environ.get("CUTOUT_SIZE", "0"))
    cr_guard_pct = float(os.environ.get("CR_CORE_GUARD_PERCENTILE", str(config.CR_BRIGHT_CORE_GUARD_PERCENTILE)))
    cr_guard_dilate = int(os.environ.get("CR_CORE_GUARD_DILATION", str(config.CR_BRIGHT_CORE_GUARD_DILATION)))
    cr_guard_radius = float(
        os.environ.get("CR_CORE_GUARD_RADIUS_PX", str(config.CR_BRIGHT_CORE_GUARD_RADIUS_PX))
    )
    unmask_sigma_inf_radius = float(os.environ.get("UNMASK_SIGMA_INF_RADIUS_PX", "0").strip() or "0")
    unmask_sigma_inf_center = os.environ.get("UNMASK_SIGMA_INF_CENTER", "nuclear").strip().lower()
    cr_guard_center = os.environ.get("CR_CORE_GUARD_CENTER", str(config.CR_BRIGHT_CORE_GUARD_CENTER)).strip().lower()
    if sr <= 0:
        raise RuntimeError(f"SR_FACTOR must be >= 1, got {sr}")
    if n_bcd <= 0:
        raise RuntimeError(f"N_BCD must be >= 1, got {n_bcd}")
    prf_tag = "prf" if use_prf else "noprf"
    label = f"STEP{step}_N{n_bcd}_SR{sr}_IND_{prf_tag.upper()}"
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, f"step{step}_n{n_bcd}_sr{sr}_independent_{prf_tag}")
    if run_tag:
        label = f"{label}_{run_tag.upper()}"
        out_dir = f"{out_dir}_{run_tag}"
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    all_cuts = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])])
    if len(all_cuts) < n_bcd:
        raise RuntimeError(f"Requested N_BCD={n_bcd} but only {len(all_cuts)} template cutouts are available")
    pool_rots = np.asarray([_frame_rotation_deg(c["raw_wcs"]) for c in all_cuts], dtype=float)
    print(
        "[step3] template pool rotation (deg, CD/pixel-scale–derived axis angle): "
        f"n={len(pool_rots)} min={float(np.min(pool_rots)):.5f} max={float(np.max(pool_rots)):.5f} "
        f"std={float(np.std(pool_rots)):.6f} — greedy diversity uses rotation+pointing center for every candidate"
    )
    cutouts = _pick_diverse_cutouts(all_cuts, n_bcd)
    cr_cfg = {
        "CR_BRIGHT_CORE_GUARD_PERCENTILE": float(cr_guard_pct),
        "CR_BRIGHT_CORE_GUARD_DILATION": int(max(0, cr_guard_dilate)),
        "CR_BRIGHT_CORE_GUARD_RADIUS_PX": float(max(0.0, cr_guard_radius)),
        "CR_BRIGHT_CORE_GUARD_CENTER": str(cr_guard_center),
    }
    def _radial_bad_sigma_counts(cut: dict, radius_px: float) -> dict:
        s = np.asarray(cut["sigma"], dtype=float)
        d = np.asarray(cut["data"], dtype=float)
        h, w = s.shape
        cx, cy = cut["raw_wcs"].world_to_pixel_values(float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC))
        cx = float(np.asarray(cx).ravel()[0])
        cy = float(np.asarray(cy).ravel()[0])
        yy, xx = np.mgrid[:h, :w]
        r = np.hypot(xx - cx, yy - cy)
        in_r = r <= float(radius_px)
        bad_sigma = ~np.isfinite(s)
        bad_data0 = d == 0
        return {
            "n_bad_sigma_r": int(np.sum(bad_sigma & in_r)),
            "n_data0_r_finite_sigma": int(np.sum(bad_data0 & in_r & np.isfinite(s))),
            "n_total_in_r": int(np.sum(in_r)),
        }

    with _temporary_config(cr_cfg):
        for cut in cutouts:
            pre_sigma = np.asarray(cut["sigma"], dtype=float).copy()
            pre_counts = _radial_bad_sigma_counts(cut, 8.0)
            apply_native_cutout_cr_mask(cut)
            post_counts = _radial_bad_sigma_counts(cut, 8.0)
            cr_effect_total = int(np.sum(np.isfinite(pre_sigma) & ~np.isfinite(cut["sigma"])))
            fname = cut.get("filename", "")
            eid = cut.get("epoch_id", None)
            print(
                f"[step3] CR mask debug {fname} epoch_id={eid}: "
                f"new_sigma_inf_total={cr_effect_total}, pre_bad_sigma_r8={pre_counts['n_bad_sigma_r']}, "
                f"post_bad_sigma_r8={post_counts['n_bad_sigma_r']}"
            )
    if cutout_size > 0:
        cutouts = [
            _crop_cutout_to_size(c, cutout_size, float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC))
            for c in cutouts
        ]
        for cut in cutouts:
            counts = _radial_bad_sigma_counts(cut, 8.0)
            fname = cut.get("filename", "")
            eid = cut.get("epoch_id", None)
            print(
                f"[step3] After crop CR debug {fname} epoch_id={eid}: "
                f"bad_sigma_r8={counts['n_bad_sigma_r']} data0_r={counts['n_data0_r_finite_sigma']} "
                f"n_total_in_r={counts['n_total_in_r']}"
            )

    # Optional: unmask pre-existing sigma=inf pixels within a radius around
    # a chosen center (useful if upstream bad-pixel/CR flags are suppressing
    # astrophysical structure).
    if unmask_sigma_inf_radius > 0.0:
        if unmask_sigma_inf_center in ("nuclear", "nuc", "host", "core"):
            c_ra = float(config.NUCLEAR_POINT_RA)
            c_dec = float(config.NUCLEAR_POINT_DEC)
        else:
            c_ra = float(config.TRANSIENT_RA)
            c_dec = float(config.TRANSIENT_DEC)

        def _unmask_sigma_inf_in_radius(cut: dict, radius_px: float) -> None:
            s = np.asarray(cut["sigma"], dtype=float)
            d = np.asarray(cut["data"], dtype=float)
            h, w = s.shape
            px, py = cut["wcs"].world_to_pixel_values(c_ra, c_dec)
            cx = float(np.asarray(px).ravel()[0])
            cy = float(np.asarray(py).ravel()[0])
            Y, X = np.mgrid[:h, :w]
            r = np.hypot(X - cx, Y - cy)
            unmask = (~np.isfinite(s)) & (d != 0) & (r <= float(radius_px))
            if not np.any(unmask):
                return
            finite = np.isfinite(s) & (d != 0)
            ref = finite & (r <= float(radius_px))
            if np.any(ref):
                med_sigma = float(np.nanmedian(s[ref]))
            else:
                med_sigma = float(np.nanmedian(s[finite])) if np.any(finite) else 1.0
            if not np.isfinite(med_sigma) or med_sigma <= 0.0:
                med_sigma = 1.0
            s2 = s.copy()
            s2[unmask] = med_sigma
            cut["sigma"] = s2

        print(
            f"[step3] Unmask sigma=inf within r<={unmask_sigma_inf_radius:.2f} "
            f"around center='{unmask_sigma_inf_center}'"
        )
        for cut in cutouts:
            fname = cut.get("filename", "")
            eid = cut.get("epoch_id", None)
            before = int(np.sum(~np.isfinite(np.asarray(cut["sigma"], dtype=float))))
            _unmask_sigma_inf_in_radius(cut, unmask_sigma_inf_radius)
            after = int(np.sum(~np.isfinite(np.asarray(cut["sigma"], dtype=float))))
            print(f"[step3] unmask sigma inf: {fname} epoch_id={eid} {before}->{after}")

    scene_wcs, scene_shape = _north_up_fixed_scene_wcs_from_cutouts(cutouts, sr)
    n_scene = int(scene_shape[0] * scene_shape[1])
    rots = [_frame_rotation_deg(c["raw_wcs"]) for c in cutouts]
    print(
        f"[step3] SR={sr}, N_BCD={n_bcd}, bcd_shape={tuple(cutouts[0]['data'].shape)}, "
        f"scene_shape={scene_shape}, n_scene={n_scene}"
    )
    print(f"[step3] selected frame rotations (deg): {[round(r, 2) for r in rots]}")
    print(
        f"[step3] CR core guard: percentile={float(cr_guard_pct):.2f}, "
        f"dilation_px={int(max(0, cr_guard_dilate))}, "
        f"radius_px={float(max(0.0, cr_guard_radius)):.2f}"
    )

    solver_cfg = {
        "SUPERSAMPLE_FACTOR": sr,
        "SCENE_WCS_STRICT_SUPERRES": True,
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": scene_ridge,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_FALLBACK_NEIGHBOR_SMOOTHNESS": 0.0,
        "FLOAT_TRANSIENT_POSITION": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": bool(project_then_prf),
        "PRF_GLS_LTWL_FULL_MAX_PIXELS": int(max(0, full_ltwl_cap)),
    }

    print(
        f"[step3] PRF convolution={'on' if use_prf else 'off'} "
        f"(PRF_ORDER_PROJECT_THEN_CONVOLVE={bool(project_then_prf)})"
    )
    print(f"[step3] SCENE_INDEPENDENT_RIDGE={scene_ridge:.3e}")
    with _temporary_config(solver_cfg):
        prf_ctx = nullcontext() if use_prf else identity_prf_operators_context()
        with prf_ctx:
            results = solver.run_gls_solve(
                cutouts,
                stars=[],
                star_initial_fluxes=[],
                gp_params={"ell": 1.0, "var": 1.0},
                regularization=(1.0, 1.0),
                deep_template=np.zeros(scene_shape, dtype=np.float64),
                template_wcs=scene_wcs,
                n_epochs=n_bcd,
            )

            metrics = fit_metrics.compute_fit_metrics(
                cutouts,
                results,
                [],
                np.zeros(0, dtype=float),
                center_ra_deg=float(config.TRANSIENT_RA),
                center_dec_deg=float(config.TRANSIENT_DEC),
                center_radius_px=3.0,
            )
            diag_pdf = write_native_fit_pdf(label, cutouts, results, out_dir)
            stack_pdf = write_stacked_residual_pdf(label, cutouts, results, out_dir)

    payload = {
        "label": label,
        "sr": sr,
        "n_bcd": n_bcd,
        "bcd_shape": [int(cutouts[0]["data"].shape[0]), int(cutouts[0]["data"].shape[1])],
        "scene_shape": [int(scene_shape[0]), int(scene_shape[1])],
        "scene_pixel_scale_arcsec": float(config.PIXEL_SCALE) / float(sr),
        "selected_rotations_deg": [float(r) for r in rots],
        "solver_config": solver_cfg,
        "gp_prior_enabled": bool(results.get("gp_prior_params", {}).get("enabled", True)),
        "metrics": {k: float(v) for k, v in metrics.items() if np.isscalar(v)},
        "diagnostic_pdf": diag_pdf,
        "stacked_pdf": stack_pdf,
        "results_keys": sorted([str(k) for k in results.keys()]),
        "gp_prior_params": results.get("gp_prior_params", {}),
    }
    out_json = os.path.join(out_dir, "step3_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[step3] wrote {out_json}")
    print(f"[step3] wrote {diag_pdf}")
    print(f"[step3] wrote {stack_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
