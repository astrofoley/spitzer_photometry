#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, native_fit_campaign, solver  # noqa: E402
from src.prf_identity_context import identity_prf_operators_context  # noqa: E402


def _scene_wcs(ny: int, nx: int) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(config.PIXEL_SCALE) / float(config.SUPERSAMPLE_FACTOR) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w


def _rot_wcs(n_pix: int, theta_deg: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(config.PIXEL_SCALE) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    w.wcs.pc = np.array([[c, -s], [s, c]])
    return w


def _inject_deltas(shape, points_flux):
    arr = np.zeros(shape, dtype=np.float64)
    for y, x, f in points_flux:
        if 0 <= int(y) < shape[0] and 0 <= int(x) < shape[1]:
            arr[int(y), int(x)] += float(f)
    return arr


def _superres(arr: np.ndarray, upsample: int = 40) -> np.ndarray:
    up = max(2, int(upsample))
    a = np.asarray(arr, dtype=float)
    return np.repeat(np.repeat(a, up, axis=0), up, axis=1)


def _resid_limits(arr: np.ndarray):
    v = np.asarray(arr, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size < 4:
        return -1.0, 1.0
    lo, hi = np.percentile(vv, [1.0, 99.0])
    lim = max(abs(float(lo)), abs(float(hi)), 1e-12)
    return -lim, lim


def _compute_metrics_from_arrays(data_bcd, model_bcd, sigma_bcd, native_wcs, center_radius_px=6.0):
    d = np.asarray(data_bcd, dtype=float)
    m = np.asarray(model_bcd, dtype=float)
    s = np.asarray(sigma_bcd, dtype=float)
    resid = d - m
    w = 1.0 / np.clip(s, 1e-12, None) ** 2
    total_chi2 = float(np.sum((resid ** 2) * w))
    total_ndof = int(d.size)
    total_reduced = float(total_chi2 / max(total_ndof, 1))

    cx, cy = native_wcs.world_to_pixel_values(float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC))
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]].astype(float)
    rr = np.hypot(xx - float(cx), yy - float(cy))
    cmask = rr <= float(center_radius_px)
    c_resid = resid[cmask]
    c_w = w[cmask]
    c_data = d[cmask]
    center_chi2 = float(np.sum((c_resid ** 2) * c_w))
    center_ndof = int(np.sum(cmask))
    center_reduced = float(center_chi2 / max(center_ndof, 1))
    center_rmse = float(np.sqrt(np.mean(c_resid ** 2))) if c_resid.size else float("nan")
    center_poisson_proxy = float(np.sqrt(np.clip(np.nanmedian(c_data), 1e-12, None))) if c_data.size else float("nan")
    center_noise_ratio = float(center_rmse / max(center_poisson_proxy, 1e-30)) if np.isfinite(center_poisson_proxy) else float("nan")
    center_dipole_mag = float(np.nanmedian(np.abs(c_resid))) if c_resid.size else float("nan")
    return {
        "total_chi2": total_chi2,
        "total_ndof": total_ndof,
        "total_reduced_chi2": total_reduced,
        "center_chi2": center_chi2,
        "center_ndof": center_ndof,
        "center_reduced_chi2": center_reduced,
        "center_rmse": center_rmse,
        "center_poisson_proxy": center_poisson_proxy,
        "center_noise_ratio": center_noise_ratio,
        "center_dipole_mag_pix_median": center_dipole_mag,
    }


def _write_9panel(path, label, data_bcd, pred_bcd, truth_scene_hi, gp_scene_hi, use_prf):
    final_truth_bcd = np.asarray(data_bcd, dtype=float)
    final_model_bcd = np.asarray(pred_bcd, dtype=float)
    final_resid_bcd = final_truth_bcd - final_model_bcd
    flux_norm_bcd = native_fit_campaign._log_norm_bcd_minmax(final_truth_bcd)
    v0, v1 = _resid_limits(final_resid_bcd)
    s0, s1 = _resid_limits((truth_scene_hi - gp_scene_hi))

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    ax = axes.ravel()

    im0 = ax[0].imshow(final_truth_bcd, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
    ax[0].set_title("BCD (synthetic, rotated)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(gp_scene_hi, origin="lower", cmap="gray", interpolation="nearest")
    ax[1].set_title("N-up super-res: GP scene (+BG)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(truth_scene_hi - gp_scene_hi, origin="lower", cmap="RdBu_r", vmin=s0, vmax=s1, interpolation="nearest")
    ax[2].set_title("N-up scene residual")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(truth_scene_hi, origin="lower", cmap="gray", interpolation="nearest")
    ax[3].set_title("N-up super-res: truth scene (+BG)")
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    ax[4].axis("off")

    im5 = ax[5].imshow(final_model_bcd, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
    title = "Model on BCD (rotated + projected"
    title += " + PRF)" if use_prf else ", no PRF)"
    ax[5].set_title(title)
    plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

    im6 = ax[6].imshow(final_truth_bcd, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
    ax[6].set_title("BCD truth model")
    plt.colorbar(im6, ax=ax[6], fraction=0.046, pad=0.04)

    im7 = ax[7].imshow(final_model_bcd, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
    ax[7].set_title("Final comparison model")
    plt.colorbar(im7, ax=ax[7], fraction=0.046, pad=0.04)

    im8 = ax[8].imshow(final_resid_bcd, origin="lower", cmap="RdBu_r", vmin=v0, vmax=v1, interpolation="nearest")
    ax[8].set_title("Residual (data - model)")
    plt.colorbar(im8, ax=ax[8], fraction=0.046, pad=0.04)

    for j in range(9):
        if j != 4:
            ax[j].axis("off")
    fig.suptitle(label)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _one_step_diagnostics(data_bcd, sigma_bcd, scene_wcs, native_wcs, scene_shape, channel, gp_scene, use_prf):
    n = int(scene_shape[0] * scene_shape[1])
    x0 = np.asarray(gp_scene, dtype=float).ravel()
    prf_ctx = nullcontext() if use_prf else identity_prf_operators_context()
    with prf_ctx:
        # Model and residual in BCD.
        model_bcd = solver._apply_frame_forward_operator(
            x0.reshape(scene_shape),
            scene_wcs,
            native_wcs,
            scene_shape,
            data_bcd.shape,
            channel,
            is_full_array=True,
        ) + 1.2
        r_bcd = np.asarray(data_bcd, float).ravel() - np.asarray(model_bcd, float).ravel()
        w = 1.0 / np.clip(np.asarray(sigma_bcd, float).ravel(), 1e-12, None) ** 2
        # Data gradient: g_data = -F^T W r
        g_data = -solver._apply_frame_adjoint_operator(
            (w * r_bcd).reshape(data_bcd.shape),
            scene_wcs,
            native_wcs,
            scene_shape,
            channel,
            is_full_array=True,
        ).ravel()

    # Prior gradient: g_prior = Q^{-1} x
    n_scene = int(scene_shape[0] * scene_shape[1])
    Qinv = solver.gp_model.build_scene_prior_inverse(
        n_scene,
        0.3797955287665477,
        99.84863766008439,
        scene_shape,
    )
    g_prior = np.asarray(Qinv @ x0, dtype=float).ravel()
    g_total = g_data + g_prior
    alpha = 1.0 / max(np.max(np.abs(g_total)), 1e-12)
    x1 = x0 - alpha * g_total
    hit_nonneg = int(np.sum(x1 < 0))
    x1 = np.maximum(x1, 0.0)
    step = x1 - x0
    return {
        "grad_data_rms": float(np.sqrt(np.mean(g_data ** 2))),
        "grad_prior_rms": float(np.sqrt(np.mean(g_prior ** 2))),
        "grad_total_rms": float(np.sqrt(np.mean(g_total ** 2))),
        "prior_to_data_grad_rms_ratio": float(np.sqrt(np.mean(g_prior ** 2)) / max(np.sqrt(np.mean(g_data ** 2)), 1e-30)),
        "step_rms": float(np.sqrt(np.mean(step ** 2))),
        "step_l2": float(np.linalg.norm(step)),
        "n_clipped_to_nonnegative": hit_nonneg,
    }


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "gp_delta_pixel_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    scene_shape = (40, 40)
    native_shape = (40, 40)
    theta_deg = 23.0
    scene_wcs = _scene_wcs(*scene_shape)
    native_wcs = _rot_wcs(native_shape[0], theta_deg)
    channel = str(getattr(config, "CHANNEL", "ch2"))
    background = 1.2
    sigma_level = 0.15
    ell = 0.3797955287665477
    var = 99.84863766008439

    truth_scene = _inject_deltas(
        scene_shape,
        [(8, 10, 5.0), (12, 28, 8.0), (20, 20, 12.0), (28, 13, 7.0), (33, 31, 9.0)],
    )

    with identity_prf_operators_context():
        data_native = solver._apply_frame_forward_operator(
            truth_scene,
            scene_wcs,
            native_wcs,
            scene_shape,
            native_shape,
            channel,
            is_full_array=True,
        ) + background

    sigma_native = np.full(native_shape, sigma_level, dtype=float)

    # Solve no-PRF and PRF with identical fixed hyperparams.
    cutout = [{
        "data": np.asarray(data_native, float),
        "sigma": np.asarray(sigma_native, float),
        "wcs": native_wcs,
        "raw_wcs": native_wcs,
        "is_full_array": True,
        "mjd": 59000.0,
        "filename": "synthetic_rotated_delta_template.fits",
        "epoch_id": 0,
        "is_template": True,
    }]

    def run_solve(use_prf: bool):
        ctx = nullcontext() if use_prf else identity_prf_operators_context()
        with ctx:
            res = solver.run_gls_solve(
                cutout,
                [],
                np.zeros(0, dtype=float),
                {"ell": ell, "var": var},
                (ell, var),
                np.zeros(scene_shape, dtype=float),
                scene_wcs,
                1,
            )
            pred = solver.predict_cutout_model(
                res, cutout, [], np.zeros(0, dtype=float), 0,
                include_gp=True, include_transient=False, include_stars=False, include_host=False, include_nuclear_point=False,
            )
        scene = np.asarray(res.get("gp_scene", res["model_scene"]), dtype=float)
        metrics = _compute_metrics_from_arrays(data_native, pred, sigma_native, native_wcs, center_radius_px=6.0)
        return scene, np.asarray(pred, float), metrics

    scene_no, pred_no, met_no = run_solve(False)
    scene_prf, pred_prf, met_prf = run_solve(True)

    # Case #4: take best-fit scene from #1 and convolve with PRF.
    scene_case1_path = os.path.join(config.DIAGNOSTIC_DIR, "gp_delta_rotated_case1_no_prf_fit", "gp_scene_bestfit.npy")
    scene_case1 = np.asarray(np.load(scene_case1_path), dtype=float)
    pred_case4 = solver._apply_frame_forward_operator(
        scene_case1, scene_wcs, native_wcs, scene_shape, native_shape, channel, is_full_array=True,
    ) + background
    met_case4 = _compute_metrics_from_arrays(data_native, pred_case4, sigma_native, native_wcs, center_radius_px=6.0)

    # One-step term diagnostics.
    diag_no = _one_step_diagnostics(data_native, sigma_native, scene_wcs, native_wcs, scene_shape, channel, scene_no, False)
    diag_prf = _one_step_diagnostics(data_native, sigma_native, scene_wcs, native_wcs, scene_shape, channel, scene_prf, True)

    truth_scene_hi = _superres(truth_scene, upsample=40) + background
    _write_9panel(
        os.path.join(out_dir, "NATIVE_FIT_DIAGNOSTIC_FIXEDPARAMS_NO_PRF.pdf"),
        "Fixed ell/var, no PRF",
        data_native, pred_no, truth_scene_hi, _superres(scene_no, 40) + background, False,
    )
    _write_9panel(
        os.path.join(out_dir, "NATIVE_FIT_DIAGNOSTIC_FIXEDPARAMS_WITH_PRF.pdf"),
        "Fixed ell/var, with PRF",
        data_native, pred_prf, truth_scene_hi, _superres(scene_prf, 40) + background, True,
    )
    _write_9panel(
        os.path.join(out_dir, "NATIVE_FIT_DIAGNOSTIC_CASE4_PRF_FROM_CASE1_SCENE.pdf"),
        "Case4: PRF forward of case1 scene",
        data_native, pred_case4, truth_scene_hi, _superres(scene_case1, 40) + background, True,
    )

    summary = {
        "fixed_hyperparams": {"ell": ell, "var": var},
        "data_definition": "rotated BCD from intrinsic delta scene, no PRF, +BG",
        "metrics_fixedparam_no_prf": met_no,
        "metrics_fixedparam_with_prf": met_prf,
        "metrics_case4_prf_from_case1_scene": met_case4,
        "pixel_diagnostics_no_prf": diag_no,
        "pixel_diagnostics_with_prf": diag_prf,
        "artifacts": {
            "pdf_no_prf": os.path.join(out_dir, "NATIVE_FIT_DIAGNOSTIC_FIXEDPARAMS_NO_PRF.pdf"),
            "pdf_with_prf": os.path.join(out_dir, "NATIVE_FIT_DIAGNOSTIC_FIXEDPARAMS_WITH_PRF.pdf"),
            "pdf_case4": os.path.join(out_dir, "NATIVE_FIT_DIAGNOSTIC_CASE4_PRF_FROM_CASE1_SCENE.pdf"),
        },
    }
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_json}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
