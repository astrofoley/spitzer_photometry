#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager, nullcontext

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, fit_metrics, native_fit_campaign, solver  # noqa: E402


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


@contextmanager
def _no_prf_context():
    orig_forward = solver.apply_spatially_varying_prf_to_scene
    orig_adjoint = solver.apply_spatially_varying_prf_adjoint
    orig_bundle = solver._get_prf_operator_bundle
    orig_apply_bundle = solver._apply_prf_operator_from_bundle
    orig_apply_adjoint_bundle = solver._apply_prf_adjoint_from_bundle
    try:
        solver.apply_spatially_varying_prf_to_scene = (
            lambda intrinsic_scene, scene_wcs, w_native, scene_shape, channel, is_full_array=False:  # noqa: ARG005
            np.asarray(intrinsic_scene, dtype=np.float64).reshape(scene_shape)
        )
        solver.apply_spatially_varying_prf_adjoint = (
            lambda y_scene, scene_wcs, w_native, scene_shape, channel, is_full_array=False:  # noqa: ARG005
            np.asarray(y_scene, dtype=np.float64).reshape(scene_shape).ravel()
        )
        solver._get_prf_operator_bundle = (
            lambda scene_wcs, w_native, scene_shape, channel, is_full_array: (None, None, None)  # noqa: ARG005
        )
        solver._apply_prf_operator_from_bundle = (
            lambda img, kernels, weights, wsum: np.asarray(img, dtype=np.float64)  # noqa: ARG005
        )
        solver._apply_prf_adjoint_from_bundle = (
            lambda y, kernels, weights, wsum: np.asarray(y, dtype=np.float64)  # noqa: ARG005
        )
        yield
    finally:
        solver.apply_spatially_varying_prf_to_scene = orig_forward
        solver.apply_spatially_varying_prf_adjoint = orig_adjoint
        solver._get_prf_operator_bundle = orig_bundle
        solver._apply_prf_operator_from_bundle = orig_apply_bundle
        solver._apply_prf_adjoint_from_bundle = orig_apply_adjoint_bundle


class _TemporaryConfig:
    def __init__(self, overrides):
        self.overrides = overrides
        self.old = {}

    def __enter__(self):
        self.old = {k: getattr(config, k) for k in self.overrides}
        for k, v in self.overrides.items():
            setattr(config, k, v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self.old.items():
            setattr(config, k, v)
        return False


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


def main() -> int:
    argv = list(sys.argv[1:])
    use_prf = "--with-prf" in argv
    fit_hyperparams = "--fit-hyperparams" in argv
    fixed_ell = None
    fixed_var = None
    tag_override = None
    label_override = None
    save_scene_npy = None
    fixed_scene_npy = None
    project_then_prf = "--project-then-prf" in argv
    for a in argv:
        if a.startswith("--fixed-ell="):
            fixed_ell = float(a.split("=", 1)[1])
        if a.startswith("--fixed-var="):
            fixed_var = float(a.split("=", 1)[1])
        if a.startswith("--tag="):
            tag_override = str(a.split("=", 1)[1]).strip()
        if a.startswith("--label="):
            label_override = str(a.split("=", 1)[1]).strip()
        if a.startswith("--save-gp-scene-npy="):
            save_scene_npy = str(a.split("=", 1)[1]).strip()
        if a.startswith("--fixed-gp-scene-npy="):
            fixed_scene_npy = str(a.split("=", 1)[1]).strip()

    if (fixed_ell is None) ^ (fixed_var is None):
        raise SystemExit("Provide both --fixed-ell and --fixed-var, or neither.")

    tag = tag_override or ("gp_delta_rotated_with_prf" if use_prf else "gp_delta_rotated_no_prf")
    label = label_override or ("ROTATED_DELTA_WITH_PRF" if use_prf else "ROTATED_DELTA_NO_PRF")
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    scene_shape = (40, 40)
    native_shape = (40, 40)
    theta_deg = 23.0
    scene_wcs = _scene_wcs(*scene_shape)
    native_wcs = _rot_wcs(native_shape[0], theta_deg)

    truth_scene = _inject_deltas(
        scene_shape,
        [
            (8, 10, 5.0),
            (12, 28, 8.0),
            (20, 20, 12.0),
            (28, 13, 7.0),
            (33, 31, 9.0),
        ],
    )
    background = 1.2
    sigma_level = 0.15

    cfg = {
        "FLOAT_TRANSIENT_POSITION": False,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "MAX_SCENE_PIXELS": 1000000,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.0,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": bool(project_then_prf),
    }

    ctx = _TemporaryConfig(cfg)
    with ctx, _no_prf_context():
        # Data must be identical across A/B/C comparisons:
        # always generate the rotated BCD from the intrinsic delta scene WITHOUT PRF.
        model_native = solver._apply_frame_forward_operator(
            truth_scene,
            scene_wcs,
            native_wcs,
            scene_shape,
            native_shape,
            str(getattr(config, "CHANNEL", "ch2")),
            is_full_array=True,
        )
        data = model_native + background
    sigma = np.full(native_shape, sigma_level, dtype=np.float64)
    cutouts = [
        {
            "data": np.asarray(data, dtype=np.float64),
            "sigma": np.asarray(sigma, dtype=np.float64),
            "wcs": native_wcs,
            "raw_wcs": native_wcs,
            "is_full_array": True,
            "mjd": 59000.0,
            "filename": "synthetic_rotated_delta_template.fits",
            "epoch_id": 0,
            "is_template": True,
        },
    ]

    prf_ctx = _no_prf_context() if not use_prf else nullcontext()
    with ctx, prf_ctx:
        def _solve_metrics(ell_in: float, var_in: float):
            results_loc = solver.run_gls_solve(
                cutouts,
                [],
                np.zeros(0, dtype=float),
                {"ell": float(ell_in), "var": float(var_in)},
                (float(ell_in), float(var_in)),
                np.zeros(scene_shape, dtype=np.float64),
                scene_wcs,
                1,
            )
            metrics_loc = fit_metrics.compute_fit_metrics(
                cutouts,
                results_loc,
                [],
                np.zeros(0, dtype=float),
                center_ra_deg=float(config.TRANSIENT_RA),
                center_dec_deg=float(config.TRANSIENT_DEC),
                center_radius_px=6.0,
            )
            return results_loc, metrics_loc

        ell = 0.5
        var = 0.05
        opt_result = None
        if fixed_ell is not None and fixed_var is not None:
            ell = float(fixed_ell)
            var = float(fixed_var)
        elif fit_hyperparams:
            def _obj(logp):
                e = float(10 ** logp[0])
                v = float(10 ** logp[1])
                _, m = _solve_metrics(e, v)
                return float(m.get("total_chi2", np.inf))

            x0 = np.array([np.log10(ell), np.log10(var)], dtype=float)
            bounds = [(-6.0, 2.0), (-8.0, 2.0)]
            opt_result = minimize(
                _obj,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxfev": 40, "maxiter": 40, "xtol": 1e-3, "ftol": 1e-3, "disp": False},
            )
            ell = float(10 ** opt_result.x[0])
            var = float(10 ** opt_result.x[1])

        if fixed_scene_npy:
            gp_scene = np.asarray(np.load(fixed_scene_npy), dtype=float).reshape(scene_shape)
            pred = solver._apply_frame_forward_operator(
                gp_scene,
                scene_wcs,
                native_wcs,
                scene_shape,
                native_shape,
                str(getattr(config, "CHANNEL", "ch2")),
                is_full_array=True,
            ) + background
            metrics = _compute_metrics_from_arrays(data, pred, sigma, native_wcs, center_radius_px=6.0)
        else:
            results = solver.run_gls_solve(
                cutouts,
                [],
                np.zeros(0, dtype=float),
                {"ell": float(ell), "var": float(var)},
                (float(ell), float(var)),
                np.zeros(scene_shape, dtype=np.float64),
                scene_wcs,
                1,
            )
            pred = solver.predict_cutout_model(
                results,
                cutouts,
                [],
                np.zeros(0, dtype=float),
                0,
                include_gp=True,
                include_transient=False,
                include_stars=False,
                include_host=False,
                include_nuclear_point=False,
            )
            metrics = fit_metrics.compute_fit_metrics(
                cutouts,
                results,
                [],
                np.zeros(0, dtype=float),
                center_ra_deg=float(config.TRANSIENT_RA),
                center_dec_deg=float(config.TRANSIENT_DEC),
                center_radius_px=6.0,
            )
            gp_scene = np.asarray(results.get("gp_scene", results["model_scene"]), dtype=float)
            if save_scene_npy:
                np.save(save_scene_npy, gp_scene)
        gp_scene_hi = _superres(gp_scene, upsample=40) + background
        truth_scene_hi = _superres(truth_scene, upsample=40) + background
        final_model_bcd = np.asarray(pred, dtype=float)
        final_truth_bcd = np.asarray(data, dtype=float)
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
        if use_prf:
            title += " + PRF)"
        else:
            title += ", no PRF)"
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
        fig.suptitle(
            f"Synthetic rotated delta diagnostics: {label}, theta={theta_deg:.1f} deg",
        )
        plt.tight_layout()
        out_pdf = os.path.join(out_dir, f"NATIVE_FIT_DIAGNOSTIC_{label}.pdf")
        fig.savefig(out_pdf, dpi=160)
        plt.close(fig)

    summary = {
        "theta_deg": theta_deg,
        "use_prf": bool(use_prf),
        "fit_hyperparams": bool(fit_hyperparams),
        "fixed_params_used": bool(fixed_ell is not None and fixed_var is not None),
        "fixed_scene_used": bool(fixed_scene_npy is not None),
        "project_then_prf_order": bool(project_then_prf),
        "ell": ell,
        "var": var,
        "background": background,
        "sigma_level": sigma_level,
        "metrics": metrics,
        "artifacts": {"pdf": out_pdf},
    }
    if opt_result is not None:
        summary["optimizer"] = {
            "method": "Powell",
            "success": bool(opt_result.success),
            "message": str(opt_result.message),
            "nfev": int(opt_result.nfev),
            "nit": int(getattr(opt_result, "nit", -1)),
            "x_log10": [float(x) for x in np.asarray(opt_result.x).ravel()],
        }
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_json}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
