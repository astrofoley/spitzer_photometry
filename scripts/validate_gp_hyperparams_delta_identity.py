#!/usr/bin/env python3
"""Validate one-scale vs two-scale GP fits on synthetic identity-operator data."""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, fit_metrics, solver  # noqa: E402


def _make_wcs(ny: int, nx: int) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(config.PIXEL_SCALE) / float(config.SUPERSAMPLE_FACTOR) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w


def _inject_deltas(shape, points_flux):
    arr = np.zeros(shape, dtype=np.float64)
    for y, x, f in points_flux:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            arr[int(y), int(x)] += float(f)
    return arr


def _inject_center_gaussian(shape, sigma_pix: float, amp: float, cx: float, cy: float):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    r2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    return float(amp) * np.exp(-0.5 * r2 / max(float(sigma_pix) ** 2, 1e-12))


@contextmanager
def _identity_prf_context():
    orig_get_bundle = solver._get_prf_operator_bundle
    orig_apply_bundle = solver._apply_prf_operator_from_bundle
    orig_apply_adjoint = solver._apply_frame_adjoint_operator
    orig_proj_s2n = solver._project_scene_to_native
    orig_proj_n2s = solver._project_native_to_scene

    def _get_bundle_identity(scene_wcs, bcd_wcs, scene_shape, channel, is_full_array):  # noqa: ARG001
        return None, None, None

    def _apply_bundle_identity(scene, kernels, weights, wsum_b):  # noqa: ARG001
        return np.asarray(scene, dtype=np.float64)

    def _apply_adjoint_identity(native_resid, scene_wcs, bcd_wcs, scene_shape, channel, is_full_array=False):  # noqa: ARG001
        return np.asarray(native_resid, dtype=np.float64).reshape(scene_shape)

    def _project_scene_to_native_identity(scene_img, scene_wcs, native_wcs, native_shape):  # noqa: ARG001
        return np.asarray(scene_img, dtype=np.float64).reshape(native_shape)

    def _project_native_to_scene_identity(native_img, native_wcs, scene_wcs, scene_shape):  # noqa: ARG001
        return np.asarray(native_img, dtype=np.float64).reshape(scene_shape)

    try:
        solver._get_prf_operator_bundle = _get_bundle_identity
        solver._apply_prf_operator_from_bundle = _apply_bundle_identity
        solver._apply_frame_adjoint_operator = _apply_adjoint_identity
        solver._project_scene_to_native = _project_scene_to_native_identity
        solver._project_native_to_scene = _project_native_to_scene_identity
        yield
    finally:
        solver._get_prf_operator_bundle = orig_get_bundle
        solver._apply_prf_operator_from_bundle = orig_apply_bundle
        solver._apply_frame_adjoint_operator = orig_apply_adjoint
        solver._project_scene_to_native = orig_proj_s2n
        solver._project_native_to_scene = orig_proj_n2s


def _temporary_cfg(overrides):
    class _Tmp:
        def __enter__(self):
            self._old = {k: getattr(config, k) for k in overrides}
            for k, v in overrides.items():
                setattr(config, k, v)
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, v in self._old.items():
                setattr(config, k, v)
            return False

    return _Tmp()


def _solve_case(cutouts, scene_wcs, scene_shape, ell, var):
    return solver.run_gls_solve(
        cutouts,
        [],
        np.zeros(0, dtype=float),
        {"ell": float(ell), "var": float(var)},
        (float(ell), float(var)),
        np.zeros(scene_shape, dtype=np.float64),
        scene_wcs,
        len(cutouts),
    )


def _solve_case_two_scale(cutouts, scene_wcs, scene_shape, ell1, var1, ell2, var2):
    return solver.run_gls_solve(
        cutouts,
        [],
        np.zeros(0, dtype=float),
        {"ell": float(ell1), "var": float(var1), "ell2": float(ell2), "var2": float(var2)},
        (float(ell1), float(var1)),
        np.zeros(scene_shape, dtype=np.float64),
        scene_wcs,
        len(cutouts),
    )


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "gp_delta_identity_validation")
    os.makedirs(out_dir, exist_ok=True)

    shape = (40, 40)
    w = _make_wcs(*shape)
    # Several sharp deltas; this should favor short-length prior over long smooth prior.
    delta_only = _inject_deltas(
        shape,
        [
            (8, 10, 5.0),
            (12, 28, 8.0),
            (20, 20, 12.0),
            (28, 13, 7.0),
            (33, 31, 9.0),
        ],
    )
    bkg_level = 0.1 * 12.0
    # Broad 2D Gaussian centered on the central delta source.
    gauss_sigma = 0.5 * shape[0]
    gauss_amp = 4.0
    gauss_center = (20.0, 20.0)
    broad_gauss = _inject_center_gaussian(shape, gauss_sigma, gauss_amp, gauss_center[1], gauss_center[0])
    truth = delta_only + broad_gauss + bkg_level
    sigma = np.full(shape, 0.15, dtype=np.float64)
    data = truth.copy()
    cutouts = [
        {
            "data": np.asarray(data, dtype=np.float64),
            "sigma": np.asarray(sigma, dtype=np.float64),
            "wcs": w,
            "raw_wcs": w,
            "is_full_array": True,
            "mjd": 59000.0,
            "filename": "synthetic_delta_template.fits",
            "epoch_id": 0,
            "is_template": True,
        },
    ]

    cfg = {
        "FLOAT_TRANSIENT_POSITION": False,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "MAX_SCENE_PIXELS": 1000000,  # ensure dense GP path (ell actually matters)
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.0,
    }

    one_scale_grid = [
        (0.2, 0.05),
        (0.5, 0.05),
        (1.0, 0.05),
        (2.0, 0.05),
        (5.0, 0.05),
        (10.0, 0.05),
        (20.0, 0.05),
        (30.0, 0.05),
    ]
    two_scale_grid = []
    for ell1, ell2 in ((0.2, 16.0), (0.5, 20.0), (1.0, 30.0)):
        for var1 in (0.02, 0.05, 0.2):
            for var2 in (0.02, 0.05, 0.2):
                two_scale_grid.append((ell1, var1, ell2, var2))

    with _temporary_cfg(cfg), _identity_prf_context():
        one_scale_rows = []
        for ell, var in one_scale_grid:
            res = _solve_case(cutouts, w, shape, ell, var)
            met = fit_metrics.compute_fit_metrics(
                cutouts, res, [], np.zeros(0), center_ra_deg=float(config.TRANSIENT_RA), center_dec_deg=float(config.TRANSIENT_DEC), center_radius_px=6.0
            )
            one_scale_rows.append({"ell": ell, "var": var, "results": res, "metrics": met})
        best_one = min(one_scale_rows, key=lambda r: float(r["metrics"]["total_reduced_chi2"]))

        two_scale_rows = []
        for ell1, var1, ell2, var2 in two_scale_grid:
            res = _solve_case_two_scale(cutouts, w, shape, ell1, var1, ell2, var2)
            met = fit_metrics.compute_fit_metrics(
                cutouts, res, [], np.zeros(0), center_ra_deg=float(config.TRANSIENT_RA), center_dec_deg=float(config.TRANSIENT_DEC), center_radius_px=6.0
            )
            two_scale_rows.append(
                {"ell1": ell1, "var1": var1, "ell2": ell2, "var2": var2, "results": res, "metrics": met},
            )
        best_two = min(two_scale_rows, key=lambda r: float(r["metrics"]["total_reduced_chi2"]))

        model_one = solver.predict_cutout_model(
            best_one["results"],
            cutouts,
            [],
            np.zeros(0),
            0,
            include_gp=True,
            include_transient=False,
            include_stars=False,
            include_host=False,
            include_nuclear_point=False,
        )
        model_two = solver.predict_cutout_model(
            best_two["results"],
            cutouts,
            [],
            np.zeros(0),
            0,
            include_gp=True,
            include_transient=False,
            include_stars=False,
            include_host=False,
            include_nuclear_point=False,
        )
    resid_one = np.asarray(data, dtype=float) - np.asarray(model_one, dtype=float)
    resid_two = np.asarray(data, dtype=float) - np.asarray(model_two, dtype=float)
    v_data = np.nanpercentile(data, [1, 99])
    v_truth = np.nanpercentile(truth, [1, 99.5])
    v_mod = np.nanpercentile(np.hstack([np.ravel(model_one), np.ravel(model_two)]), [1, 99])
    rv = np.nanpercentile(np.hstack([np.ravel(resid_one), np.ravel(resid_two)]), [1, 99])
    rlim = max(abs(float(rv[0])), abs(float(rv[1])), 1e-6)
    per_ell_plots = {}
    for label, subtitle, model_arr, resid_arr in (
        ("one_scale", f"one-scale GP (ell={best_one['ell']}, var={best_one['var']})", model_one, resid_one),
        (
            "two_scale",
            "two-scale GP (ell1={:.2f}, var1={:.2g}, ell2={:.2f}, var2={:.2g})".format(
                best_two["ell1"], best_two["var1"], best_two["ell2"], best_two["var2"],
            ),
            model_two,
            resid_two,
        ),
    ):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        im = ax[0].imshow(data, origin="lower", cmap="magma", vmin=float(v_data[0]), vmax=float(v_data[1]))
        ax[0].set_title("Input data (no noise)")
        plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        im = ax[1].imshow(model_arr, origin="lower", cmap="magma", vmin=float(v_mod[0]), vmax=float(v_mod[1]))
        ax[1].set_title(subtitle)
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        im = ax[2].imshow(resid_arr, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
        ax[2].set_title(f"Residual ({label})")
        plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        for a in ax.ravel():
            a.axis("off")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"delta_identity_input_model_residual_{label}_ell.png")
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        per_ell_plots[label] = out_png

    summary = {
        "setup": {
            "shape": list(shape),
            "n_delta_sources": 5,
            "identity_prf": True,
            "uniform_background_level": bkg_level,
            "broad_gaussian": {
                "sigma_pix": gauss_sigma,
                "amplitude": gauss_amp,
                "center_yx": [gauss_center[0], gauss_center[1]],
            },
            "one_scale_grid": [{"ell": r[0], "var": r[1]} for r in one_scale_grid],
            "two_scale_grid_size": len(two_scale_grid),
        },
        "best_one_scale": {
            "ell": best_one["ell"],
            "var": best_one["var"],
            "metrics": best_one["metrics"],
        },
        "best_two_scale": {
            "ell1": best_two["ell1"],
            "var1": best_two["var1"],
            "ell2": best_two["ell2"],
            "var2": best_two["var2"],
            "metrics": best_two["metrics"],
        },
        "delta": {
            "total_reduced_chi2": float(best_two["metrics"]["total_reduced_chi2"] - best_one["metrics"]["total_reduced_chi2"]),
            "center_reduced_chi2": float(best_two["metrics"]["center_reduced_chi2"] - best_one["metrics"]["center_reduced_chi2"]),
        },
        "artifacts": {
            "input_model_residual_per_ell_png": per_ell_plots,
        },
    }
    out_json = os.path.join(out_dir, "delta_identity_validation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_json}")
    print(
        "total_reduced_chi2: one_scale={:.6f} two_scale={:.6f} delta={:.6f}".format(
            float(best_one["metrics"]["total_reduced_chi2"]),
            float(best_two["metrics"]["total_reduced_chi2"]),
            float(summary["delta"]["total_reduced_chi2"]),
        ),
    )
    print(
        "center_reduced_chi2: one_scale={:.6f} two_scale={:.6f} delta={:.6f}".format(
            float(best_one["metrics"]["center_reduced_chi2"]),
            float(best_two["metrics"]["center_reduced_chi2"]),
            float(summary["delta"]["center_reduced_chi2"]),
        ),
    )
    for k, v in per_ell_plots.items():
        print(f"Wrote {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
