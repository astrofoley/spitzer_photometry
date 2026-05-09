#!/usr/bin/env python3
"""Real N=1 fit with log-space objective on linear forward model."""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, solver  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _analysis_mask,
    _reindex_epochs,
    _resid_limits,
    _temporary_config,
    _valid_mask,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
)


def _chi2_for_params_log(cut_linear, floor, scene_wcs, scene_shape, ell, var):
    res = solver.run_gls_solve(
        [cut_linear],
        [],
        np.zeros(0, dtype=float),
        {"ell": float(ell), "var": float(var)},
        (float(ell), float(var)),
        np.zeros(scene_shape),
        scene_wcs,
        1,
    )
    pred_linear = np.asarray(
        solver.predict_cutout_model(
            res,
            [cut_linear],
            [],
            [],
            0,
            include_gp=True,
            include_transient=False,
            include_stars=False,
            include_host=False,
            include_nuclear_point=False,
        ),
        dtype=float,
    )
    data = np.asarray(cut_linear["data"], dtype=float)
    sigma = np.asarray(cut_linear["sigma"], dtype=float)
    mask = _analysis_mask(cut_linear)
    data_clip = np.clip(data, float(floor), None)
    pred_clip = np.clip(pred_linear, float(floor), None)
    data_log = np.log(data_clip)
    pred_log = np.log(pred_clip)
    sigma_log = np.full_like(sigma, np.inf, dtype=float)
    pos = mask & np.isfinite(sigma) & (sigma > 0) & np.isfinite(data_clip) & (data_clip > 0)
    sigma_log[pos] = np.clip(sigma[pos] / data_clip[pos], 1e-12, 1e12)
    w = np.zeros_like(data_log, dtype=float)
    good = mask & np.isfinite(sigma_log) & (sigma_log > 0)
    w[good] = 1.0 / np.clip(sigma_log[good], 1e-12, None) ** 2
    resid_log = data_log - pred_log
    chi2 = float(np.sum(w * resid_log * resid_log))
    ndof = int(np.sum(good))
    return chi2, ndof, res, pred_log, pred_linear


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_bg_gp_hyperopt_single_image_logflux")
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cut = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])[0]
    apply_native_cutout_cr_mask(cut)
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])

    data = np.asarray(cut["data"], dtype=float)
    sigma = np.asarray(cut["sigma"], dtype=float)
    vm = _analysis_mask(cut)
    pos = vm & np.isfinite(data) & (data > 0) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(pos):
        raise SystemExit("No positive valid pixels for log transform.")
    floor = float(np.nanpercentile(data[pos], 1.0))
    floor = max(floor, 1e-12)
    data_clip = np.clip(data, floor, None)

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    eval_rows: list[dict] = []
    best = {"chi2": np.inf, "ell": None, "var": None, "res": None, "pred_log": None, "pred_linear": None, "ndof": 0}

    def _obj(theta):
        ln_ell, ln_var = float(theta[0]), float(theta[1])
        ell = float(np.exp(ln_ell))
        var = float(np.exp(ln_var))
        chi2, ndof, res, pred_log, pred_linear = _chi2_for_params_log(cut, floor, scene_wcs, scene_shape, ell, var)
        row = {"ell": ell, "var": var, "chi2": chi2, "ndof": ndof, "red_chi2": chi2 / max(ndof, 1)}
        eval_rows.append(row)
        if chi2 < float(best["chi2"]):
            best.update({"chi2": chi2, "ell": ell, "var": var, "res": res, "pred_log": pred_log, "pred_linear": pred_linear, "ndof": ndof})
        print(
            "eval {:03d}: ell={:.6g} var={:.6g} log-chi2={:.6f} red={:.6f}".format(
                len(eval_rows), ell, var, chi2, chi2 / max(ndof, 1),
            ),
        )
        sys.stdout.flush()
        return chi2

    x0 = np.array([np.log(20.0), np.log(1e-2)], dtype=float)
    with _temporary_config(overrides):
        opt = minimize(
            _obj,
            x0,
            method="Powell",
            bounds=[(np.log(1e-5), np.log(5e1)), (np.log(1e-10), np.log(1e3))],
            options={"maxiter": 40, "maxfev": 10, "xtol": 1e-3, "ftol": 1e-6},
        )

    if best["res"] is None or best["pred_log"] is None or best["pred_linear"] is None:
        raise SystemExit("No valid optimization result.")

    pred_log = np.asarray(best["pred_log"], dtype=float)
    pred_linearized = np.asarray(best["pred_linear"], dtype=float)
    resid_lin = data - pred_linearized
    resid_log = np.log(np.clip(data, floor, None)) - pred_log
    vm_valid = _valid_mask(cut)
    dlim = np.nanpercentile(data[vm], [1, 99]) if np.any(vm) else [np.nanmin(data), np.nanmax(data)]
    lv0, lv1 = _resid_limits(np.where(vm, resid_lin, 0.0))
    gv0, gv1 = _resid_limits(np.where(vm, resid_log, 0.0))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = ax[0].imshow(np.where(vm_valid, data, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[0].set_title("Data")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(np.where(vm_valid, pred_linearized, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[1].set_title("Best model (exp back-transform)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(np.where(vm, resid_lin, 0.0), origin="lower", cmap="RdBu_r", vmin=lv0, vmax=lv1)
    ax[2].set_title("Residual in linear units")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    for a in ax:
        a.axis("off")
    fig.suptitle(
        "Log-flux fit best: ell={:.6g}, var={:.6g}, reduced log-chi2={:.6f}".format(
            float(best["ell"]), float(best["var"]), float(best["chi2"]) / max(int(best["ndof"]), 1),
        ),
    )
    plt.tight_layout()
    out_png = os.path.join(out_dir, "N1_BG_GP_HYPEROPT_LOGFLUX_MODEL_RESID.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    trace_png = os.path.join(out_dir, "N1_BG_GP_HYPEROPT_LOGFLUX_TRACE.png")
    if eval_rows:
        idx = np.arange(1, len(eval_rows) + 1)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        ax[0].plot(idx, [r["ell"] for r in eval_rows], marker="o", ms=2)
        ax[0].set_yscale("log")
        ax[0].set_title("ell")
        ax[1].plot(idx, [r["var"] for r in eval_rows], marker="o", ms=2)
        ax[1].set_yscale("log")
        ax[1].set_title("var")
        ax[2].plot(idx, [r["red_chi2"] for r in eval_rows], marker="o", ms=2)
        ax[2].set_title("reduced log-chi2")
        for a in ax:
            a.set_xlabel("evaluation")
        plt.tight_layout()
        fig.savefig(trace_png, dpi=170)
        plt.close(fig)

    out_json = os.path.join(out_dir, "n1_bg_gp_hyperopt_logflux_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimizer": {
                    "method": "Powell",
                    "success": bool(opt.success),
                    "message": str(opt.message),
                    "nfev": int(opt.nfev),
                    "nit": int(getattr(opt, "nit", -1)),
                },
                "log_transform": {"floor": floor, "n_positive_valid_pixels": int(np.sum(pos))},
                "best_fit": {
                    "ell": float(best["ell"]),
                    "var": float(best["var"]),
                    "log_chi2": float(best["chi2"]),
                    "ndof": int(best["ndof"]),
                    "reduced_log_chi2": float(best["chi2"]) / max(int(best["ndof"]), 1),
                    "linear_resid_rms": float(np.sqrt(np.mean((np.where(vm, resid_lin, 0.0)) ** 2))),
                    "log_resid_rms": float(np.sqrt(np.mean((np.where(vm, resid_log, 0.0)) ** 2))),
                },
                "n_evaluations": int(len(eval_rows)),
                "evaluations": eval_rows,
                "artifacts": {"model_png": out_png, "trace_png": trace_png},
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    print(f"Wrote {trace_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

