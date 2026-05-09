#!/usr/bin/env python3
"""Real N=1 fit: background + single GP with optimized ell/var."""
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


def _chi2_for_params(cutout, scene_wcs, scene_shape, ell, var):
    res = solver.run_gls_solve(
        [cutout],
        [],
        np.zeros(0, dtype=float),
        {"ell": float(ell), "var": float(var)},
        (float(ell), float(var)),
        np.zeros(scene_shape),
        scene_wcs,
        1,
    )
    pred = np.asarray(
        solver.predict_cutout_model(
            res,
            [cutout],
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
    data = np.asarray(cutout["data"], dtype=float)
    sigma = np.asarray(cutout["sigma"], dtype=float)
    mask = _analysis_mask(cutout)
    w = np.zeros_like(data, dtype=float)
    good = mask & np.isfinite(sigma) & (sigma > 0)
    w[good] = 1.0 / np.clip(sigma[good], 1e-12, None) ** 2
    resid = data - pred
    chi2 = float(np.sum(w * resid * resid))
    ndof = int(np.sum(good))
    return chi2, ndof, res, pred


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_bg_gp_hyperopt_single_image")
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cut = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])[0]
    apply_native_cutout_cr_mask(cut)
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    eval_rows: list[dict] = []
    best = {"chi2": np.inf, "ell": None, "var": None, "res": None, "pred": None, "ndof": 0}

    def _obj(theta):
        ln_ell, ln_var = float(theta[0]), float(theta[1])
        ell = float(np.exp(ln_ell))
        var = float(np.exp(ln_var))
        chi2, ndof, res, pred = _chi2_for_params(cut, scene_wcs, scene_shape, ell, var)
        row = {"ell": ell, "var": var, "chi2": chi2, "ndof": ndof, "red_chi2": chi2 / max(ndof, 1)}
        eval_rows.append(row)
        if chi2 < float(best["chi2"]):
            best.update({"chi2": chi2, "ell": ell, "var": var, "res": res, "pred": pred, "ndof": ndof})
        print(
            "eval {:03d}: ell={:.6g} var={:.6g} chi2={:.6f} red={:.6f}".format(
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

    if best["res"] is None or best["pred"] is None:
        raise SystemExit("No valid optimization result.")

    data = np.asarray(cut["data"], dtype=float)
    pred = np.asarray(best["pred"], dtype=float)
    resid = data - pred
    vm_valid = _valid_mask(cut)
    vm_analysis = _analysis_mask(cut)
    dlim = np.nanpercentile(data[vm_analysis], [1, 99]) if np.any(vm_analysis) else [np.nanmin(data), np.nanmax(data)]
    rv0, rv1 = _resid_limits(np.where(vm_analysis, resid, 0.0))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = ax[0].imshow(np.where(vm_valid, data, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[0].set_title("Data")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(np.where(vm_valid, pred, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[1].set_title("Best-fit model (BG + GP)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(np.where(vm_analysis, resid, 0.0), origin="lower", cmap="RdBu_r", vmin=rv0, vmax=rv1)
    ax[2].set_title("Residual")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    for a in ax:
        a.axis("off")
    fig.suptitle(
        "Best fit: ell={:.6g}, var={:.6g}, red_chi2={:.6f}".format(
            float(best["ell"]), float(best["var"]), float(best["chi2"]) / max(int(best["ndof"]), 1),
        ),
    )
    plt.tight_layout()
    out_png = os.path.join(out_dir, "N1_BG_GP_HYPEROPT_MODEL_RESID.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    trace_png = os.path.join(out_dir, "N1_BG_GP_HYPEROPT_TRACE.png")
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
        ax[2].set_title("reduced chi2")
        for a in ax:
            a.set_xlabel("evaluation")
        plt.tight_layout()
        fig.savefig(trace_png, dpi=170)
        plt.close(fig)

    out_json = os.path.join(out_dir, "n1_bg_gp_hyperopt_summary.json")
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
                "best_fit": {
                    "ell": float(best["ell"]),
                    "var": float(best["var"]),
                    "chi2": float(best["chi2"]),
                    "ndof": int(best["ndof"]),
                    "reduced_chi2": float(best["chi2"]) / max(int(best["ndof"]), 1),
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

