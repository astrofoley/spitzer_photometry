#!/usr/bin/env python3
"""Experimental N=1 run comparing linear-flux vs log-flux fitting."""
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
    _analysis_mask,
    _reindex_epochs,
    _resid_limits,
    _temporary_config,
    _valid_mask,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
)


def _run_single(cutout, scene_wcs, scene_shape, stars, init_star_fluxes, ell, var):
    res = solver.run_gls_solve(
        [cutout],
        stars,
        init_star_fluxes,
        {"ell": float(ell), "var": float(var)},
        (float(ell), float(var)),
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
    return res, pred


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_logflux_experiment")
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cut = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])[0]
    apply_native_cutout_cr_mask(cut)
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
    nr = float(real_case["centers"]["nuc_ra"])
    nd = float(real_case["centers"]["nuc_dec"])

    ell_opt, var_opt = gp_model.optimize_hyperparameters([cut])
    ell_opt = float(ell_opt)
    var_opt = float(max(float(var_opt), 1e-30))

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    with _temporary_config(overrides):
        # Baseline linear-flux fit
        res_lin, pred_lin = _run_single(cut, scene_wcs, scene_shape, stars, init_star_fluxes, ell_opt, var_opt)
        met_lin = fit_metrics.compute_fit_metrics(
            [cut],
            res_lin,
            stars,
            res_lin.get("star_fluxes", init_star_fluxes),
            center_ra_deg=nr,
            center_dec_deg=nd,
            center_radius_px=3.0,
        )

        # Log-flux transformed fit (experimental approximation).
        cut_log = dict(cut)
        data = np.asarray(cut["data"], dtype=float)
        sig = np.asarray(cut["sigma"], dtype=float)
        vm = _analysis_mask(cut)
        pos = vm & np.isfinite(data) & (data > 0) & np.isfinite(sig) & (sig > 0)
        if not np.any(pos):
            raise SystemExit("No positive valid pixels for log-flux transform.")
        floor = float(np.nanpercentile(data[pos], 1.0))
        floor = max(floor, 1e-12)
        data_clipped = np.clip(data, floor, None)
        data_log = np.log(data_clipped)
        sigma_log = np.full_like(sig, np.inf, dtype=float)
        sigma_log[pos] = np.clip(sig[pos] / data_clipped[pos], 1e-12, 1e12)
        cut_log["data"] = data_log
        cut_log["sigma"] = sigma_log

        res_log, pred_log = _run_single(cut_log, scene_wcs, scene_shape, stars, init_star_fluxes, ell_opt, var_opt)
        # Back-transform model to linear flux for qualitative comparison.
        pred_log_linearized = np.exp(pred_log)
        resid_log_linearized = data - pred_log_linearized
        # Also compute residual in log domain against transformed data.
        resid_log_domain = data_log - pred_log

    # Plots
    vm_valid = _valid_mask(cut)
    dlim = np.nanpercentile(data[vm], [1, 99]) if np.any(vm) else [np.nanmin(data), np.nanmax(data)]
    rv0, rv1 = _resid_limits(np.where(vm, data - pred_lin, 0.0))
    lv0, lv1 = _resid_limits(np.where(vm, resid_log_linearized, 0.0))
    logrv0, logrv1 = _resid_limits(np.where(vm, resid_log_domain, 0.0))

    out_png = os.path.join(out_dir, "N1_LOGFLUX_EXPERIMENT.png")
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    im = ax[0, 0].imshow(np.where(vm_valid, pred_lin, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[0, 0].set_title("Linear fit model")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    im = ax[0, 1].imshow(np.where(vm, data - pred_lin, 0.0), origin="lower", cmap="RdBu_r", vmin=rv0, vmax=rv1)
    ax[0, 1].set_title("Linear residual (data-model)")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)
    im = ax[0, 2].imshow(np.where(vm_valid, data, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[0, 2].set_title("Data")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im = ax[1, 0].imshow(np.where(vm_valid, pred_log_linearized, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[1, 0].set_title("Log-fit model (exp back-transform)")
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
    im = ax[1, 1].imshow(np.where(vm, resid_log_linearized, 0.0), origin="lower", cmap="RdBu_r", vmin=lv0, vmax=lv1)
    ax[1, 1].set_title("Residual in linear units")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
    im = ax[1, 2].imshow(np.where(vm, resid_log_domain, 0.0), origin="lower", cmap="RdBu_r", vmin=logrv0, vmax=logrv1)
    ax[1, 2].set_title("Residual in log domain")
    plt.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    out_json = os.path.join(out_dir, "n1_logflux_experiment_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ell": ell_opt,
                "var": var_opt,
                "linear_metrics": {k: float(v) for k, v in met_lin.items() if isinstance(v, (int, float))},
                "log_transform": {
                    "floor": floor,
                    "n_positive_valid_pixels": int(np.sum(pos)),
                },
                "residual_stats": {
                    "linear_fit_resid_rms": float(np.sqrt(np.mean((np.where(vm, data - pred_lin, 0.0)) ** 2))),
                    "log_fit_resid_rms_linear_units": float(np.sqrt(np.mean((np.where(vm, resid_log_linearized, 0.0)) ** 2))),
                    "log_fit_resid_rms_log_units": float(np.sqrt(np.mean((np.where(vm, resid_log_domain, 0.0)) ** 2))),
                },
                "artifact_png": out_png,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

