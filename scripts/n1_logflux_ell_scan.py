#!/usr/bin/env python3
"""N=1 ell scan in log-flux space (single GP)."""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_logflux_ell_scan")
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

    # Build log-space cutout.
    data = np.asarray(cut["data"], dtype=float)
    sig = np.asarray(cut["sigma"], dtype=float)
    vm = _analysis_mask(cut)
    pos = vm & np.isfinite(data) & (data > 0) & np.isfinite(sig) & (sig > 0)
    if not np.any(pos):
        raise SystemExit("No positive valid pixels for log-space scan.")
    floor = float(np.nanpercentile(data[pos], 1.0))
    floor = max(floor, 1e-12)
    dclip = np.clip(data, floor, None)
    data_log = np.log(dclip)
    sigma_log = np.full_like(sig, np.inf, dtype=float)
    sigma_log[pos] = np.clip(sig[pos] / dclip[pos], 1e-12, 1e12)
    cut_log = dict(cut)
    cut_log["data"] = data_log
    cut_log["sigma"] = sigma_log

    _, var_opt = gp_model.optimize_hyperparameters([cut_log])
    var_opt = float(max(float(var_opt), 1e-30))
    ell_vals = [20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    rows = []
    models_log = []
    models_linear = []
    with _temporary_config(overrides):
        for ell in ell_vals:
            res = solver.run_gls_solve(
                [cut_log],
                stars,
                init_star_fluxes,
                {"ell": float(ell), "var": float(var_opt)},
                (float(ell), float(var_opt)),
                np.zeros(scene_shape),
                scene_wcs,
                1,
            )
            pred_log = np.asarray(
                solver.predict_cutout_model(
                    res,
                    [cut_log],
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
            pred_lin = np.exp(pred_log)
            models_log.append(pred_log.copy())
            models_linear.append(pred_lin.copy())

            # Chi2-style metrics in log domain (using transformed cutout).
            mlog = fit_metrics.compute_fit_metrics(
                [cut_log],
                res,
                stars,
                res.get("star_fluxes", init_star_fluxes),
                center_ra_deg=nr,
                center_dec_deg=nd,
                center_radius_px=3.0,
            )
            rows.append(
                {
                    "ell": float(ell),
                    "var": float(var_opt),
                    "log_domain_total_reduced_chi2": float(mlog.get("total_reduced_chi2", np.nan)),
                    "log_domain_center_reduced_chi2": float(mlog.get("center_reduced_chi2", np.nan)),
                },
            )

    # Successive deltas
    drel_log = [None]
    drel_lin = [None]
    for i in range(1, len(models_log)):
        dlog = models_log[i] - models_log[i - 1]
        dlin = models_linear[i] - models_linear[i - 1]
        drel_log.append(float(np.sqrt(np.mean(dlog * dlog)) / max(np.sqrt(np.mean(models_log[i - 1] ** 2)), 1e-30)))
        drel_lin.append(float(np.sqrt(np.mean(dlin * dlin)) / max(np.sqrt(np.mean(models_linear[i - 1] ** 2)), 1e-30)))
    for i, r in enumerate(rows):
        r["delta_rel_vs_prev_log_model"] = None if drel_log[i] is None else float(drel_log[i])
        r["delta_rel_vs_prev_linearized_model"] = None if drel_lin[i] is None else float(drel_lin[i])

    # Plots
    model_pdf = os.path.join(out_dir, "N1_LOGFLUX_ELL_MODEL_AND_RESIDUALS.pdf")
    delta_pdf = os.path.join(out_dir, "N1_LOGFLUX_ELL_SUCCESSIVE_DELTAS.pdf")
    vm_valid = _valid_mask(cut)
    dlim = np.nanpercentile(data[vm], [1, 99]) if np.any(vm) else [np.nanmin(data), np.nanmax(data)]

    with PdfPages(model_pdf) as pdf:
        for i, ell in enumerate(ell_vals):
            pred_lin = models_linear[i]
            pred_log = models_log[i]
            resid_lin = data - pred_lin
            resid_log = data_log - pred_log
            rl0, rl1 = _resid_limits(np.where(vm, resid_lin, 0.0))
            rg0, rg1 = _resid_limits(np.where(vm, resid_log, 0.0))
            fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
            im0 = ax[0].imshow(np.where(vm_valid, pred_lin, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
            ax[0].set_title(f"Model (exp back-transform), ell={ell:g}")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            im1 = ax[1].imshow(np.where(vm, resid_lin, 0.0), origin="lower", cmap="RdBu_r", vmin=rl0, vmax=rl1)
            ax[1].set_title("Residual in linear units")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            im2 = ax[2].imshow(np.where(vm, resid_log, 0.0), origin="lower", cmap="RdBu_r", vmin=rg0, vmax=rg1)
            ax[2].set_title("Residual in log domain")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
            for a in ax:
                a.axis("off")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(delta_pdf) as pdf:
        for i in range(1, len(ell_vals)):
            e0, e1 = ell_vals[i - 1], ell_vals[i]
            dlog = models_log[i] - models_log[i - 1]
            dlin = models_linear[i] - models_linear[i - 1]
            l0, l1 = _resid_limits(dlin)
            g0, g1 = _resid_limits(dlog)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            im0 = ax[0].imshow(dlin, origin="lower", cmap="RdBu_r", vmin=l0, vmax=l1)
            ax[0].set_title(f"Linearized model diff: {e1:g}-{e0:g}")
            ax[0].axis("off")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            im1 = ax[1].imshow(dlog, origin="lower", cmap="RdBu_r", vmin=g0, vmax=g1)
            ax[1].set_title(f"Log model diff: {e1:g}-{e0:g}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    out_json = os.path.join(out_dir, "n1_logflux_ell_scan_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "floor": floor,
                "n_positive_valid_pixels": int(np.sum(pos)),
                "var_fixed": var_opt,
                "rows": rows,
                "artifacts": {
                    "model_and_residual_pdf": model_pdf,
                    "successive_delta_pdf": delta_pdf,
                },
            },
            f,
            indent=2,
        )

    print(f"Wrote {out_json}")
    print(f"Wrote {model_pdf}")
    print(f"Wrote {delta_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

