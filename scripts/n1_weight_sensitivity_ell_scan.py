#!/usr/bin/env python3
"""Compare ell sensitivity with native vs uniform pixel weighting (N=1)."""
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
    _temporary_config,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
)


def _run_case(cutout, scene_wcs, scene_shape, stars, init_star_fluxes, ell_vals, var_fixed):
    rows = []
    models = []
    for ell in ell_vals:
        res = solver.run_gls_solve(
            [cutout],
            stars,
            init_star_fluxes,
            {"ell": float(ell), "var": float(var_fixed)},
            (float(ell), float(var_fixed)),
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
        models.append(pred.copy())
        m = fit_metrics.compute_fit_metrics(
            [cutout],
            res,
            stars,
            res.get("star_fluxes", init_star_fluxes),
            center_ra_deg=float(config.TRANSIENT_RA),
            center_dec_deg=float(config.TRANSIENT_DEC),
            center_radius_px=3.0,
        )
        rows.append(
            {
                "ell": float(ell),
                "total_reduced_chi2": float(m.get("total_reduced_chi2", np.nan)),
                "center_reduced_chi2": float(m.get("center_reduced_chi2", np.nan)),
            },
        )
    drel = [None]
    for i in range(1, len(models)):
        d = models[i] - models[i - 1]
        num = float(np.sqrt(np.mean(d * d)))
        den = float(np.sqrt(np.mean(models[i - 1] * models[i - 1])))
        drel.append(num / max(den, 1e-30))
    for r, dr in zip(rows, drel):
        r["delta_rel_vs_prev_model"] = None if dr is None else float(dr)
    return rows, models


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_weight_sensitivity_ell_scan")
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cutout = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])[0]
    apply_native_cutout_cr_mask(cutout)
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
    _, var_opt = gp_model.optimize_hyperparameters([cutout])
    var_opt = float(max(float(var_opt), 1e-30))
    ell_vals = [20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    with _temporary_config(overrides):
        rows_native, models_native = _run_case(cutout, scene_wcs, scene_shape, stars, init_star_fluxes, ell_vals, var_opt)

        cut_uniform = dict(cutout)
        sig = np.asarray(cutout["sigma"], dtype=float)
        valid = np.isfinite(sig) & (sig < 1e20)
        med = float(np.nanmedian(sig[valid])) if np.any(valid) else 1.0
        su = np.full_like(sig, med, dtype=float)
        su[~valid] = np.inf
        cut_uniform["sigma"] = su
        rows_uniform, models_uniform = _run_case(cut_uniform, scene_wcs, scene_shape, stars, init_star_fluxes, ell_vals, var_opt)

    comp_rows = []
    for i, ell in enumerate(ell_vals):
        comp_rows.append(
            {
                "ell": float(ell),
                "native": rows_native[i],
                "uniform": rows_uniform[i],
            },
        )

    summary = {
        "var_fixed": var_opt,
        "ell_values": ell_vals,
        "comparison": comp_rows,
        "uniform_sigma_value": med,
    }
    out_json = os.path.join(out_dir, "n1_weight_sensitivity_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Diagnostic plot of sensitivity curves
    out_png = os.path.join(out_dir, "N1_WEIGHT_SENSITIVITY_CURVES.png")
    x = np.arange(len(ell_vals))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, [r["delta_rel_vs_prev_model"] or np.nan for r in rows_native], marker="o", label="native")
    ax[0].plot(x, [r["delta_rel_vs_prev_model"] or np.nan for r in rows_uniform], marker="o", label="uniform")
    ax[0].set_xticks(x, [f"{e:g}" for e in ell_vals], rotation=45)
    ax[0].set_yscale("log")
    ax[0].set_title("Model delta_rel vs previous ell")
    ax[0].set_xlabel("ell")
    ax[0].legend()
    ax[1].plot(x, [r["center_reduced_chi2"] for r in rows_native], marker="o", label="native")
    ax[1].plot(x, [r["center_reduced_chi2"] for r in rows_uniform], marker="o", label="uniform")
    ax[1].set_xticks(x, [f"{e:g}" for e in ell_vals], rotation=45)
    ax[1].set_title("Center reduced chi2 vs ell")
    ax[1].set_xlabel("ell")
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # PDF with successive model differences (native vs uniform)
    out_pdf = os.path.join(out_dir, "N1_WEIGHT_SENSITIVITY_DELTAS.pdf")
    with PdfPages(out_pdf) as pdf:
        for i in range(1, len(ell_vals)):
            d_nat = models_native[i] - models_native[i - 1]
            d_uni = models_uniform[i] - models_uniform[i - 1]
            lim = max(np.nanpercentile(np.abs(d_nat), 99), np.nanpercentile(np.abs(d_uni), 99), 1e-20)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            im0 = ax[0].imshow(d_nat, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[0].set_title(f"native: ell {ell_vals[i]:g} - {ell_vals[i-1]:g}")
            ax[0].axis("off")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            im1 = ax[1].imshow(d_uni, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[1].set_title(f"uniform: ell {ell_vals[i]:g} - {ell_vals[i-1]:g}")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

