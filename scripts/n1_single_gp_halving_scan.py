#!/usr/bin/env python3
"""Single-GP N=1 halving scan with model/residual and successive-delta plots."""
from __future__ import annotations

import argparse
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


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="N=1 single-GP halving scan on real template data")
    p.add_argument(
        "--output-dir",
        default=os.path.join(config.DIAGNOSTIC_DIR, "n1_single_gp_halving_scan"),
        help="Output directory for summary and plots",
    )
    p.add_argument("--ell-start", type=float, default=20.0, help="Starting GP ell")
    p.add_argument("--ell-min", type=float, default=1e-5, help="Minimum GP ell")
    p.add_argument("--halve-factor", type=float, default=2.0, help="Divide ell by this each step")
    p.add_argument(
        "--plateau-rel-thresh",
        type=float,
        default=1e-3,
        help="Stop when relative RMS of successive GP deltas drops below this threshold",
    )
    p.add_argument(
        "--plateau-consecutive",
        type=int,
        default=2,
        help="Consecutive plateau steps required before stopping",
    )
    return p


def _ell_sequence(start: float, ell_min: float, halve_factor: float) -> list[float]:
    vals: list[float] = []
    ell = float(start)
    while ell >= float(ell_min):
        vals.append(float(ell))
        ell /= float(halve_factor)
    return vals


def main() -> int:
    args = _make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cutouts = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])
    for c in cutouts:
        apply_native_cutout_cr_mask(c)

    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
    nuc_ra, nuc_dec = real_case["centers"]["nuc_ra"], real_case["centers"]["nuc_dec"]

    _, var_opt = gp_model.optimize_hyperparameters([cutouts[0]])
    var_opt = float(max(float(var_opt), 1e-30))

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    seq = _ell_sequence(args.ell_start, args.ell_min, args.halve_factor)
    run_rows: list[dict] = []
    gp_scenes: list[np.ndarray] = []
    bcd_models: list[np.ndarray] = []
    plateau_hits = 0

    with _temporary_config(overrides):
        for k, ell in enumerate(seq):
            res = solver.run_gls_solve(
                cutouts,
                stars,
                init_star_fluxes,
                {"ell": float(ell), "var": var_opt},
                (float(ell), var_opt),
                np.zeros(scene_shape),
                scene_wcs,
                len(cutouts),
            )
            if not isinstance(res, dict):
                break
            pred = solver.predict_cutout_model(
                res,
                cutouts,
                [],
                [],
                0,
                include_gp=True,
                include_transient=False,
                include_stars=False,
                include_host=False,
                include_nuclear_point=False,
            )
            data = np.asarray(cutouts[0]["data"], dtype=float)
            resid = data - np.asarray(pred, dtype=float)
            metrics = fit_metrics.compute_fit_metrics(
                cutouts,
                res,
                stars,
                res.get("star_fluxes", init_star_fluxes),
                center_ra_deg=float(nuc_ra),
                center_dec_deg=float(nuc_dec),
                center_radius_px=3.0,
            )
            scene = np.asarray(res.get("gp_scene", res["model_scene"]), dtype=float)
            gp_scenes.append(scene.copy())
            bcd_models.append(np.asarray(pred, dtype=float).copy())

            rel_delta = None
            delta_rms = None
            if len(gp_scenes) > 1:
                d = bcd_models[-1] - bcd_models[-2]
                delta_rms = float(np.sqrt(np.mean(d * d)))
                base_rms = float(np.sqrt(np.mean(bcd_models[-2] * bcd_models[-2])))
                rel_delta = float(delta_rms / max(base_rms, 1e-30))
                if rel_delta < float(args.plateau_rel_thresh):
                    plateau_hits += 1
                else:
                    plateau_hits = 0

            run_rows.append(
                {
                    "idx": k,
                    "ell": float(ell),
                    "var": var_opt,
                    "metrics": {kk: float(vv) for kk, vv in metrics.items() if isinstance(vv, (int, float))},
                    "delta_rms_vs_prev": delta_rms,
                    "delta_rel_rms_vs_prev": rel_delta,
                },
            )
            if plateau_hits >= int(args.plateau_consecutive):
                break

    model_pdf = os.path.join(args.output_dir, "N1_SINGLE_GP_MODEL_AND_RESIDUALS.pdf")
    delta_pdf = os.path.join(args.output_dir, "N1_SINGLE_GP_SUCCESSIVE_DELTAS.pdf")

    data = np.asarray(cutouts[0]["data"], dtype=float)
    vm_valid = _valid_mask(cutouts[0])
    vm_analysis = _analysis_mask(cutouts[0])
    d_lim = np.nanpercentile(data[vm_analysis], [1, 99]) if np.any(vm_analysis) else [np.nanmin(data), np.nanmax(data)]

    with PdfPages(model_pdf) as pdf:
        for row in run_rows:
            i = int(row["idx"])
            ell = float(row["ell"])
            pred = np.asarray(bcd_models[i], dtype=float)
            resid = data - np.asarray(pred, dtype=float)
            pred_disp = np.where(vm_valid, pred, 0.0)
            resid_disp = np.where(vm_analysis, resid, 0.0)
            rv0, rv1 = _resid_limits(resid_disp)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            im0 = ax[0].imshow(pred_disp, origin="lower", cmap="gray", vmin=float(d_lim[0]), vmax=float(d_lim[1]), interpolation="nearest")
            ax[0].set_title(f"Model on BCD (ell={ell:.6g})")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            im1 = ax[1].imshow(resid_disp, origin="lower", cmap="RdBu_r", vmin=rv0, vmax=rv1, interpolation="nearest")
            ax[1].set_title("Fit residual (data - model)")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            for a in ax:
                a.axis("off")
            m = row["metrics"]
            fig.suptitle(
                "N=1 single-GP: ell={:.6g}, var={:.3e}, total_red_chi2={:.4f}, center_red_chi2={:.4f}".format(
                    ell,
                    var_opt,
                    float(m.get("total_reduced_chi2", np.nan)),
                    float(m.get("center_reduced_chi2", np.nan)),
                ),
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(delta_pdf) as pdf:
        for i in range(len(bcd_models) - 1):
            e0 = float(run_rows[i]["ell"])
            e1 = float(run_rows[i + 1]["ell"])
            d = np.asarray(bcd_models[i + 1], dtype=float) - np.asarray(bcd_models[i], dtype=float)
            dv0, dv1 = _resid_limits(d)
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=dv0, vmax=dv1, interpolation="nearest")
            ax.set_title(f"Model difference: ell={e1:.6g} minus ell={e0:.6g}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            meta = run_rows[i + 1]
            fig.suptitle(
                "delta_rms={:.3e}, rel_delta_rms={}".format(
                    float(meta["delta_rms_vs_prev"]) if meta["delta_rms_vs_prev"] is not None else float("nan"),
                    "nan" if meta["delta_rel_rms_vs_prev"] is None else f"{float(meta['delta_rel_rms_vs_prev']):.3e}",
                ),
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    summary = {
        "var_fixed": var_opt,
        "rows": run_rows,
        "artifacts": {
            "model_and_residual_pdf": model_pdf,
            "successive_delta_pdf": delta_pdf,
        },
    }
    out_json = os.path.join(args.output_dir, "n1_single_gp_halving_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"Wrote {model_pdf}")
    print(f"Wrote {delta_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

