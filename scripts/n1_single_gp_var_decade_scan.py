#!/usr/bin/env python3
"""Single-GP N=1 fixed-ell var-decade scan with incremental outputs each step."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, fit_metrics, solver  # noqa: E402
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
    p = argparse.ArgumentParser(description="N=1 single-GP fixed-ell decade-var scan")
    p.add_argument("--output-dir", default=os.path.join(config.DIAGNOSTIC_DIR, "n1_single_gp_var_decade_scan"))
    p.add_argument("--ell-fixed", type=float, default=0.5)
    p.add_argument("--var-start", type=float, default=1e8)
    p.add_argument("--var-min", type=float, default=1e-6)
    p.add_argument("--var-factor", type=float, default=10.0)
    return p


def _var_sequence(start: float, var_min: float, factor: float) -> list[float]:
    vals: list[float] = []
    v = float(start)
    while v >= float(var_min):
        vals.append(float(v))
        v /= float(factor)
    return vals


def _write_summary(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> int:
    args = _make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    per_step_dir = os.path.join(args.output_dir, "per_step")
    os.makedirs(per_step_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cutouts = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])
    for c in cutouts:
        apply_native_cutout_cr_mask(c)

    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
    nuc_ra, nuc_dec = real_case["centers"]["nuc_ra"], real_case["centers"]["nuc_dec"]
    vars_to_run = _var_sequence(float(args.var_start), float(args.var_min), float(args.var_factor))

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
    }

    run_rows: list[dict] = []
    models: list[np.ndarray] = []

    data = np.asarray(cutouts[0]["data"], dtype=float)
    vm_valid = _valid_mask(cutouts[0])
    vm_analysis = _analysis_mask(cutouts[0])
    d_lim = np.nanpercentile(data[vm_analysis], [1, 99]) if np.any(vm_analysis) else [np.nanmin(data), np.nanmax(data)]
    summary_path = os.path.join(args.output_dir, "n1_single_gp_var_decade_summary.json")

    with _temporary_config(overrides):
        for idx, var in enumerate(vars_to_run):
            var = float(max(var, 1e-30))
            res = solver.run_gls_solve(
                cutouts,
                stars,
                init_star_fluxes,
                {"ell": float(args.ell_fixed), "var": var},
                (float(args.ell_fixed), var),
                np.zeros(scene_shape),
                scene_wcs,
                len(cutouts),
            )
            if not isinstance(res, dict):
                break
            pred = np.asarray(
                solver.predict_cutout_model(
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
                ),
                dtype=float,
            )
            models.append(pred.copy())
            resid = data - pred
            metrics = fit_metrics.compute_fit_metrics(
                cutouts,
                res,
                stars,
                res.get("star_fluxes", init_star_fluxes),
                center_ra_deg=float(nuc_ra),
                center_dec_deg=float(nuc_dec),
                center_radius_px=3.0,
            )

            delta_rms = None
            rel_delta = None
            if len(models) > 1:
                d = models[-1] - models[-2]
                delta_rms = float(np.sqrt(np.mean(d * d)))
                base_rms = float(np.sqrt(np.mean(models[-2] * models[-2])))
                rel_delta = float(delta_rms / max(base_rms, 1e-30))

            row = {
                "idx": idx,
                "ell": float(args.ell_fixed),
                "var": var,
                "metrics": {kk: float(vv) for kk, vv in metrics.items() if isinstance(vv, (int, float))},
                "delta_rms_vs_prev": delta_rms,
                "delta_rel_rms_vs_prev": rel_delta,
            }
            run_rows.append(row)

            # Per-step model/residual plot
            pred_disp = np.where(vm_valid, pred, 0.0)
            resid_disp = np.where(vm_analysis, resid, 0.0)
            rv0, rv1 = _resid_limits(resid_disp)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            im0 = ax[0].imshow(pred_disp, origin="lower", cmap="gray", vmin=float(d_lim[0]), vmax=float(d_lim[1]), interpolation="nearest")
            ax[0].set_title(f"Model on BCD (ell={args.ell_fixed:.6g}, var={var:.3e})")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            im1 = ax[1].imshow(resid_disp, origin="lower", cmap="RdBu_r", vmin=rv0, vmax=rv1, interpolation="nearest")
            ax[1].set_title("Fit residual (data - model)")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            for a in ax:
                a.axis("off")
            fig.suptitle(
                "Step {} | total_red_chi2={:.4f}, center_red_chi2={:.4f}".format(
                    idx,
                    float(metrics.get("total_reduced_chi2", np.nan)),
                    float(metrics.get("center_reduced_chi2", np.nan)),
                ),
            )
            plt.tight_layout()
            out_step = os.path.join(per_step_dir, f"step_{idx:02d}_model_residual.png")
            fig.savefig(out_step, dpi=160)
            plt.close(fig)

            # Per-step successive delta plot
            delta_path = None
            if len(models) > 1:
                prev_var = run_rows[-2]["var"]
                d = models[-1] - models[-2]
                dv0, dv1 = _resid_limits(d)
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                im = ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=dv0, vmax=dv1, interpolation="nearest")
                ax.set_title(f"Model difference: var={var:.3e} - {prev_var:.3e}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                delta_path = os.path.join(per_step_dir, f"step_{idx:02d}_delta_vs_prev.png")
                fig.savefig(delta_path, dpi=160)
                plt.close(fig)

            summary = {
                "ell_fixed": float(args.ell_fixed),
                "var_start": float(args.var_start),
                "var_min": float(args.var_min),
                "var_factor": float(args.var_factor),
                "rows": run_rows,
                "artifacts": {
                    "per_step_dir": per_step_dir,
                },
            }
            _write_summary(summary_path, summary)
            print(
                "step={} var={:.3e} total_red_chi2={:.6f} center_red_chi2={:.6f} delta_rel={}".format(
                    idx,
                    var,
                    float(metrics.get("total_reduced_chi2", np.nan)),
                    float(metrics.get("center_reduced_chi2", np.nan)),
                    "nan" if rel_delta is None else f"{rel_delta:.3e}",
                ),
            )
            print(f"wrote {out_step}")
            if delta_path is not None:
                print(f"wrote {delta_path}")
            sys.stdout.flush()

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

