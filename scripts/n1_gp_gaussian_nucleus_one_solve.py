#!/usr/bin/env python3
"""Single N=1 native solve with optional Gaussian nucleus component."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src import fit_metrics, gp_model, solver  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _intrinsic_components,
    _split_gp_components_from_prior,
    _reindex_epochs,
    _temporary_config,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
    write_iteration_metric_plot,
    write_native_fit_pdf,
    write_stacked_residual_pdf,
)


def main() -> int:
    p = argparse.ArgumentParser(description="N=1 one-shot: GP (optimized) with optional Gaussian nucleus")
    p.add_argument(
        "--output-dir",
        default=os.path.join(config.DIAGNOSTIC_DIR, "iterative_campaign", "gp_gaussian_nucleus_one"),
        help="Directory for PDFs, plots, and summary JSON",
    )
    p.add_argument(
        "--sigma-pair",
        default="0.8,1.8",
        help="Comma-separated scene-pixel sigmas for two nonnegative Gaussian basis columns (fixed RA/Dec)",
    )
    p.add_argument(
        "--disable-gaussian-nucleus",
        action="store_true",
        help="Turn off host Gaussian nucleus terms (pure GP + background solve)",
    )
    p.add_argument(
        "--gp2-ell",
        type=float,
        default=None,
        help="Optional second GP length scale (scene px) for two-scale GP prior.",
    )
    p.add_argument(
        "--gp2-var",
        type=float,
        default=None,
        help="Optional second GP variance for two-scale GP prior.",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cutouts = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])
    for c in cutouts:
        apply_native_cutout_cr_mask(c)

    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    centers = dict(real_case["centers"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)

    nr, nd = centers.get("nuc_ra"), centers.get("nuc_dec")
    if nr is None or nd is None:
        raise SystemExit("Galaxy nucleus coordinates missing in prepared case (nuc_ra / nuc_dec).")

    sigmas = tuple(float(x.strip()) for x in args.sigma_pair.split(",") if x.strip())
    if not args.disable_gaussian_nucleus and len(sigmas) < 1:
        raise SystemExit("--sigma-pair must list at least one positive sigma in scene pixels")

    print("Optimizing GP hyperparameters on first template cutout...")
    ell_opt, var_opt = gp_model.optimize_hyperparameters([cutouts[0]])
    ell_opt = float(ell_opt)
    var_opt = float(max(float(var_opt), 1e-30))
    print(f"  ell_opt={ell_opt:.6f}  var_opt={var_opt:.6e}")
    if (args.gp2_ell is None) ^ (args.gp2_var is None):
        raise SystemExit("Provide both --gp2-ell and --gp2-var, or neither.")
    gp_params = {"ell": ell_opt, "var": var_opt}
    if args.gp2_ell is not None and args.gp2_var is not None:
        gp_params["ell2"] = float(args.gp2_ell)
        gp_params["var2"] = float(max(float(args.gp2_var), 1e-30))

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": (not args.disable_gaussian_nucleus),
        "HOST_CORE_RA": float(nr),
        "HOST_CORE_DEC": float(nd),
        "HOST_GAUSSIAN_SIGMA_PX_LIST": sigmas,
        "HOST_GAUSSIAN_MIN_OFFSET_PX": 0.0,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
        "USE_NUCLEAR_POINT_SOURCE": False,
    }

    stage_name = "N1"
    with _temporary_config(overrides):
        results = solver.run_gls_solve(
            cutouts,
            stars,
            init_star_fluxes,
            gp_params,
            (ell_opt, var_opt),
            np.zeros(scene_shape),
            scene_wcs,
            len(cutouts),
        )
        if results is None or not isinstance(results, dict):
            raise SystemExit("run_gls_solve returned no results")

        metrics = fit_metrics.compute_fit_metrics(
            cutouts,
            results,
            stars,
            results.get("star_fluxes", init_star_fluxes),
            center_ra_deg=float(nr),
            center_dec_deg=float(nd),
            center_radius_px=3.0,
        )
        gp_scene, _, _ = _intrinsic_components(results)
        gp_c1, gp_c2 = _split_gp_components_from_prior(results, gp_scene)
        g1 = np.asarray(gp_c1, dtype=np.float64).ravel()
        g2 = np.asarray(gp_c2, dtype=np.float64).ravel()
        cc = float("nan")
        if np.std(g1) > 0 and np.std(g2) > 0:
            cc = float(np.corrcoef(g1, g2)[0, 1])
        gy1, gx1 = np.gradient(np.asarray(gp_c1, dtype=np.float64))
        gy2, gx2 = np.gradient(np.asarray(gp_c2, dtype=np.float64))
        grad1 = float(np.sqrt(np.mean(gx1 * gx1 + gy1 * gy1)))
        grad2 = float(np.sqrt(np.mean(gx2 * gx2 + gy2 * gy2)))
        rms1 = float(np.sqrt(np.mean(np.asarray(gp_c1, dtype=np.float64) ** 2)))
        rms2 = float(np.sqrt(np.mean(np.asarray(gp_c2, dtype=np.float64) ** 2)))

        row = {
            "iteration": 0,
            "use_point": False,
            "ell": ell_opt,
            "var_mult": float(var_opt / 1e-7),
            **metrics,
            "host_effective_sigma_px": float(results.get("host_effective_sigma_px", float("nan"))),
            "host_core_flux": float(results.get("host_core_flux", 0.0)),
        }
        if "host_core_fluxes" in results:
            row["host_core_fluxes"] = [float(x) for x in np.asarray(results["host_core_fluxes"]).ravel()]
        if "host_gaussian_sigmas_px" in results:
            row["host_gaussian_sigmas_px"] = [float(x) for x in np.asarray(results["host_gaussian_sigmas_px"]).ravel()]

        diag = write_native_fit_pdf(stage_name, cutouts, results, args.output_dir)
        stack = write_stacked_residual_pdf(stage_name, cutouts, results, args.output_dir)
        plot_path = write_iteration_metric_plot(stage_name, [row], args.output_dir)

    summary = {
        "stage": stage_name,
        "n_bcd": 1,
        "iterations": 1,
        "gp_hyperparams": gp_params,
        "gaussian_nucleus": {
            "enabled": bool(not args.disable_gaussian_nucleus),
            "ra_deg": float(nr),
            "dec_deg": float(nd),
            "sigma_basis_px": list(sigmas),
            "fitted_host_amplitudes": row.get("host_core_fluxes"),
            "effective_sigma_px": row.get("host_effective_sigma_px"),
        },
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "artifacts": {
            "diagnostic_pdf": diag,
            "stacked_pdf": stack,
            "iteration_metrics_png": plot_path,
        },
        "gp_component_diagnostics": {
            "component_corrcoef": cc,
            "component1_rms": rms1,
            "component2_rms": rms2,
            "component1_grad_rms": grad1,
            "component2_grad_rms": grad2,
            "grad_rms_ratio_2_over_1": float(grad2 / max(grad1, 1e-30)),
        },
    }
    out_json = os.path.join(args.output_dir, "n1_gp_gaussian_nucleus_one_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {out_json}")
    print(f"Wrote {diag}")
    print(f"Wrote {stack}")
    if plot_path:
        print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
