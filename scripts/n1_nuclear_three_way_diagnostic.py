#!/usr/bin/env python3
"""Three-way nuclear diagnostic:
  A) GP only (no Gaussians, no point source)
  B) GP + free-position nuclear point source
  C) Background + free-position nuclear point source only (no GP)

Tests whether the nucleus residual is explained by:
 - A GP fitting failure (A vs B comparison)
 - A genuine point-source component (B vs C comparison)
 - GP/point-source degeneracy (B vs C comparison)
"""
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
    _reindex_epochs,
    _temporary_config,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
    write_native_fit_pdf,
    write_stacked_residual_pdf,
)


def _solve_case(label, cutouts, stars, init_star_fluxes, gp_params, scene_shape, scene_wcs, overrides):
    print(f"\n{'='*60}")
    print(f"  Running test: {label}")
    print(f"{'='*60}")
    sys.stdout.flush()
    with _temporary_config(overrides):
        results = solver.run_gls_solve(
            cutouts,
            stars,
            init_star_fluxes,
            gp_params,
            (gp_params["ell"], gp_params["var"]),
            np.zeros(scene_shape),
            scene_wcs,
            len(cutouts),
        )
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Three-way nuclear residual diagnostic")
    p.add_argument(
        "--output-dir",
        default=os.path.join(config.DIAGNOSTIC_DIR, "iterative_campaign", "nuclear_three_way"),
        help="Output directory for PDFs and JSON",
    )
    p.add_argument("--gp2-ell", type=float, default=2.0, help="Second GP length scale (scene px)")
    p.add_argument("--gp2-var", type=float, default=1e-6, help="Second GP variance")
    p.add_argument(
        "--nps-ridge",
        type=float,
        default=0.0,
        help="Ridge on nuclear point position (0 = only data constraints; default: 0)",
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

    print("Optimizing GP hyperparameters on first template cutout...")
    ell_opt, var_opt = gp_model.optimize_hyperparameters([cutouts[0]])
    ell_opt = float(ell_opt)
    var_opt = float(max(float(var_opt), 1e-30))
    print(f"  ell_opt={ell_opt:.6f}  var_opt={var_opt:.6e}")

    gp_params = {"ell": ell_opt, "var": var_opt}
    if args.gp2_ell is not None and args.gp2_var is not None:
        gp_params["ell2"] = float(args.gp2_ell)
        gp_params["var2"] = float(args.gp2_var)

    # Shared config items common to all tests
    _base = {
        "USE_HOST_GAUSSIAN_CORE": False,          # No Gaussians in any test
        "HOST_CORE_RA": float(nr),
        "HOST_CORE_DEC": float(nd),
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
        "GP_COMPONENTS_NONNEGATIVE": False,
        "NUCLEAR_POINT_RA": float(nr),
        "NUCLEAR_POINT_DEC": float(nd),
        "NUCLEAR_POINT_NONNEGATIVE": True,
        "FLOAT_NUCLEAR_POINT_POSITION": True,
        "NUCLEAR_POINT_POS_RIDGE": float(args.nps_ridge),
    }

    # ── Test A: GP only ──────────────────────────────────────────────────────
    overrides_A = {**_base, "USE_NUCLEAR_POINT_SOURCE": False}
    results_A = _solve_case("A: GP only", cutouts, stars, init_star_fluxes, gp_params, scene_shape, scene_wcs, overrides_A)

    # ── Test B: GP + free-position nuclear point source ──────────────────────
    overrides_B = {**_base, "USE_NUCLEAR_POINT_SOURCE": True}
    results_B = _solve_case("B: GP + NPS (free pos)", cutouts, stars, init_star_fluxes, gp_params, scene_shape, scene_wcs, overrides_B)

    # ── Test C: Background + NPS only (no GP = very strong prior) ────────────
    # Force GP to zero by making the variance tiny (strong zero-mean prior).
    gp_params_null = {"ell": ell_opt, "var": 1e-30}  # Effectively kills GP scene
    overrides_C = {
        **_base,
        "USE_NUCLEAR_POINT_SOURCE": True,
        "GP_COMPONENTS_NONNEGATIVE": False,  # doesn't matter, GP is killed
    }
    results_C = _solve_case(
        "C: Background + NPS only (no GP)",
        cutouts, stars, init_star_fluxes,
        gp_params_null,
        scene_shape, scene_wcs, overrides_C,
    )

    # ── Metrics for all three ────────────────────────────────────────────────
    summary_rows = []
    for label, results, ovr in [
        ("A_gp_only", results_A, overrides_A),
        ("B_gp_plus_nps", results_B, overrides_B),
        ("C_bkg_plus_nps", results_C, overrides_C),
    ]:
        with _temporary_config(ovr):
            met = fit_metrics.compute_fit_metrics(
                cutouts,
                results,
                stars,
                results.get("star_fluxes", init_star_fluxes),
                center_ra_deg=float(nr),
                center_dec_deg=float(nd),
                center_radius_px=3.0,
            )
        nps_flux = float(results.get("nuclear_point_flux", 0.0)) if results is not None else float("nan")
        nps_dra = float(results.get("nuclear_point_dra_deg", 0.0)) * 3600.0 if results is not None else float("nan")
        nps_ddec = float(results.get("nuclear_point_ddec_deg", 0.0)) * 3600.0 if results is not None else float("nan")
        summary_rows.append({
            "test": label,
            "nps_flux": nps_flux,
            "nps_dra_arcsec": nps_dra,
            "nps_ddec_arcsec": nps_ddec,
            **met,
        })

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n\n=== THREE-WAY NUCLEAR DIAGNOSTIC RESULTS ===")
    dq = '"'
    print(f"{'Test':<22} {'total_chi2_r':>12} {'center_chi2_r':>14} {'nps_flux':>14} {f'dRA{dq}':>8} {f'dDec{dq}':>8}")
    print("-" * 82)
    for row in summary_rows:
        print(
            f"{row['test']:<22} "
            f"{row['total_reduced_chi2']:>12.4f} "
            f"{row['center_reduced_chi2']:>14.2f} "
            f"{row['nps_flux']:>14.4e} "
            f"{row['nps_dra_arcsec']:>8.3f} "
            f"{row['nps_ddec_arcsec']:>8.3f}"
        )

    # ── Write PDFs ───────────────────────────────────────────────────────────
    pdf_paths = {}
    for label, results, ovr in [
        ("A_gp_only", results_A, overrides_A),
        ("B_gp_plus_nps", results_B, overrides_B),
        ("C_bkg_plus_nps", results_C, overrides_C),
    ]:
        if results is None:
            continue
        with _temporary_config(ovr):
            pdf = write_native_fit_pdf(label, cutouts, results, args.output_dir)
            stk = write_stacked_residual_pdf(label, cutouts, results, args.output_dir)
        pdf_paths[label] = {"diagnostic_pdf": pdf, "stacked_pdf": stk}
        print(f"Wrote {label}: {pdf}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, "nuclear_three_way_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "gp_hyperparams": gp_params,
            "nucleus_coords": {"ra": float(nr), "dec": float(nd)},
            "results": summary_rows,
            "artifacts": pdf_paths,
        }, f, indent=2, default=str)
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
