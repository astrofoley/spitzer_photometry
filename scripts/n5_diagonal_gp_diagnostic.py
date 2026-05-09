#!/usr/bin/env python3
"""Diagnostic: diagonal (independent-pixel) GP + background, N template BCDs.

No Gaussians, no point sources. Run with --no-prf to replace PRF convolution
with identity (projection-only forward model).

Key diagnostic:
  With PRF:    Hessian diagonal approximation is INCONSISTENT
               (H ignores PRF coupling, RHS includes it via adjoint)
  Without PRF: Hessian diagonal approximation is EXACT
               (A = P projection only; H_jj = (P^T W P)_jj exactly)

If center_chi2 drops dramatically without PRF → the Hessian/PRF inconsistency
is the root cause of the nuclear residuals.
If center_chi2 stays high without PRF → the projection model or data is the problem.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src import fit_metrics, solver  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _reindex_epochs,
    _temporary_config,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
    write_native_fit_pdf,
    write_stacked_residual_pdf,
)


@contextlib.contextmanager
def _no_prf_context():
    """Replace all PRF operators with identity (scene passthrough, no convolution).
    This makes the forward model A = P (projection only) and the Hessian diagonal
    approximation H_jj = (P^T W P)_jj becomes exact.
    """
    orig_forward = solver.apply_spatially_varying_prf_to_scene
    orig_adjoint = solver.apply_spatially_varying_prf_adjoint
    orig_bundle = solver._get_prf_operator_bundle
    orig_apply_bundle = solver._apply_prf_operator_from_bundle
    orig_apply_adjoint_bundle = solver._apply_prf_adjoint_from_bundle
    try:
        solver.apply_spatially_varying_prf_to_scene = (
            lambda intrinsic_scene, scene_wcs, w_native, scene_shape, channel, is_full_array=False:
            np.asarray(intrinsic_scene, dtype=np.float64).reshape(scene_shape)
        )
        solver.apply_spatially_varying_prf_adjoint = (
            lambda y_scene, scene_wcs, w_native, scene_shape, channel, is_full_array=False:
            np.asarray(y_scene, dtype=np.float64).reshape(scene_shape).ravel()
        )
        solver._get_prf_operator_bundle = (
            lambda scene_wcs, w_native, scene_shape, channel, is_full_array: (None, None, None)
        )
        solver._apply_prf_operator_from_bundle = (
            lambda img, kernels, weights, wsum: np.asarray(img, dtype=np.float64)
        )
        solver._apply_prf_adjoint_from_bundle = (
            lambda y, kernels, weights, wsum: np.asarray(y, dtype=np.float64)
        )
        yield
    finally:
        solver.apply_spatially_varying_prf_to_scene = orig_forward
        solver.apply_spatially_varying_prf_adjoint = orig_adjoint
        solver._get_prf_operator_bundle = orig_bundle
        solver._apply_prf_operator_from_bundle = orig_apply_bundle
        solver._apply_prf_adjoint_from_bundle = orig_apply_adjoint_bundle


def main() -> int:
    p = argparse.ArgumentParser(
        description="Independent-pixel (diagonal GP) diagnostic with N template BCDs"
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(config.DIAGNOSTIC_DIR, "iterative_campaign", "diagonal_gp_n5"),
        help="Output directory for PDFs and JSON",
    )
    p.add_argument("--n-templates", type=int, default=5, help="Number of template BCDs (default: 5)")
    p.add_argument("--eps", type=float, default=1e-10, help="Diagonal GP precision ε (default: 1e-10)")
    p.add_argument(
        "--no-prf", action="store_true",
        help="Replace PRF with identity operator (projection-only). Makes Hessian diagonal exact.",
    )
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    n_tpl = min(args.n_templates, len(real_case["template_cutouts"]))
    print(f"Using {n_tpl} template BCDs (requested {args.n_templates}, "
          f"available {len(real_case['template_cutouts'])})")

    cutouts = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:n_tpl]])
    for c in cutouts:
        apply_native_cutout_cr_mask(c)

    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    centers = dict(real_case["centers"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)

    nr, nd = centers.get("nuc_ra"), centers.get("nuc_dec")
    if nr is None or nd is None:
        raise SystemExit("Galaxy nucleus coordinates missing in prepared case.")

    gp_params = {"ell": 2.0, "var": 1e-6}

    overrides = {
        "GP_KERNEL_TYPE": "diagonal",
        "GP_DIAGONAL_EPS": float(args.eps),
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "GP_CENTRAL_MONOTONIC_STRENGTH_FRAC": 0.0,
        "NATIVE_SCENE_SUPPORT_THRESHOLD": 0.6,
        "GP_COMPONENTS_NONNEGATIVE": False,
    }

    prf_label = "NO_PRF" if args.no_prf else "WITH_PRF"
    stage = f"DIAG_GP_N{n_tpl}_{prf_label}"
    print(f"\nRunning: {stage}")
    print(f"  diagonal GP (ε={args.eps:.1e}), {n_tpl} BCDs, "
          f"PRF={'OFF — identity operator' if args.no_prf else 'ON'}")
    if args.no_prf:
        print("  NOTE: Without PRF, Hessian diagonal is exact. Low chi2 → Hessian/PRF inconsistency confirmed.")
    sys.stdout.flush()

    prf_ctx = _no_prf_context() if args.no_prf else contextlib.nullcontext()

    with _temporary_config(overrides), prf_ctx:
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

    if results is None:
        raise SystemExit("Solver returned no results.")

    with _temporary_config(overrides), prf_ctx:
        metrics = fit_metrics.compute_fit_metrics(
            cutouts,
            results,
            stars,
            results.get("star_fluxes", init_star_fluxes),
            center_ra_deg=float(nr),
            center_dec_deg=float(nd),
            center_radius_px=3.0,
        )

    print(f"\n=== {stage} RESULTS ===")
    print(f"  n_templates          = {n_tpl}")
    print(f"  prf_active           = {not args.no_prf}")
    print(f"  total_reduced_chi2   = {metrics['total_reduced_chi2']:.4f}")
    print(f"  center_reduced_chi2  = {metrics['center_reduced_chi2']:.2f}")
    print(f"  center_chi2          = {metrics['center_chi2']:.1f}  (ndof={metrics['center_ndof']})")

    with _temporary_config(overrides), prf_ctx:
        pdf = write_native_fit_pdf(stage, cutouts, results, args.output_dir)
        stk = write_stacked_residual_pdf(stage, cutouts, results, args.output_dir)
    print(f"\nWrote diagnostic PDF: {pdf}")
    print(f"Wrote stacked PDF:    {stk}")

    out_json = os.path.join(args.output_dir, f"{stage.lower()}_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "stage": stage,
            "n_templates": n_tpl,
            "diagonal_eps": float(args.eps),
            "prf_active": not args.no_prf,
            "gp_kernel": "diagonal",
            "metrics": metrics,
            "artifacts": {"diagnostic_pdf": pdf, "stacked_pdf": stk},
        }, f, indent=2, default=str)
    print(f"Wrote summary JSON:   {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
