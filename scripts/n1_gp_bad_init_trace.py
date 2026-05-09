#!/usr/bin/env python3
"""Single N=1 GP-only multi-start trace with per-iteration optimizer logging."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

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

_GRID_WORKER_STATE = None


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trace GP hyperparameter optimization for one N=1 native fit")
    p.add_argument(
        "--output-dir",
        default=os.path.join(config.DIAGNOSTIC_DIR, "iterative_campaign", "n1_gp_trace_bad_init"),
        help="Output directory for trace logs and plots",
    )
    p.add_argument(
        "--include-control-start",
        action="store_true",
        help="Also include a moderate control start near previous defaults",
    )
    p.add_argument(
        "--single-start",
        default=None,
        choices=["center", "lowEll_lowVar", "highEll_lowVar", "lowEll_highVar", "highEll_highVar"],
        help="Run only one configured start label",
    )
    p.add_argument("--objective", choices=["center", "total"], default="center", help="Objective used by optimizer")
    p.add_argument("--maxfev", type=int, default=6, help="Maximum objective evaluations per start")
    p.add_argument("--fd-rel-step", type=float, default=1e-2, help="Finite-difference relative step for gradients")
    p.add_argument(
        "--optimizer",
        choices=["powell", "lbfgsb"],
        default="powell",
        help="Optimizer for log(ell),log(var). Powell enforces real step exploration.",
    )
    p.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip heavy per-run diagnostic PDFs during sweeps",
    )
    p.add_argument(
        "--run-grid-if-stuck",
        action="store_true",
        help="If none of the 5 starts moves both parameters enough, run coarse ell/var grid sweep",
    )
    p.add_argument(
        "--grid-only",
        action="store_true",
        help="Skip optimizer starts and run only fixed ell/var chi2 surface grid",
    )
    p.add_argument("--grid-n-ell", type=int, default=7, help="Number of ell grid points for grid-only scan")
    p.add_argument("--grid-n-var", type=int, default=7, help="Number of var grid points for grid-only scan")
    p.add_argument("--grid-workers", type=int, default=1, help="Parallel worker processes for grid scan")
    return p


def _write_trace_plot(
    eval_trace: List[Dict[str, float]],
    iter_trace: List[Dict[str, float]],
    out_png: str,
    title_prefix: str,
) -> None:
    eidx = np.arange(len(eval_trace), dtype=float)
    ell = np.array([float(r["ell"]) for r in eval_trace], dtype=float)
    var = np.array([float(r["var"]) for r in eval_trace], dtype=float)
    ctr = np.array([float(r["center_reduced_chi2"]) for r in eval_trace], dtype=float)
    tot = np.array([float(r["total_reduced_chi2"]) for r in eval_trace], dtype=float)
    obj = np.array([float(r["objective_value"]) for r in eval_trace], dtype=float)
    best = np.minimum.accumulate(obj)

    iidx = np.arange(len(iter_trace), dtype=float)
    iell = np.array([float(r["ell"]) for r in iter_trace], dtype=float) if iter_trace else np.array([])
    ivar = np.array([float(r["var"]) for r in iter_trace], dtype=float) if iter_trace else np.array([])
    istep = np.array([float(r["step_norm"]) for r in iter_trace], dtype=float) if iter_trace else np.array([])

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax = ax.ravel()
    ax[0].plot(eidx, ell, marker="o", lw=1.3, label="evaluated")
    if iter_trace:
        ax[0].plot(iidx, iell, marker="s", lw=1.0, label="accepted_iterates")
        ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title(f"{title_prefix}: ell")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("ell")
    ax[0].set_yscale("log")
    ax[0].grid(alpha=0.25)

    ax[1].plot(eidx, var, marker="o", lw=1.3, color="tab:orange", label="evaluated")
    if iter_trace:
        ax[1].plot(iidx, ivar, marker="s", lw=1.0, color="tab:red", label="accepted_iterates")
        ax[1].legend(loc="best", fontsize=8)
    ax[1].set_title(f"{title_prefix}: var")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("var")
    ax[1].set_yscale("log")
    ax[1].grid(alpha=0.25)

    ax[2].plot(eidx, ctr, marker="o", lw=1.3, label="center reduced chi2")
    ax[2].plot(eidx, tot, marker="s", lw=1.1, label="total reduced chi2")
    ax[2].set_title(f"{title_prefix}: fit metrics")
    ax[2].set_xlabel("evaluation")
    ax[2].set_ylabel("reduced chi2")
    ax[2].grid(alpha=0.25)
    ax[2].legend(loc="best", fontsize=8)

    ax[3].plot(eidx, obj, marker="o", lw=1.3, label="objective")
    ax[3].plot(eidx, best, lw=1.0, ls="--", label="best-so-far")
    if iter_trace:
        ax[3].plot(iidx, istep, marker="^", lw=1.0, label="iterate_step_norm")
    ax[3].set_title(f"{title_prefix}: objective progression")
    ax[3].set_xlabel("step")
    ax[3].set_ylabel("objective / step norm")
    ax[3].grid(alpha=0.25)
    ax[3].legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _write_grid_maps(rows: List[Dict[str, float]], out_png: str) -> None:
    ell = np.array([float(r["ell"]) for r in rows], dtype=float)
    var = np.array([float(r["var"]) for r in rows], dtype=float)
    ctr = np.array([float(r["center_reduced_chi2"]) for r in rows], dtype=float)
    tot = np.array([float(r["total_reduced_chi2"]) for r in rows], dtype=float)
    dctr = ctr - np.nanmin(ctr)
    dtot = tot - np.nanmin(tot)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, (z, ttl) in enumerate(((dctr, "Delta center reduced chi2"), (dtot, "Delta total reduced chi2"))):
        tcf = ax[i].tricontourf(ell, var, z, levels=14)
        ax[i].tricontour(ell, var, z, levels=7, colors="k", linewidths=0.5)
        ax[i].scatter(ell, var, s=22, c="white", edgecolor="k", linewidths=0.5)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("ell")
        ax[i].set_ylabel("var")
        ax[i].set_title(ttl)
        plt.colorbar(tcf, ax=ax[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _ran_moves_both(run_row: Dict[str, object], frac: float = 0.2) -> bool:
    init = run_row["init"]
    best = run_row["best"]
    if best is None:
        return False
    d_ell = abs(float(best["ell"]) - float(init["ell"])) / max(abs(float(init["ell"])), 1e-12)
    d_var = abs(float(best["var"]) - float(init["var"])) / max(abs(float(init["var"])), 1e-12)
    return (d_ell >= frac) and (d_var >= frac)


def _run_start(
    *,
    label: str,
    init_ell: float,
    init_var: float,
    args,
    cutouts,
    scene_wcs,
    scene_shape,
    stars,
    init_star_fluxes,
    nr: float,
    nd: float,
) -> Dict[str, object]:
    eval_trace: List[Dict[str, float]] = []
    iter_trace: List[Dict[str, float]] = []
    best_row: Dict[str, float] | None = None
    best_results = None
    t0 = time.perf_counter()
    prev_iterate = None

    overrides = {
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
    }

    def _objective(log_params: np.ndarray) -> float:
        nonlocal best_row, best_results
        ell = float(np.exp(float(log_params[0])))
        var = float(np.exp(float(log_params[1])))
        eval_idx = len(eval_trace)
        try:
            with _temporary_config(overrides):
                results = solver.run_gls_solve(
                    cutouts,
                    stars,
                    init_star_fluxes,
                    {"ell": ell, "var": var},
                    (ell, var),
                    np.zeros(scene_shape),
                    scene_wcs,
                    len(cutouts),
                )
            metrics = fit_metrics.compute_fit_metrics(
                cutouts,
                results,
                stars,
                results.get("star_fluxes", init_star_fluxes),
                center_ra_deg=float(nr),
                center_dec_deg=float(nd),
                center_radius_px=3.0,
            )
            obj = float(metrics["center_reduced_chi2"] if args.objective == "center" else metrics["total_reduced_chi2"])
            row: Dict[str, float] = {
                "evaluation": float(eval_idx),
                "elapsed_s": float(time.perf_counter() - t0),
                "ell": ell,
                "var": var,
                "objective_value": obj,
                "center_reduced_chi2": float(metrics["center_reduced_chi2"]),
                "total_reduced_chi2": float(metrics["total_reduced_chi2"]),
            }
            eval_trace.append(row)
            if best_row is None or obj < float(best_row["objective_value"]):
                best_row = dict(row)
                best_results = results
            print(
                f"[{label} eval {eval_idx:02d}] ell={ell:.6g} var={var:.6g} "
                f"obj={obj:.6f} center={metrics['center_reduced_chi2']:.6f} total={metrics['total_reduced_chi2']:.6f}",
            )
            return obj
        except Exception as exc:
            print(f"[{label} eval {eval_idx:02d}] FAILED: {exc}")
            eval_trace.append(
                {
                    "evaluation": float(eval_idx),
                    "elapsed_s": float(time.perf_counter() - t0),
                    "ell": ell,
                    "var": var,
                    "objective_value": float("inf"),
                    "center_reduced_chi2": float("inf"),
                    "total_reduced_chi2": float("inf"),
                },
            )
            return float("inf")

    def _cb(xk: np.ndarray) -> None:
        nonlocal prev_iterate
        ell = float(np.exp(float(xk[0])))
        var = float(np.exp(float(xk[1])))
        if prev_iterate is None:
            step_norm = 0.0
        else:
            step_norm = float(np.linalg.norm(np.asarray(xk) - np.asarray(prev_iterate)))
        prev_iterate = np.asarray(xk, dtype=float).copy()
        iter_trace.append(
            {
                "iteration": float(len(iter_trace)),
                "elapsed_s": float(time.perf_counter() - t0),
                "ell": ell,
                "var": var,
                "step_norm": step_norm,
            },
        )
        print(f"[{label} iter {len(iter_trace)-1:02d}] ell={ell:.6g} var={var:.6g} step_norm={step_norm:.4e}")

    x0 = np.array(
        [np.log(max(init_ell, 1e-12)), np.log(max(init_var, 1e-20))],
        dtype=float,
    )
    bounds = [(np.log(0.01), np.log(100.0)), (np.log(1e-12), np.log(10.0))]
    if args.optimizer == "powell":
        opt = minimize(
            _objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            callback=_cb,
            options={
                "maxfev": int(max(4, args.maxfev)),
                "maxiter": int(max(2, args.maxfev // 2)),
                "xtol": 1e-3,
                "ftol": 1e-6,
            },
        )
    else:
        opt = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            callback=_cb,
            options={
                "maxfun": int(max(1, args.maxfev)),
                "maxiter": int(max(1, args.maxfev)),
                "ftol": 1e-8,
                "finite_diff_rel_step": float(max(args.fd_rel_step, 1e-6)),
            },
        )

    if best_results is None or best_row is None:
        raise SystemExit("No successful optimization evaluations were recorded")

    base = os.path.join(args.output_dir, label)
    os.makedirs(base, exist_ok=True)
    trace_json = os.path.join(base, "gp_param_trace.json")
    with open(trace_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": args.objective,
                "optimizer": args.optimizer,
                "start_label": label,
                "init": {"ell": float(init_ell), "var": float(init_var)},
                "optimizer_result": {
                    "success": bool(opt.success),
                    "status": int(opt.status),
                    "message": str(opt.message),
                    "nfev": int(getattr(opt, "nfev", len(eval_trace))),
                    "nit": int(getattr(opt, "nit", -1)),
                },
                "best_evaluation": best_row,
                "evaluate_trace": eval_trace,
                "iterate_trace": iter_trace,
            },
            f,
            indent=2,
            default=str,
        )

    trace_png = os.path.join(base, "GP_PARAM_TRACE_N1.png")
    _write_trace_plot(eval_trace, iter_trace, trace_png, title_prefix=label)

    diag_pdf = None
    stack_pdf = None
    if not args.skip_diagnostics:
        with _temporary_config(overrides):
            diag_pdf = write_native_fit_pdf(f"N1_TRACE_{label}", cutouts, best_results, base)
            stack_pdf = write_stacked_residual_pdf(f"N1_TRACE_{label}", cutouts, best_results, base)

    summary_json = os.path.join(base, "n1_gp_trace_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": args.objective,
                "optimizer": args.optimizer,
                "start_label": label,
                "init": {"ell": float(init_ell), "var": float(init_var)},
                "best": best_row,
                "optimizer": {
                    "success": bool(opt.success),
                    "status": int(opt.status),
                    "message": str(opt.message),
                    "nfev": int(getattr(opt, "nfev", len(eval_trace))),
                    "nit": int(getattr(opt, "nit", -1)),
                },
                "artifacts": {
                    "trace_json": trace_json,
                    "trace_plot_png": trace_png,
                    "diagnostic_pdf": diag_pdf,
                    "stacked_pdf": stack_pdf,
                },
            },
            f,
            indent=2,
            default=str,
        )

    return {
        "label": label,
        "init": {"ell": init_ell, "var": init_var},
        "best": best_row,
        "optimizer": {
            "success": bool(opt.success),
            "status": int(opt.status),
            "message": str(opt.message),
            "nfev": int(getattr(opt, "nfev", len(eval_trace))),
            "nit": int(getattr(opt, "nit", -1)),
        },
        "artifacts": {
            "summary_json": summary_json,
            "trace_json": trace_json,
            "trace_plot_png": trace_png,
            "diagnostic_pdf": diag_pdf,
            "stacked_pdf": stack_pdf,
        },
    }


def _starts(include_control: bool) -> List[Tuple[str, float, float]]:
    # Center first, then four corners away from hard bounds.
    starts: List[Tuple[str, float, float]] = [
        ("center", 2.0, 1.0),
        ("lowEll_lowVar", 0.03, 1e-10),
        ("highEll_lowVar", 60.0, 1e-10),
        ("lowEll_highVar", 0.03, 5.0),
        ("highEll_highVar", 60.0, 5.0),
    ]
    if include_control:
        starts.append(("center", 2.0, 1.0))
    return starts


def _grid_rows(
    cutouts,
    scene_wcs,
    scene_shape,
    stars,
    init_star_fluxes,
    nr: float,
    nd: float,
    n_ell: int = 5,
    n_var: int = 5,
) -> List[Dict[str, float]]:
    ell_grid = np.geomspace(0.02, 80.0, int(max(2, n_ell)))
    var_grid = np.geomspace(1e-11, 5.0, int(max(2, n_var)))
    points = [(float(e), float(v)) for e in ell_grid for v in var_grid]
    rows: List[Dict[str, float]] = []
    for e, v in points:
        row = _grid_eval_point(e, v, cutouts, scene_wcs, scene_shape, stars, init_star_fluxes, nr, nd)
        rows.append(row)
        print(
            f"[grid] ell={float(e):.5g} var={float(v):.5g} "
            f"center={row['center_reduced_chi2']:.6f} total={row['total_reduced_chi2']:.6f}",
        )
        if row["total_reduced_chi2"] <= 2.0 and row["center_reduced_chi2"] <= 10.0:
            return rows
    return rows


def _grid_eval_point(
    e: float,
    v: float,
    cutouts,
    scene_wcs,
    scene_shape,
    stars,
    init_star_fluxes,
    nr: float,
    nd: float,
) -> Dict[str, float]:
    overrides = {"USE_HOST_GAUSSIAN_CORE": False, "USE_NUCLEAR_POINT_SOURCE": False}
    with _temporary_config(overrides):
        res = solver.run_gls_solve(
            cutouts,
            stars,
            init_star_fluxes,
            {"ell": float(e), "var": float(v)},
            (float(e), float(v)),
            np.zeros(scene_shape),
            scene_wcs,
            len(cutouts),
        )
    met = fit_metrics.compute_fit_metrics(
        cutouts,
        res,
        stars,
        res.get("star_fluxes", init_star_fluxes),
        center_ra_deg=float(nr),
        center_dec_deg=float(nd),
        center_radius_px=3.0,
    )
    return {
        "ell": float(e),
        "var": float(v),
        "center_reduced_chi2": float(met["center_reduced_chi2"]),
        "total_reduced_chi2": float(met["total_reduced_chi2"]),
    }


def _grid_worker_init() -> None:
    global _GRID_WORKER_STATE
    real_case = prepare_real_template_case()
    cutouts = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])
    for c in cutouts:
        apply_native_cutout_cr_mask(c)
    centers = dict(real_case["centers"])
    _GRID_WORKER_STATE = {
        "cutouts": cutouts,
        "scene_wcs": real_case["scene_wcs"],
        "scene_shape": tuple(real_case["scene_shape"]),
        "stars": list(real_case["all_stars"]),
        "init_star_fluxes": np.asarray(real_case["init_star_fluxes"], dtype=float),
        "nr": float(centers["nuc_ra"]),
        "nd": float(centers["nuc_dec"]),
    }


def _grid_worker_point(point: Tuple[float, float]) -> Dict[str, float]:
    global _GRID_WORKER_STATE
    if _GRID_WORKER_STATE is None:
        _grid_worker_init()
    e, v = point
    s = _GRID_WORKER_STATE
    return _grid_eval_point(
        float(e),
        float(v),
        s["cutouts"],
        s["scene_wcs"],
        s["scene_shape"],
        s["stars"],
        s["init_star_fluxes"],
        s["nr"],
        s["nd"],
    )


def main() -> int:
    args = _make_parser().parse_args()
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
    nr = centers.get("nuc_ra")
    nd = centers.get("nuc_dec")
    if nr is None or nd is None:
        raise SystemExit("Nucleus coordinates missing in prepared case")

    all_rows = []
    grid_info = None
    if args.grid_only:
        print("Running grid-only fixed-hyperparameter chi2 surface scan...")
        if int(args.grid_workers) <= 1:
            grid_rows = _grid_rows(
                cutouts,
                scene_wcs,
                scene_shape,
                stars,
                init_star_fluxes,
                float(nr),
                float(nd),
                n_ell=int(args.grid_n_ell),
                n_var=int(args.grid_n_var),
            )
        else:
            ell_grid = np.geomspace(0.02, 80.0, int(max(2, args.grid_n_ell)))
            var_grid = np.geomspace(1e-11, 5.0, int(max(2, args.grid_n_var)))
            points = [(float(e), float(v)) for e in ell_grid for v in var_grid]
            grid_rows = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.grid_workers), initializer=_grid_worker_init) as ex:
                for row in ex.map(_grid_worker_point, points):
                    grid_rows.append(row)
                    print(
                        f"[grid] ell={row['ell']:.5g} var={row['var']:.5g} "
                        f"center={row['center_reduced_chi2']:.6f} total={row['total_reduced_chi2']:.6f}",
                    )
        grid_json = os.path.join(args.output_dir, "grid_scan_metrics.json")
        with open(grid_json, "w", encoding="utf-8") as f:
            json.dump(grid_rows, f, indent=2, default=str)
        grid_png = os.path.join(args.output_dir, "DELTA_CHI2_MAPS.png")
        _write_grid_maps(grid_rows, grid_png)
        best_center = min(grid_rows, key=lambda r: float(r["center_reduced_chi2"]))
        best_total = min(grid_rows, key=lambda r: float(r["total_reduced_chi2"]))
        grid_info = {
            "grid_json": grid_json,
            "delta_chi2_map_png": grid_png,
            "n_points": len(grid_rows),
            "best_center": best_center,
            "best_total": best_total,
        }
        campaign_json = os.path.join(args.output_dir, "multi_start_trace_summary.json")
        with open(campaign_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "objective": args.objective,
                    "optimizer": args.optimizer,
                    "maxfev": int(args.maxfev),
                    "fd_rel_step": float(args.fd_rel_step),
                    "grid_workers": int(args.grid_workers),
                    "runs": [],
                    "grid_followup": grid_info,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Wrote {grid_json}")
        print(f"Wrote {grid_png}")
        print(f"Wrote {campaign_json}")
        return 0

    starts = _starts(args.include_control_start)
    if args.single_start is not None:
        starts = [s for s in starts if s[0] == args.single_start]
        if not starts:
            raise SystemExit(f"--single-start '{args.single_start}' not present in active start set")
    for label, init_ell, init_var in starts:
        print(f"\n=== Start {label}: ell0={init_ell:.4g}, var0={init_var:.4g} ===")
        row = _run_start(
            label=label,
            init_ell=float(init_ell),
            init_var=float(init_var),
            args=args,
            cutouts=cutouts,
            scene_wcs=scene_wcs,
            scene_shape=scene_shape,
            stars=stars,
            init_star_fluxes=init_star_fluxes,
            nr=float(nr),
            nd=float(nd),
        )
        all_rows.append(row)
        print(f"Wrote {row['artifacts']['summary_json']}")
        if row["best"]["total_reduced_chi2"] <= 2.0 and row["best"]["center_reduced_chi2"] <= 10.0:
            print("Early stop: reached good chi^2 threshold.")
            break
        if label != "center":
            center_row = next((r for r in all_rows if r.get("label") == "center"), None)
            if center_row is not None:
                c0 = float(center_row["best"]["center_reduced_chi2"])
                c1 = float(row["best"]["center_reduced_chi2"])
                if np.isfinite(c0) and c0 > 0 and (c0 - c1) / c0 >= 0.2:
                    print("Early stop: dramatic center chi^2 improvement versus center start.")
                    break

    campaign_json = os.path.join(args.output_dir, "multi_start_trace_summary.json")
    grid_info = None
    if args.run_grid_if_stuck:
        moved_any = any(_ran_moves_both(r, frac=0.2) for r in all_rows)
        if not moved_any:
            print("No run moved both parameters much; running coarse grid sweep...")
            if int(args.grid_workers) <= 1:
                grid_rows = _grid_rows(
                    cutouts,
                    scene_wcs,
                    scene_shape,
                    stars,
                    init_star_fluxes,
                    float(nr),
                    float(nd),
                )
            else:
                ell_grid = np.geomspace(0.02, 80.0, 5)
                var_grid = np.geomspace(1e-11, 5.0, 5)
                points = [(float(e), float(v)) for e in ell_grid for v in var_grid]
                grid_rows = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.grid_workers), initializer=_grid_worker_init) as ex:
                    for row in ex.map(_grid_worker_point, points):
                        grid_rows.append(row)
                        print(
                            f"[grid] ell={row['ell']:.5g} var={row['var']:.5g} "
                            f"center={row['center_reduced_chi2']:.6f} total={row['total_reduced_chi2']:.6f}",
                        )
            grid_json = os.path.join(args.output_dir, "grid_scan_metrics.json")
            with open(grid_json, "w", encoding="utf-8") as f:
                json.dump(grid_rows, f, indent=2, default=str)
            grid_png = os.path.join(args.output_dir, "DELTA_CHI2_MAPS.png")
            _write_grid_maps(grid_rows, grid_png)
            grid_info = {"grid_json": grid_json, "delta_chi2_map_png": grid_png, "n_points": len(grid_rows)}
            print(f"Wrote {grid_json}")
            print(f"Wrote {grid_png}")

    with open(campaign_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": args.objective,
                "optimizer": args.optimizer,
                "maxfev": int(args.maxfev),
                "fd_rel_step": float(args.fd_rel_step),
                "grid_workers": int(args.grid_workers),
                "runs": all_rows,
                "grid_followup": grid_info,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Wrote {campaign_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
