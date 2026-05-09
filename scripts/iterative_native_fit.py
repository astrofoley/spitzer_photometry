#!/usr/bin/env python3
"""Single-command iterative native fit campaign."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    prepare_real_template_case,
    run_gp_tier_sequence,
    run_stage,
    write_campaign_summary,
)

# (stage label for PDFs/logs, number of template BCDs)
_SINGLE_STAGE_MODES = {
    "n1_only": ("N1", 1),
    "n2_only": ("N2", 2),
}


def parse_args():
    p = argparse.ArgumentParser(description="Iterative native-fit campaign")
    p.add_argument(
        "--mode",
        default="n1_only",
        choices=["full_campaign", "n1_only", "n2_only"],
        help="n1_only (default): one BCD; n2_only: two BCDs — each defaults to a single fit with nuclear point on. full_campaign: N1, N10, Nall (gated).",
    )
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--center-radius-px", type=float, default=3.0)
    p.add_argument("--center-reduced-chi2-target", type=float, default=1.5)
    p.add_argument("--output-dir", default=os.path.join(config.DIAGNOSTIC_DIR, "iterative_campaign"))
    p.add_argument("--nall", type=int, default=None, help="Override all-template count cap")
    p.add_argument("--allow-synthetic", action="store_true", help="Explicitly allow synthetic debug mode")
    p.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    p.add_argument(
        "--allow-point-source",
        action="store_true",
        help="Try nuclear point on and off each iteration; best trial by center reduced chi2.",
    )
    p.add_argument(
        "--require-nuclear-point-source",
        action="store_true",
        help="Force nuclear point on (e.g. full_campaign). n1_only/n2_only already default to a single nuclear run.",
    )
    p.add_argument(
        "--no-nuclear-point-source",
        action="store_true",
        help="For n1_only/n2_only: fit without the unresolved nuclear point (overrides default).",
    )
    p.add_argument(
        "--gp-tier-auto",
        action="store_true",
        help="For n1_only/n2_only only: run automated GP tier A→B→C under output_dir/gp_tier_tuning/… "
        "(manifest + HUMAN_REVIEW_TIER_D.md); Tier D is never automated.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.data_source == "synthetic" and not args.allow_synthetic:
        raise SystemExit("Synthetic data mode requires --allow-synthetic (safety rule).")
    if args.require_nuclear_point_source and args.no_nuclear_point_source:
        raise SystemExit("Use only one of --require-nuclear-point-source and --no-nuclear-point-source.")
    if args.gp_tier_auto and args.mode not in _SINGLE_STAGE_MODES:
        raise SystemExit("--gp-tier-auto is only supported with --mode n1_only or n2_only.")
    # Single-stage modes default to one solve with nuclear point on (testing workflow).
    # --allow-point-source: compare with/without nuclear instead. --no-nuclear-point-source: GP only.
    if args.allow_point_source:
        req_np = False
    elif args.no_nuclear_point_source:
        req_np = False
    elif bool(args.require_nuclear_point_source):
        req_np = True
    elif args.mode in _SINGLE_STAGE_MODES:
        req_np = True
    else:
        req_np = False
    os.makedirs(args.output_dir, exist_ok=True)
    # Canonical Nall outputs must only exist if Nall stage executes.
    for p in (
        os.path.join(args.output_dir, "NATIVE_FIT_DIAGNOSTIC_Nall.pdf"),
        os.path.join(args.output_dir, "STACKED_RESIDUALS_Nall.pdf"),
    ):
        if os.path.exists(p):
            os.remove(p)
    stages = []
    iter_logs = {}
    cmd = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    real_case = None
    if args.data_source == "real":
        real_case = prepare_real_template_case()

    if args.mode in _SINGLE_STAGE_MODES:
        stage_name, n_bcd = _SINGLE_STAGE_MODES[args.mode]
        if args.gp_tier_auto:
            if args.data_source != "real":
                raise SystemExit("--gp-tier-auto requires real data (--data-source real).")
            if real_case is None:
                raise SystemExit("--gp-tier-auto requires a prepared real template case.")
            manifest = run_gp_tier_sequence(
                stage_name,
                n_bcd,
                real_case=real_case,
                base_output_dir=args.output_dir,
                center_radius_px=args.center_radius_px,
                reduced_chi2_target=args.center_reduced_chi2_target,
                require_nuclear_point=req_np,
                data_source=args.data_source,
                use_point=not args.no_nuclear_point_source,
            )
            run_root = str(manifest.get("run_root", ""))
            mp = str(manifest.get("manifest_json", os.path.join(run_root, "GP_TIER_RUN_MANIFEST.json")))
            human = str(manifest.get("human_review_md", os.path.join(run_root, "HUMAN_REVIEW_TIER_D.md")))
            print(f"GP tier run root: {run_root}")
            print(f"Wrote {mp}")
            print(f"Wrote {human}")
            print(f"Campaign status: {stage_name} gp-tier-auto (A→B→C); see manifest for gate results")
            return 0

        s_run, l_run = run_stage(
            stage_name,
            n_bcd,
            max_iterations=args.max_iters,
            center_radius_px=args.center_radius_px,
            output_dir=args.output_dir,
            reduced_chi2_target=args.center_reduced_chi2_target,
            real_case=real_case,
            data_source=args.data_source,
            allow_point_source=args.allow_point_source,
            require_nuclear_point=req_np,
            aggressive_recovery=True,
        )
        stages.append(s_run)
        iter_logs[stage_name] = l_run
        promote = False
        iter_logs["N10"] = [{"skipped": True, "reason": f"{args.mode} mode"}]
        iter_logs["Nall"] = [{"skipped": True, "reason": f"{args.mode} mode"}]
        summary_json, summary_md = write_campaign_summary(args.output_dir, stages, iter_logs, command=cmd)
        best_cfg = os.path.join(args.output_dir, "best_config_snapshot.json")
        with open(best_cfg, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stages": [s.stage_name for s in stages],
                    "latest_knobs": stages[-1].best_knobs if stages else {},
                    "promotion_to_nall": False,
                },
                f,
                indent=2,
            )
        print(f"Wrote {summary_json}")
        print(f"Wrote {summary_md}")
        print(f"Wrote {best_cfg}")
        print(f"Campaign status: {stage_name} only, {n_bcd} BCD(s) (N10 and Nall skipped)")
        return 0

    # full_campaign: Stage N1
    s1, l1 = run_stage(
        "N1",
        1,
        max_iterations=args.max_iters,
        center_radius_px=args.center_radius_px,
        output_dir=args.output_dir,
        reduced_chi2_target=args.center_reduced_chi2_target,
        real_case=real_case,
        data_source=args.data_source,
        allow_point_source=args.allow_point_source,
        require_nuclear_point=req_np,
        aggressive_recovery=True,
    )
    stages.append(s1)
    iter_logs["N1"] = l1

    # Stage N10
    s10, l10 = run_stage(
        "N10",
        10,
        max_iterations=args.max_iters,
        center_radius_px=args.center_radius_px,
        output_dir=args.output_dir,
        reduced_chi2_target=args.center_reduced_chi2_target,
        real_case=real_case,
        data_source=args.data_source,
        allow_point_source=args.allow_point_source,
        require_nuclear_point=req_np,
        aggressive_recovery=True,
    )
    stages.append(s10)
    iter_logs["N10"] = l10

    # Promotion gate requested by user:
    # run all templates if primary criterion met OR within 2x Poisson after 20 iterations.
    # Gate: N10 passes if primary, or fallback only after full 20 iterations.
    promote = bool(s10.met_primary or (s10.iterations_run >= 20 and s10.met_fallback))
    if promote:
        nall = int(args.nall) if args.nall is not None else (len(real_case["template_cutouts"]) if real_case else 30)
        sall, lall = run_stage(
            "Nall",
            nall,
            max_iterations=args.max_iters,
            center_radius_px=args.center_radius_px,
            output_dir=args.output_dir,
            reduced_chi2_target=args.center_reduced_chi2_target,
            real_case=real_case,
            data_source=args.data_source,
            allow_point_source=args.allow_point_source,
            require_nuclear_point=req_np,
            aggressive_recovery=True,
        )
        stages.append(sall)
        iter_logs["Nall"] = lall

        # Required final-naming deliverables
        final_diag = os.path.join(args.output_dir, "NATIVE_FIT_DIAGNOSTIC_Nall.pdf")
        final_stack = os.path.join(args.output_dir, "STACKED_RESIDUALS_Nall.pdf")
        if sall.diagnostic_pdf != final_diag:
            os.replace(sall.diagnostic_pdf, final_diag)
            sall.diagnostic_pdf = final_diag
        if sall.stacked_pdf != final_stack:
            os.replace(sall.stacked_pdf, final_stack)
            sall.stacked_pdf = final_stack
    else:
        # Do not emit canonical Nall outputs when gate fails.
        iter_logs["Nall"] = [{
            "skipped": True,
            "reason": "N10 did not meet primary criterion or 20-iteration fallback criterion; Nall not executed",
        }]

    summary_json, summary_md = write_campaign_summary(args.output_dir, stages, iter_logs, command=cmd)
    best_cfg = os.path.join(args.output_dir, "best_config_snapshot.json")
    with open(best_cfg, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stages": [s.stage_name for s in stages],
                "latest_knobs": stages[-1].best_knobs if stages else {},
                "promotion_to_nall": promote,
            },
            f,
            indent=2,
        )

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {best_cfg}")
    if promote:
        print("Campaign status: success (Nall executed)")
        return 0
    print("Campaign status: partial (Nall gate not met; no Nall PDFs emitted)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
