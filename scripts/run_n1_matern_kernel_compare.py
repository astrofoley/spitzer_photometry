#!/usr/bin/env python3
"""N=1, one iteration: compare GP scene priors Matérn 1/2 vs 3/2 on real template data.

Writes PDFs and iteration metrics under output/diagnostics/ (see config.DIAGNOSTIC_DIR).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src.native_fit_campaign import (  # noqa: E402
    _temporary_config,
    prepare_real_template_case,
    run_stage,
)


def main() -> int:
    base = os.path.join(config.DIAGNOSTIC_DIR, "n1_matern_kernel_compare")
    os.makedirs(base, exist_ok=True)
    real_case = prepare_real_template_case()
    summary: dict = {"output_root": base, "runs": []}
    for order in ("matern12", "matern32"):
        out = os.path.join(base, order)
        os.makedirs(out, exist_ok=True)
        with _temporary_config({"GP_MATERN_ORDER": order}):
            stage, iter_log = run_stage(
                "N1",
                1,
                max_iterations=1,
                center_radius_px=3.0,
                output_dir=out,
                reduced_chi2_target=1.5,
                real_case=real_case,
                data_source="real",
                allow_point_source=False,
                require_nuclear_point=True,
                aggressive_recovery=True,
            )
        summary["runs"].append(
            {
                "GP_MATERN_ORDER": order,
                "best_iteration": stage.best_iteration,
                "best_knobs": stage.best_knobs,
                "best_metrics": stage.best_metrics,
                "diagnostic_pdf": stage.diagnostic_pdf,
                "stacked_pdf": stage.stacked_pdf,
                "iter_log": iter_log,
            }
        )
        print(f"[{order}] diagnostic_pdf={stage.diagnostic_pdf}")
        print(f"[{order}] stacked_pdf={stage.stacked_pdf}")
        print(f"[{order}] best_metrics={stage.best_metrics}")
    out_json = os.path.join(base, "compare_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
