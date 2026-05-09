"""Run N=3 for 10 iterations with floating GP hyperparameters and diagnostics."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

import argparse

from src import config, fit_metrics, native_fit_campaign


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--supersample", type=int, default=5, help="SUPERSAMPLE_FACTOR for scene/grid.")
    p.add_argument(
        "--diag-superres-display",
        type=int,
        default=5,
        help="DIAG_SUPERRES_DISPLAY_FACTOR for N-up panel upsampling in plots.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory (default includes supersample params).",
    )
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(
        config.OUTPUT_DIR,
        f"n3_iter10_hyperopt_trace_ss{args.supersample}_diag{args.diag_superres_display}",
    )
    os.makedirs(out_dir, exist_ok=True)

    with native_fit_campaign._temporary_config(
        {
            "SUPERSAMPLE_FACTOR": int(args.supersample),
            "DIAG_SUPERRES_DISPLAY_FACTOR": int(args.diag_superres_display),
        }
    ):
        real_case = native_fit_campaign.prepare_real_template_case()
        cutouts = native_fit_campaign._reindex_epochs(list(real_case["template_cutouts"][:3]))
        scene_wcs = real_case["scene_wcs"]
        scene_shape = tuple(real_case["scene_shape"])
        centers = dict(real_case["centers"])
        stars = list(real_case["all_stars"])
        init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
        for c in cutouts:
            native_fit_campaign.apply_native_cutout_cr_mask(c)

        base_var = 1.0e-7
        ell_seed = 1.8
        var_seed = 1.0e-7
        rows = []
        best = None
        best_metric = np.inf
        best_iter = -1

        for it in range(10):
            print(f"[N3-HYPEROPT] iteration {it + 1}/10")
            var_mult = float(var_seed / base_var)
            row = native_fit_campaign._solve_campaign_trial(
                cutouts=cutouts,
                scene_wcs=scene_wcs,
                scene_shape=scene_shape,
                centers=centers,
                stars=stars,
                init_star_fluxes=init_star_fluxes,
                ell=float(ell_seed),
                base_var=float(base_var),
                var_mult=float(var_mult),
                use_point=False,
                require_nuclear_point=False,
                smooth=float(getattr(config, "GP_FALLBACK_NEIGHBOR_SMOOTHNESS", 0.35)),
                merge_extra={},
                trial_label=f"N3_IT{it+1:02d}",
                center_radius_px=3.0,
                optimize_gp_params=True,
            )
            rows.append(row)
            if not row.get("ok", False):
                continue
            ell_seed = float(row.get("ell_effective", ell_seed))
            var_seed = float(row.get("var_product", var_seed))
            results = row["results"]
            metric = float(row.get("center_reduced_chi2", np.inf))
            if np.isfinite(metric) and metric < best_metric:
                best_metric = metric
                best = results
                best_iter = it + 1

            iter_dir = os.path.join(out_dir, f"iter_{it+1:02d}")
            os.makedirs(iter_dir, exist_ok=True)
            native_fit_campaign.write_native_fit_pdf(f"N3_IT{it+1:02d}", cutouts, results, iter_dir)
            native_fit_campaign.write_stacked_residual_pdf(
                f"N3_IT{it+1:02d}", cutouts, results, iter_dir
            )

        # Convergence plot
        it_idx = []
        ell_vals = []
        var_vals = []
        center_vals = []
        total_vals = []
        for i, r in enumerate(rows, start=1):
            if not r.get("ok", False):
                continue
            it_idx.append(i)
            ell_vals.append(float(r.get("ell_effective", np.nan)))
            var_vals.append(float(r.get("var_product", np.nan)))
            center_vals.append(float(r.get("center_reduced_chi2", np.nan)))
            total_vals.append(float(r.get("total_reduced_chi2", np.nan)))

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()
        axes[0].plot(it_idx, ell_vals, marker="o")
        axes[0].set_title("ell vs iteration")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("ell")
        axes[0].grid(alpha=0.3)
        axes[1].plot(it_idx, var_vals, marker="o")
        axes[1].set_yscale("log")
        axes[1].set_title("var vs iteration")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("var")
        axes[1].grid(alpha=0.3)
        axes[2].plot(it_idx, center_vals, marker="o")
        axes[2].set_title("center reduced chi2")
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("chi2")
        axes[2].grid(alpha=0.3)
        axes[3].plot(it_idx, total_vals, marker="o")
        axes[3].set_title("total reduced chi2")
        axes[3].set_xlabel("iteration")
        axes[3].set_ylabel("chi2")
        axes[3].grid(alpha=0.3)
        plt.tight_layout()
        conv_plot = os.path.join(out_dir, "HYPERPARAM_CONVERGENCE_N3.png")
        fig.savefig(conv_plot, dpi=160)
        plt.close(fig)

        summary = {
            "out_dir": out_dir,
            "best_iteration": best_iter,
            "best_center_reduced_chi2": best_metric,
            "rows": rows,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        if best is not None:
            native_fit_campaign.write_native_fit_pdf("N3_BEST", cutouts, best, out_dir)
            native_fit_campaign.write_stacked_residual_pdf("N3_BEST", cutouts, best, out_dir)
            best_metrics = fit_metrics.compute_fit_metrics(
                cutouts,
                best,
                stars,
                best.get("star_fluxes", init_star_fluxes),
                center_ra_deg=float(centers["nuc_ra"] or centers["transient_ra"]),
                center_dec_deg=float(centers["nuc_dec"] or centers["transient_dec"]),
                center_radius_px=3.0,
            )
            print("BEST_METRICS", best_metrics)
        print("OUT_DIR", out_dir)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
