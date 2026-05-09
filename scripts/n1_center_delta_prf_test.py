#!/usr/bin/env python3
"""N=1 test: center delta-only model through real PRF + fitted background."""
from __future__ import annotations

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
    _valid_mask,
    apply_native_cutout_cr_mask,
    prepare_real_template_case,
)


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "n1_center_delta_prf_test")
    os.makedirs(out_dir, exist_ok=True)

    real_case = prepare_real_template_case()
    cut = _reindex_epochs([dict(c) for c in list(real_case["template_cutouts"])[:1]])[0]
    apply_native_cutout_cr_mask(cut)
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    nr = float(real_case["centers"]["nuc_ra"])
    nd = float(real_case["centers"]["nuc_dec"])

    data = np.asarray(cut["data"], dtype=float)
    sig = np.asarray(cut["sigma"], dtype=float)
    vm = _analysis_mask(cut)
    w = np.zeros_like(data, dtype=float)
    good = vm & np.isfinite(sig) & (sig > 0)
    w[good] = 1.0 / (sig[good] ** 2)
    if not np.any(good):
        raise SystemExit("No valid weighted pixels for delta PRF test.")

    # Build a pure delta intrinsic scene at galaxy center.
    delta_scene = np.zeros(scene_shape, dtype=float)
    sx, sy = scene_wcs.world_to_pixel_values(nr, nd)
    solver._add_delta_to_image(delta_scene, float(sx), float(sy), 1.0)

    chan = "ch2" if "ch2" in cut["filename"] else "ch1"
    prf_native = solver._apply_frame_forward_operator(
        delta_scene,
        scene_wcs,
        cut["raw_wcs"],
        scene_shape,
        data.shape,
        chan,
        is_full_array=bool(cut.get("is_full_array", False)),
    )

    # Weighted linear solve for amplitude + background.
    p = np.asarray(prf_native, dtype=float).ravel()
    d = data.ravel()
    ww = w.ravel()
    A00 = float(np.sum(ww * p * p))
    A01 = float(np.sum(ww * p))
    A11 = float(np.sum(ww))
    b0 = float(np.sum(ww * p * d))
    b1 = float(np.sum(ww * d))
    M = np.array([[A00, A01], [A01, A11]], dtype=float)
    rhs = np.array([b0, b1], dtype=float)
    amp, bg = np.linalg.solve(M, rhs)

    model = amp * np.asarray(prf_native, dtype=float) + bg
    resid = data - model

    results_like = {
        "gp_scene": np.zeros(scene_shape, dtype=float),
        "model_scene": np.zeros(scene_shape, dtype=float),
        "scene_wcs": scene_wcs,
        "scene_shape": scene_shape,
        "bcd_backgrounds": np.array([bg], dtype=float),
        "host_core_flux": 0.0,
        "nuclear_point_flux": float(amp),
    }
    metrics = fit_metrics.compute_fit_metrics(
        [cut],
        results_like,
        [],
        np.zeros(0, dtype=float),
        center_ra_deg=nr,
        center_dec_deg=nd,
        center_radius_px=3.0,
    )

    # Plot
    vm_valid = _valid_mask(cut)
    dlim = np.nanpercentile(data[vm], [1, 99]) if np.any(vm) else [np.nanmin(data), np.nanmax(data)]
    rv0, rv1 = _resid_limits(np.where(vm, resid, 0.0))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = ax[0].imshow(np.where(vm_valid, data, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[0].set_title("Data")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(np.where(vm_valid, model, 0.0), origin="lower", cmap="gray", vmin=float(dlim[0]), vmax=float(dlim[1]))
    ax[1].set_title("Delta@center * PRF + BG")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(np.where(vm, resid, 0.0), origin="lower", cmap="RdBu_r", vmin=rv0, vmax=rv1)
    ax[2].set_title("Residual")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    for a in ax:
        a.axis("off")
    fig.suptitle(
        "N=1 center-delta PRF test | amp={:.3e} bg={:.3e} | total_red_chi2={:.3f} center_red_chi2={:.3f}".format(
            float(amp),
            float(bg),
            float(metrics.get("total_reduced_chi2", np.nan)),
            float(metrics.get("center_reduced_chi2", np.nan)),
        ),
    )
    plt.tight_layout()
    out_png = os.path.join(out_dir, "N1_CENTER_DELTA_PRF_TEST.png")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    out_json = os.path.join(out_dir, "n1_center_delta_prf_test_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "amp": float(amp),
                "background": float(bg),
                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                "artifact_png": out_png,
            },
            f,
            indent=2,
        )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

