#!/usr/bin/env python3
"""Tune native-cutout CR masking to preserve nucleus while retaining outside flags."""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config
from src.native_fit_campaign import prepare_real_template_case


def _valid_mask(cutout: Dict[str, object]) -> np.ndarray:
    d = np.asarray(cutout["data"], dtype=float)
    s = np.asarray(cutout["sigma"], dtype=float)
    return (d != 0) & np.isfinite(s) & (s < 1e20)


def _local_peak_pixel(cutout: Dict[str, object]) -> Tuple[float, float]:
    d = np.asarray(cutout["data"], dtype=float)
    vm = _valid_mask(cutout)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    win = (np.hypot(xx - cx, yy - cy) <= 12.0) & vm
    if np.any(win):
        idx = int(np.nanargmax(np.where(win, d, -np.inf)))
        py, px = np.unravel_index(idx, d.shape)
        return float(px), float(py)
    return float(cx), float(cy)


def _baseline_mask(cutout: Dict[str, object], sigma_thresh: float = 7.0) -> np.ndarray:
    d = np.asarray(cutout["data"], dtype=float)
    s = np.asarray(cutout["sigma"], dtype=float)
    vm = _valid_mask(cutout)
    if np.sum(vm) < 16:
        return np.zeros_like(d, dtype=bool)
    med = median_filter(d, size=5, mode="nearest")
    resid = d - med
    with np.errstate(divide="ignore", invalid="ignore"):
        nsig = resid / np.clip(s, 1e-30, None)
    return vm & np.isfinite(nsig) & (nsig > float(sigma_thresh))


def _mask_algo(cutout: Dict[str, object], *, sigma_thresh: float, bright_pct: float, dilate_iter: int) -> np.ndarray:
    d = np.asarray(cutout["data"], dtype=float)
    vm = _valid_mask(cutout)
    cr = _baseline_mask(cutout, sigma_thresh=sigma_thresh)
    loc = median_filter(d, size=9, mode="nearest")
    vv = loc[vm & np.isfinite(loc)]
    if vv.size > 16:
        thr = float(np.percentile(vv, float(bright_pct)))
        bright = vm & (loc >= thr)
        bright = binary_dilation(bright, iterations=int(dilate_iter))
        cr &= ~bright
    return cr


def _region_masks(cutout: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    vm = _valid_mask(cutout)
    tx, ty = _local_peak_pixel(cutout)
    yy, xx = np.mgrid[0 : vm.shape[0], 0 : vm.shape[1]]
    rr = np.hypot(xx - float(tx), yy - float(ty))
    inside = vm & (rr <= 5.0)
    outside = vm & (rr > 10.0)
    return inside, outside


def evaluate(cutouts: List[Dict[str, object]], baseline_flags: np.ndarray, params: Dict[str, object]) -> Dict[str, float]:
    inside_total = 0
    inside_masked = 0
    outside_baseline = 0
    outside_recovered = 0
    for i, c in enumerate(cutouts):
        cand = _mask_algo(
            c,
            sigma_thresh=float(params["sigma_thresh"]),
            bright_pct=float(params["bright_pct"]),
            dilate_iter=int(params["dilate_iter"]),
        )
        inside, outside = _region_masks(c)
        base = baseline_flags[i].astype(bool)
        inside_total += int(np.sum(inside))
        inside_masked += int(np.sum(cand & inside))
        outside_baseline += int(np.sum(base & outside))
        outside_recovered += int(np.sum(cand & base & outside))
    inside_retained = 1.0 - (inside_masked / max(inside_total, 1))
    outside_recall = outside_recovered / max(outside_baseline, 1)
    return {
        "inside_retained_frac": float(inside_retained),
        "outside_recall_vs_baseline": float(outside_recall),
        "inside_total": int(inside_total),
        "inside_masked": int(inside_masked),
        "outside_baseline_flagged": int(outside_baseline),
        "outside_recovered_flagged": int(outside_recovered),
    }


def main() -> int:
    out_dir = os.path.join(config.DIAGNOSTIC_DIR, "iterative_campaign")
    os.makedirs(out_dir, exist_ok=True)
    case = prepare_real_template_case()
    cutouts = list(case["template_cutouts"])

    # Baseline: current pre-tuning behavior snapshot (without modifying current code path).
    baseline = []
    for c in cutouts:
        b = _baseline_mask(c, sigma_thresh=7.0)
        baseline.append(b.astype(np.uint8))
    baseline_arr = np.asarray(baseline, dtype=np.uint8)
    np.save(os.path.join(out_dir, "cr_flags_baseline_current.npy"), baseline_arr)

    candidates = []
    for sigma_thresh in (6.0, 7.0, 8.0, 9.0):
        for bright_pct in (99.0, 99.3, 99.5, 99.7, 99.9):
            for dilate_iter in (1, 2, 3):
                candidates.append(
                    {
                        "sigma_thresh": sigma_thresh,
                        "bright_pct": bright_pct,
                        "dilate_iter": dilate_iter,
                    }
                )

    history: List[Dict[str, object]] = []
    best = None
    for p in candidates:
        m = evaluate(cutouts, baseline_arr, p)
        row = {**p, **m}
        history.append(row)
        ok = (m["inside_retained_frac"] >= 0.90) and (m["outside_recall_vs_baseline"] >= 0.90)
        if ok:
            best = row
            break
        if best is None:
            best = row
        else:
            bscore = (best["inside_retained_frac"] >= 0.90, best["outside_recall_vs_baseline"])
            nscore = (row["inside_retained_frac"] >= 0.90, row["outside_recall_vs_baseline"])
            if nscore > bscore:
                best = row

    payload = {
        "criteria": {
            "inside_retained_min": 0.90,
            "outside_recall_vs_baseline_min": 0.90,
        },
        "n_template_bcd": int(len(cutouts)),
        "baseline_file": os.path.join(out_dir, "cr_flags_baseline_current.npy"),
        "best": best,
        "met_criteria": bool(
            best
            and best["inside_retained_frac"] >= 0.90
            and best["outside_recall_vs_baseline"] >= 0.90
        ),
        "history": history,
    }
    out_json = os.path.join(out_dir, "cr_mask_tuning_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_json}")
    if payload["met_criteria"]:
        print(
            "Selected params:",
            f"sigma_thresh={best['sigma_thresh']}",
            f"bright_pct={best['bright_pct']}",
            f"dilate_iter={best['dilate_iter']}",
        )
        return 0
    print("No candidate met criteria; review history for next algorithm iteration.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
