#!/usr/bin/env python3
"""
Histogram of PRF values on the outer 1-pixel border of a peak-normalized PRF image.
Default mode uses the native interpolated PRF array (recommended for checking
whether PRF borders are truly zero). Optional mode uses the reprojected scene
stamp PRF used by the forward model.

Usage (from repo root):
  python scripts/prf_edge_histogram.py
  python scripts/prf_edge_histogram.py --channel ch1 --ref-flux-jy 1e-3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config, solver  # noqa: E402


def _border_mask(h: int, w: int, width: int = 1) -> np.ndarray:
    width = max(1, int(width))
    width = min(width, h // 2, w // 2)
    m = np.zeros((h, w), dtype=bool)
    m[:width, :] = True
    m[-width:, :] = True
    m[:, :width] = True
    m[:, -width:] = True
    return m


def _hist_range(values: np.ndarray, pad_frac: float = 0.1, floor: float = 1e-12):
    """
    Robust plotting range tightly around data.
    Falls back to a tiny symmetric range when all values are identical.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi <= lo:
        c = lo
        half = max(abs(c) * pad_frac, floor)
        return (c - half, c + half)
    span = hi - lo
    pad = max(span * pad_frac, floor)
    return (lo - pad, hi + pad)


def main():
    p = argparse.ArgumentParser(description='PRF edge pixel histogram (normalized PRF)')
    p.add_argument('--channel', default=None, help='ch1 or ch2 (default: config.CHANNEL)')
    p.add_argument(
        '--mode',
        choices=('native', 'reprojected'),
        default='native',
        help='Histogram native PRF edge or reprojected scene-stamp PRF edge',
    )
    p.add_argument(
        '--edge-width',
        type=int,
        default=1,
        help='Outer border width in pixels used for the edge histogram',
    )
    p.add_argument(
        '--ref-flux-jy',
        type=float,
        default=1.0,
        help='Reference point-source flux (Jy) for predicted edge Jy/pixel',
    )
    args = p.parse_args()

    chan = (args.channel or config.CHANNEL or 'ch2').lower()
    if 'ch2' in chan:
        chan = 'ch2'
    else:
        chan = 'ch1'

    n_pix = int(config.ANALYSIS_BOX_SIZE * config.SUPERSAMPLE_FACTOR)
    target_ra = float(config.TRANSIENT_RA)
    target_dec = float(config.TRANSIENT_DEC)
    tx, ty = 128.0, 128.0
    prf_model = solver.load_prf(chan, float(tx), float(ty))
    if args.mode == 'native':
        img = np.asarray(prf_model, dtype=float)
        s = float(np.sum(img))
        if s > 0:
            img = img / s
    else:
        scene_wcs = WCS(naxis=2)
        scene_wcs.wcs.crpix = [n_pix / 2, n_pix / 2]
        scene_wcs.wcs.crval = [target_ra, target_dec]
        scene_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        scale = config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR / 3600.0
        scene_wcs.wcs.cdelt = [-scale, scale]
        scene_wcs.wcs.pc = np.eye(2)
        raw_wcs = scene_wcs.deepcopy()
        scene_shape = (n_pix, n_pix)
        col = solver.generate_prf_fast(
            scene_wcs,
            raw_wcs,
            prf_model,
            target_ra,
            target_dec,
            scene_shape,
            channel=chan,
            is_full_array=True,
        )
        img = col.reshape(scene_shape)
        s = float(np.sum(img))
        if s > 0:
            img = img / s

    h, w = img.shape
    border = _border_mask(h, w, width=args.edge_width)
    peak = float(np.nanmax(np.abs(img)))
    if peak > 0:
        img = img / peak
    edge_vals_norm = img[border]
    ref_f = float(args.ref_flux_jy)
    edge_pred_jy = edge_vals_norm * ref_f
    rng_norm = _hist_range(edge_vals_norm)
    rng_pred = _hist_range(edge_pred_jy)
    n_bins = 40

    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(edge_vals_norm, bins=n_bins, range=rng_norm, color='steelblue', alpha=0.85)
    axes[0].set_xlabel(f'PRF value (outer {args.edge_width}-px border, peak=1)')
    axes[0].set_ylabel('count')
    axes[0].set_title(f'Normalized PRF border histogram ({chan}, {args.mode})')
    axes[0].set_xlim(rng_norm)

    axes[1].hist(edge_pred_jy, bins=n_bins, range=rng_pred, color='darkorange', alpha=0.85)
    axes[1].set_xlabel(f'Border × F (F = {ref_f:g} Jy) → pred Jy/pixel')
    axes[1].set_ylabel('count')
    axes[1].set_title('Linear prediction on edge (solver convention)')
    axes[1].set_xlim(rng_pred)

    if args.mode == 'native':
        supt = f'Edge = outer {args.edge_width}-pixel frame of native normalized PRF image'
    else:
        supt = f'Edge = outer {args.edge_width}-pixel frame after reproject + renorm onto scene stamp'
    plt.suptitle(supt, fontsize=10)
    plt.tight_layout()
    out = os.path.join(
        config.DIAGNOSTIC_DIR,
        f'PRF_EDGE_HIST_{chan}_{args.mode}_w{int(args.edge_width)}.png',
    )
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Wrote {out}')
    print(
        f'Border stats (peak=1): min={edge_vals_norm.min():.3e} max={edge_vals_norm.max():.3e} '
        f'median={np.median(edge_vals_norm):.3e}',
    )


if __name__ == '__main__':
    main()
