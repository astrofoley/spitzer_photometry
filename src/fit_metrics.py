"""Fit quality metrics for iterative native-fit campaigns."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import residual_metrics, solver


def _valid_mask(cutout: dict) -> np.ndarray:
    d = np.asarray(cutout["data"], dtype=float)
    s = np.asarray(cutout["sigma"], dtype=float)
    return (d != 0) & np.isfinite(s) & (s < 1e20)


def _center_mask(shape: Tuple[int, int], cx: float, cy: float, radius_px: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    rr = np.hypot(xx - float(cx), yy - float(cy))
    return rr <= float(radius_px)


def compute_fit_metrics(
    cutouts,
    results,
    stars,
    star_fluxes,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
    center_radius_px: float = 3.0,
) -> Dict[str, float]:
    """Return total + center reduced-chi2 and residual structure metrics."""
    tot_chi2 = 0.0
    tot_ndof = 0
    ctr_chi2 = 0.0
    ctr_ndof = 0
    ctr_poisson_vals = []
    dip_mag_vals = []

    for i, c in enumerate(cutouts):
        data = np.asarray(c["data"], dtype=float)
        sigma = np.asarray(c["sigma"], dtype=float)
        pred = solver.predict_cutout_model(
            results,
            cutouts,
            stars,
            star_fluxes,
            i,
            include_gp=True,
            include_transient=True,
            include_stars=True,
            include_host=True,
            include_nuclear_point=True,
        )
        resid = data - pred
        vm = _valid_mask(c)
        if not np.any(vm):
            continue
        w = 1.0 / np.clip(sigma, 1e-30, None) ** 2
        tot_chi2 += float(np.sum((resid[vm] ** 2) * w[vm]))
        tot_ndof += int(np.sum(vm))

        cx, cy = c["wcs"].world_to_pixel_values(float(center_ra_deg), float(center_dec_deg))
        cm = _center_mask(data.shape, float(cx), float(cy), float(center_radius_px))
        m_ctr = vm & cm
        if np.any(m_ctr):
            ctr_chi2 += float(np.sum((resid[m_ctr] ** 2) * w[m_ctr]))
            ctr_ndof += int(np.sum(m_ctr))
            ctr_poisson_vals.append(float(np.sqrt(max(np.nanmedian(np.abs(data[m_ctr])), 0.0))))
            _, _, dip = residual_metrics.dipole_moment_xy(resid, m_ctr)
            dip_mag_vals.append(float(dip))

    # model complexity is not tracked here; this is a practical reduced-chi2 proxy.
    total_reduced = float(tot_chi2 / max(tot_ndof, 1))
    center_reduced = float(ctr_chi2 / max(ctr_ndof, 1))
    center_poisson = float(np.nanmedian(ctr_poisson_vals)) if ctr_poisson_vals else float("nan")
    center_rmse = float(np.sqrt(ctr_chi2 / max(ctr_ndof, 1)))
    center_noise_ratio = (
        float(center_rmse / center_poisson)
        if np.isfinite(center_poisson) and center_poisson > 0
        else float("nan")
    )
    return {
        "total_chi2": float(tot_chi2),
        "total_ndof": int(tot_ndof),
        "total_reduced_chi2": total_reduced,
        "center_chi2": float(ctr_chi2),
        "center_ndof": int(ctr_ndof),
        "center_reduced_chi2": center_reduced,
        "center_rmse": center_rmse,
        "center_poisson_proxy": center_poisson,
        "center_noise_ratio": center_noise_ratio,
        "center_dipole_mag_pix_median": float(np.nanmedian(dip_mag_vals)) if dip_mag_vals else 0.0,
    }
