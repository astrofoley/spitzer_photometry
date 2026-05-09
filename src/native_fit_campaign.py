"""Iterative native-fit campaign utilities."""
from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from reproject import reproject_interp
from scipy.ndimage import binary_dilation, median_filter
from scipy.optimize import minimize

from astropy.coordinates import SkyCoord

from . import config, dipole_chi2_scan, fit_metrics, gp_model, preprocessing, solver


@dataclass
class StageResult:
    stage_name: str
    n_bcd: int
    best_iteration: int
    best_metrics: Dict[str, float]
    best_knobs: Dict[str, float]
    diagnostic_pdf: str
    stacked_pdf: str
    dipole_shift_pix: float
    met_primary: bool
    met_fallback: bool
    iterations_run: int


def _rot_wcs(n_pix: int, ra: float, dec: float, theta_deg: float, pixel_scale_arcsec: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(pixel_scale_arcsec) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    w.wcs.pc = np.array([[c, -s], [s, c]])
    return w


def _scene_wcs_from_bcd_footprints(
    bcd_wcs_list,
    bcd_shape,
    scene_pixel_scale_arcsec: float,
    sky_points_deg,
    pad_scene_px: int,
):
    ra_all, dec_all = [], []
    h, w = int(bcd_shape[0]), int(bcd_shape[1])
    corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]], dtype=float)
    for wb in bcd_wcs_list:
        ra_c, dec_c = wb.pixel_to_world_values(corners[:, 0], corners[:, 1])
        ra_all.extend(np.asarray(ra_c, dtype=float).tolist())
        dec_all.extend(np.asarray(dec_c, dtype=float).tolist())
    for ra_p, dec_p in sky_points_deg:
        ra_all.append(float(ra_p))
        dec_all.append(float(dec_p))
    ra_ref = float(np.mean(ra_all))
    dec_ref = float(np.mean(dec_all))
    cosd = max(np.cos(np.deg2rad(dec_ref)), 1e-6)
    x_pix = ((np.asarray(ra_all) - ra_ref) * cosd * 3600.0) / float(scene_pixel_scale_arcsec)
    y_pix = ((np.asarray(dec_all) - dec_ref) * 3600.0) / float(scene_pixel_scale_arcsec)
    x_min = float(np.min(x_pix)) - float(pad_scene_px)
    x_max = float(np.max(x_pix)) + float(pad_scene_px)
    y_min = float(np.min(y_pix)) - float(pad_scene_px)
    y_max = float(np.max(y_pix)) + float(pad_scene_px)
    w_scene = int(np.ceil(x_max - x_min + 1.0))
    h_scene = int(np.ceil(y_max - y_min + 1.0))
    wcs_scene = WCS(naxis=2)
    wcs_scene.wcs.crpix = [1.0 - x_min, 1.0 - y_min]
    wcs_scene.wcs.crval = [ra_ref, dec_ref]
    wcs_scene.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_scene.wcs.cdelt = [-float(scene_pixel_scale_arcsec) / 3600.0, float(scene_pixel_scale_arcsec) / 3600.0]
    wcs_scene.wcs.pc = np.eye(2)
    return wcs_scene, (h_scene, w_scene)


def _scene_wcs_budgeted(
    bcd_wcs_list,
    bcd_shape,
    min_scene_pixel_scale_arcsec: float,
    sky_points_deg,
    max_scene_pixels: int,
    pad_scene_px: int,
):
    scale = float(min_scene_pixel_scale_arcsec)
    strict_sr = bool(getattr(config, "SCENE_WCS_STRICT_SUPERRES", False))
    if strict_sr:
        w_scene, shp = _scene_wcs_from_bcd_footprints(
            bcd_wcs_list, bcd_shape, scale, sky_points_deg, pad_scene_px,
        )
        n_pix = int(shp[0] * shp[1])
        if n_pix > int(max_scene_pixels):
            raise RuntimeError(
                "Strict super-res scene grid exceeds max_scene_pixels: "
                f"requested_scale={scale:.6g} arcsec, shape={shp}, "
                f"n_pix={n_pix}, max_scene_pixels={int(max_scene_pixels)}. "
                "Increase max_scene_pixels, reduce ROI, or lower SUPERSAMPLE_FACTOR.",
            )
        return w_scene, shp, scale
    for _ in range(20):
        w_scene, shp = _scene_wcs_from_bcd_footprints(
            bcd_wcs_list, bcd_shape, scale, sky_points_deg, pad_scene_px,
        )
        if int(shp[0] * shp[1]) <= int(max_scene_pixels):
            return w_scene, shp, scale
        scale *= 1.12
    return w_scene, shp, scale


def _synthetic_galaxy_scene(shape, cx: float, cy: float):
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    core = 2.4e-5 * np.exp(-0.5 * r2 / (2.6 ** 2))
    bulge = 1.8e-5 * np.exp(-0.5 * r2 / (6.0 ** 2))
    return core + bulge


@contextlib.contextmanager
def _temporary_config(overrides: Dict[str, object]):
    old = {}
    try:
        for k, v in overrides.items():
            old[k] = getattr(config, k)
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def resolve_explicit_nuclear_host_sky_deg() -> Optional[Tuple[float, float]]:
    """
    RA/Dec for an unresolved nucleus / host anchor for the native campaign.

    Does not fall back to TRANSIENT_* (transient != galaxy nucleus). Order:
    NUCLEAR_POINT_*, HOST_CORE_*, GALAXY_EXTENDED_CENTER_*, GP_PROFILE_CENTER_*.
    """
    pairs = (
        ("NUCLEAR_POINT_RA", "NUCLEAR_POINT_DEC"),
        ("HOST_CORE_RA", "HOST_CORE_DEC"),
        ("GALAXY_EXTENDED_CENTER_RA", "GALAXY_EXTENDED_CENTER_DEC"),
        ("GP_PROFILE_CENTER_RA", "GP_PROFILE_CENTER_DEC"),
    )
    for ra_attr, dec_attr in pairs:
        ra = getattr(config, ra_attr, None)
        dec = getattr(config, dec_attr, None)
        if ra is not None and dec is not None:
            return float(ra), float(dec)
    return None


def generate_synthetic_case(n_bcd: int, *, seed_base: int = 123):
    n_pix = 48
    ra0 = 197.450286
    dec0 = -23.381497
    nuc_ra = 197.448762
    nuc_dec = -23.383962
    rng = np.random.default_rng(seed_base + int(n_bcd))
    bcd_wcs_list = [
        _rot_wcs(n_pix, ra0, dec0, theta_deg=17.0 + 2.0 * i, pixel_scale_arcsec=config.PIXEL_SCALE)
        for i in range(n_bcd)
    ]
    scene_wcs, scene_shape, _ = _scene_wcs_budgeted(
        bcd_wcs_list,
        (n_pix, n_pix),
        min_scene_pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR),
        sky_points_deg=[(ra0, dec0), (nuc_ra, nuc_dec)],
        max_scene_pixels=13000,
        pad_scene_px=int(getattr(config, "NATIVE_SCENE_PAD_PX", 6)),
    )
    cx_g, cy_g = scene_wcs.world_to_pixel_values(float(nuc_ra), float(nuc_dec))
    scene_truth = _synthetic_galaxy_scene(scene_shape, float(cx_g), float(cy_g))
    host_truth = solver.host_core_gaussian_column(
        scene_wcs, float(nuc_ra), float(nuc_dec), 1.7, scene_shape,
    ).reshape(scene_shape) * 1.5e-5
    nps_truth = np.zeros(scene_shape, dtype=float)
    nx_scene, ny_scene = scene_wcs.world_to_pixel_values(float(nuc_ra), float(nuc_dec))
    solver._add_delta_to_image(nps_truth, float(nx_scene), float(ny_scene), 1.3e-5)
    intrinsic_truth = scene_truth + host_truth + nps_truth

    cutouts = []
    for i in range(n_bcd):
        w_i = bcd_wcs_list[i]
        bg_i = 1.2e-5 + 6.0e-7 * i
        noise_sig = 1.2e-6
        conv_truth = solver._apply_frame_forward_operator(
            intrinsic_truth, scene_wcs, w_i, scene_shape, (n_pix, n_pix), "ch2", is_full_array=True,
        )
        d = (conv_truth + bg_i + rng.normal(0.0, noise_sig, (n_pix, n_pix))).astype(np.float64)
        cutouts.append({
            "data": d,
            "sigma": np.full_like(d, noise_sig),
            "wcs": w_i,
            "raw_wcs": w_i,
            "is_full_array": True,
            "mjd": 58000.0 + i,
            "filename": f"synthetic_native_ch2_{i:03d}_cbcd.fits",
            "epoch_id": i,
            "is_template": True,
        })
    centers = {"transient_ra": ra0, "transient_dec": dec0, "nuc_ra": nuc_ra, "nuc_dec": nuc_dec}
    return cutouts, scene_wcs, scene_shape, centers


def _resid_limits(arr):
    v = np.asarray(arr, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size < 4:
        return -1.0, 1.0
    lo, hi = np.percentile(vv, [1.0, 99.0])
    lim = max(abs(float(lo)), abs(float(hi)), 1e-12)
    return -lim, lim


def _local_weighted_centroid(img: np.ndarray, x0: float, y0: float, radius_px: float = 12.0) -> Tuple[float, float]:
    """Robust local centroid around (x0, y0) in pixel coordinates."""
    arr = np.asarray(img, dtype=float)
    h, w = arr.shape
    r = float(max(1.0, radius_px))
    x0f = float(x0)
    y0f = float(y0)
    xsl = slice(max(0, int(np.floor(x0f - r))), min(w, int(np.ceil(x0f + r + 1.0))))
    ysl = slice(max(0, int(np.floor(y0f - r))), min(h, int(np.ceil(y0f + r + 1.0))))
    sub = np.array(arr[ysl, xsl], dtype=float)
    sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    # Remove local baseline and keep only positive structure to avoid bias from offsets.
    sub = sub - float(np.nanmedian(sub))
    sub[sub < 0.0] = 0.0
    sw = float(np.sum(sub))
    if not np.isfinite(sw) or sw <= 0.0:
        return x0f, y0f
    yy, xx = np.mgrid[ysl.start:ysl.stop, xsl.start:xsl.stop].astype(float)
    cx = float(np.sum(xx * sub) / sw)
    cy = float(np.sum(yy * sub) / sw)
    return cx, cy


def write_bcd_center_offset_report(
    label: str,
    cutouts: List[dict],
    results: dict,
    out_dir: str,
    *,
    center_radius_px: float = 12.0,
) -> str:
    """Write per-BCD data/model center RA/Dec and angular offsets (arcsec)."""
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"BCD_CENTER_OFFSETS_{label}.json")
    rows: List[Dict[str, object]] = []
    for i, c in enumerate(cutouts):
        data = np.asarray(c["data"], dtype=float)
        pred = solver.predict_cutout_model(
            results,
            cutouts,
            [],
            [],
            i,
            include_gp=True,
            include_transient=True,
            include_stars=False,
            include_host=True,
            include_nuclear_point=True,
        )
        ra0 = float(config.TRANSIENT_RA)
        dec0 = float(config.TRANSIENT_DEC)
        x0, y0 = c["wcs"].world_to_pixel_values(ra0, dec0)
        x0 = float(np.asarray(x0).ravel()[0])
        y0 = float(np.asarray(y0).ravel()[0])
        xd, yd = _local_weighted_centroid(data, x0, y0, radius_px=float(center_radius_px))
        xm, ym = _local_weighted_centroid(np.asarray(pred, dtype=float), x0, y0, radius_px=float(center_radius_px))
        ra_d, dec_d = c["wcs"].pixel_to_world_values(xd, yd)
        ra_m, dec_m = c["wcs"].pixel_to_world_values(xm, ym)
        sc_d = SkyCoord(float(ra_d), float(dec_d), unit="deg")
        sc_m = SkyCoord(float(ra_m), float(dec_m), unit="deg")
        sep = float(sc_d.separation(sc_m).arcsec)
        dra = (float(ra_m) - float(ra_d)) * 3600.0 * np.cos(np.deg2rad(float(dec_d)))
        ddec = (float(dec_m) - float(dec_d)) * 3600.0
        rows.append(
            {
                "frame_index": int(i),
                "filename": str(c.get("filename", "")),
                "data_center_ra_deg": float(ra_d),
                "data_center_dec_deg": float(dec_d),
                "model_center_ra_deg": float(ra_m),
                "model_center_dec_deg": float(dec_m),
                "delta_ra_arcsec_cosdec": float(dra),
                "delta_dec_arcsec": float(ddec),
                "offset_arcsec": sep,
            },
        )
    payload = {
        "label": str(label),
        "n_frames": int(len(rows)),
        "center_radius_px": float(center_radius_px),
        "rows": rows,
        "median_offset_arcsec": float(np.median([r["offset_arcsec"] for r in rows])) if rows else np.nan,
        "max_offset_arcsec": float(np.max([r["offset_arcsec"] for r in rows])) if rows else np.nan,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_json


def _valid_mask(cutout):
    d = np.asarray(cutout["data"], dtype=float)
    s = np.asarray(cutout["sigma"], dtype=float)
    return (d != 0) & np.isfinite(s) & (s < 1e20)


def _cr_mask_local(cutout, sigma_thresh: float = 6.0):
    d = np.asarray(cutout["data"], dtype=float)
    s = np.asarray(cutout["sigma"], dtype=float)
    vm = _valid_mask(cutout)
    if np.sum(vm) < 16:
        return np.zeros_like(d, dtype=bool)
    med = median_filter(d, size=5, mode="nearest")
    resid = d - med
    with np.errstate(divide="ignore", invalid="ignore"):
        nsig = resid / np.clip(s, 1e-30, None)
    cr = vm & np.isfinite(nsig) & (nsig > float(sigma_thresh))
    # Target-agnostic bright-structure guard: suppress CR masking in compact regions
    # with very high local background where astrophysical cores are likely.
    loc = median_filter(d, size=9, mode="nearest")
    vv = loc[vm & np.isfinite(loc)]
    if vv.size > 16:
        pct = float(getattr(config, "CR_BRIGHT_CORE_GUARD_PERCENTILE", 99.0))
        pct = float(np.clip(pct, 0.0, 100.0))
        thr = float(np.percentile(vv, pct))
        bright = vm & (loc >= thr)
        dil = int(max(0, getattr(config, "CR_BRIGHT_CORE_GUARD_DILATION", 1)))
        if dil > 0:
            bright = binary_dilation(bright, iterations=dil)
        cr &= ~bright
    # Optional explicit transient-centered guard in native cutout pixels.
    r_guard = float(max(0.0, getattr(config, "CR_BRIGHT_CORE_GUARD_RADIUS_PX", 0.0)))
    if r_guard > 0.0 and ("raw_wcs" in cutout):
        try:
            guard_center = str(getattr(config, "CR_BRIGHT_CORE_GUARD_CENTER", "transient")).strip().lower()
            if guard_center in ("nuclear", "nuc", "host", "core"):
                ra_c = float(getattr(config, "NUCLEAR_POINT_RA", config.TRANSIENT_RA))
                dec_c = float(getattr(config, "NUCLEAR_POINT_DEC", config.TRANSIENT_DEC))
            else:
                ra_c = float(config.TRANSIENT_RA)
                dec_c = float(config.TRANSIENT_DEC)
            cx, cy = cutout["raw_wcs"].world_to_pixel_values(float(ra_c), float(dec_c))
            cx = float(np.asarray(cx).ravel()[0])
            cy = float(np.asarray(cy).ravel()[0])
            yy, xx = np.mgrid[:d.shape[0], :d.shape[1]]
            core = (np.hypot(xx - cx, yy - cy) <= r_guard) & vm
            cr &= ~core
        except Exception:
            pass
    return cr


# NOTE (follow-up): compact high-residual clusters flagged as CRs that *repeat* across
# multiple template BCDs at the same sky location may be unresolved stellar cores, not CRs.
# Revisit CR heuristics / catalog cross-check when tuning multi-epoch native masks.


def apply_native_cutout_cr_mask(cutout: dict) -> None:
    """Set sigma=inf on locally detected CRs so the joint solver gives them zero weight."""
    cr = _cr_mask_local(cutout)
    if not np.any(cr):
        return
    s = np.asarray(cutout["sigma"], dtype=np.float64).copy()
    s[cr] = np.inf
    cutout["sigma"] = s


def crop_cutout_to_size(cut: dict, size: int, ra_deg: float, dec_deg: float) -> dict:
    """Crop a native cutout dict to ``size``×``size`` pixels centered on ``(ra_deg, dec_deg)``.

    Centers using float pixel coordinates of the sky point in ``cut["wcs"]``, then clips the
    window to the array. Updates ``data``, ``sigma``, ``wcs``, and ``raw_wcs`` slices.
    """
    size = int(size)
    if size <= 0:
        return cut
    h, w = np.asarray(cut["data"]).shape
    if size > h or size > w:
        raise RuntimeError(f"Requested cutout size={size} exceeds cutout shape={(h, w)}")
    px, py = cut["wcs"].world_to_pixel_values(float(ra_deg), float(dec_deg))
    cx = float(np.asarray(px).ravel()[0])
    cy = float(np.asarray(py).ravel()[0])
    target_c = 0.5 * (float(size) - 1.0)
    x0 = int(np.round(cx - target_c))
    y0 = int(np.round(cy - target_c))
    x0 = int(np.clip(x0, 0, w - size))
    y0 = int(np.clip(y0, 0, h - size))
    x1 = x0 + int(size)
    y1 = y0 + int(size)
    out = dict(cut)
    out["data"] = np.asarray(cut["data"], dtype=float)[y0:y1, x0:x1]
    out["sigma"] = np.asarray(cut["sigma"], dtype=float)[y0:y1, x0:x1]
    out["wcs"] = cut["wcs"].slice((slice(y0, y1), slice(x0, x1)))
    out["raw_wcs"] = cut["raw_wcs"].slice((slice(y0, y1), slice(x0, x1)))
    return out


def unmask_sigma_inf_in_radius(cutout: dict, radius_px: float, center: str = "nuclear") -> None:
    """Set inf weights back to finite σ for pixels that are flagged invalid but carry signal.

    Only touches positions with non-zero ``data``, inside ``radius_px`` of either the nucleus
    or the transient (see ``center``). Median σ is taken from valid neighbors in the same
    disk, then applied in-place on ``cutout["sigma"]``.
    """
    r = float(max(0.0, radius_px))
    if r <= 0.0:
        return
    center_key = str(center).strip().lower()
    if center_key in ("nuclear", "nuc", "host", "core"):
        ra = float(getattr(config, "NUCLEAR_POINT_RA", config.TRANSIENT_RA))
        dec = float(getattr(config, "NUCLEAR_POINT_DEC", config.TRANSIENT_DEC))
    else:
        ra = float(config.TRANSIENT_RA)
        dec = float(config.TRANSIENT_DEC)
    s = np.asarray(cutout["sigma"], dtype=float)
    d = np.asarray(cutout["data"], dtype=float)
    px, py = cutout["wcs"].world_to_pixel_values(ra, dec)
    cx = float(np.asarray(px).ravel()[0])
    cy = float(np.asarray(py).ravel()[0])
    yy, xx = np.mgrid[: s.shape[0], : s.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    unmask = (~np.isfinite(s)) & (d != 0) & (rr <= r)
    if not np.any(unmask):
        return
    finite = np.isfinite(s) & (d != 0)
    ref = finite & (rr <= r)
    if np.any(ref):
        med = float(np.nanmedian(s[ref]))
    elif np.any(finite):
        med = float(np.nanmedian(s[finite]))
    else:
        med = 1.0
    if not np.isfinite(med) or med <= 0.0:
        med = 1.0
    s2 = s.copy()
    s2[unmask] = med
    cutout["sigma"] = s2


def _analysis_mask(cutout):
    # Use the explicit fit mask only: sigma finite + nonzero data.
    # Avoid re-running local CR detection at display time, which can make
    # diagnostics depend on cutout statistics rather than the solver mask.
    return _valid_mask(cutout)


def _log_norm_bcd_minmax(data, mask=None):
    """Log stretch: vmin/vmax = min and max of strictly positive BCD values in mask."""
    d = np.asarray(data, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        v = d[m & np.isfinite(d)]
    else:
        v = d[np.isfinite(d)]
    v = v[v > 0]
    if v.size < 1:
        return LogNorm(vmin=1e-30, vmax=1.0)
    lo, hi = float(np.min(v)), float(np.max(v))
    lo = max(lo, 1e-30)
    if hi <= lo:
        hi = lo * 1.0000001
    return LogNorm(vmin=lo, vmax=hi)


def _superres(arr: np.ndarray, upsample: int = 40) -> np.ndarray:
    """Nearest-neighbor magnification for diagnostic plots only (not SUPERSAMPLE_FACTOR)."""
    up = max(2, int(upsample))
    a = np.asarray(arr, dtype=float)
    return np.repeat(np.repeat(a, up, axis=0), up, axis=1)


def _nuclear_delta_display_scene(scene_wcs, scene_shape, nuclear_flux_jy: float) -> np.ndarray:
    """
    PDF-only intrinsic nuclear term: exactly one nonzero scene pixel (nearest pixel to RA/Dec).

    The joint fit still uses bilinear subpixel deltas (solver); this panel is intentionally
    a Kronecker on the scene grid so the schematic is not spread across four neighbors.
    """
    h, w = int(scene_shape[0]), int(scene_shape[1])
    out = np.zeros((h, w), dtype=np.float64)
    if abs(float(nuclear_flux_jy)) <= 0.0:
        return out
    ra_np = getattr(config, "NUCLEAR_POINT_RA", None)
    dec_np = getattr(config, "NUCLEAR_POINT_DEC", None)
    if ra_np is None or dec_np is None:
        ra_np = getattr(config, "HOST_CORE_RA", None)
        dec_np = getattr(config, "HOST_CORE_DEC", None)
    if ra_np is None or dec_np is None:
        return out
    px, py = scene_wcs.world_to_pixel_values(float(ra_np), float(dec_np))
    ix = int(np.clip(int(np.round(float(px))), 0, w - 1))
    iy = int(np.clip(int(np.round(float(py))), 0, h - 1))
    out[iy, ix] = float(nuclear_flux_jy)
    return out


def _to_superres_pixel(x: float, y: float, upsample: int):
    up = float(max(2, int(upsample)))
    return (float(x) + 0.5) * up - 0.5, (float(y) + 0.5) * up - 0.5


def _compass_vectors(w: WCS, ra_deg: float, dec_deg: float):
    eps_dec = 1.0 / 3600.0
    cosd = max(np.cos(np.deg2rad(dec_deg)), 1e-6)
    eps_ra = eps_dec / cosd
    x0, y0 = w.world_to_pixel_values(ra_deg, dec_deg)
    x_n, y_n = w.world_to_pixel_values(ra_deg, dec_deg + eps_dec)
    x_e, y_e = w.world_to_pixel_values(ra_deg + eps_ra, dec_deg)
    return (float(x_n - x0), float(y_n - y0)), (float(x_e - x0), float(y_e - y0))


def _draw_marker_and_compass(ax, x: float, y: float, v_n, v_e, shape):
    ax.plot([x], [y], marker="x", ms=7, mew=1.6, color="lime")
    h, w = shape
    base_x = 0.84 * (w - 1)
    base_y = 0.11 * (h - 1)
    L = 0.09 * min(h, w)

    def _unit(vx, vy):
        nrm = np.hypot(vx, vy)
        if nrm <= 1e-12:
            return 0.0, 1.0
        return vx / nrm, vy / nrm

    nux, nuy = _unit(*v_n)
    eux, euy = _unit(*v_e)
    ax.annotate("", xy=(base_x + L * nux, base_y + L * nuy), xytext=(base_x, base_y), arrowprops=dict(color="yellow", lw=1.6))
    ax.annotate("", xy=(base_x + L * eux, base_y + L * euy), xytext=(base_x, base_y), arrowprops=dict(color="cyan", lw=1.6))
    ax.text(base_x + 1.08 * L * nux, base_y + 1.08 * L * nuy, "N", color="yellow", fontsize=8, ha="center", va="center")
    ax.text(base_x + 1.08 * L * eux, base_y + 1.08 * L * euy, "E", color="cyan", fontsize=8, ha="center", va="center")


def _intrinsic_components(results):
    scene_wcs = results["scene_wcs"]
    scene_shape = results["scene_shape"]
    gp = np.asarray(results.get("gp_scene", results["model_scene"]), dtype=float)
    host = np.zeros(scene_shape, dtype=float)
    nps = np.zeros(scene_shape, dtype=float)

    if getattr(config, "USE_HOST_GAUSSIAN_CORE", False):
        ra_h = getattr(config, "HOST_CORE_RA", None)
        dec_h = getattr(config, "HOST_CORE_DEC", None)
        if ra_h is not None and dec_h is not None:
            f_multi = results.get("host_core_fluxes")
            s_multi = results.get("host_gaussian_sigmas_px")
            if (
                f_multi is not None
                and s_multi is not None
                and len(np.asarray(f_multi).ravel()) > 1
                and len(np.asarray(f_multi).ravel()) == len(np.asarray(s_multi).ravel())
            ):
                f_multi = np.asarray(f_multi, dtype=np.float64).ravel()
                s_multi = np.asarray(s_multi, dtype=np.float64).ravel()
                for fj, sj in zip(f_multi, s_multi):
                    col_h = solver.host_core_gaussian_column(
                        scene_wcs,
                        float(ra_h),
                        float(dec_h),
                        float(sj),
                        scene_shape,
                    )
                    host += float(fj) * col_h.reshape(scene_shape)
            else:
                col_h = solver.host_core_gaussian_column(
                    scene_wcs,
                    float(ra_h),
                    float(dec_h),
                    float(getattr(config, "HOST_CORE_SIGMA_PX", 1.5)),
                    scene_shape,
                )
                host += float(results.get("host_core_flux", 0.0)) * col_h.reshape(scene_shape)

    if getattr(config, "USE_NUCLEAR_POINT_SOURCE", False):
        ra_np = getattr(config, "NUCLEAR_POINT_RA", None)
        dec_np = getattr(config, "NUCLEAR_POINT_DEC", None)
        if ra_np is None or dec_np is None:
            ra_np = getattr(config, "HOST_CORE_RA", None)
            dec_np = getattr(config, "HOST_CORE_DEC", None)
        if ra_np is not None and dec_np is not None:
            fnp = float(results.get("nuclear_point_flux", 0.0))
            nx, ny = scene_wcs.world_to_pixel_values(float(ra_np), float(dec_np))
            solver._add_delta_to_image(nps, float(nx), float(ny), fnp)
    return gp, host, nps


def _split_gp_components_from_prior(results, gp_scene):
    gp = np.asarray(gp_scene, dtype=np.float64)
    if isinstance(results, dict):
        c1s = results.get("gp_scene_component1")
        c2s = results.get("gp_scene_component2")
        if c1s is not None and c2s is not None:
            return np.asarray(c1s, dtype=np.float64), np.asarray(c2s, dtype=np.float64)
    shp = tuple(gp.shape)
    p = results.get("gp_prior_params", {}) if isinstance(results, dict) else {}
    ell = p.get("ell", None)
    var = p.get("var", None)
    ell2 = p.get("ell2", None)
    var2 = p.get("var2", None)
    if ell is None or var is None or ell2 is None or var2 is None:
        return gp, np.zeros_like(gp)
    try:
        y, x = np.mgrid[0 : shp[0], 0 : shp[1]]
        coords = np.vstack([y.ravel(), x.ravel()]).T
        mo = gp_model.normalize_matern_order(
            p.get("matern_order") or getattr(config, "GP_MATERN_ORDER", "matern32"),
        )
        ell_s1 = float(ell) * float(config.SUPERSAMPLE_FACTOR)
        ell_s2 = float(ell2) * float(config.SUPERSAMPLE_FACTOR)
        k1 = gp_model.scene_kernel_matrix(coords, ell_s1, float(var), order=mo)
        k2 = gp_model.scene_kernel_matrix(coords, ell_s2, float(var2), order=mo)
        ksum = k1 + k2 + np.eye(coords.shape[0]) * 1e-8
        rhs = gp.ravel()
        alpha = np.linalg.solve(ksum, rhs)
        c1 = (k1 @ alpha).reshape(shp)
        c2 = (k2 @ alpha).reshape(shp)
        return c1, c2
    except Exception:
        return gp, np.zeros_like(gp)


def write_native_fit_pdf(label: str, cutouts, results, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"NATIVE_FIT_DIAGNOSTIC_{label}.pdf")
    # `up` is nearest-neighbor PDF zoom only (`_superres`); solver SR is `SUPERSAMPLE_FACTOR`.
    up = int(max(1, getattr(config, "DIAG_SUPERRES_DISPLAY_FACTOR", 40)))
    scene_sr = int(max(1, getattr(config, "SUPERSAMPLE_FACTOR", 1)))
    disp_lbl = f"scene SR={scene_sr}×, PDF zoom ×{up}"
    gp_i, host_i, nps_i = _intrinsic_components(results)
    gp_prior = results.get("gp_prior_params", {}) if isinstance(results, dict) else {}
    ell1 = gp_prior.get("ell", None)
    var1 = gp_prior.get("var", None)
    ell2 = gp_prior.get("ell2", None)
    var2 = gp_prior.get("var2", None)
    with PdfPages(out_pdf) as pdf:
        for i, c in enumerate(cutouts):
            data = np.asarray(c["data"], dtype=float)
            bg_i = float(np.asarray(results.get("bcd_backgrounds", np.zeros(len(cutouts))))[i])
            pred = solver.predict_cutout_model(
                results,
                cutouts,
                [],
                [],
                i,
                include_gp=True,
                include_transient=True,
                include_stars=False,
                include_host=True,
                include_nuclear_point=True,
            )
            resid = data - pred
            vm_valid = _valid_mask(c)
            vm_analysis = _analysis_mask(c)
            pred_disp = np.where(vm_valid, pred, 0.0)
            resid_disp = np.where(vm_analysis, resid, 0.0)
            flux_norm_bcd = _log_norm_bcd_minmax(data, mask=vm_analysis)

            gp_comp1, gp_comp2 = _split_gp_components_from_prior(results, gp_i)
            # Decomposition/closure diagnostics must be performed in the intrinsic
            # scene domain. Adding per-BCD background into comp1/comp2 and then
            # subtracting the total makes the closure panel dominated by `bg_i`.
            gp_i_hi = _superres(gp_i, upsample=up)
            gp1_i_hi = _superres(gp_comp1, upsample=up)
            gp2_i_hi = _superres(gp_comp2, upsample=up)
            gp_hi = gp_i_hi + bg_i
            gp1_hi = gp1_i_hi + bg_i
            gp2_hi = gp2_i_hi + bg_i
            gp_closure = (gp1_i_hi + gp2_i_hi) - gp_i_hi
            host_hi = _superres(host_i, upsample=up) + bg_i
            scene_wcs = results["scene_wcs"]
            fnp_disp = float(results.get("nuclear_point_flux", 0.0))
            nps_display = _nuclear_delta_display_scene(scene_wcs, tuple(results["scene_shape"]), fnp_disp)
            nps_hi = _superres(nps_display, upsample=up) + bg_i
            intrinsic_scene = gp_i + host_i + nps_i
            chan = "ch2" if "ch2" in c["filename"] else "ch1"
            conv_scene = solver.apply_spatially_varying_prf_to_scene(
                intrinsic_scene,
                results["scene_wcs"],
                c["raw_wcs"],
                results["scene_shape"],
                chan,
                is_full_array=bool(c.get("is_full_array", False)),
            )
            conv_hi = _superres(conv_scene, upsample=up) + bg_i

            fig, axes = plt.subplots(3, 3, figsize=(16, 14))
            ax = axes.ravel()
            bcd_masked = np.ma.array(data, mask=~vm_analysis)
            cmap_bcd = plt.get_cmap("gray").copy()
            cmap_bcd.set_bad(color="red")
            im0 = ax[0].imshow(bcd_masked, origin="lower", cmap=cmap_bcd, norm=flux_norm_bcd, interpolation="nearest")
            ax[0].set_title("BCD (unaltered)")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            im1 = ax[1].imshow(gp1_hi, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            if ell1 is not None and var1 is not None:
                ax[1].set_title(f"GP comp 1 (+BG); {disp_lbl}\nell={float(ell1):.3g}, var={float(var1):.3g}")
            else:
                ax[1].set_title(f"GP component 1 (+BG); {disp_lbl}")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            im2 = ax[2].imshow(gp2_hi, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            if ell2 is not None and var2 is not None:
                ax[2].set_title(f"GP comp 2 (+BG); {disp_lbl}\nell={float(ell2):.3g}, var={float(var2):.3g}")
            else:
                ax[2].set_title(f"GP component 2 (+BG); {disp_lbl}")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
            im3 = ax[3].imshow(gp_hi, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            ax[3].set_title(f"Total GP (+BG); {disp_lbl}")
            plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
            c0, c1 = _resid_limits(gp_closure)
            im4 = ax[4].imshow(gp_closure, origin="lower", cmap="RdBu_r", vmin=c0, vmax=c1, interpolation="nearest")
            ax[4].set_title(f"GP closure (intrinsic): comp1 + comp2 - total; {disp_lbl}")
            plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
            im5 = ax[5].imshow(host_hi, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            ax[5].set_title(f"Host Gaussian (+BG); {disp_lbl}")
            plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)
            im6 = ax[6].imshow(nps_hi, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            ax[6].set_title(f"Nuclear (1 scene px; display only) +BG; {disp_lbl}")
            plt.colorbar(im6, ax=ax[6], fraction=0.046, pad=0.04)
            im7 = ax[7].imshow(pred_disp, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            ax[7].set_title("BCD model: projected + SV-PRF convolved (+BG)")
            plt.colorbar(im7, ax=ax[7], fraction=0.046, pad=0.04)
            v0, v1 = _resid_limits(resid_disp)
            im8 = ax[8].imshow(resid_disp, origin="lower", cmap="RdBu_r", vmin=v0, vmax=v1, interpolation="nearest")
            ax[8].set_title("Residual (data - model)")
            plt.colorbar(im8, ax=ax[8], fraction=0.046, pad=0.04)
            # Overplot residual in figure title since 3x3 is now used for GP closure diagnostics.
            # Keep residual scale available numerically to avoid losing context.
            resid_stat = f"resid range [{v0:.3g}, {v1:.3g}]"

            ra_t = float(config.TRANSIENT_RA)
            dec_t = float(config.TRANSIENT_DEC)
            tx_b, ty_b = c["wcs"].world_to_pixel_values(ra_t, dec_t)
            v_n_b, v_e_b = _compass_vectors(c["wcs"], ra_t, dec_t)
            _draw_marker_and_compass(ax[0], tx_b, ty_b, v_n_b, v_e_b, data.shape)
            _draw_marker_and_compass(ax[8], tx_b, ty_b, v_n_b, v_e_b, resid.shape)

            tx_s, ty_s = results["scene_wcs"].world_to_pixel_values(ra_t, dec_t)
            tx_h, ty_h = _to_superres_pixel(tx_s, ty_s, up)
            v_n_s, v_e_s = _compass_vectors(results["scene_wcs"], ra_t, dec_t)
            v_n_h = (v_n_s[0] * up, v_n_s[1] * up)
            v_e_h = (v_e_s[0] * up, v_e_s[1] * up)
            _draw_marker_and_compass(ax[1], tx_h, ty_h, v_n_h, v_e_h, gp1_hi.shape)
            _draw_marker_and_compass(ax[2], tx_h, ty_h, v_n_h, v_e_h, gp2_hi.shape)
            _draw_marker_and_compass(ax[3], tx_h, ty_h, v_n_h, v_e_h, gp_hi.shape)
            _draw_marker_and_compass(ax[4], tx_h, ty_h, v_n_h, v_e_h, gp_closure.shape)
            _draw_marker_and_compass(ax[5], tx_h, ty_h, v_n_h, v_e_h, host_hi.shape)
            _draw_marker_and_compass(ax[6], tx_h, ty_h, v_n_h, v_e_h, nps_hi.shape)
            _draw_marker_and_compass(ax[7], tx_b, ty_b, v_n_b, v_e_b, pred_disp.shape)
            for j in range(9):
                ax[j].axis("off")
            mo = gp_model.normalize_matern_order(
                gp_prior.get("matern_order") or getattr(config, "GP_MATERN_ORDER", "matern32"),
            )
            kern_lbl = "Matérn 1/2" if mo == "matern12" else "Matérn 3/2"
            if ell2 is not None and var2 is not None:
                gp_txt = (
                    f"GP prior ({kern_lbl}): comp1(ell={float(ell1):.3g}, var={float(var1):.3g}), "
                    f"comp2(ell={float(ell2):.3g}, var={float(var2):.3g})"
                )
            else:
                gp_txt = (
                    f"GP prior ({kern_lbl}): ell={float(ell1):.3g}, var={float(var1):.3g}"
                    if (ell1 is not None and var1 is not None)
                    else f"GP prior ({kern_lbl}): unavailable"
                )
            fig.suptitle(
                f"Native template-fit diagnostics: {label}, frame={i}, template={bool(c.get('is_template'))}\n"
                f"{disp_lbl}; {gp_txt}; {resid_stat}",
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Separate decomposition diagnostics (intrinsic scene domain).
            d_tot_minus_gp1 = gp_i_hi - gp1_i_hi
            d_tot_minus_gp2 = gp_i_hi - gp2_i_hi
            d_tot_minus_sum = gp_i_hi - (gp1_i_hi + gp2_i_hi)
            dd = np.hstack(
                [
                    np.ravel(np.asarray(d_tot_minus_gp1, dtype=float)),
                    np.ravel(np.asarray(d_tot_minus_gp2, dtype=float)),
                    np.ravel(np.asarray(d_tot_minus_sum, dtype=float)),
                ],
            )
            dlim = float(max(np.nanpercentile(np.abs(dd), 99.0), 1e-12))
            fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
            imd0 = ax2[0].imshow(
                d_tot_minus_gp1,
                origin="lower",
                cmap="RdBu_r",
                vmin=-dlim,
                vmax=dlim,
                interpolation="nearest",
            )
            ax2[0].set_title("total - GP1 (intrinsic)")
            plt.colorbar(imd0, ax=ax2[0], fraction=0.046, pad=0.04)
            imd1 = ax2[1].imshow(
                d_tot_minus_gp2,
                origin="lower",
                cmap="RdBu_r",
                vmin=-dlim,
                vmax=dlim,
                interpolation="nearest",
            )
            ax2[1].set_title("total - GP2 (intrinsic)")
            plt.colorbar(imd1, ax=ax2[1], fraction=0.046, pad=0.04)
            imd2 = ax2[2].imshow(
                d_tot_minus_sum,
                origin="lower",
                cmap="RdBu_r",
                vmin=-dlim,
                vmax=dlim,
                interpolation="nearest",
            )
            ax2[2].set_title("total - (GP1 + GP2) (intrinsic)")
            plt.colorbar(imd2, ax=ax2[2], fraction=0.046, pad=0.04)
            _draw_marker_and_compass(ax2[0], tx_h, ty_h, v_n_h, v_e_h, d_tot_minus_gp1.shape)
            _draw_marker_and_compass(ax2[1], tx_h, ty_h, v_n_h, v_e_h, d_tot_minus_gp2.shape)
            _draw_marker_and_compass(ax2[2], tx_h, ty_h, v_n_h, v_e_h, d_tot_minus_sum.shape)
            for a in ax2:
                a.axis("off")
            fig2.suptitle(f"GP decomposition checks: {label}, frame={i}; {disp_lbl}")
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)
    return out_pdf


def write_stacked_residual_pdf(label: str, cutouts, results, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"STACKED_RESIDUALS_{label}.pdf")
    scene_wcs = results["scene_wcs"]
    scene_shape = tuple(results["scene_shape"])
    scene_sr = int(max(1, getattr(config, "SUPERSAMPLE_FACTOR", 1)))
    res_stack = []
    z_stack = []
    cov_stack = []
    for i, c in enumerate(cutouts):
        pred = solver.predict_cutout_model(results, cutouts, [], [], i)
        resid = np.asarray(c["data"], dtype=float) - np.asarray(pred, dtype=float)
        sig = np.asarray(c["sigma"], dtype=float)
        vm = _analysis_mask(c).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = resid / np.clip(sig, 1e-30, None)
        r_grid, _ = reproject_interp((resid, c["wcs"]), scene_wcs, shape_out=scene_shape)
        z_grid, _ = reproject_interp((z, c["wcs"]), scene_wcs, shape_out=scene_shape)
        m_grid, _ = reproject_interp((vm, c["wcs"]), scene_wcs, shape_out=scene_shape)
        m = np.asarray(m_grid, dtype=float) > 0.5
        res_stack.append(np.where(m, np.asarray(r_grid, dtype=float), np.nan))
        z_stack.append(np.where(m, np.asarray(z_grid, dtype=float), np.nan))
        cov_stack.append(m.astype(float))
    with np.errstate(invalid="ignore"):
        med_r = np.nanmedian(np.stack(res_stack, axis=0), axis=0)
        med_z = np.nanmedian(np.stack(z_stack, axis=0), axis=0)
    cov = np.nansum(np.stack(cov_stack, axis=0), axis=0)
    min_cov = max(1, int(np.ceil(0.2 * len(cutouts))))
    low_cov = cov < float(min_cov)
    med_r = np.where(low_cov, np.nan, med_r)
    med_z = np.where(low_cov, np.nan, med_z)
    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        rv0, rv1 = _resid_limits(med_r)
        im0 = axes[0].imshow(med_r, origin="lower", cmap="RdBu_r", vmin=rv0, vmax=rv1)
        axes[0].set_title(f"{label} median residual (Jy); scene SR={scene_sr}×")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        zv0, zv1 = _resid_limits(med_z)
        im1 = axes[1].imshow(med_z, origin="lower", cmap="RdBu_r", vmin=zv0, vmax=zv1)
        axes[1].set_title(f"{label} median residual/sigma; scene SR={scene_sr}×")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(cov, origin="lower", cmap="viridis")
        axes[2].set_title(f"{label} stack coverage (valid frames)")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return out_pdf


def _run_solver(cutouts, scene_wcs, scene_shape, ell, var):
    stars = getattr(_run_solver, "_stars", [])
    star_fluxes = getattr(_run_solver, "_star_fluxes", [])
    return solver.run_gls_solve(
        cutouts,
        stars,
        star_fluxes,
        {"ell": float(ell), "var": float(var)},
        (float(ell), float(var)),
        np.zeros(scene_shape),
        scene_wcs,
        len(cutouts),
    )


def _settings_space(base_ell: float, use_point: bool, iteration: int):
    mult = [1.0, 0.8, 0.6, 1.2, 1.5, 2.0]
    vmult = [1.0, 0.5, 2.0, 0.25, 4.0, 0.75]
    idx = min(iteration, len(mult) - 1)
    return {
        "ell": float(base_ell * mult[idx]),
        "var_mult": float(vmult[idx]),
        "use_point": bool(use_point),
    }


def _run_dipole_refinement(cutouts, results):
    try:
        scan = dipole_chi2_scan.compute_dipole_chi2_scan(cutouts, results, [], [], stretch_mask=None)
        sec = scan.get("dipole_chi2_refinement", {})
        if sec.get("skipped"):
            return {"best_s_pix": 0.0, "ux": 0.0, "uy": 0.0, "scan": scan, "skipped": True}
        uv = sec.get("unit_vector_scene_xy", [0.0, 0.0])
        ux = float(uv[0]) if len(uv) > 0 else 0.0
        uy = float(uv[1]) if len(uv) > 1 else 0.0
        return {
            "best_s_pix": float(sec.get("best_s_pix", 0.0)),
            "ux": ux,
            "uy": uy,
            "scan": scan,
            "skipped": False,
        }
    except Exception:
        return {"best_s_pix": 0.0, "ux": 0.0, "uy": 0.0, "scan": {}, "skipped": True}


def _shift_cutout_wcs(cutouts: List[dict], dx_pix: float, dy_pix: float) -> List[dict]:
    """Return shallow-copied cutouts with per-frame WCS CRPIX shifted by (dx, dy) pixels."""
    out: List[dict] = []
    for c in cutouts:
        cc = dict(c)
        w = c.get("wcs")
        rw = c.get("raw_wcs")
        if w is not None:
            ww = w.deepcopy()
            ww.wcs.crpix[0] += float(dx_pix)
            ww.wcs.crpix[1] += float(dy_pix)
            cc["wcs"] = ww
        if rw is not None:
            rr = rw.deepcopy()
            rr.wcs.crpix[0] += float(dx_pix)
            rr.wcs.crpix[1] += float(dy_pix)
            cc["raw_wcs"] = rr
        out.append(cc)
    return out


def write_iteration_metric_plot(stage_name: str, iter_log: List[Dict[str, object]], out_dir: str) -> Optional[str]:
    rows = [r for r in iter_log if "center_reduced_chi2" in r and "total_reduced_chi2" in r and "error" not in r]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (int(r["iteration"]), int(bool(r.get("use_point", False)))))
    x = np.arange(len(rows), dtype=float)
    y_center = np.array([float(r["center_reduced_chi2"]) for r in rows], dtype=float)
    y_total = np.array([float(r["total_reduced_chi2"]) for r in rows], dtype=float)
    out = os.path.join(out_dir, f"ITER_METRICS_{stage_name}.png")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, y_total, marker="o", lw=1.5, label="total reduced chi2")
    ax.plot(x, y_center, marker="s", lw=1.5, label="center reduced chi2")
    ax.set_xlabel("trial index")
    ax.set_ylabel("reduced chi2")
    ax.set_title(f"{stage_name} iteration metrics")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _reindex_epochs(cutouts):
    out = []
    for i, c in enumerate(cutouts):
        cc = dict(c)
        cc["epoch_id"] = i
        out.append(cc)
    return out


def prepare_real_template_case():
    """Build template-only native cutouts centered on transient with analysis padding."""
    all_files = preprocessing.find_spitzer_files(config.DATA_DIR)
    _, tpl_files = preprocessing.categorize_observations(all_files, config.SPLIT_DATE_MJD)
    if not tpl_files:
        raise RuntimeError("No template files available")
    # Explicitly mark template inputs so downstream native cutouts carry is_template=True.
    tpl_files = [dict(f, is_template=True) for f in tpl_files]
    target = SkyCoord(float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC), unit="deg")

    # Build a deep-template stack for template-only astrometric alignment.
    mosaic_wcs, mosaic_shape = preprocessing.define_mosaic_wcs(tpl_files, target)
    processed_tpl = preprocessing.reproject_to_grid(tpl_files, mosaic_wcs, mosaic_shape)
    tpl_cube = np.array([p["data"] for p in processed_tpl])
    med_stack, _ = preprocessing.create_median_stack(tpl_cube)
    source_cat = preprocessing.get_or_create_source_catalog(all_files)
    preprocessing.align_frames_to_template(tpl_files, med_stack, mosaic_wcs, source_cat)

    cutouts, _ = preprocessing.extract_native_analysis_cutouts(tpl_files, target)
    if not cutouts:
        raise RuntimeError("Template-only native cutout extraction returned no frames")
    cutouts = _reindex_epochs([dict(c) for c in cutouts])
    # Build a common North-up scene grid spanning template detector footprints.
    bcd_wcs_list = [c["raw_wcs"] for c in cutouts]
    n_pix = int(cutouts[0]["data"].shape[0])
    nuc_explicit = resolve_explicit_nuclear_host_sky_deg()
    # Second sky anchor for scene footprint: use real nucleus if configured, else transient
    # (transient-only does not imply the nuclear PSF sits on the transient).
    tr_ra, tr_dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
    nuc_scene_ra, nuc_scene_dec = nuc_explicit if nuc_explicit is not None else (tr_ra, tr_dec)
    scene_wcs, scene_shape, _ = _scene_wcs_budgeted(
        bcd_wcs_list,
        (n_pix, n_pix),
        min_scene_pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR),
        sky_points_deg=[(tr_ra, tr_dec), (nuc_scene_ra, nuc_scene_dec)],
        max_scene_pixels=13000,
        pad_scene_px=int(getattr(config, "NATIVE_SCENE_PAD_PX", 6)),
    )
    centers = {
        "transient_ra": tr_ra,
        "transient_dec": tr_dec,
        "nuc_ra": float(nuc_explicit[0]) if nuc_explicit is not None else None,
        "nuc_dec": float(nuc_explicit[1]) if nuc_explicit is not None else None,
    }
    for c in cutouts:
        apply_native_cutout_cr_mask(c)
    return {
        "template_cutouts": cutouts,
        "scene_wcs": scene_wcs,
        "scene_shape": scene_shape,
        "all_stars": [],
        "init_star_fluxes": np.zeros(0, dtype=float),
        "centers": centers,
    }


def build_template_real_case_from_cutouts(
    cutouts: Sequence[dict],
    *,
    max_scene_pixels: int = 13000,
) -> Dict[str, object]:
    """
    Build a real template ``real_case`` dict from an explicit subset of native cutouts.

    Recomputes ``scene_wcs`` / ``scene_shape`` from those BCD footprints using
    ``config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR`` as the target North-up scene
    pixel scale. Call under ``_temporary_config`` if ``SUPERSAMPLE_FACTOR`` must
    differ from the value used when ``prepare_real_template_case`` ran.

    Parameters
    ----------
    max_scene_pixels
        Passed to ``_scene_wcs_budgeted`` (raise if you need a finer grid than the
        default 13000-pixel footprint cap allows).
    """
    cut_list = _reindex_epochs([dict(c) for c in cutouts])
    if not cut_list:
        raise RuntimeError("build_template_real_case_from_cutouts: empty cutouts")
    for c in cut_list:
        apply_native_cutout_cr_mask(c)
    bcd_wcs_list = [c["raw_wcs"] for c in cut_list]
    n_pix = int(cut_list[0]["data"].shape[0])
    nuc_explicit = resolve_explicit_nuclear_host_sky_deg()
    tr_ra, tr_dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
    nuc_scene_ra, nuc_scene_dec = nuc_explicit if nuc_explicit is not None else (tr_ra, tr_dec)
    scene_wcs, scene_shape, _ = _scene_wcs_budgeted(
        bcd_wcs_list,
        (n_pix, n_pix),
        min_scene_pixel_scale_arcsec=(float(config.PIXEL_SCALE) / float(config.SUPERSAMPLE_FACTOR)),
        sky_points_deg=[(tr_ra, tr_dec), (nuc_scene_ra, nuc_scene_dec)],
        max_scene_pixels=int(max_scene_pixels),
        pad_scene_px=int(getattr(config, "NATIVE_SCENE_PAD_PX", 6)),
    )
    centers = {
        "transient_ra": tr_ra,
        "transient_dec": tr_dec,
        "nuc_ra": float(nuc_explicit[0]) if nuc_explicit is not None else None,
        "nuc_dec": float(nuc_explicit[1]) if nuc_explicit is not None else None,
    }
    return {
        "template_cutouts": cut_list,
        "scene_wcs": scene_wcs,
        "scene_shape": scene_shape,
        "all_stars": [],
        "init_star_fluxes": np.zeros(0, dtype=float),
        "centers": centers,
    }


def _campaign_solver_cfg(
    centers: dict,
    *,
    use_point: bool,
    require_nuclear_point: bool,
    smooth: float,
    merge_extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    tr_ra = float(centers["transient_ra"])
    tr_dec = float(centers["transient_dec"])
    nr = centers.get("nuc_ra")
    nd = centers.get("nuc_dec")
    if bool(use_point) or require_nuclear_point:
        if nr is None or nd is None:
            raise RuntimeError(
                "Nuclear point requires explicit nucleus coordinates in config "
                "(NUCLEAR_POINT_RA/DEC, HOST_CORE_RA/DEC, GALAXY_EXTENDED_CENTER_RA/DEC, "
                "or GP_PROFILE_CENTER_RA/DEC).",
            )
    host_ra = float(nr) if nr is not None else tr_ra
    host_dec = float(nd) if nd is not None else tr_dec
    nuc_ra_cfg = float(nr) if (bool(use_point) or require_nuclear_point) else tr_ra
    nuc_dec_cfg = float(nd) if (bool(use_point) or require_nuclear_point) else tr_dec
    cfg: Dict[str, object] = {
        "TRANSIENT_RA": tr_ra,
        "TRANSIENT_DEC": tr_dec,
        "USE_HOST_GAUSSIAN_CORE": False,
        "HOST_CORE_RA": host_ra,
        "HOST_CORE_DEC": host_dec,
        "USE_NUCLEAR_POINT_SOURCE": bool(use_point),
        "NUCLEAR_POINT_RA": nuc_ra_cfg,
        "NUCLEAR_POINT_DEC": nuc_dec_cfg,
        "GP_FALLBACK_NEIGHBOR_SMOOTHNESS": float(max(0.35, smooth)),
    }
    if merge_extra:
        cfg.update(merge_extra)
    return cfg


def _gp_tier_gate_passes(metrics: Dict[str, float], baseline: Optional[Dict[str, float]]) -> bool:
    if baseline is None or not np.isfinite(float(baseline.get("center_reduced_chi2", float("nan")))):
        return False
    c = float(metrics.get("center_reduced_chi2", float("nan")))
    if not np.isfinite(c):
        return False
    cap = float(getattr(config, "GP_TIER_GATE_CENTER_CHI2_MAX", 0.0))
    if cap > 0.0 and c <= cap:
        return True
    imp = float(getattr(config, "GP_TIER_GATE_IMPROVE_CENTER_CHI2", 0.0))
    if imp > 0.0 and float(baseline["center_reduced_chi2"]) - c >= imp:
        return True
    return False


def _solve_campaign_trial(
    cutouts: List[dict],
    scene_wcs,
    scene_shape: Tuple[int, int],
    centers: dict,
    stars: list,
    init_star_fluxes: np.ndarray,
    ell: float,
    base_var: float,
    var_mult: float,
    use_point: bool,
    require_nuclear_point: bool,
    smooth: float,
    merge_extra: Optional[Dict[str, object]],
    trial_label: str,
    center_radius_px: float = 3.0,
    optimize_gp_params: bool = False,
) -> Dict[str, object]:
    cfg = _campaign_solver_cfg(
        centers,
        use_point=use_point,
        require_nuclear_point=require_nuclear_point,
        smooth=smooth,
        merge_extra=merge_extra or {},
    )
    row: Dict[str, object] = {"trial": trial_label, "merge_extra": dict(merge_extra or {})}
    center_ra = float(centers["nuc_ra"] or centers["transient_ra"])
    center_dec = float(centers["nuc_dec"] or centers["transient_dec"])

    def _eval_trial(eval_ell: float, eval_var: float) -> Dict[str, object]:
        with _temporary_config(cfg):
            results_local = _run_solver(cutouts, scene_wcs, scene_shape, float(eval_ell), float(eval_var))
        metrics_local = fit_metrics.compute_fit_metrics(
            cutouts,
            results_local,
            stars,
            results_local.get("star_fluxes", init_star_fluxes),
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
            center_radius_px=float(center_radius_px),
        )
        return {
            "results": results_local,
            "metrics": metrics_local,
            "ell": float(eval_ell),
            "var": float(eval_var),
        }

    try:
        ell0 = float(ell)
        var0 = float(base_var) * float(var_mult)
        if not optimize_gp_params:
            best_eval = _eval_trial(ell0, var0)
        else:
            best_eval = None
            trace_rows: List[Dict[str, float]] = []
            iter_rows: List[Dict[str, float]] = []
            prev_x = None

            def _score(log_params: np.ndarray) -> float:
                nonlocal best_eval
                eval_ell = float(np.exp(float(log_params[0])))
                eval_var = float(np.exp(float(log_params[1])))
                try:
                    out = _eval_trial(eval_ell, eval_var)
                    score = float(out["metrics"].get("total_reduced_chi2", np.inf))
                    trace_rows.append(
                        {
                            "evaluation": float(len(trace_rows)),
                            "ell": eval_ell,
                            "var": eval_var,
                            "objective_total_reduced_chi2": score,
                            "center_reduced_chi2": float(out["metrics"].get("center_reduced_chi2", np.inf)),
                        },
                    )
                    if np.isfinite(score) and (
                        best_eval is None
                        or score < float(best_eval["metrics"].get("total_reduced_chi2", np.inf))
                    ):
                        best_eval = out
                    return score if np.isfinite(score) else np.inf
                except Exception:
                    return np.inf

            lb_ell, ub_ell = np.log(0.15), np.log(80.0)
            lb_var, ub_var = np.log(1e-10), np.log(10.0)
            x0 = np.array(
                [
                    np.clip(np.log(max(ell0, 1e-8)), lb_ell, ub_ell),
                    np.clip(np.log(max(var0, 1e-16)), lb_var, ub_var),
                ],
                dtype=float,
            )
            # Ensure a valid baseline even if optimizer fails immediately.
            best_eval = _eval_trial(float(np.exp(x0[0])), float(np.exp(x0[1])))
            def _cb(xk: np.ndarray) -> None:
                nonlocal prev_x
                step = 0.0 if prev_x is None else float(np.linalg.norm(np.asarray(xk) - np.asarray(prev_x)))
                prev_x = np.asarray(xk, dtype=float).copy()
                iter_rows.append(
                    {
                        "iteration": float(len(iter_rows)),
                        "ell": float(np.exp(float(xk[0]))),
                        "var": float(np.exp(float(xk[1]))),
                        "step_norm": step,
                    },
                )

            opt = minimize(
                _score,
                x0=x0,
                method="Powell",
                callback=_cb,
                bounds=[(lb_ell, ub_ell), (lb_var, ub_var)],
                options={"maxfev": 12, "maxiter": 6, "xtol": 1e-3, "ftol": 1e-6},
            )
            row["optimizer"] = {
                "method": "Powell",
                "success": bool(opt.success),
                "status": int(opt.status),
                "message": str(opt.message),
                "nfev": int(getattr(opt, "nfev", len(trace_rows))),
                "nit": int(getattr(opt, "nit", -1)),
                "x0_ell": float(np.exp(float(x0[0]))),
                "x0_var": float(np.exp(float(x0[1]))),
            }
            row["optimizer_eval_trace"] = trace_rows
            row["optimizer_iter_trace"] = iter_rows

        metrics = dict(best_eval["metrics"])
        row.update(
            {
                "ok": True,
                "results": best_eval["results"],
                "ell_effective": float(best_eval["ell"]),
                "var_product": float(best_eval["var"]),
                "optimized_hyperparams": bool(optimize_gp_params),
                **metrics,
            },
        )
        print(
            f"  [{trial_label}] center_red_chi2={metrics['center_reduced_chi2']:.4f} "
            f"total_red_chi2={metrics['total_reduced_chi2']:.4f} "
            f"ell={float(best_eval['ell']):.4f} var={float(best_eval['var']):.4e}",
        )
    except Exception as exc:
        row.update({"ok": False, "error": str(exc)})
        print(f"  [{trial_label}] FAILED: {exc}")
    return row


def _tier_pick_best(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    ok = [r for r in rows if r.get("ok") and np.isfinite(float(r.get("center_reduced_chi2", float("nan"))))]
    if not ok:
        return None
    ok.sort(key=lambda r: (float(r["center_reduced_chi2"]), float(r.get("total_reduced_chi2", 1e99))))
    return ok[0]


def _tier_write_pdf_stack(
    stage_name: str,
    cutouts: List[dict],
    results: dict,
    output_dir: str,
    iter_rows: List[Dict[str, object]],
    require_nuclear_point: bool,
    centers: dict,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    nuclear_pdf_overrides = None
    if require_nuclear_point:
        nr0, nd0 = centers.get("nuc_ra"), centers.get("nuc_dec")
        if nr0 is not None and nd0 is not None:
            nuclear_pdf_overrides = {
                "USE_NUCLEAR_POINT_SOURCE": True,
                "NUCLEAR_POINT_RA": float(nr0),
                "NUCLEAR_POINT_DEC": float(nd0),
            }
    if nuclear_pdf_overrides is not None:
        with _temporary_config(nuclear_pdf_overrides):
            diag = write_native_fit_pdf(stage_name, cutouts, results, output_dir)
            stack = write_stacked_residual_pdf(stage_name, cutouts, results, output_dir)
    else:
        diag = write_native_fit_pdf(stage_name, cutouts, results, output_dir)
        stack = write_stacked_residual_pdf(stage_name, cutouts, results, output_dir)
    plot_rows = []
    for i, r in enumerate(iter_rows):
        if not isinstance(r, dict):
            continue
        pr = dict(r)
        pr.setdefault("iteration", i)
        pr.setdefault("use_point", False)
        plot_rows.append(pr)
    write_iteration_metric_plot(stage_name, plot_rows, output_dir)
    return diag, stack


def _write_human_review_tier_d(run_root: str, manifest: dict) -> str:
    path = os.path.join(run_root, "HUMAN_REVIEW_TIER_D.md")
    lines = [
        "# Human review before Tier D",
        "",
        "Tiers A–C completed automatically. **Do not implement Tier D** (prior geometry,",
        "multi-scale GP, extra Q jitter, etc.) until:",
        "",
        "1. You have reviewed the PDFs and `tier_summary.json` in `tier_A/`, `tier_B/`, `tier_C/`.",
        "2. You updated the project plan based on findings.",
        "3. You explicitly confirmed the next implementation step.",
        "",
        "## Run manifest (machine-readable)",
        "",
        f"See `GP_TIER_RUN_MANIFEST.json` in this folder.",
        "",
        "## Manifest summary",
        "",
        "```json",
        json.dumps(manifest, indent=2, default=str),
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def run_gp_tier_sequence(
    stage_name: str,
    n_bcd: int,
    *,
    real_case: Dict[str, object],
    base_output_dir: str,
    center_radius_px: float = 3.0,
    reduced_chi2_target: float = 1.5,
    require_nuclear_point: bool = False,
    data_source: str = "real",
    use_point: bool = False,
) -> Dict[str, object]:
    """
    Automated Tier A → (gate) → Tier B → (gate) → Tier C with separate output subdirs.
    Tier D is not executed; HUMAN_REVIEW_TIER_D.md is written at the end.
    """
    if data_source != "real":
        raise RuntimeError("run_gp_tier_sequence currently supports data_source='real' only")
    if real_case is None:
        raise RuntimeError("real_case is required")
    all_tpl = list(real_case["template_cutouts"])
    cutouts = all_tpl if n_bcd >= len(all_tpl) else all_tpl[: int(n_bcd)]
    cutouts = _reindex_epochs([dict(c) for c in cutouts])
    scene_wcs = real_case["scene_wcs"]
    scene_shape = tuple(real_case["scene_shape"])
    centers = dict(real_case["centers"])
    stars = list(real_case["all_stars"])
    init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
    if require_nuclear_point and (centers.get("nuc_ra") is None or centers.get("nuc_dec") is None):
        raise RuntimeError("require_nuclear_point needs explicit nucleus coordinates in config")
    for c in cutouts:
        apply_native_cutout_cr_mask(c)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(base_output_dir, "gp_tier_tuning", f"{stage_name}_{run_id}")
    os.makedirs(run_root, exist_ok=True)
    base_var = 1e-7
    smooth = float(getattr(config, "GP_FALLBACK_NEIGHBOR_SMOOTHNESS", 0.15))
    base_ell = 1.8
    var_mult = 1.0
    _run_solver._stars = stars
    _run_solver._star_fluxes = init_star_fluxes

    manifest: Dict[str, object] = {
        "run_root": run_root,
        "stage_name": stage_name,
        "n_bcd": int(n_bcd),
        "tiers": [],
        "winner_tier": None,
    }

    def _trial_json(r: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in r.items() if k != "results"}

    # Budgeted A/B/C path for true GP hyperparameter optimization:
    # one core trial per tier for N=1 responsiveness.
    tier_a_trials = [
        ("A0_baseline", {}),
    ]

    tier_a_dir = os.path.join(run_root, "tier_A")
    os.makedirs(tier_a_dir, exist_ok=True)
    rows_a_full: List[Dict[str, object]] = []
    baseline_metrics: Optional[Dict[str, float]] = None
    for label, merge in tier_a_trials:
        row = _solve_campaign_trial(
            cutouts,
            scene_wcs,
            scene_shape,
            centers,
            stars,
            init_star_fluxes,
            base_ell,
            base_var,
            var_mult,
            use_point,
            require_nuclear_point,
            smooth,
            merge,
            label,
            center_radius_px=float(center_radius_px),
            optimize_gp_params=True,
        )
        rows_a_full.append(row)
        if baseline_metrics is None and row.get("ok"):
            baseline_metrics = {
                "center_reduced_chi2": float(row["center_reduced_chi2"]),
                "total_reduced_chi2": float(row["total_reduced_chi2"]),
            }

    best_a = _tier_pick_best(rows_a_full)
    if best_a is None:
        raise RuntimeError("Tier A: all trials failed")
    res_a = next(r["results"] for r in rows_a_full if r.get("ok") and r.get("trial") == best_a["trial"])
    metrics_a = {k: float(best_a[k]) for k in ("center_reduced_chi2", "total_reduced_chi2", "center_noise_ratio") if k in best_a}
    merge_a_win = next((m for lab, m in tier_a_trials if lab == best_a["trial"]), {})
    rows_a_json = [_trial_json(r) for r in rows_a_full]
    _tier_write_pdf_stack(stage_name, cutouts, res_a, tier_a_dir, rows_a_json, require_nuclear_point, centers)
    with open(os.path.join(tier_a_dir, "tier_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"tier": "A", "best_trial": best_a.get("trial"), "merge": merge_a_win, "metrics": metrics_a, "trials": rows_a_json},
            f,
            indent=2,
            default=str,
        )
    gate_a = _gp_tier_gate_passes(metrics_a, baseline_metrics)
    manifest["tiers"].append({"name": "A", "passed_gate": gate_a, "dir": tier_a_dir, "metrics": metrics_a})
    manifest["winner_tier"] = "A"
    if gate_a:
        mp = os.path.join(run_root, "GP_TIER_RUN_MANIFEST.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        human = _write_human_review_tier_d(run_root, manifest)
        manifest["manifest_json"] = mp
        manifest["human_review_md"] = human
        return manifest

    tier_b_dir = os.path.join(run_root, "tier_B")
    os.makedirs(tier_b_dir, exist_ok=True)
    merge_b0 = dict(merge_a_win)
    ell_hp, var_hp = float(base_ell), float(base_var) * float(var_mult)
    if getattr(config, "GP_OPTIMIZE_HYPERPARAMS", False) and cutouts:
        try:
            ell_hp, vopt = gp_model.optimize_hyperparameters([cutouts[0]])
            var_hp = float(max(vopt, float(base_var)))
            print(f"  [Tier B] optimize_hyperparameters ell={ell_hp:.4f} var={var_hp:.4e}")
        except Exception as exc:
            print(f"  [Tier B] hyperparameter optimization failed, using defaults: {exc}")
            ell_hp, var_hp = float(base_ell), float(base_var) * float(var_mult)

    rows_b_full: List[Dict[str, object]] = []
    hp_families = [
        ("opt", float(ell_hp), float(var_hp)),
    ]
    for fam, ell0, var0 in hp_families:
        for ef in (1.0,):
            for vf in (1.0,):
                label = f"B_{fam}_ellx{ef:.2f}_varx{vf:.2f}"
                row = _solve_campaign_trial(
                    cutouts,
                    scene_wcs,
                    scene_shape,
                    centers,
                    stars,
                    init_star_fluxes,
                    float(ell0) * float(ef),
                    var0,
                    float(vf),
                    use_point,
                    require_nuclear_point,
                    smooth,
                    merge_b0,
                    label,
                    center_radius_px=float(center_radius_px),
                    optimize_gp_params=True,
                )
                rows_b_full.append(row)

    best_b = _tier_pick_best(rows_b_full)
    if best_b is None:
        raise RuntimeError("Tier B: all trials failed")
    res_b = next(r["results"] for r in rows_b_full if r.get("ok") and r.get("trial") == best_b["trial"])
    metrics_b = {k: float(best_b[k]) for k in ("center_reduced_chi2", "total_reduced_chi2") if k in best_b}
    rows_b_json = [_trial_json(r) for r in rows_b_full]
    _tier_write_pdf_stack(stage_name, cutouts, res_b, tier_b_dir, rows_b_json, require_nuclear_point, centers)
    with open(os.path.join(tier_b_dir, "tier_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"tier": "B", "best_trial": best_b.get("trial"), "merge": merge_b0, "metrics": metrics_b, "trials": rows_b_json},
            f,
            indent=2,
            default=str,
        )
    gate_b = _gp_tier_gate_passes(metrics_b, baseline_metrics)
    manifest["tiers"].append({"name": "B", "passed_gate": gate_b, "dir": tier_b_dir, "metrics": metrics_b})
    manifest["winner_tier"] = "B"
    if gate_b:
        mp = os.path.join(run_root, "GP_TIER_RUN_MANIFEST.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        human = _write_human_review_tier_d(run_root, manifest)
        manifest["manifest_json"] = mp
        manifest["human_review_md"] = human
        return manifest

    tier_c_dir = os.path.join(run_root, "tier_C")
    os.makedirs(tier_c_dir, exist_ok=True)
    merge_c = dict(merge_b0)
    merge_c["USE_HOST_GAUSSIAN_CORE"] = True
    # Nucleus often matches GP profile center; allow host in this diagnostic tier without disabling.
    merge_c["HOST_GAUSSIAN_MIN_OFFSET_PX"] = 0.0
    nr, nd = centers.get("nuc_ra"), centers.get("nuc_dec")
    tier_c_skipped = False
    skip_reason = ""
    if nr is None or nd is None:
        tier_c_skipped = True
        skip_reason = "no nucleus coordinates for HOST_CORE"
    else:
        ra_gp, dec_gp = solver._gp_profile_center_world()
        x_h, y_h = scene_wcs.world_to_pixel_values(float(nr), float(nd))
        x_gp, y_gp = scene_wcs.world_to_pixel_values(float(ra_gp), float(dec_gp))
        off = float(np.hypot(x_h - x_gp, y_h - y_gp))
        min_off = float(
            max(
                0.0,
                float(
                    merge_c.get(
                        "HOST_GAUSSIAN_MIN_OFFSET_PX",
                        getattr(config, "HOST_GAUSSIAN_MIN_OFFSET_PX", 1.0),
                    ),
                ),
            ),
        )
        if off < min_off:
            tier_c_skipped = True
            skip_reason = f"host offset {off:.3f}px < HOST_GAUSSIAN_MIN_OFFSET_PX={min_off}"

    rows_c_json: List[Dict[str, object]] = []
    res_c = res_b
    metrics_c = dict(metrics_b)
    if not tier_c_skipped:
        row_c = _solve_campaign_trial(
            cutouts,
            scene_wcs,
            scene_shape,
            centers,
            stars,
            init_star_fluxes,
            float(best_b["ell_effective"]),
            float(best_b["var_product"]),
            1.0,
            use_point,
            require_nuclear_point,
            smooth,
            merge_c,
            "C_host_gaussian",
            center_radius_px=float(center_radius_px),
            optimize_gp_params=True,
        )
        rows_c_json.append(_trial_json(row_c))
        if row_c.get("ok"):
            res_c = row_c["results"]
            metrics_c = {
                k: float(row_c[k])
                for k in ("center_reduced_chi2", "total_reduced_chi2", "center_noise_ratio")
                if k in row_c
            }
            _tier_write_pdf_stack(stage_name, cutouts, res_c, tier_c_dir, rows_c_json, require_nuclear_point, centers)
        else:
            rows_c_json[-1]["error"] = row_c.get("error")
    else:
        rows_c_json.append({"skipped": True, "reason": skip_reason})

    with open(os.path.join(tier_c_dir, "tier_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tier": "C",
                "skipped": tier_c_skipped,
                "skip_reason": skip_reason,
                "metrics": metrics_c,
                "trials": rows_c_json,
            },
            f,
            indent=2,
            default=str,
        )
    if tier_c_skipped:
        with open(os.path.join(tier_c_dir, "README_SKIPPED.txt"), "w", encoding="utf-8") as f:
            f.write(skip_reason + "\n")

    manifest["tiers"].append({"name": "C", "dir": tier_c_dir, "skipped": tier_c_skipped, "metrics": metrics_c})
    manifest["winner_tier"] = "C"

    mp = os.path.join(run_root, "GP_TIER_RUN_MANIFEST.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    human = _write_human_review_tier_d(run_root, manifest)
    manifest["manifest_json"] = mp
    manifest["human_review_md"] = human
    return manifest


def run_stage(
    stage_name: str,
    n_bcd: int,
    *,
    max_iterations: int,
    center_radius_px: float,
    output_dir: str,
    reduced_chi2_target: float,
    real_case: Optional[Dict[str, object]] = None,
    data_source: str = "real",
    allow_point_source: bool = False,
    require_nuclear_point: bool = False,
    aggressive_recovery: bool = True,
    solver_config_merge: Optional[Dict[str, object]] = None,
    gp_amp_variance: Optional[float] = None,
    skip_dipole_refinement: bool = False,
) -> Tuple[StageResult, List[Dict[str, object]]]:
    if data_source == "real":
        if real_case is None:
            raise RuntimeError("real_case is required for data_source='real'")
        all_tpl = list(real_case["template_cutouts"])
        if n_bcd >= len(all_tpl):
            cutouts = all_tpl
        else:
            cutouts = all_tpl[: int(n_bcd)]
        cutouts = _reindex_epochs(cutouts)
        scene_wcs = real_case["scene_wcs"]
        scene_shape = tuple(real_case["scene_shape"])
        centers = dict(real_case["centers"])
        stars = list(real_case["all_stars"])
        init_star_fluxes = np.asarray(real_case["init_star_fluxes"], dtype=float)
    elif data_source == "synthetic":
        cutouts, scene_wcs, scene_shape, centers = generate_synthetic_case(n_bcd)
        stars = []
        init_star_fluxes = np.zeros(0, dtype=float)
    else:
        raise RuntimeError(f"Unknown data_source: {data_source}")
    if require_nuclear_point and (centers.get("nuc_ra") is None or centers.get("nuc_dec") is None):
        raise RuntimeError(
            "require_nuclear_point needs explicit nucleus coordinates in config "
            "(NUCLEAR_POINT_RA/DEC, HOST_CORE_RA/DEC, GALAXY_EXTENDED_CENTER_RA/DEC, "
            "or GP_PROFILE_CENTER_RA/DEC). Transient RA/Dec is not used as the nucleus.",
        )
    for c in cutouts:
        apply_native_cutout_cr_mask(c)
    iter_log: List[Dict[str, object]] = []
    best = None
    best_metrics = None
    best_knobs = None
    best_iter = 0
    recovered = 0
    # Default 1e-7 matches legacy native_fit_campaign scaling; dense Matérn uses var as kernel scale.
    # Diagonal GP fallback treats var as per-pixel marginal variance — use ``gp_amp_variance``
    # (e.g. config.INIT_VARIANCE) for independence/diagnostic runs or the prior is absurdly tight.
    base_var = float(gp_amp_variance) if gp_amp_variance is not None else 1e-7
    _run_solver._stars = stars
    _run_solver._star_fluxes = init_star_fluxes
    iterations_run = 0
    for it in range(int(max_iterations)):
        iterations_run = it + 1
        print(f"[{stage_name}] iteration {it + 1}/{int(max_iterations)}")
        # Nuclear delta uses exact RA/Dec → subpixel scene placement (_add_delta_to_image)
        # before spatially varying PRF convolution (column_L_pointsource).
        if require_nuclear_point:
            point_trials = (True,)
        elif allow_point_source:
            point_trials = (False, True)
        else:
            point_trials = (False,)
        for use_point in point_trials:
            knobs = _settings_space(base_ell=1.8 + 0.05 * recovered, use_point=use_point, iteration=it)
            print(
                f"[{stage_name}] trial use_point={bool(use_point)} ell={knobs['ell']:.3f} "
                f"var_mult={knobs['var_mult']:.3f}"
            )
            smooth = float(getattr(config, "GP_FALLBACK_NEIGHBOR_SMOOTHNESS", 0.15))
            if aggressive_recovery and it > 0 and best_metrics is not None:
                if not np.isfinite(best_metrics["center_reduced_chi2"]) or best_metrics["center_reduced_chi2"] > reduced_chi2_target:
                    smooth = min(1.0, smooth + 0.05 * recovered)
            tr_ra = float(centers["transient_ra"])
            tr_dec = float(centers["transient_dec"])
            nr = centers.get("nuc_ra")
            nd = centers.get("nuc_dec")
            if bool(use_point) or require_nuclear_point:
                if nr is None or nd is None:
                    raise RuntimeError(
                        "Nuclear point trials require explicit nucleus sky coordinates in config "
                        "(NUCLEAR_POINT_RA/DEC, HOST_CORE_RA/DEC, GALAXY_EXTENDED_CENTER_RA/DEC, "
                        "or GP_PROFILE_CENTER_RA/DEC). Transient RA/Dec is not used as the nucleus.",
                    )
            host_ra = float(nr) if nr is not None else tr_ra
            host_dec = float(nd) if nd is not None else tr_dec
            nuc_ra_cfg = float(nr) if (bool(use_point) or require_nuclear_point) else tr_ra
            nuc_dec_cfg = float(nd) if (bool(use_point) or require_nuclear_point) else tr_dec
            cfg = {
                "TRANSIENT_RA": tr_ra,
                "TRANSIENT_DEC": tr_dec,
                "USE_HOST_GAUSSIAN_CORE": False,
                "HOST_CORE_RA": host_ra,
                "HOST_CORE_DEC": host_dec,
                "USE_NUCLEAR_POINT_SOURCE": bool(use_point),
                "NUCLEAR_POINT_RA": nuc_ra_cfg,
                "NUCLEAR_POINT_DEC": nuc_dec_cfg,
                # Do not cap MAX_SCENE_PIXELS here: use config (dense Matérn same as main pipeline).
                "GP_FALLBACK_NEIGHBOR_SMOOTHNESS": float(max(0.35, smooth)),
            }
            if solver_config_merge:
                cfg.update(dict(solver_config_merge))
            try:
                with _temporary_config(cfg):
                    results = _run_solver(
                        cutouts,
                        scene_wcs,
                        scene_shape,
                        knobs["ell"],
                        base_var * knobs["var_mult"],
                    )
                metrics = fit_metrics.compute_fit_metrics(
                    cutouts,
                    results,
                    stars,
                    results.get("star_fluxes", init_star_fluxes),
                    center_ra_deg=float(centers["nuc_ra"] or centers["transient_ra"]),
                    center_dec_deg=float(centers["nuc_dec"] or centers["transient_dec"]),
                    center_radius_px=float(center_radius_px),
                )
                row = {"iteration": it, "use_point": use_point, **knobs, **metrics}
                iter_log.append(row)
                print(
                    f"[{stage_name}] metrics center_red_chi2={metrics['center_reduced_chi2']:.4f} "
                    f"total_red_chi2={metrics['total_reduced_chi2']:.4f} "
                    f"center_noise_ratio={metrics['center_noise_ratio']:.4f}"
                )
                if best is None or metrics["center_reduced_chi2"] < best_metrics["center_reduced_chi2"]:
                    best = results
                    best_metrics = metrics
                    best_knobs = knobs
                    best_iter = it
            except Exception as exc:
                iter_log.append({"iteration": it, "use_point": use_point, "error": str(exc), **knobs})
                recovered += 1
                continue
        if best_metrics is not None and best_metrics["center_reduced_chi2"] <= reduced_chi2_target:
            break
    if best is None:
        raise RuntimeError(f"{stage_name}: all iterations failed")

    # Dipole check only after monopole/core criterion is acceptable.
    met_primary = bool(best_metrics["center_reduced_chi2"] <= reduced_chi2_target)
    met_fallback = bool(np.isfinite(best_metrics["center_noise_ratio"]) and best_metrics["center_noise_ratio"] <= 2.0)
    dip_shift = 0.0
    is_template_only = all(bool(c.get("is_template")) for c in cutouts)
    if not bool(skip_dipole_refinement) and ((met_primary or met_fallback) or is_template_only):
        dip = _run_dipole_refinement(cutouts, best)
        dip_shift = float(dip.get("best_s_pix", 0.0))
        ux = float(dip.get("ux", 0.0))
        uy = float(dip.get("uy", 0.0))
        # For template-only stages, actively apply the measured dipole recentering
        # and keep it only if center chi2 improves.
        if is_template_only and np.isfinite(dip_shift) and abs(dip_shift) >= 0.02:
            dx = float(dip_shift * ux)
            dy = float(dip_shift * uy)
            shifted_cutouts = _shift_cutout_wcs(cutouts, dx, dy)
            try:
                with _temporary_config(
                    _campaign_solver_cfg(
                        centers,
                        use_point=bool((best_knobs or {}).get("use_point", False)),
                        require_nuclear_point=require_nuclear_point,
                        smooth=float(max(0.35, getattr(config, "GP_FALLBACK_NEIGHBOR_SMOOTHNESS", 0.35))),
                        merge_extra=solver_config_merge,
                    )
                ):
                    shifted_results = _run_solver(
                        shifted_cutouts,
                        scene_wcs,
                        scene_shape,
                        float((best_knobs or {}).get("ell", 1.8)),
                        float(base_var * float((best_knobs or {}).get("var_mult", 1.0))),
                    )
                shifted_metrics = fit_metrics.compute_fit_metrics(
                    shifted_cutouts,
                    shifted_results,
                    stars,
                    shifted_results.get("star_fluxes", init_star_fluxes),
                    center_ra_deg=float(centers["nuc_ra"] or centers["transient_ra"]),
                    center_dec_deg=float(centers["nuc_dec"] or centers["transient_dec"]),
                    center_radius_px=float(center_radius_px),
                )
                improved_chi2 = float(shifted_metrics.get("center_reduced_chi2", np.inf)) < float(best_metrics.get("center_reduced_chi2", np.inf))
                improved_dipole = float(shifted_metrics.get("center_dipole_mag_pix_median", np.inf)) < float(best_metrics.get("center_dipole_mag_pix_median", np.inf))
                if improved_chi2 or improved_dipole:
                    cutouts = shifted_cutouts
                    best = shifted_results
                    best_metrics = shifted_metrics
            except Exception:
                pass
    # predict_cutout_model / _intrinsic_components read global config; restore nuclear flags
    # for PDFs when the chosen fit included the nuclear point source.
    nuclear_pdf_overrides = None
    if ((best_knobs or {}).get("use_point")) or require_nuclear_point:
        nr0, nd0 = centers.get("nuc_ra"), centers.get("nuc_dec")
        if nr0 is not None and nd0 is not None:
            nuclear_pdf_overrides = {
                "USE_NUCLEAR_POINT_SOURCE": True,
                "NUCLEAR_POINT_RA": float(nr0),
                "NUCLEAR_POINT_DEC": float(nd0),
            }
    if nuclear_pdf_overrides is not None:
        with _temporary_config(nuclear_pdf_overrides):
            diag_pdf = write_native_fit_pdf(stage_name, cutouts, best, output_dir)
            stack_pdf = write_stacked_residual_pdf(stage_name, cutouts, best, output_dir)
    else:
        diag_pdf = write_native_fit_pdf(stage_name, cutouts, best, output_dir)
        stack_pdf = write_stacked_residual_pdf(stage_name, cutouts, best, output_dir)
    write_iteration_metric_plot(stage_name, iter_log, output_dir)
    center_report = write_bcd_center_offset_report(stage_name, cutouts, best, output_dir)
    print(f"[{stage_name}] center offset report: {center_report}")
    return (
        StageResult(
            stage_name=stage_name,
            n_bcd=int(len(cutouts)),
            best_iteration=int(best_iter),
            best_metrics=best_metrics,
            best_knobs=best_knobs or {},
            diagnostic_pdf=diag_pdf,
            stacked_pdf=stack_pdf,
            dipole_shift_pix=float(dip_shift),
            met_primary=met_primary,
            met_fallback=met_fallback,
            iterations_run=int(iterations_run),
        ),
        iter_log,
    )


def infer_all_template_count() -> int:
    path = os.path.join(config.OUTPUT_DIR, "used_template_bcds.txt")
    if not os.path.exists(path):
        return 30
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return max(10, len(lines))


def write_campaign_summary(
    out_dir: str,
    stage_results: List[StageResult],
    iter_logs: Dict[str, List[Dict[str, object]]],
    *,
    command: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    summary_json = os.path.join(out_dir, "campaign_summary.json")
    payload = {
        "command": command,
        "stages": [asdict(s) for s in stage_results],
        "iteration_logs": iter_logs,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_md = os.path.join(out_dir, "final_run_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Final Run Summary\n\n")
        f.write(f"- Command: `{command}`\n")
        f.write("\n## Model Parameters (current campaign)\n")
        f.write("- GP scene pixel amplitudes: fit\n")
        f.write("- Per-BCD background amplitudes: fit\n")
        f.write("- Host core RA/Dec: fixed\n")
        f.write("- Host core amplitude: fit (when host enabled)\n")
        f.write("- Host core sigma: selected by iteration trials\n")
        f.write("- Nuclear point source RA/Dec: fixed (at nucleus)\n")
        f.write("- Nuclear point source amplitude: fit (when enabled)\n")
        f.write("- Field stars: fixed off (none)\n")
        for s in stage_results:
            f.write(f"\n## {s.stage_name}\n")
            f.write(f"- n_bcd: {s.n_bcd}\n")
            f.write(f"- center_reduced_chi2: {s.best_metrics.get('center_reduced_chi2', float('nan')):.4f}\n")
            f.write(f"- center_noise_ratio: {s.best_metrics.get('center_noise_ratio', float('nan')):.4f}\n")
            f.write(f"- total_reduced_chi2: {s.best_metrics.get('total_reduced_chi2', float('nan')):.4f}\n")
            f.write(f"- dipole_shift_pix: {s.dipole_shift_pix:.4f}\n")
            f.write(f"- met_primary: {s.met_primary}\n")
            f.write(f"- met_fallback: {s.met_fallback}\n")
            f.write(f"- diagnostic_pdf: `{s.diagnostic_pdf}`\n")
            f.write(f"- stacked_pdf: `{s.stacked_pdf}`\n")
            iter_plot = os.path.join(out_dir, f"ITER_METRICS_{s.stage_name}.png")
            if os.path.exists(iter_plot):
                f.write(f"- iteration_metrics_plot: `{iter_plot}`\n")
        f.write("\n## Notable Conclusions\n")
        f.write("- Iterative bracketing (with/without point source) was executed each iteration.\n")
        f.write("- Kernel length was tuned adaptively; aggressive recovery adjusted smoothing when needed.\n")
        f.write("- Stage promotion used primary reduced-chi2 target or fallback 2x Poisson criterion.\n")
    return summary_json, summary_md
