"""src/solver.py"""
import sys
import os
import logging
import warnings
from typing import List, Tuple

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.optimize import Bounds, minimize
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from . import config, gp_model
from .pmap_correction import iracpc_pmap_corr

# Spitzer BCD headers often list both CD and CDELT; Astropy prefers CD and
# emits repetitive "cdelt ignored" warnings whenever the WCS is inspected.
for _cls in (RuntimeWarning, UserWarning):
    warnings.filterwarnings(
        "ignore",
        category=_cls,
        message=r".*[Cc]delt.*[Ii]gnored.*",
    )

_log = logging.getLogger(__name__)
_GEOMETRY_CACHE = {}
_PROJECTION_CACHE = {}

GRID_CENTERS = np.array([25, 77, 129, 181, 233])


def _read_fits_image_array(path):
    """Load primary HDU data; ignore common SSC-style END-card header noise."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)
        with fits.open(path, memmap=False) as hdul:
            return np.nan_to_num(hdul[0].data, nan=0.0)

def get_gaussian_kernel_array(size=2001, fwhm=2.0, oversample=100):
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    sigma = (fwhm * oversample) / 2.355
    r2 = (x - cx)**2 + (y - cy)**2
    kernel = np.exp(-r2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_window_function(data):
    h, w = data.shape
    alpha = float(getattr(config, 'PRF_APODIZATION_EDGE', 0.1))
    if alpha <= 0.0:
        return data
    x = np.linspace(0, 1, w); y = np.linspace(0, 1, h)
    wx = np.ones_like(x); mask_x = (x < alpha/2)
    wx[mask_x] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[mask_x] - alpha/2)))
    mask_x = (x > 1 - alpha/2)
    wx[mask_x] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[mask_x] - 1 + alpha/2)))
    wy = np.ones_like(y); mask_y = (y < alpha/2)
    wy[mask_y] = 0.5 * (1 + np.cos(2*np.pi/alpha * (y[mask_y] - alpha/2)))
    mask_y = (y > 1 - alpha/2)
    wy[mask_y] = 0.5 * (1 + np.cos(2*np.pi/alpha * (y[mask_y] - 1 + alpha/2)))
    return data * np.outer(wy, wx)


def trim_prf_zero_padding(prf_img, eps=0.0):
    """
    Remove fully padded outer rows/cols where all values are <= eps.

    This trims SSC-style zero borders so edge behavior reflects the real PRF wings
    rather than artificial file padding.
    """
    arr = np.asarray(prf_img, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    mag = np.abs(arr)
    rows = np.any(mag > float(eps), axis=1)
    cols = np.any(mag > float(eps), axis=0)
    if (not np.any(rows)) or (not np.any(cols)):
        return arr
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y0, y1 = int(y_idx[0]), int(y_idx[-1]) + 1
    x0, x1 = int(x_idx[0]), int(x_idx[-1]) + 1
    return arr[y0:y1, x0:x1]


def load_real_prf_file(channel, row_idx, col_idx):
    if not (0 <= row_idx < 5 and 0 <= col_idx < 5): return None
    r_val = GRID_CENTERS[row_idx]; c_val = GRID_CENTERS[col_idx]
    ch_num = '2' if 'ch2' in channel.lower() else '1'
    filename = f"apex_sh_IRACPC{ch_num}_col{c_val:03d}_row{r_val:03d}_x100.fits"
    patterns = [filename, f"prf_{channel}_row{row_idx}_col{col_idx}.fits"]
    for pat in patterns:
        path = os.path.join(config.PRF_DIR, pat)
        if os.path.exists(path):
            try:
                data = _read_fits_image_array(path)
                if bool(getattr(config, "PRF_APPLY_APODIZATION", False)):
                    return apply_window_function(data)
                return data
            except OSError:
                continue
    return None

def load_interpolated_prf(channel, x_det, y_det):
    grid_centers = GRID_CENTERS
    x = np.clip(x_det, grid_centers[0], grid_centers[-1])
    y = np.clip(y_det, grid_centers[0], grid_centers[-1])
    ix = np.searchsorted(grid_centers, x, side='right'); iy = np.searchsorted(grid_centers, y, side='right')
    ix = max(1, min(ix, 4)); iy = max(1, min(iy, 4))
    x0, x1 = grid_centers[ix-1], grid_centers[ix]
    y0, y1 = grid_centers[iy-1], grid_centers[iy]
    u = (x - x0) / (x1 - x0); v = (y - y0) / (y1 - y0)
    p00 = load_real_prf_file(channel, iy-1, ix-1)
    p10 = load_real_prf_file(channel, iy-1, ix)
    p01 = load_real_prf_file(channel, iy, ix-1)
    p11 = load_real_prf_file(channel, iy, ix)
    if all(p is None for p in [p00, p10, p01, p11]):
        return get_gaussian_kernel_array(size=2001, fwhm=2.0, oversample=config.PRF_OVERSAMPLE)
    if p00 is None: p00 = p10 if p10 else (p01 if p01 else p11)
    if p10 is None: p10 = p00;
    if p01 is None: p01 = p00;
    if p11 is None: p11 = p10
    w00 = (1-u)*(1-v); w10 = u*(1-v); w01 = (1-u)*v; w11 = u*v
    eff_prf = w00*p00 + w10*p10 + w01*p01 + w11*p11
    eff_prf = trim_prf_zero_padding(eff_prf, eps=0.0)
    if np.sum(eff_prf) > 0: eff_prf /= np.sum(eff_prf)
    return eff_prf

def load_prf(channel, x_det=None, y_det=None):
    if x_det is None: x_det = 128
    if y_det is None: y_det = 128
    return load_interpolated_prf(channel, x_det, y_det)

# --- Core Solver ---

DEBUG_DUMPED = False

# (id(scene_wcs), id(raw_wcs), h, w, channel, is_full, n_anchors) -> (kernels, weights, wsum)
_PRF_OPERATOR_BUNDLE_CACHE = {}
# (id(scene_wcs), id(raw_wcs), h, w, channel, is_full) -> dense matrix A, shape (n_scene, n_scene)
# where vec(L(img)) = A @ vec(img)
_PRF_EXACT_OPERATOR_CACHE = {}
_PRF_NATIVE_OPERATOR_BUNDLE_CACHE = {}


def _center_kernel_for_convolution(ker):
    """
    Recenter a point-response image so its peak sits at array center.

    `convolved_delta_column` returns a response image centered at the source
    location in scene coordinates. For FFT convolution kernels, that response
    must be shifted to zero-lag (image center), otherwise the source is
    duplicated/translated incorrectly across the field.
    """
    k = np.asarray(ker, dtype=np.float64)
    if k.ndim != 2 or k.size == 0:
        return k
    iy, ix = np.unravel_index(int(np.nanargmax(np.abs(k))), k.shape)
    cy = k.shape[0] // 2
    cx = k.shape[1] // 2
    dy = int(cy - iy)
    dx = int(cx - ix)
    return np.roll(np.roll(k, dy, axis=0), dx, axis=1)


def generate_prf_fast(scene_wcs, raw_wcs, prf_model, ra, dec, scene_shape, channel='ch1', is_full_array=False):
    global DEBUG_DUMPED
    if prf_model is None: return np.zeros(scene_shape[0]*scene_shape[1])
    
    ph, pw = prf_model.shape
    prf_wcs = WCS(naxis=2)
    prf_wcs.wcs.crpix = [pw / 2.0, ph / 2.0]
    prf_wcs.wcs.crval = [ra, dec]
    prf_wcs.wcs.ctype = raw_wcs.wcs.ctype
    
    if hasattr(raw_wcs.wcs, 'cd'):
        prf_wcs.wcs.cd = raw_wcs.wcs.cd / float(config.PRF_OVERSAMPLE)
    else:
        prf_wcs.wcs.cdelt = raw_wcs.wcs.cdelt / float(config.PRF_OVERSAMPLE)
        prf_wcs.wcs.pc = raw_wcs.wcs.pc
    
    blur = getattr(config, 'PRF_PREBLUR_IN_OVERSAMPLE_PIXELS', None)
    if blur is not None and float(blur) > 0.0:
        prf_smooth = gaussian_filter(prf_model, float(blur))
    else:
        prf_smooth = prf_model
    
    try:
        out_arr, _ = reproject_interp((prf_smooth, prf_wcs), scene_wcs, shape_out=scene_shape)
        out_arr = np.nan_to_num(out_arr)
    except Exception as exc:
        _log.warning(
            "PRF reproject failed for (ra,dec)=(%.6f,%.6f): %s; returning zeros.",
            ra, dec, exc,
        )
        return np.zeros(scene_shape[0] * scene_shape[1])
    
    curr_sum = np.sum(out_arr)
    if curr_sum > 0:
        out_arr /= curr_sum
        
    # --- Optional intrapixel correction (photometry-style gain) ---
    tx, ty = raw_wcs.world_to_pixel_values(ra, dec)
    gain_msg = "Gain=1.0 (disabled)"
    if bool(getattr(config, "PRF_APPLY_PMAP_GAIN", False)):
        try:
            # Strict usage: Pass Full Array flag, no periodic logic.
            pmap_dir = getattr(config, 'PMAP_DIR', os.path.join(config.BASE_DIR, 'data', 'pmap_fits'))
            corr_val = iracpc_pmap_corr(1.0, tx, ty, channel,
                                        pmap_dir=pmap_dir,
                                        threshold_occ=False,
                                        full_array=is_full_array)

            if np.isfinite(corr_val) and corr_val > 0:
                gain = 1.0 / corr_val
                out_arr *= gain
                gain_msg = f"Gain={gain:.4f}"
            else:
                gain_msg = "Gain=1.0 (OOB/Masked)"
        except Exception as e:
            print(f"   [PMap Warning] Failed for ({tx:.2f}, {ty:.2f}): {e}")
            gain_msg = "Gain=1.0 (Error)"
        
    if not DEBUG_DUMPED:
        out_path = os.path.join(config.DIAGNOSTIC_DIR, 'DEBUG_PRF_GENERATED.fits')
        os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyWarning)
            hdu = fits.PrimaryHDU(np.asarray(out_arr, dtype=np.float64))
            hdu.writeto(out_path, overwrite=True, output_verify='ignore')
        print(f"   [DEBUG PRF] Loc: ({tx:.2f}, {ty:.2f}) [Full={is_full_array}] -> {gain_msg}")
        DEBUG_DUMPED = True

    return out_arr.flatten()


def _get_prf_operator_bundle(scene_wcs, w_native, scene_shape, channel, is_full_array):
    """
    Normalized anchor kernels and Gaussian blend weights for the spatially varying PRF operator L.
    Returns (kernels, weights, wsum) with L(img) = sum_a fftconv(img, ker_a) * wt_a / wsum.
    """
    h, w = int(scene_shape[0]), int(scene_shape[1])
    n_anch = int(max(1, getattr(config, 'PRF_SPATIAL_ANCHORS_PER_AXIS', 3)))
    bkey = (id(scene_wcs), id(w_native), h, w, channel, bool(is_full_array), n_anch)
    if bkey in _PRF_OPERATOR_BUNDLE_CACHE:
        return _PRF_OPERATOR_BUNDLE_CACHE[bkey]
    n = n_anch
    kernels = []
    weights = []
    if n <= 1:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ra_c, dec_c = scene_wcs.pixel_to_world_values(cx, cy)
        col = convolved_delta_column(
            scene_wcs, w_native, scene_shape, channel,
            float(np.asarray(ra_c).ravel()[0]),
            float(np.asarray(dec_c).ravel()[0]),
            is_full_array=is_full_array,
        )
        ker = _center_kernel_for_convolution(col.reshape(scene_shape))
        sk = float(np.sum(ker))
        if sk > 0:
            ker /= sk
        kernels.append(np.asarray(ker, dtype=np.float64))
        weights.append(np.ones((h, w), dtype=np.float64))
    else:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        xs = np.linspace(0.0, w - 1.0, n)
        ys = np.linspace(0.0, h - 1.0, n)
        sigma = max(h, w) / max(2.0 * n, 1.0)
        for ay in ys:
            for ax in xs:
                ra_a, dec_a = scene_wcs.pixel_to_world_values(float(ax), float(ay))
                col = convolved_delta_column(
                    scene_wcs, w_native, scene_shape, channel,
                    float(np.asarray(ra_a).ravel()[0]),
                    float(np.asarray(dec_a).ravel()[0]),
                    is_full_array=is_full_array,
                )
                ker = _center_kernel_for_convolution(col.reshape(scene_shape))
                sk = float(np.sum(ker))
                if sk > 0:
                    ker /= sk
                wt = np.exp(-0.5 * (((xx - ax) / sigma) ** 2 + ((yy - ay) / sigma) ** 2))
                kernels.append(np.asarray(ker, dtype=np.float64))
                weights.append(wt.astype(np.float64))
    wsum = np.zeros((h, w), dtype=np.float64)
    for wt in weights:
        wsum += wt
    wsum = np.clip(wsum, 1e-30, None)
    _PRF_OPERATOR_BUNDLE_CACHE[bkey] = (kernels, weights, wsum)
    return kernels, weights, wsum


def _apply_prf_operator_from_bundle(img, kernels, weights, wsum):
    img = np.asarray(img, dtype=np.float64)
    out = np.zeros_like(img, dtype=np.float64)
    for ker, wt in zip(kernels, weights):
        # Use circular convolution on the scene grid (FFT) with a kernel whose peak
        # is at the array center (ifftshift to zero-lag). This yields a consistent
        # linear operator with a well-defined Euclidean adjoint.
        ker0 = np.fft.ifftshift(np.asarray(ker, dtype=np.float64))
        K = np.fft.fft2(ker0)
        out += np.fft.ifft2(np.fft.fft2(img) * K).real * wt
    return out / wsum


def _apply_prf_adjoint_from_bundle(y, kernels, weights, wsum):
    """Adjoint of L for the Euclidean inner product (matches Jᵀ in WLS when composed with W)."""
    z = np.asarray(y, dtype=np.float64) / wsum
    acc = np.zeros_like(z, dtype=np.float64)
    for ker, wt in zip(kernels, weights):
        sig = z * wt
        ker0 = np.fft.ifftshift(np.asarray(ker, dtype=np.float64))
        K = np.fft.fft2(ker0)
        acc += np.fft.ifft2(np.fft.fft2(sig) * np.conj(K)).real
    return acc


def _get_prf_operator_bundle_native(native_wcs, native_shape, channel, is_full_array):
    """
    Spatially varying PRF bundle on the native detector grid.
    Returns kernels/weights/wsum such that L_native(img_native) is applied on BCD pixels.
    """
    h, w = int(native_shape[0]), int(native_shape[1])
    n_anch = int(max(1, getattr(config, 'PRF_SPATIAL_ANCHORS_PER_AXIS', 3)))
    bkey = (id(native_wcs), h, w, channel, bool(is_full_array), n_anch)
    if bkey in _PRF_NATIVE_OPERATOR_BUNDLE_CACHE:
        return _PRF_NATIVE_OPERATOR_BUNDLE_CACHE[bkey]

    kernels = []
    weights = []
    if n_anch <= 1:
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        ra_c, dec_c = native_wcs.pixel_to_world_values(cx, cy)
        col = convolved_delta_column(
            native_wcs, native_wcs, native_shape, channel,
            float(np.asarray(ra_c).ravel()[0]),
            float(np.asarray(dec_c).ravel()[0]),
            is_full_array=is_full_array,
        )
        ker = _center_kernel_for_convolution(col.reshape(native_shape))
        sk = float(np.sum(ker))
        if sk > 0:
            ker /= sk
        kernels.append(np.asarray(ker, dtype=np.float64))
        weights.append(np.ones((h, w), dtype=np.float64))
    else:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        xs = np.linspace(0.0, w - 1.0, n_anch)
        ys = np.linspace(0.0, h - 1.0, n_anch)
        sigma = max(h, w) / max(2.0 * n_anch, 1.0)
        for ay in ys:
            for ax in xs:
                ra_a, dec_a = native_wcs.pixel_to_world_values(float(ax), float(ay))
                col = convolved_delta_column(
                    native_wcs, native_wcs, native_shape, channel,
                    float(np.asarray(ra_a).ravel()[0]),
                    float(np.asarray(dec_a).ravel()[0]),
                    is_full_array=is_full_array,
                )
                ker = _center_kernel_for_convolution(col.reshape(native_shape))
                sk = float(np.sum(ker))
                if sk > 0:
                    ker /= sk
                wt = np.exp(-0.5 * (((xx - ax) / sigma) ** 2 + ((yy - ay) / sigma) ** 2))
                kernels.append(np.asarray(ker, dtype=np.float64))
                weights.append(wt.astype(np.float64))
    wsum = np.zeros((h, w), dtype=np.float64)
    for wt in weights:
        wsum += wt
    wsum = np.clip(wsum, 1e-30, None)
    _PRF_NATIVE_OPERATOR_BUNDLE_CACHE[bkey] = (kernels, weights, wsum)
    return kernels, weights, wsum


def _apply_prf_operator_native(img_native, native_wcs, native_shape, channel, is_full_array):
    kernels, weights, wsum = _get_prf_operator_bundle_native(
        native_wcs, native_shape, channel, is_full_array,
    )
    return _apply_prf_operator_from_bundle(
        np.asarray(img_native, dtype=np.float64).reshape(native_shape),
        kernels,
        weights,
        wsum,
    )


def _apply_prf_adjoint_native(y_native, native_wcs, native_shape, channel, is_full_array):
    kernels, weights, wsum = _get_prf_operator_bundle_native(
        native_wcs, native_shape, channel, is_full_array,
    )
    return _apply_prf_adjoint_from_bundle(
        np.asarray(y_native, dtype=np.float64).reshape(native_shape),
        kernels,
        weights,
        wsum,
    )


def _prf_operator_mode(scene_shape):
    mode = str(getattr(config, "PRF_OPERATOR_MODE", "anchor")).strip().lower()
    n_scene = int(scene_shape[0]) * int(scene_shape[1])
    if mode == "exact":
        return "exact"
    if mode == "auto":
        cap = int(max(1, getattr(config, "PRF_OPERATOR_EXACT_MAX_PIXELS", 2500)))
        return "exact" if n_scene <= cap else "anchor"
    return "anchor"


def _get_prf_exact_operator_matrix(scene_wcs, w_native, scene_shape, channel, is_full_array):
    h, w = int(scene_shape[0]), int(scene_shape[1])
    key = (id(scene_wcs), id(w_native), h, w, channel, bool(is_full_array))
    A = _PRF_EXACT_OPERATOR_CACHE.get(key)
    if A is not None:
        return A
    n_scene = h * w
    A = np.zeros((n_scene, n_scene), dtype=np.float64)
    for j in range(n_scene):
        y, x = divmod(j, w)
        ra_j, dec_j = scene_wcs.pixel_to_world_values(float(x), float(y))
        col = convolved_delta_column(
            scene_wcs, w_native, scene_shape, channel,
            float(np.asarray(ra_j).ravel()[0]),
            float(np.asarray(dec_j).ravel()[0]),
            is_full_array=is_full_array,
        )
        A[:, j] = np.asarray(col, dtype=np.float64).ravel()
    _PRF_EXACT_OPERATOR_CACHE[key] = A
    return A


def _apply_prf_operator(scene_img, scene_wcs, w_native, scene_shape, channel, is_full_array):
    mode = _prf_operator_mode(scene_shape)
    img = np.asarray(scene_img, dtype=np.float64).reshape(tuple(scene_shape))
    if mode == "exact":
        A = _get_prf_exact_operator_matrix(scene_wcs, w_native, scene_shape, channel, is_full_array)
        return (A @ img.ravel()).reshape(tuple(scene_shape))
    kernels, weights, wsum = _get_prf_operator_bundle(
        scene_wcs, w_native, scene_shape, channel, is_full_array,
    )
    return _apply_prf_operator_from_bundle(img, kernels, weights, wsum)


def _apply_prf_adjoint(y_scene, scene_wcs, w_native, scene_shape, channel, is_full_array):
    mode = _prf_operator_mode(scene_shape)
    y2 = np.asarray(y_scene, dtype=np.float64).reshape(tuple(scene_shape))
    if mode == "exact":
        A = _get_prf_exact_operator_matrix(scene_wcs, w_native, scene_shape, channel, is_full_array)
        return (A.T @ y2.ravel()).reshape(tuple(scene_shape))
    kernels, weights, wsum = _get_prf_operator_bundle(
        scene_wcs, w_native, scene_shape, channel, is_full_array,
    )
    return _apply_prf_adjoint_from_bundle(y2, kernels, weights, wsum)


def _wcs_key(w: WCS):
    ww = w.wcs
    return (
        tuple(np.asarray(ww.crpix, dtype=float).ravel().tolist()),
        tuple(np.asarray(ww.crval, dtype=float).ravel().tolist()),
        tuple(np.asarray(getattr(ww, "cdelt", [np.nan, np.nan]), dtype=float).ravel().tolist()),
        tuple(np.asarray(getattr(ww, "pc", np.eye(2)), dtype=float).ravel().tolist()),
        tuple(getattr(ww, "ctype", ["", ""])),
    )


def _sip_distortion_fingerprint(w: WCS) -> tuple:
    """
    Fingerprint FITS SIP distortion for the sparse projection-operator cache.

    Linear parameters are already hashed in ``_wcs_key``; including SIP avoids
    reusing a cached projection matrix when two frames share the same linear
    astrometry but differ in distortion polynomials.
    """
    sip = getattr(w, "sip", None)
    if sip is None:
        return ("nosip",)
    parts = []
    for name in ("a", "b", "ap", "bp"):
        arr = getattr(sip, name, None)
        if arr is None:
            parts.append((name, None))
            continue
        parts.append((name, tuple(np.asarray(arr, dtype=np.float64).ravel().tolist())))
    return ("sip", tuple(parts))


def _wcs_projection_cache_key(w: WCS) -> tuple:
    return (_wcs_key(w), _sip_distortion_fingerprint(w))


def _projection_matrix(scene_wcs: WCS, native_wcs: WCS, scene_shape, native_shape):
    """
    Build sparse projection operator P such that:
      native_vec ~= P @ scene_vec

    We use bilinear interpolation on scene pixel coordinates for each native pixel center.
    The adjoint projection uses P^T, which satisfies the defining inner-product identity
    for the Euclidean dot product on the discretized grids.
    """
    scene_shape = (int(scene_shape[0]), int(scene_shape[1]))
    native_shape = (int(native_shape[0]), int(native_shape[1]))
    key = (
        _wcs_projection_cache_key(scene_wcs),
        _wcs_projection_cache_key(native_wcs),
        scene_shape,
        native_shape,
    )
    M = _PROJECTION_CACHE.get(key)
    if M is not None:
        return M

    hs, ws = scene_shape
    hn, wn = native_shape

    # Native pixel centers -> world -> scene pixel coordinates
    yy, xx = np.mgrid[0:hn, 0:wn].astype(np.float64)
    ra, dec = native_wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
    xs, ys = scene_wcs.world_to_pixel_values(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    good = np.isfinite(xs) & np.isfinite(ys)
    xs_g = xs[good]
    ys_g = ys[good]
    row_ids = np.nonzero(good)[0].astype(np.int64)

    # Bilinear weights on scene grid
    x0 = np.floor(xs_g).astype(np.int64)
    y0 = np.floor(ys_g).astype(np.int64)
    fx = xs_g - x0
    fy = ys_g - y0
    x1 = x0 + 1
    y1 = y0 + 1

    def _accum(xi, yi, wi, row_ids_in, row_acc, col_acc, data_acc):
        ok = (xi >= 0) & (xi < ws) & (yi >= 0) & (yi < hs) & np.isfinite(wi) & (wi != 0.0)
        if not np.any(ok):
            return
        rr = row_ids_in[ok]
        cc = (yi[ok] * ws + xi[ok]).astype(np.int64)
        dd = wi[ok].astype(np.float64)
        row_acc.append(rr)
        col_acc.append(cc)
        data_acc.append(dd)

    row_acc = []
    col_acc = []
    data_acc = []

    # four neighbors
    _accum(x0, y0, (1 - fx) * (1 - fy), row_ids, row_acc, col_acc, data_acc)
    _accum(x1, y0, fx * (1 - fy), row_ids, row_acc, col_acc, data_acc)
    _accum(x0, y1, (1 - fx) * fy, row_ids, row_acc, col_acc, data_acc)
    _accum(x1, y1, fx * fy, row_ids, row_acc, col_acc, data_acc)

    if row_acc:
        r = np.concatenate(row_acc, axis=0)
        c = np.concatenate(col_acc, axis=0)
        d = np.concatenate(data_acc, axis=0)
    else:
        r = np.zeros(0, dtype=np.int64)
        c = np.zeros(0, dtype=np.int64)
        d = np.zeros(0, dtype=np.float64)

    M = coo_matrix((d, (r, c)), shape=(hn * wn, hs * ws)).tocsr()
    _PROJECTION_CACHE[key] = M
    return M


def _project_scene_to_native(scene_img, scene_wcs, native_wcs, native_shape):
    """Project scene-grid image to native BCD grid."""
    arr = np.asarray(scene_img, dtype=np.float64)
    M = _projection_matrix(scene_wcs, native_wcs, arr.shape, tuple(native_shape))
    out = (M @ arr.ravel()).reshape(tuple(native_shape))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _project_native_to_scene(native_img, native_wcs, scene_wcs, scene_shape):
    """Adjoint projection native->scene via transpose of sparse projection operator."""
    arr = np.asarray(native_img, dtype=np.float64)
    M = _projection_matrix(scene_wcs, native_wcs, tuple(scene_shape), arr.shape)
    out = (M.T @ arr.ravel()).reshape(tuple(scene_shape))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_frame_forward_operator(
    scene_img,
    scene_wcs,
    native_wcs,
    scene_shape,
    native_shape,
    channel,
    *,
    is_full_array=False,
):
    """
    Frame forward operator.
      Default legacy order: F_i = P_i L_scene,i
      Optional diagnostics order: F_i = L_native,i P_i
    """
    if bool(getattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", False)):
        proj_native = _project_scene_to_native(scene_img, scene_wcs, native_wcs, native_shape)
        return _apply_prf_operator_native(
            proj_native, native_wcs, native_shape, channel, is_full_array=is_full_array,
        )
    conv_scene = apply_spatially_varying_prf_to_scene(
        scene_img, scene_wcs, native_wcs, scene_shape, channel, is_full_array=is_full_array,
    )
    return _project_scene_to_native(conv_scene, scene_wcs, native_wcs, native_shape)


def _apply_frame_adjoint_operator(
    native_img,
    scene_wcs,
    native_wcs,
    scene_shape,
    channel,
    *,
    is_full_array=False,
):
    """
    Adjoint consistent with selected forward order.
      Legacy: F^T = L_scene^T P^T
      Optional diagnostics: F^T = P^T L_native^T
    """
    if bool(getattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", False)):
        y_native_conv_adj = _apply_prf_adjoint_native(
            native_img, native_wcs, native_img.shape, channel, is_full_array=is_full_array,
        )
        return _project_native_to_scene(y_native_conv_adj, native_wcs, scene_wcs, scene_shape).reshape(scene_shape)
    y_scene = _project_native_to_scene(native_img, native_wcs, scene_wcs, scene_shape)
    return apply_spatially_varying_prf_adjoint(
        y_scene, scene_wcs, native_wcs, scene_shape, channel, is_full_array=is_full_array,
    ).reshape(scene_shape)


def _estimate_prf_support_radius_px(kernels, frac=0.999):
    """
    Estimate a conservative PRF support radius in scene pixels from anchor kernels.
    """
    if not kernels:
        return 0
    ker = np.asarray(kernels[len(kernels) // 2], dtype=np.float64)
    h, w = ker.shape
    cy = 0.5 * (h - 1)
    cx = 0.5 * (w - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rr = np.hypot(xx - cx, yy - cy).ravel()
    vv = np.clip(ker.ravel(), 0.0, None)
    if np.sum(vv) <= 0:
        return 0
    order = np.argsort(rr)
    rr_s = rr[order]
    csum = np.cumsum(vv[order]) / np.sum(vv)
    idx = int(np.searchsorted(csum, float(frac), side='left'))
    idx = int(np.clip(idx, 0, len(rr_s) - 1))
    return int(np.ceil(rr_s[idx]))


def apply_spatially_varying_prf_adjoint(
    y_scene,
    scene_wcs,
    w_native,
    scene_shape,
    channel,
    *,
    is_full_array=False,
):
    """Lᵀ y with y on the scene grid (2D or flattened)."""
    y2 = np.asarray(y_scene, dtype=np.float64).reshape(scene_shape)
    return _apply_prf_adjoint(
        y2, scene_wcs, w_native, scene_shape, channel, is_full_array=is_full_array,
    ).ravel()


def column_L_pointsource(
    scene_wcs,
    w_native,
    scene_shape,
    channel,
    ra_deg,
    dec_deg,
    *,
    is_full_array=False,
):
    """B column: L(δ) for a unit bilinear subpixel delta at exact (RA,Dec), then PRF convolution."""
    h, w = int(scene_shape[0]), int(scene_shape[1])
    img = np.zeros((h, w), dtype=np.float64)
    px, py = scene_wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
    _add_delta_to_image(img, float(px), float(py), 1.0)
    return apply_spatially_varying_prf_to_scene(
        img, scene_wcs, w_native, scene_shape, channel, is_full_array=is_full_array,
    ).ravel()


def _transient_prf_pos_derivatives(
    scene_wcs, w_native, ra0, dec0, scene_shape, channel, is_full_array, eps_deg,
):
    """Central finite-difference columns (flattened) for ∂/∂(RA,Dec) of L(δ)."""

    def _col(ra, dec):
        return column_L_pointsource(
            scene_wcs, w_native, scene_shape, channel,
            float(ra), float(dec), is_full_array=is_full_array,
        )

    col_m_ra = _col(ra0 - eps_deg, dec0)
    col_p_ra = _col(ra0 + eps_deg, dec0)
    col_m_dec = _col(ra0, dec0 - eps_deg)
    col_p_dec = _col(ra0, dec0 + eps_deg)
    d_ra = (col_p_ra - col_m_ra) / (2.0 * eps_deg)
    d_dec = (col_p_dec - col_m_dec) / (2.0 * eps_deg)
    return d_ra, d_dec


def convolved_delta_column(
    scene_wcs,
    w_native,
    scene_shape,
    channel,
    ra_deg,
    dec_deg,
    is_full_array=False,
):
    """
    Render an intrinsic sky delta-function at (RA,Dec) through the spatially-varying PRF.

    Mathematically this is δ(ra,dec) ⊗ PRF_i on the analysis scene grid for frame i.
    """
    tx, ty = w_native.world_to_pixel_values(ra_deg, dec_deg)
    prf = load_prf(channel, tx, ty)
    return generate_prf_fast(
        scene_wcs, w_native, prf, ra_deg, dec_deg, scene_shape,
        channel=channel, is_full_array=is_full_array,
    )


def _add_delta_to_image(img, x, y, amp):
    """Add a subpixel delta via bilinear weights in image pixel coordinates."""
    h, w = img.shape
    if not np.isfinite(x) or not np.isfinite(y):
        return
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = float(x - x0)
    fy = float(y - y0)
    for dx in (0, 1):
        for dy in (0, 1):
            xi = x0 + dx
            yi = y0 + dy
            if 0 <= xi < w and 0 <= yi < h:
                wx = (1.0 - fx) if dx == 0 else fx
                wy = (1.0 - fy) if dy == 0 else fy
                img[yi, xi] += float(amp) * wx * wy


def apply_spatially_varying_prf_to_scene(
    intrinsic_scene,
    scene_wcs,
    w_native,
    scene_shape,
    channel,
    *,
    is_full_array=False,
):
    """
    Approximate full-model convolution with spatially varying PRF across a frame.

    Uses anchor PRFs (N x N across the frame), convolves the *entire* intrinsic scene with each
    anchor kernel, then blends the convolved outputs with smooth spatial weights.
    """
    img = np.asarray(intrinsic_scene, dtype=np.float64).reshape(scene_shape)
    return _apply_prf_operator(
        img, scene_wcs, w_native, scene_shape, channel, is_full_array=is_full_array,
    )


def host_core_gaussian_column(scene_wcs, ra_deg, dec_deg, sigma_pix, scene_shape):
    """Normalized circular Gaussian on the analysis stamp (sum=1), sky-fixed."""
    h, w = int(scene_shape[0]), int(scene_shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ox, oy = scene_wcs.world_to_pixel_values(ra_deg, dec_deg)
    sp = max(float(sigma_pix), 1e-3)
    r2 = (xx - ox) ** 2 + (yy - oy) ** 2
    g = np.exp(-0.5 * r2 / sp ** 2)
    s = float(np.sum(g))
    if s > 0:
        g /= s
    return g.ravel()


def _gp_profile_center_world():
    """Reference center for GP-profile radial constraints."""
    ra = getattr(config, 'GP_PROFILE_CENTER_RA', None)
    dec = getattr(config, 'GP_PROFILE_CENTER_DEC', None)
    if ra is not None and dec is not None:
        return float(ra), float(dec)
    ra = getattr(config, 'NUCLEAR_POINT_RA', None)
    dec = getattr(config, 'NUCLEAR_POINT_DEC', None)
    if ra is not None and dec is not None:
        return float(ra), float(dec)
    ra = getattr(config, 'HOST_CORE_RA', None)
    dec = getattr(config, 'HOST_CORE_DEC', None)
    if ra is not None and dec is not None:
        return float(ra), float(dec)
    ra = getattr(config, 'GALAXY_EXTENDED_CENTER_RA', None)
    dec = getattr(config, 'GALAXY_EXTENDED_CENTER_DEC', None)
    if ra is not None and dec is not None:
        return float(ra), float(dec)
    return float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)


def _build_central_annulus_terms(scene_wcs, scene_shape):
    """
    Build adjacent-annulus mean-difference terms around the GP profile center.
    Each term encodes v^T x = mean(outer) - mean(inner).
    """
    radii = tuple(getattr(config, 'GP_CENTRAL_MONOTONIC_RADII_PX', (0.0, 1.5, 3.0, 4.5, 6.0)))
    if len(radii) < 3:
        return []
    edges = np.asarray(radii, dtype=float)
    if not np.all(np.diff(edges) > 0):
        return []
    h, w = int(scene_shape[0]), int(scene_shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ra_c, dec_c = _gp_profile_center_world()
    cx, cy = scene_wcs.world_to_pixel_values(ra_c, dec_c)
    rr = np.hypot(xx - float(cx), yy - float(cy))
    ring_indices = []
    for r0, r1 in zip(edges[:-1], edges[1:]):
        m = (rr >= float(r0)) & (rr < float(r1))
        idx = np.where(m.ravel())[0]
        if len(idx) < 4:
            ring_indices.append(None)
        else:
            ring_indices.append(idx)
    terms = []
    for i in range(len(ring_indices) - 1):
        idx_in = ring_indices[i]
        idx_out = ring_indices[i + 1]
        if idx_in is None or idx_out is None:
            continue
        w_in = np.full(len(idx_in), 1.0 / len(idx_in), dtype=np.float64)
        w_out = np.full(len(idx_out), 1.0 / len(idx_out), dtype=np.float64)
        terms.append((idx_in, w_in, idx_out, w_out))
    return terms


def _solve_map_bounds(H, rhs, lb, ub, x0=None):
    """Minimize 0.5 θᵀHθ − rhsᵀθ subject to lb ≤ θ ≤ ub (H symmetric SPD)."""
    H = np.asarray(H, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64).ravel()
    n = H.shape[0]
    H = 0.5 * (H + H.T)
    if x0 is None:
        try:
            x0 = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            x0 = lstsq(H, rhs, check_finite=False, lapack_driver='gelsy')[0]
    x0 = np.clip(x0, lb, ub)

    def fun(theta):
        return 0.5 * float(theta @ H @ theta) - float(rhs @ theta)

    def jac(theta):
        return H @ theta - rhs

    bnds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(
        fun,
        x0,
        method='L-BFGS-B',
        jac=jac,
        bounds=bnds,
        options={'maxiter': 400, 'ftol': 1e-12},
    )
    if not res.success:
        _log.warning("Bounded MAP (L-BFGS-B) did not converge: %s; falling back to trust-constr.", res.message)
        res = minimize(
            fun,
            x0,
            method='trust-constr',
            jac=jac,
            hess=lambda _th: H,
            bounds=Bounds(lb, ub),
            options={'verbose': 0, 'maxiter': 300},
        )
        if not res.success:
            _log.warning("Bounded MAP (trust-constr) did not converge: %s", res.message)
    return res.x


def run_gls_solve(cutouts, stars, star_initial_fluxes, gp_params, regularization, deep_template, template_wcs, n_epochs):
    """
    Joint MAP fit: GP prior on static scene, one transient flux per science BCD,
    star fluxes (shared across all BCDs), per-BCD backgrounds.
    Template BCDs have no transient term.

    gp_params: optional dict with keys 'ell', 'var' overriding regularization length-scale and variance.
        Optional keys 'ell2' and 'var2' enable a second GP scale via K = K1 + K2.
    deep_template: optional 2D array; if shape matches the scene stamp, used only for consistency checks
        (future: weak prior toward template). template_wcs reserved for the same.
    """
    global DEBUG_DUMPED
    DEBUG_DUMPED = False
    
    if not cutouts: return None
    scene_shape = (cutouts[0]['data'].shape[0], cutouts[0]['data'].shape[1])
    if deep_template is not None and np.ndim(deep_template) == 2:
        scene_shape = (int(deep_template.shape[0]), int(deep_template.shape[1]))
    n_scene = scene_shape[0]*scene_shape[1]
    
    results = {
        'transient_fluxes': np.zeros(len(cutouts)),
        'transient_errs': np.zeros(len(cutouts)),
        'transient_epoch_fluxes': np.zeros(0),
        'transient_epoch_errs': np.zeros(0),
        'science_epoch_ids': np.zeros(0, dtype=int),
        'transient_epoch_index_by_id': {},
        'transient_bg_cov_by_epoch_id': {},
        'star_fluxes': np.zeros(len(stars)),
        'star_errs': np.zeros(len(stars)),
        'epoch_backgrounds': np.zeros(n_epochs),
        'model_scene': np.zeros(scene_shape),
        'scene_wcs': None,
        'scene_shape': scene_shape,
        'transient_dra_deg': 0.0,
        'transient_ddec_deg': 0.0,
        'transient_dra_err_deg': 0.0,
        'transient_ddec_err_deg': 0.0,
        'host_core_flux': 0.0,
        'host_core_err': 0.0,
        'nuclear_point_flux': 0.0,
        'nuclear_point_err': 0.0,
        'bcd_backgrounds': np.zeros(len(cutouts)),
        'gp_prior_params': {},
    }

    try:
        print("   [Solver] Step 1: Geometry Setup...")
        sys.stdout.flush()
        
        ell, var = regularization
        ell2 = None
        var2 = None
        if isinstance(gp_params, dict):
            if gp_params.get('ell') is not None:
                ell = float(gp_params['ell'])
            if gp_params.get('var') is not None:
                var = float(gp_params['var'])
            if gp_params.get('ell2') is not None and gp_params.get('var2') is not None:
                ell2 = float(gp_params['ell2'])
                var2 = float(gp_params['var2'])
        results['gp_prior_params'] = {
            'ell': float(ell),
            'var': float(var),
            'ell2': None if ell2 is None else float(ell2),
            'var2': None if var2 is None else float(var2),
            'matern_order': gp_model.normalize_matern_order(getattr(config, 'GP_MATERN_ORDER', 'matern32')),
        }
        cut_sig = tuple((str(c.get('filename', '')), int(bool(c.get('is_template')))) for c in cutouts)
        tw = template_wcs
        tw_sig = None
        if tw is not None:
            tw_sig = (
                tuple(np.asarray(tw.wcs.crpix, dtype=float).tolist()),
                tuple(np.asarray(tw.wcs.crval, dtype=float).tolist()),
                tuple(np.asarray(tw.wcs.cdelt, dtype=float).tolist()),
            )
        cache_key = (
            scene_shape,
            cut_sig,
            tw_sig,
            float(getattr(config, 'NATIVE_SCENE_SUPPORT_THRESHOLD', 0.85)),
            int(getattr(config, 'SUPERSAMPLE_FACTOR', 1)),
            bool(getattr(config, 'USE_HOST_GAUSSIAN_CORE', False)),
            float(getattr(config, 'HOST_GAUSSIAN_MIN_OFFSET_PX', 1.0)),
            tuple(getattr(config, 'HOST_GAUSSIAN_SIGMA_PX_LIST', ()) or ()),
            getattr(config, 'HOST_CORE_RA', None),
            getattr(config, 'HOST_CORE_DEC', None),
            bool(getattr(config, 'USE_NUCLEAR_POINT_SOURCE', False)),
            getattr(config, 'NUCLEAR_POINT_RA', None),
            getattr(config, 'NUCLEAR_POINT_DEC', None),
        )
        g = _GEOMETRY_CACHE.get(cache_key)
        cache_hit = g is not None
        if g is None:
            target_loc = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')
            if template_wcs is not None:
                scene_wcs = template_wcs.deepcopy()
            else:
                scene_wcs = WCS(naxis=2)
                scene_wcs.wcs.crpix = [scene_shape[1]/2, scene_shape[0]/2]
                scene_wcs.wcs.crval = [target_loc.ra.deg, target_loc.dec.deg]
                scene_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                scale = config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR / 3600.0
                scene_wcs.wcs.cdelt = [-scale, scale]
                scene_wcs.wcs.pc = np.eye(2)
            if deep_template is not None and np.ndim(deep_template) == 2 and deep_template.shape != scene_shape:
                _log.warning(
                    "deep_template shape %s != scene_shape %s; ignoring for prior anchor.",
                    deep_template.shape,
                    scene_shape,
                )
            monotonic_terms = _build_central_annulus_terms(scene_wcs, scene_shape)
            science_epoch_ids = sorted({int(c['epoch_id']) for c in cutouts if not c.get('is_template')})
            science_frame_indices = [i for i, c in enumerate(cutouts) if not c.get('is_template')]
            n_sci_ep = len(science_frame_indices)
            sci_frame_to_idx = {int(i): k for k, i in enumerate(science_frame_indices)}
            sci_ep_to_idx = {eid: k for k, eid in enumerate(science_epoch_ids)}
            scene_data_support = np.zeros(scene_shape, dtype=bool)
            base_supp_thr = float(getattr(config, 'NATIVE_SCENE_SUPPORT_THRESHOLD', 0.85))
            # Native-valid support is projected onto an SR scene grid; at higher SR the
            # support weight is distributed across ~SR^2 scene pixels. Scale threshold
            # accordingly so support locking is SR-invariant.
            sr = float(max(1, int(getattr(config, 'SUPERSAMPLE_FACTOR', 1))))
            supp_thr = float(np.clip(base_supp_thr / (sr * sr), 1e-6, 1.0))
            for entry in cutouts:
                data2 = np.asarray(entry['data'], dtype=np.float64)
                sigma2 = np.asarray(entry['sigma'], dtype=np.float64)
                native_valid = (data2 != 0) & np.isfinite(sigma2) & (sigma2 < 1e20)
                if not np.any(native_valid):
                    continue
                w_bcd = entry['raw_wcs']
                support_scene = _project_native_to_scene(
                    native_valid.astype(np.float64),
                    w_bcd,
                    scene_wcs,
                    scene_shape,
                )
                scene_data_support |= np.isfinite(support_scene) & (support_scene > supp_thr)
            scene_lock_mask = ~scene_data_support
            scene_lock_idx = np.where(scene_lock_mask.ravel())[0]
            n_host = 0
            host_scene_cols: List[np.ndarray] = []
            host_gaussian_sigmas_px: Tuple[float, ...] = ()
            if getattr(config, 'USE_HOST_GAUSSIAN_CORE', False):
                ra_h = getattr(config, 'HOST_CORE_RA', None)
                dec_h = getattr(config, 'HOST_CORE_DEC', None)
                if ra_h is not None and dec_h is not None:
                    ra_gpc, dec_gpc = _gp_profile_center_world()
                    x_h, y_h = scene_wcs.world_to_pixel_values(float(ra_h), float(dec_h))
                    x_gp, y_gp = scene_wcs.world_to_pixel_values(float(ra_gpc), float(dec_gpc))
                    off_pix = float(np.hypot(float(x_h) - float(x_gp), float(y_h) - float(y_gp)))
                    min_off = float(max(0.0, getattr(config, 'HOST_GAUSSIAN_MIN_OFFSET_PX', 1.0)))
                    if off_pix >= min_off:
                        sig_list = getattr(config, "HOST_GAUSSIAN_SIGMA_PX_LIST", None)
                        if sig_list is None or (isinstance(sig_list, (list, tuple)) and len(sig_list) == 0):
                            sig_list = (float(getattr(config, "HOST_CORE_SIGMA_PX", 1.5)),)
                        else:
                            sig_list = tuple(float(s) for s in sig_list)
                        host_gaussian_sigmas_px = sig_list
                        n_host = len(sig_list)
                        host_scene_cols = [
                            host_core_gaussian_column(
                                scene_wcs,
                                float(ra_h),
                                float(dec_h),
                                float(sp),
                                scene_shape,
                            )
                            for sp in sig_list
                        ]
            n_nps = 0
            col_nps_vec = None
            if getattr(config, 'USE_NUCLEAR_POINT_SOURCE', False):
                ra_np = getattr(config, 'NUCLEAR_POINT_RA', None)
                dec_np = getattr(config, 'NUCLEAR_POINT_DEC', None)
                if ra_np is None or dec_np is None:
                    ra_np = getattr(config, 'HOST_CORE_RA', None)
                    dec_np = getattr(config, 'HOST_CORE_DEC', None)
                if ra_np is not None and dec_np is not None:
                    n_nps = 1
                    col_nps_vec = (float(ra_np), float(dec_np))
            g = {
                'scene_wcs': scene_wcs,
                'monotonic_terms': monotonic_terms,
                'science_epoch_ids': science_epoch_ids,
                'science_frame_indices': science_frame_indices,
                'sci_frame_to_idx': sci_frame_to_idx,
                'sci_ep_to_idx': sci_ep_to_idx,
                'scene_data_support': scene_data_support,
                'scene_lock_idx': scene_lock_idx,
                'n_host': n_host,
                'host_scene_cols': host_scene_cols,
                'host_gaussian_sigmas_px': host_gaussian_sigmas_px,
                'n_nps': n_nps,
                'col_nps_vec': col_nps_vec,
            }
            _GEOMETRY_CACHE[cache_key] = g
        if cache_hit:
            print("   [Solver] Step 1a: Geometry cache HIT (reused precomputed geometry)")
        else:
            print("   [Solver] Step 1a: Geometry cache MISS (built geometry)")
        scene_wcs = g['scene_wcs']
        monotonic_terms = g['monotonic_terms']
        science_epoch_ids = g['science_epoch_ids']
        science_frame_indices = g['science_frame_indices']
        n_sci_ep = len(science_frame_indices)
        sci_frame_to_idx = g['sci_frame_to_idx']
        sci_ep_to_idx = g['sci_ep_to_idx']
        scene_data_support = g['scene_data_support']
        scene_lock_idx = g['scene_lock_idx']
        n_host = g['n_host']
        host_scene_cols = g['host_scene_cols']
        host_gaussian_sigmas_px = g['host_gaussian_sigmas_px']
        n_nps = g['n_nps']
        col_nps_vec = g['col_nps_vec']
        results['scene_wcs'] = scene_wcs
        results['science_epoch_ids'] = np.asarray(science_epoch_ids, dtype=int)
        results['transient_epoch_index_by_id'] = dict(sci_ep_to_idx)
        use_scene_gp_prior = bool(getattr(config, 'USE_SCENE_GP_PRIOR', True))
        if use_scene_gp_prior:
            use_two_scale_components = (ell2 is not None) and (var2 is not None)
            Q1_inv = gp_model.build_scene_prior_inverse(n_scene, ell, var, scene_shape)
            Q2_inv = gp_model.build_scene_prior_inverse(n_scene, float(ell2), float(var2), scene_shape) if use_two_scale_components else None
        else:
            use_two_scale_components = False
            ridge = float(max(0.0, getattr(config, 'SCENE_INDEPENDENT_RIDGE', 1e-12)))
            Q1_inv = np.eye(n_scene, dtype=np.float64) * ridge
            Q2_inv = None
            ell2 = None
            var2 = None
        results['gp_prior_params']['enabled'] = bool(use_scene_gp_prior)
        n_scene_blocks = 2 if use_two_scale_components else 1
        n_scene_total = n_scene * n_scene_blocks
        solver_stars = list(stars)
        solver_init = list(star_initial_fluxes)
        n_stars = len(solver_stars)
        n_bg = len(cutouts)  # sky term per BCD
        use_nonneg = bool(getattr(config, 'TRANSIENT_NONNEGATIVE', True)) and n_sci_ep > 0
        float_pos = bool(getattr(config, 'FLOAT_TRANSIENT_POSITION', False)) and n_sci_ep > 0
        eps_deg = float(getattr(config, 'TRANSIENT_POS_FD_STEP_ARCSEC', 0.05)) / 3600.0
        pos_ridge = float(getattr(config, 'TRANSIENT_POS_RIDGE', 1e12))

        def build_system(include_offset, f0_epoch):
            """
            f0_epoch: None, or length n_sci_ep vector of fluxes for position linearization scale.
            """
            if include_offset:
                idx_trans = n_scene_total
                idx_off = n_scene_total + n_sci_ep
                idx_stars = idx_off + 2
            else:
                idx_trans = n_scene_total
                idx_off = None
                idx_stars = idx_trans + n_sci_ep
            idx_star_end = idx_stars + n_stars
            idx_nps = idx_star_end + n_host
            idx_bg = idx_nps + n_nps
            n_params = idx_bg + n_bg
            # TODO: For large `n_params` (e.g. SR runs with big scene footprints),
            # replace this dense normal-equation build with a sparse/block solve
            # to avoid O(n_params^2) memory and O(n_params^3) runtime.
            Hloc = np.zeros((n_params, n_params))
            rhsloc = np.zeros(n_params)
            Hloc[:n_scene, :n_scene] = Q1_inv
            if use_two_scale_components and Q2_inv is not None:
                Hloc[n_scene:2 * n_scene, n_scene:2 * n_scene] = Q2_inv

            for i, entry in enumerate(cutouts):
                data2 = np.asarray(entry['data'], dtype=np.float64)
                sigma2 = np.asarray(entry['sigma'], dtype=np.float64)
                native_shape = data2.shape
                d = data2.ravel()
                s = sigma2.ravel()
                mask = (entry['data'] != 0) & np.isfinite(entry['sigma'])
                if np.sum(mask) == 0:
                    continue

                w_data = np.zeros_like(d, dtype=np.float64)
                w_data[mask.flatten()] = 1.0 / (np.clip(s[mask.flatten()], 1e-9, None)**2)

                chan = 'ch2' if 'ch2' in entry['filename'] else 'ch1'
                w_bcd = entry['raw_wcs']
                is_full = entry.get('is_full_array', False)
                kernels, weights, wsum_b = _get_prf_operator_bundle(
                    scene_wcs, w_bcd, scene_shape, chan, is_full,
                )
                # Build a chi^2-valid native mask from scene pixels that are actually
                # constrained by BCD data, then trim edges by PRF support.
                cov_native = _project_scene_to_native(
                    scene_data_support.astype(np.float64),
                    scene_wcs,
                    w_bcd,
                    native_shape,
                )
                mask_cov = np.isfinite(cov_native) & (cov_native > 0.999)
                trim_px = min(_estimate_prf_support_radius_px(kernels, frac=0.999), 8)
                # Potential follow-up knob: optionally exclude additional edge pixels beyond
                # PRF-support trimming to suppress scene-border leakage into native data.
                extra_trim = int(max(0, getattr(config, 'PRF_CHI2_EXTRA_EDGE_EXCLUSION_PX', 0)))
                trim_px += extra_trim
                if trim_px > 0:
                    mask_cov = binary_erosion(mask_cov, iterations=trim_px, border_value=0)
                # NOTE: keep fit-time weights tied to the base valid-data mask.
                # `mask_cov` is a strict support/edge trim and can be too aggressive
                # as a hard objective gate, leading to unstable solves.
                # We still compute it here for diagnostics/future soft-gating use.

                h_s, w_s = scene_shape
                ltwl_full_cap = int(getattr(config, 'PRF_GLS_LTWL_FULL_MAX_PIXELS', 0))
                ltwl_diag_cap = int(getattr(config, 'PRF_GLS_LTWL_DIAG_MAX_PIXELS', 0))
                if ltwl_full_cap > 0 and n_scene <= ltwl_full_cap:
                    # Full dense scene-block coupling: H_scene += A^T W A
                    # where A maps scene pixels -> native data for this frame.
                    Acols = np.zeros((w_data.size, n_scene), dtype=np.float64)
                    for jj in range(n_scene):
                        ei = np.zeros((h_s, w_s), dtype=np.float64)
                        ei.ravel()[jj] = 1.0
                        fwd_native = _apply_frame_forward_operator(
                            ei, scene_wcs, w_bcd, scene_shape, native_shape, chan, is_full_array=is_full,
                        ).ravel()
                        Acols[:, jj] = fwd_native
                    H_scene = Acols.T @ (Acols * w_data[:, None])
                    Hloc[:n_scene, :n_scene] += H_scene
                    if use_two_scale_components:
                        Hloc[n_scene:2 * n_scene, n_scene:2 * n_scene] += H_scene
                        Hloc[:n_scene, n_scene:2 * n_scene] += H_scene
                        Hloc[n_scene:2 * n_scene, :n_scene] += H_scene
                elif ltwl_diag_cap > 0 and n_scene <= ltwl_diag_cap:
                    for jj in range(n_scene):
                        ei = np.zeros((h_s, w_s), dtype=np.float64)
                        ei.ravel()[jj] = 1.0
                        fwd_native = _apply_frame_forward_operator(
                            ei, scene_wcs, w_bcd, scene_shape, native_shape, chan, is_full_array=is_full,
                        ).ravel()
                        Hloc[jj, jj] += np.dot(fwd_native, fwd_native * w_data)
                else:
                    w_scene_diag = _project_native_to_scene(
                        w_data.reshape(native_shape), w_bcd, scene_wcs, scene_shape,
                    ).ravel()
                    np.fill_diagonal(
                        Hloc[:n_scene, :n_scene],
                        np.diag(Hloc[:n_scene, :n_scene]) + w_scene_diag,
                    )
                    if use_two_scale_components:
                        np.fill_diagonal(
                            Hloc[n_scene:2 * n_scene, n_scene:2 * n_scene],
                            np.diag(Hloc[n_scene:2 * n_scene, n_scene:2 * n_scene]) + w_scene_diag,
                        )
                        np.fill_diagonal(
                            Hloc[:n_scene, n_scene:2 * n_scene],
                            np.diag(Hloc[:n_scene, n_scene:2 * n_scene]) + w_scene_diag,
                        )
                        np.fill_diagonal(
                            Hloc[n_scene:2 * n_scene, :n_scene],
                            np.diag(Hloc[n_scene:2 * n_scene, :n_scene]) + w_scene_diag,
                        )
                lt_data = _apply_frame_adjoint_operator(
                    (w_data * d).reshape(native_shape),
                    scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                ).ravel()
                rhsloc[:n_scene] += lt_data
                if use_two_scale_components:
                    rhsloc[n_scene:2 * n_scene] += lt_data

                is_tpl = bool(entry.get('is_template'))
                it = None
                col_t = None
                b_hosts: List[np.ndarray] = []
                if not is_tpl and n_sci_ep > 0:
                    col_t = column_L_pointsource(
                        scene_wcs, w_bcd, scene_shape, chan,
                        config.TRANSIENT_RA, config.TRANSIENT_DEC,
                        is_full_array=is_full,
                    )
                    col_t_native = _project_scene_to_native(
                        col_t.reshape(scene_shape), scene_wcs, w_bcd, native_shape,
                    ).ravel()
                    ie = sci_frame_to_idx[int(i)]
                    it = idx_trans + ie
                    Hloc[it, it] += np.dot(col_t_native, col_t_native * w_data)
                    rhsloc[it] += np.dot(col_t_native, w_data * d)
                    lt_t = _apply_frame_adjoint_operator(
                        (col_t_native * w_data).reshape(native_shape),
                        scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                    ).ravel()
                    Hloc[:n_scene, it] += lt_t
                    Hloc[it, :n_scene] += lt_t
                    if use_two_scale_components:
                        Hloc[n_scene:2 * n_scene, it] += lt_t
                        Hloc[it, n_scene:2 * n_scene] += lt_t

                cols_s = []
                cols_s_native = []
                for s_obj in solver_stars:
                    col_s = column_L_pointsource(
                        scene_wcs, w_bcd, scene_shape, chan,
                        s_obj.ra.deg, s_obj.dec.deg,
                        is_full_array=is_full,
                    )
                    cols_s.append(col_s)
                    cols_s_native.append(
                        _project_scene_to_native(col_s.reshape(scene_shape), scene_wcs, w_bcd, native_shape).ravel()
                    )

                if cols_s_native:
                    S = np.column_stack(cols_s_native)
                    Hloc[idx_stars:idx_star_end, idx_stars:idx_star_end] += S.T @ (S * w_data[:, None])
                    rhsloc[idx_stars:idx_star_end] += S.T @ (w_data * d)
                    for js in range(S.shape[1]):
                        wb = (w_data * S[:, js]).reshape(native_shape)
                        lj = _apply_frame_adjoint_operator(
                            wb, scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                        ).ravel()
                        Hloc[:n_scene, idx_stars + js] += lj
                        Hloc[idx_stars + js, :n_scene] += lj
                        if use_two_scale_components:
                            Hloc[n_scene:2 * n_scene, idx_stars + js] += lj
                            Hloc[idx_stars + js, n_scene:2 * n_scene] += lj
                    if it is not None:
                        Hloc[it, idx_stars:idx_star_end] += col_t_native @ (S * w_data[:, None])
                        Hloc[idx_stars:idx_star_end, it] += (S * w_data[:, None]).T @ col_t_native

                ib = idx_bg + int(i)
                Hloc[ib, ib] += np.sum(w_data)
                rhsloc[ib] += np.dot(w_data, d)
                lt_bg = _apply_frame_adjoint_operator(
                    w_data.reshape(native_shape),
                    scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                ).ravel()
                Hloc[:n_scene, ib] += lt_bg
                Hloc[ib, :n_scene] += lt_bg
                if use_two_scale_components:
                    Hloc[n_scene:2 * n_scene, ib] += lt_bg
                    Hloc[ib, n_scene:2 * n_scene] += lt_bg

                if cols_s_native:
                    Hloc[idx_stars:idx_star_end, ib] += np.sum(S * w_data[:, None], axis=0)
                    Hloc[ib, idx_stars:idx_star_end] += np.sum(S * w_data[:, None], axis=0)
                if it is not None:
                    Hloc[it, ib] += np.sum(col_t_native * w_data)
                    Hloc[ib, it] += np.sum(col_t_native * w_data)

                if n_host > 0 and len(host_scene_cols) == n_host:
                    b_hosts = []
                    for ch in host_scene_cols:
                        b_host_scene = _apply_prf_operator_from_bundle(
                            ch.reshape(scene_shape), kernels, weights, wsum_b,
                        )
                        b_hosts.append(
                            _project_scene_to_native(
                                b_host_scene, scene_wcs, w_bcd, native_shape,
                            ).ravel(),
                        )
                    for jh, b_host in enumerate(b_hosts):
                        ih = idx_star_end + jh
                        Hloc[ih, ih] += np.dot(b_host, b_host * w_data)
                        rhsloc[ih] += np.dot(b_host, w_data * d)
                        lt_h = _apply_frame_adjoint_operator(
                            (b_host * w_data).reshape(native_shape),
                            scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                        ).ravel()
                        Hloc[:n_scene, ih] += lt_h
                        Hloc[ih, :n_scene] += lt_h
                        if use_two_scale_components:
                            Hloc[n_scene:2 * n_scene, ih] += lt_h
                            Hloc[ih, n_scene:2 * n_scene] += lt_h
                        if cols_s_native:
                            vh = b_host @ (S * w_data[:, None])
                            Hloc[ih, idx_stars:idx_star_end] += vh
                            Hloc[idx_stars:idx_star_end, ih] += vh
                        if it is not None:
                            Hloc[it, ih] += np.dot(col_t_native, b_host * w_data)
                            Hloc[ih, it] = Hloc[it, ih]
                        Hloc[ih, ib] += np.sum(b_host * w_data)
                        Hloc[ib, ih] = Hloc[ih, ib]
                    for jh in range(n_host):
                        for kh in range(jh + 1, n_host):
                            bh_j, bh_k = b_hosts[jh], b_hosts[kh]
                            ih_j, ih_k = idx_star_end + jh, idx_star_end + kh
                            cjk = float(np.dot(bh_j, bh_k * w_data))
                            Hloc[ih_j, ih_k] += cjk
                            Hloc[ih_k, ih_j] += cjk

                if n_nps > 0 and col_nps_vec is not None:
                    inp = idx_nps
                    ra_np, dec_np = col_nps_vec
                    col_np = column_L_pointsource(
                        scene_wcs, w_bcd, scene_shape, chan,
                        ra_np, dec_np,
                        is_full_array=is_full,
                    )
                    col_np_native = _project_scene_to_native(
                        col_np.reshape(scene_shape), scene_wcs, w_bcd, native_shape,
                    ).ravel()
                    Hloc[inp, inp] += np.dot(col_np_native, col_np_native * w_data)
                    rhsloc[inp] += np.dot(col_np_native, w_data * d)
                    lt_np = _apply_frame_adjoint_operator(
                        (col_np_native * w_data).reshape(native_shape),
                        scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                    ).ravel()
                    Hloc[:n_scene, inp] += lt_np
                    Hloc[inp, :n_scene] += lt_np
                    if use_two_scale_components:
                        Hloc[n_scene:2 * n_scene, inp] += lt_np
                        Hloc[inp, n_scene:2 * n_scene] += lt_np
                    if cols_s_native:
                        vnp = col_np_native @ (S * w_data[:, None])
                        Hloc[inp, idx_stars:idx_star_end] += vnp
                        Hloc[idx_stars:idx_star_end, inp] += vnp
                    if it is not None:
                        vtnp = np.dot(col_t_native, col_np_native * w_data)
                        Hloc[it, inp] += vtnp
                        Hloc[inp, it] += vtnp
                    if n_host > 0 and len(b_hosts) == n_host:
                        for jh, b_host in enumerate(b_hosts):
                            ih = idx_star_end + jh
                            vhnp = np.dot(b_host, col_np_native * w_data)
                            Hloc[ih, inp] += vhnp
                            Hloc[inp, ih] += vhnp
                    Hloc[inp, ib] += np.sum(col_np_native * w_data)
                    Hloc[ib, inp] += np.sum(col_np_native * w_data)

                if (
                    include_offset
                    and idx_off is not None
                    and (not is_tpl)
                    and n_sci_ep > 0
                    and f0_epoch is not None
                    and col_t is not None
                ):
                    ie = sci_frame_to_idx[int(i)]
                    scale = float(f0_epoch[ie])
                    d_ra_c, d_dec_c = _transient_prf_pos_derivatives(
                        scene_wcs, w_bcd,
                        config.TRANSIENT_RA, config.TRANSIENT_DEC, scene_shape,
                        chan, is_full, eps_deg,
                    )
                    coldra = scale * d_ra_c
                    coldec = scale * d_dec_c
                    coldra_native = _project_scene_to_native(
                        coldra.reshape(scene_shape), scene_wcs, w_bcd, native_shape,
                    ).ravel()
                    coldec_native = _project_scene_to_native(
                        coldec.reshape(scene_shape), scene_wcs, w_bcd, native_shape,
                    ).ravel()
                    ia, ibp = idx_off, idx_off + 1
                    Hloc[ia, ia] += np.dot(coldra_native, coldra_native * w_data)
                    Hloc[ibp, ibp] += np.dot(coldec_native, coldec_native * w_data)
                    c_off = np.dot(coldra_native, coldec_native * w_data)
                    Hloc[ia, ibp] += c_off
                    Hloc[ibp, ia] += c_off
                    rhsloc[ia] += np.dot(coldra_native, w_data * d)
                    rhsloc[ibp] += np.dot(coldec_native, w_data * d)
                    lt_ra = _apply_frame_adjoint_operator(
                        (coldra_native * w_data).reshape(native_shape),
                        scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                    ).ravel()
                    Hloc[:n_scene, ia] += lt_ra
                    Hloc[ia, :n_scene] += lt_ra
                    if use_two_scale_components:
                        Hloc[n_scene:2 * n_scene, ia] += lt_ra
                        Hloc[ia, n_scene:2 * n_scene] += lt_ra
                    lt_dec = _apply_frame_adjoint_operator(
                        (coldec_native * w_data).reshape(native_shape),
                        scene_wcs, w_bcd, scene_shape, chan, is_full_array=is_full,
                    ).ravel()
                    Hloc[:n_scene, ibp] += lt_dec
                    Hloc[ibp, :n_scene] += lt_dec
                    if use_two_scale_components:
                        Hloc[n_scene:2 * n_scene, ibp] += lt_dec
                        Hloc[ibp, n_scene:2 * n_scene] += lt_dec
                    Hloc[it, ia] += np.dot(col_t_native, coldra_native * w_data)
                    Hloc[ia, it] = Hloc[it, ia]
                    Hloc[it, ibp] += np.dot(col_t_native, coldec_native * w_data)
                    Hloc[ibp, it] = Hloc[it, ibp]
                    if cols_s_native:
                        v_ra = coldra_native @ (S * w_data[:, None])
                        v_dec = coldec_native @ (S * w_data[:, None])
                        Hloc[ia, idx_stars:idx_star_end] += v_ra
                        Hloc[idx_stars:idx_star_end, ia] += v_ra
                        Hloc[ibp, idx_stars:idx_star_end] += v_dec
                        Hloc[idx_stars:idx_star_end, ibp] += v_dec
                    Hloc[ia, ib] += np.sum(coldra_native * w_data)
                    Hloc[ib, ia] = Hloc[ia, ib]
                    Hloc[ibp, ib] += np.sum(coldec_native * w_data)
                    Hloc[ib, ibp] = Hloc[ibp, ib]
                    if n_host > 0 and len(b_hosts) == n_host:
                        for jh, b_host in enumerate(b_hosts):
                            ih = idx_star_end + jh
                            vhr = np.dot(coldra_native, b_host * w_data)
                            Hloc[ia, ih] += vhr
                            Hloc[ih, ia] += vhr
                            vhd = np.dot(coldec_native, b_host * w_data)
                            Hloc[ibp, ih] += vhd
                            Hloc[ih, ibp] += vhd

                    if n_nps > 0 and col_nps_vec is not None:
                        inp = idx_nps
                        ra_np, dec_np = col_nps_vec
                        col_np_off = column_L_pointsource(
                            scene_wcs, w_bcd, scene_shape, chan,
                            ra_np, dec_np,
                            is_full_array=is_full,
                        )
                        col_np_off_native = _project_scene_to_native(
                            col_np_off.reshape(scene_shape), scene_wcs, w_bcd, native_shape,
                        ).ravel()
                        vnr = np.dot(coldra_native, col_np_off_native * w_data)
                        vnd = np.dot(coldec_native, col_np_off_native * w_data)
                        Hloc[ia, inp] += vnr
                        Hloc[inp, ia] += vnr
                        Hloc[ibp, inp] += vnd
                        Hloc[inp, ibp] += vnd

            if n_stars > 0:
                star_diag = np.diag(Hloc[idx_stars:idx_star_end, idx_stars:idx_star_end])
                avg_w = np.median(star_diag)
                lam = max((avg_w / len(cutouts)) * 5.0, 1.0)
                for k, f_init in enumerate(solver_init):
                    im = idx_stars + k
                    Hloc[im, im] += lam
                    rhsloc[im] += lam * f_init

            if include_offset and idx_off is not None:
                Hloc[idx_off, idx_off] += pos_ridge
                Hloc[idx_off + 1, idx_off + 1] += pos_ridge

            if bool(getattr(config, 'ENFORCE_GP_CENTRAL_MONOTONICITY', True)) and monotonic_terms:
                d_scene = np.diag(Hloc[:n_scene, :n_scene])
                d_scene = d_scene[np.isfinite(d_scene) & (d_scene > 0)]
                if len(d_scene) > 0:
                    lam = float(np.median(d_scene)) * float(
                        max(0.0, getattr(config, 'GP_CENTRAL_MONOTONIC_STRENGTH_FRAC', 0.03))
                    )
                    tol = float(getattr(config, 'GP_CENTRAL_MONOTONIC_ALLOWED_DROP_JY', 0.0))
                    if lam > 0.0:
                        for idx_in, w_in, idx_out, w_out in monotonic_terms:
                            Hloc[np.ix_(idx_out, idx_out)] += lam * np.outer(w_out, w_out)
                            Hloc[np.ix_(idx_in, idx_in)] += lam * np.outer(w_in, w_in)
                            cblk = lam * np.outer(w_out, w_in)
                            Hloc[np.ix_(idx_out, idx_in)] -= cblk
                            Hloc[np.ix_(idx_in, idx_out)] -= cblk.T
                            rhsloc[idx_out] += lam * tol * w_out
                            rhsloc[idx_in] -= lam * tol * w_in

            if len(scene_lock_idx) > 0:
                # Hard lock unconstrained/buffer scene pixels to zero intrinsic flux.
                Hloc[scene_lock_idx, :] = 0.0
                Hloc[:, scene_lock_idx] = 0.0
                Hloc[scene_lock_idx, scene_lock_idx] = 1.0
                rhsloc[scene_lock_idx] = 0.0
                if use_two_scale_components:
                    idx2 = scene_lock_idx + n_scene
                    Hloc[idx2, :] = 0.0
                    Hloc[:, idx2] = 0.0
                    Hloc[idx2, idx2] = 1.0
                    rhsloc[idx2] = 0.0

            return Hloc, rhsloc, idx_trans, idx_off, idx_stars, idx_star_end, idx_bg, n_params

        def _solve_H(
            Hm,
            rhsm,
            idx_trans_l,
            idx_off_l,
            n_sci_ep_l,
            idx_host_start=-1,
            n_host_nonneg=0,
            idx_nps_l=-1,
        ):
            n_p = Hm.shape[0]
            lb = np.full(n_p, -np.inf, dtype=np.float64)
            ub = np.full(n_p, np.inf, dtype=np.float64)
            use_bounds = False
            if use_nonneg and n_sci_ep_l > 0:
                lb[idx_trans_l:idx_trans_l + n_sci_ep_l] = 0.0
                use_bounds = True
            if n_host_nonneg > 0 and idx_host_start >= 0 and getattr(config, 'HOST_CORE_NONNEGATIVE', True):
                for jh in range(int(n_host_nonneg)):
                    lb[idx_host_start + jh] = 0.0
                use_bounds = True
            if n_nps > 0 and idx_nps_l >= 0 and getattr(config, 'NUCLEAR_POINT_NONNEGATIVE', True):
                lb[idx_nps_l] = 0.0
                use_bounds = True
            if use_two_scale_components and getattr(config, 'GP_COMPONENTS_NONNEGATIVE', True):
                lb[:n_scene_total] = 0.0
                use_bounds = True
            if use_bounds:
                print("   [Solver] Bounded MAP (nonnegative constraints)...")
                sys.stdout.flush()
                try:
                    xu = np.linalg.solve(Hm, rhsm)
                except np.linalg.LinAlgError:
                    xu = lstsq(Hm, rhsm, check_finite=False, lapack_driver='gelsy')[0]
                # Large systems can stall in constrained optimizers; use projected unconstrained solve.
                if n_p > 2000:
                    return np.clip(xu, lb, ub)
                return _solve_map_bounds(Hm, rhsm, lb, ub, x0=xu)
            try:
                return np.linalg.solve(Hm, rhsm)
            except np.linalg.LinAlgError:
                return lstsq(Hm, rhsm, check_finite=False, lapack_driver='gelsy')[0]

        print("   [Solver] Step 2: Matrix fill (base pass)...")
        sys.stdout.flush()
        H, rhs, idx_trans, idx_off, idx_stars, idx_star_end, idx_bg, n_params = build_system(False, None)

        print("   [Solver] Step 3: Linear / bounded MAP solve (base)...")
        sys.stdout.flush()
        idx_host_start = idx_star_end if n_host > 0 else -1
        idx_nps_for_bounds = (idx_star_end + n_host) if n_nps > 0 else -1
        sol = _solve_H(H, rhs, idx_trans, idx_off, n_sci_ep, idx_host_start, n_host, idx_nps_for_bounds)

        f_epoch = sol[idx_trans:idx_trans + n_sci_ep].copy() if n_sci_ep > 0 else np.zeros(0)

        if float_pos and n_sci_ep > 0:
            print("   [Solver] Step 3b: Matrix fill + solve (transient position linearization)...")
            sys.stdout.flush()
            H, rhs, idx_trans, idx_off, idx_stars, idx_star_end, idx_bg, n_params = build_system(True, f_epoch)
            sol = _solve_H(H, rhs, idx_trans, idx_off, n_sci_ep, idx_host_start, n_host, idx_nps_for_bounds)
            results['transient_dra_deg'] = float(sol[idx_off])
            results['transient_ddec_deg'] = float(sol[idx_off + 1])
        else:
            idx_off = None

            if use_scene_gp_prior and (not use_two_scale_components) and bool(getattr(config, 'ENFORCE_GP_CENTRAL_MONOTONICITY', True)) and monotonic_terms:
                n_enforce = int(max(0, getattr(config, 'GP_CENTRAL_MONOTONIC_ENFORCEMENT_ITERS', 1)))
                tol = float(getattr(config, 'GP_CENTRAL_MONOTONIC_ALLOWED_DROP_JY', 0.0))
                boost = float(max(0.0, getattr(config, 'GP_CENTRAL_MONOTONIC_VIOLATION_BOOST', 8.0)))
                if n_enforce > 0 and boost > 0.0:
                    for _ in range(n_enforce):
                        x_scene = np.asarray(sol[:n_scene], dtype=np.float64)
                        d_scene = np.diag(H[:n_scene, :n_scene])
                        d_scene = d_scene[np.isfinite(d_scene) & (d_scene > 0)]
                        if len(d_scene) == 0:
                            break
                        lam = float(np.median(d_scene)) * float(
                            max(0.0, getattr(config, 'GP_CENTRAL_MONOTONIC_STRENGTH_FRAC', 0.03))
                        ) * boost
                        if lam <= 0:
                            break
                        n_viol = 0
                        for idx_in, w_in, idx_out, w_out in monotonic_terms:
                            m_in = float(np.dot(w_in, x_scene[idx_in]))
                            m_out = float(np.dot(w_out, x_scene[idx_out]))
                            if (m_out - m_in) <= tol:
                                continue
                            n_viol += 1
                            # One-sided hinge surrogate active only on violated annulus pairs.
                            H[np.ix_(idx_out, idx_out)] += lam * np.outer(w_out, w_out)
                            H[np.ix_(idx_in, idx_in)] += lam * np.outer(w_in, w_in)
                            cblk = lam * np.outer(w_out, w_in)
                            H[np.ix_(idx_out, idx_in)] -= cblk
                            H[np.ix_(idx_in, idx_out)] -= cblk.T
                            rhs[idx_out] += lam * tol * w_out
                            rhs[idx_in] -= lam * tol * w_in
                        if n_viol == 0:
                            break
                        sol = _solve_H(H, rhs, idx_trans, idx_off, n_sci_ep, idx_host_start, n_host, idx_nps_for_bounds)

        if use_two_scale_components:
            gp1 = sol[:n_scene].reshape(scene_shape)
            gp2 = sol[n_scene:2 * n_scene].reshape(scene_shape)
            results['gp_scene_component1'] = gp1.copy()
            results['gp_scene_component2'] = gp2.copy()
            results['model_scene'] = gp1 + gp2
        else:
            results['model_scene'] = sol[:n_scene].reshape(scene_shape)
        results['gp_scene'] = results['model_scene'].copy()

        if n_sci_ep > 0:
            results['transient_fluxes'] = np.zeros(len(cutouts))
            results['transient_errs'] = np.zeros(len(cutouts))
            ef_bcd = sol[idx_trans:idx_trans + n_sci_ep].copy()
            for i in science_frame_indices:
                ie = sci_frame_to_idx[int(i)]
                results['transient_fluxes'][i] = ef_bcd[ie]
            # Keep epoch summaries for compatibility
            ep_flux = []
            for eid in science_epoch_ids:
                vals = [results['transient_fluxes'][i] for i, c in enumerate(cutouts) if (not c.get('is_template')) and int(c['epoch_id']) == int(eid)]
                ep_flux.append(float(np.mean(vals)) if vals else 0.0)
            results['transient_epoch_fluxes'] = np.asarray(ep_flux, dtype=float)
        else:
            results['transient_epoch_fluxes'] = np.zeros(0)
            results['transient_fluxes'] = np.zeros(len(cutouts))
            results['transient_errs'] = np.zeros(len(cutouts))

        fitted_star_fluxes = sol[idx_stars:idx_star_end]
        results['star_fluxes'] = fitted_star_fluxes.copy()

        if n_host > 0:
            flux_h = np.asarray(sol[idx_star_end : idx_star_end + n_host], dtype=np.float64)
            results['host_core_flux'] = float(np.sum(flux_h))
            sig_arr = np.asarray(host_gaussian_sigmas_px, dtype=np.float64)
            if n_host > 1:
                results['host_core_fluxes'] = flux_h.copy()
                results['host_gaussian_sigmas_px'] = sig_arr.copy()
                wpos = float(np.sum(np.clip(flux_h, 0.0, None)))
                if wpos > 0.0 and len(sig_arr) == len(flux_h):
                    results['host_effective_sigma_px'] = float(
                        np.sum(np.clip(flux_h, 0.0, None) * sig_arr) / wpos,
                    )
                elif len(sig_arr) > 0:
                    results['host_effective_sigma_px'] = float(sig_arr[0])
            else:
                results['host_effective_sigma_px'] = float(sig_arr[0]) if len(sig_arr) > 0 else float(
                    getattr(config, "HOST_CORE_SIGMA_PX", 1.5),
                )
        if n_nps > 0:
            results['nuclear_point_flux'] = float(sol[idx_star_end + n_host])

        results['bcd_backgrounds'] = sol[idx_bg:idx_bg + n_bg].copy()
        ep_bg = np.zeros(n_epochs, dtype=float)
        for eid in range(n_epochs):
            vals = [results['bcd_backgrounds'][i] for i, c in enumerate(cutouts) if int(c['epoch_id']) == int(eid)]
            ep_bg[eid] = float(np.mean(vals)) if vals else 0.0
        results['epoch_backgrounds'] = ep_bg

        try:
            cov = np.linalg.inv(H)
            sig = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            if n_sci_ep > 0:
                e_err = sig[idx_trans:idx_trans + n_sci_ep].copy()
                for i in science_frame_indices:
                    ie = sci_frame_to_idx[int(i)]
                    results['transient_errs'][i] = e_err[ie]
                ep_err = []
                for eid in science_epoch_ids:
                    vals = [results['transient_errs'][i] for i, c in enumerate(cutouts) if (not c.get('is_template')) and int(c['epoch_id']) == int(eid)]
                    ep_err.append(float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0)
                results['transient_epoch_errs'] = np.asarray(ep_err, dtype=float)
            else:
                results['transient_epoch_errs'] = np.zeros(0)
            if idx_off is not None:
                results['transient_dra_err_deg'] = float(sig[idx_off])
                results['transient_ddec_err_deg'] = float(sig[idx_off + 1])
            results['star_errs'] = sig[idx_stars:idx_star_end].copy()
            if n_host > 0:
                se_h = np.asarray(sig[idx_star_end : idx_star_end + n_host], dtype=np.float64)
                if n_host > 1:
                    results['host_core_errs'] = se_h.copy()
                    results['host_core_err'] = float(np.sqrt(np.sum(np.square(se_h))))
                else:
                    results['host_core_err'] = float(se_h[0])
            if n_nps > 0:
                results['nuclear_point_err'] = float(sig[idx_star_end + n_host])

            cov_bg = {}
            for eid in science_epoch_ids:
                frame_hits = [i for i, c in enumerate(cutouts) if (not c.get('is_template')) and int(c['epoch_id']) == int(eid)]
                if not frame_hits:
                    continue
                i0 = frame_hits[0]
                i_f = idx_trans + sci_frame_to_idx[int(i0)]
                i_b = idx_bg + int(i0)
                sub = cov[np.ix_([i_f, i_b], [i_f, i_b])]
                cov_bg[int(eid)] = {
                    'var_transient': float(sub[0, 0]),
                    'var_background': float(sub[1, 1]),
                    'cov_transient_background': float(sub[0, 1]),
                }
            results['transient_bg_cov_by_epoch_id'] = cov_bg
        except np.linalg.LinAlgError:
            _log.warning("Could not invert Hessian H for parameter uncertainties; errs left at zero.")

        if float_pos and idx_off is not None:
            dar = results['transient_dra_deg']
            dde = results['transient_ddec_deg']
            cosdec = np.cos(np.deg2rad(config.TRANSIENT_DEC))
            d_as = 3600.0 * np.hypot(dar * cosdec, dde)
            print(
                f"   [Solver] Fitted transient offset: dRA={dar:.4e} deg, dDec={dde:.4e} deg "
                f"(|Δ|≈{d_as:.4f} arcsec)"
            )
        
    except Exception as e:
        print(f"CRITICAL SOLVER ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def predict_cutout_model(
    results,
    cutouts,
    stars,
    star_fluxes,
    frame_index,
    *,
    include_gp=True,
    include_transient=True,
    include_stars=True,
    include_host=True,
    include_nuclear_point=True,
    transient_flux_override=None,
    host_position_override=None,
):
    """
    Prediction path for an intrinsic scene model:
      intrinsic = GP galaxy + resolved host + deltas(stars, nuclear point, transient)
      model = background + intrinsic convolved with spatially varying PRF on this BCD.
    Point sources are represented as deltas in intrinsic space; PRF smearing happens only
    in the full-scene convolution.
    """
    scene_wcs = results['scene_wcs']
    scene_shape = results['scene_shape']
    gp = results.get('gp_scene', results['model_scene'])
    gp = np.asarray(gp, dtype=np.float64)
    entry = cutouts[frame_index]
    if 'bcd_backgrounds' in results and len(np.asarray(results.get('bcd_backgrounds'))) > frame_index:
        bg = float(np.asarray(results['bcd_backgrounds'], dtype=np.float64)[frame_index])
    else:
        bg = float(results['epoch_backgrounds'][entry['epoch_id']])
    native_shape = tuple(np.asarray(entry['data']).shape)
    n_pix_native = int(native_shape[0]) * int(native_shape[1])
    pred = np.full(n_pix_native, bg, dtype=np.float64)

    tra_ra = float(config.TRANSIENT_RA) + float(results.get('transient_dra_deg', 0.0))
    tra_dec = float(config.TRANSIENT_DEC) + float(results.get('transient_ddec_deg', 0.0))

    chan = 'ch2' if 'ch2' in entry['filename'] else 'ch1'
    w_native = entry['raw_wcs']
    is_full = entry.get('is_full_array', False)
    h, w = scene_shape
    intrinsic = np.zeros((h, w), dtype=np.float64)
    has_intrinsic = False

    if include_gp:
        intrinsic += gp
        has_intrinsic = True

    if include_transient and not entry.get('is_template'):
        if transient_flux_override is not None:
            ft = float(transient_flux_override)
        else:
            tf = results.get('transient_fluxes')
            if tf is not None and len(np.asarray(tf)) > frame_index:
                ft = float(np.asarray(tf, dtype=np.float64)[frame_index])
            else:
                eid = int(entry['epoch_id'])
                ep_idx = results.get('transient_epoch_index_by_id', {}).get(eid)
                te = results.get('transient_epoch_fluxes')
                if ep_idx is None or te is None or len(np.asarray(te)) <= ep_idx:
                    ft = 0.0
                else:
                    ft = float(np.asarray(te, dtype=np.float64)[ep_idx])
        tx, ty = scene_wcs.world_to_pixel_values(tra_ra, tra_dec)
        _add_delta_to_image(intrinsic, float(tx), float(ty), ft)
        has_intrinsic = has_intrinsic or (abs(ft) > 0.0)

    if include_stars and len(stars) > 0:
        sf = np.asarray(star_fluxes, dtype=np.float64).ravel()
        for j, s_obj in enumerate(stars):
            if j >= len(sf) or sf[j] <= 0.0:
                continue
            sx, sy = scene_wcs.world_to_pixel_values(s_obj.ra.deg, s_obj.dec.deg)
            _add_delta_to_image(intrinsic, float(sx), float(sy), sf[j])
            has_intrinsic = True

    if include_host:
        ra_h = dec_h = None
        if host_position_override is not None:
            ra_h = float(host_position_override[0])
            dec_h = float(host_position_override[1])
        elif getattr(config, 'USE_HOST_GAUSSIAN_CORE', False):
            ra_h = getattr(config, 'HOST_CORE_RA', None)
            dec_h = getattr(config, 'HOST_CORE_DEC', None)
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
                    col_h = host_core_gaussian_column(
                        scene_wcs,
                        float(ra_h),
                        float(dec_h),
                        float(sj),
                        scene_shape,
                    )
                    intrinsic += float(fj) * col_h.reshape(scene_shape)
                    has_intrinsic = has_intrinsic or (abs(float(fj)) > 0.0)
            else:
                col_h = host_core_gaussian_column(
                    scene_wcs,
                    ra_h,
                    dec_h,
                    float(getattr(config, 'HOST_CORE_SIGMA_PX', 1.5)),
                    scene_shape,
                )
                fh = float(results.get('host_core_flux', 0.0))
                intrinsic += fh * col_h.reshape(scene_shape)
                has_intrinsic = has_intrinsic or (abs(fh) > 0.0)

    if include_nuclear_point and getattr(config, 'USE_NUCLEAR_POINT_SOURCE', False):
        ra_np = getattr(config, 'NUCLEAR_POINT_RA', None)
        dec_np = getattr(config, 'NUCLEAR_POINT_DEC', None)
        if ra_np is None or dec_np is None:
            ra_np = getattr(config, 'HOST_CORE_RA', None)
            dec_np = getattr(config, 'HOST_CORE_DEC', None)
        if ra_np is not None and dec_np is not None:
            fnp = float(results.get('nuclear_point_flux', 0.0))
            nx, ny = scene_wcs.world_to_pixel_values(float(ra_np), float(dec_np))
            _add_delta_to_image(intrinsic, float(nx), float(ny), fnp)
            has_intrinsic = has_intrinsic or (abs(fnp) > 0.0)

    if has_intrinsic:
        conv = _apply_frame_forward_operator(
            intrinsic, scene_wcs, w_native, scene_shape, native_shape, chan, is_full_array=is_full,
        )
        pred += conv.ravel()

    return pred.reshape(native_shape)


def reconstruct_full_model(results, star_coords, star_fluxes, config_channel, cutouts):
    scene_model = results['model_scene'].copy()
    scene_wcs = results['scene_wcs']
    if scene_wcs is None: return scene_model
    
    scene_shape = results['scene_shape']
    n = min(len(star_fluxes), len(star_coords))
    
    chan = 'ch2' if 'ch2' in config_channel else 'ch1'
    accum = np.zeros_like(scene_model)
    counts = 0
    stride = max(1, len(cutouts) // 50)
    
    for i in range(0, len(cutouts), stride):
        w_native = cutouts[i]['raw_wcs']
        is_full = cutouts[i].get('is_full_array', False)
        
        h, w = scene_shape
        intrinsic = np.zeros((h, w), dtype=np.float64)
        for j in range(n):
            if star_fluxes[j] <= 0:
                continue
            sx, sy = scene_wcs.world_to_pixel_values(
                star_coords[j].ra.deg, star_coords[j].dec.deg,
            )
            _add_delta_to_image(intrinsic, float(sx), float(sy), float(star_fluxes[j]))
        field = apply_spatially_varying_prf_to_scene(
            intrinsic, scene_wcs, w_native, scene_shape, chan, is_full_array=is_full,
        )
        accum += field
        counts += 1
        
    if counts > 0: scene_model += accum / counts
    return scene_model
