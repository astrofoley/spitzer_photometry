"""Residual spatial structure metrics for QA (ACF, dipole, whiteness)."""
import numpy as np


def _acf2d_centered(arr, valid_mask):
    """Normalized 2D autocorrelation; peak at center = 1."""
    a = np.where(valid_mask, arr.astype(float), 0.0)
    a -= np.nanmean(a[valid_mask]) if np.any(valid_mask) else 0.0
    h, w = a.shape
    pad_h, pad_w = h, w
    ap = np.zeros((2 * pad_h, 2 * pad_w), dtype=float)
    ap[:h, :w] = a
    fa = np.fft.rfft2(ap)
    p = np.fft.irfft2(fa * np.conj(fa), s=ap.shape)
    p = np.fft.fftshift(p)
    cy, cx = p.shape[0] // 2, p.shape[1] // 2
    peak = p[cy, cx]
    if peak <= 0 or not np.isfinite(peak):
        return p * 0.0
    return p / peak


def acf_e90_scale_pix(acf):
    """Radius (pix) where azimuthally averaged ACF drops to 0.5 (FWHM-like)."""
    h, w = acf.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(xx - cx, yy - cy).astype(float)
    r_flat = r.ravel()
    a_flat = acf.ravel()
    m = np.isfinite(a_flat) & (r_flat > 0.5)
    if np.sum(m) < 10:
        return float('nan')
    bins = np.arange(0.5, min(cy, cx) + 0.5, 0.5)
    prof = []
    rb = []
    for i in range(len(bins) - 1):
        sel = m & (r_flat >= bins[i]) & (r_flat < bins[i + 1])
        if np.sum(sel) > 0:
            prof.append(float(np.nanmean(a_flat[sel])))
            rb.append(0.5 * (bins[i] + bins[i + 1]))
    if len(prof) < 2:
        return float('nan')
    prof = np.asarray(prof)
    rb = np.asarray(rb)
    above = prof > 0.5
    if not np.any(above):
        return float(rb[-1])
    i0 = int(np.argmax(above))
    if i0 == 0:
        return float(rb[0])
    t = (0.5 - prof[i0 - 1]) / (prof[i0] - prof[i0 - 1] + 1e-30)
    t = float(np.clip(t, 0.0, 1.0))
    return float(rb[i0 - 1] + t * (rb[i0] - rb[i0 - 1]))


def dipole_moment_xy(resid, valid_mask):
    """Flux-weighted centroid offset from image center (dipole proxy)."""
    r = np.asarray(resid, dtype=float)
    h, w = r.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    m = valid_mask & np.isfinite(r)
    if np.sum(m) < 5:
        return 0.0, 0.0, 0.0
    wt = np.abs(r[m])
    s = np.sum(wt)
    if s <= 0:
        return 0.0, 0.0, 0.0
    mx = np.sum((xx[m] - cx) * wt) / s
    my = np.sum((yy[m] - cy) * wt) / s
    return float(mx), float(my), float(np.hypot(mx, my))


def lag1_correlation_z(z, valid_mask):
    """Mean corr of z(x,y) with z(x+1,y) and z(x,y+1) on valid shifted pairs."""
    z = np.where(valid_mask, z.astype(float), np.nan)
    m = valid_mask
    c = []
    zr = z[:, 1:]
    zl = z[:, :-1]
    mr = m[:, 1:] & m[:, :-1]
    if np.sum(mr) > 5:
        a, b = zr[mr], zl[mr]
        c.append(np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b))) / (np.nanstd(a) * np.nanstd(b) + 1e-30))
    zd = z[1:, :]
    zu = z[:-1, :]
    md = m[1:, :] & m[:-1, :]
    if np.sum(md) > 5:
        a, b = zd[md], zu[md]
        c.append(np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b))) / (np.nanstd(a) * np.nanstd(b) + 1e-30))
    if not c:
        return float('nan')
    return float(np.nanmean(c))


def summarize_frame_residual(resid, sigma, valid_mask):
    """Single-frame metrics dict."""
    r = np.asarray(resid, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    m = valid_mask & np.isfinite(r) & np.isfinite(sig)
    out = {}
    if np.sum(m) < 16:
        return out
    acf = _acf2d_centered(r, m)
    out['acf_e90_scale_pix'] = acf_e90_scale_pix(acf)
    dx, dy, dmag = dipole_moment_xy(r, m)
    out['dipole_mx'] = dx
    out['dipole_my'] = dy
    out['dipole_mag_pix'] = dmag
    with np.errstate(divide='ignore', invalid='ignore'):
        z = r / np.clip(sig, 1e-30, None)
    zm = m & np.isfinite(z)
    out['z_mean'] = float(np.nanmean(z[zm]))
    out['z_std'] = float(np.nanstd(z[zm]))
    out['z_lag1_corr'] = lag1_correlation_z(z, zm)
    return out


def prf_autocorr_scale_on_grid(scene_wcs, w_native, channel, ra, dec, scene_shape, is_full):
    """ACF e90 scale of normalized PRF stamp (same grid as data)."""
    from .solver import load_prf, generate_prf_fast

    tx, ty = w_native.world_to_pixel_values(ra, dec)
    prf = load_prf(channel, tx, ty)
    col = generate_prf_fast(
        scene_wcs, w_native, prf, ra, dec, scene_shape,
        channel=channel, is_full_array=is_full,
    )
    h, w = scene_shape
    img = col.reshape(h, w)
    vm = img > 1e-20 * np.nanmax(img)
    acf = _acf2d_centered(img, vm)
    return acf_e90_scale_pix(acf)
