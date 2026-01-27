"""src/pmap_correction.py"""
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

# Cache to avoid re-reading FITS files for every star
_CACHE = {}

def _extract_varying_vector(arr):
    """
    Extracts the 1D coordinate vector from a 2D meshgrid FITS file.
    """
    if arr.ndim == 1:
        return arr
    
    std_0 = np.std(arr[:, 0]) # Variation down the column
    std_1 = np.std(arr[0, :]) # Variation across the row
    
    if std_1 > std_0: return arr[0, :]
    elif std_0 > std_1: return arr[:, 0]
    else: return arr[:, 0]

def _load_channel_data(channel, pmap_dir, combine_gauss=False):
    cache_key = (channel, combine_gauss)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    ch_str = str(channel).replace('ch', '')
    if str(ch_str) == '1':
        suffix = '1_500x500_0043_120828.fits'
    elif str(ch_str) == '2':
        suffix = '2_0p1s_x4_500x500_0043_120124.fits'
    else:
        raise ValueError(f"Unsupported Channel: {channel}")

    # File Paths
    f_xgrid = os.path.join(pmap_dir, f'xgrid_ch{suffix}')
    f_ygrid = os.path.join(pmap_dir, f'ygrid_ch{suffix}')
    f_occu = os.path.join(pmap_dir, f'occu_ch{suffix}')
    
    if combine_gauss:
        f_pmap = os.path.join(pmap_dir, f'pmap_combined_ch{suffix}')
    else:
        f_pmap = os.path.join(pmap_dir, f'pmap_ch{suffix}')

    for f in [f_xgrid, f_ygrid, f_pmap, f_occu]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"[PMap] Missing required file: {f}")

    # Load Data
    xgrid_raw = fits.getdata(f_xgrid).astype(np.float64)
    ygrid_raw = fits.getdata(f_ygrid).astype(np.float64)
    pmap = fits.getdata(f_pmap).astype(np.float64)
    occu = fits.getdata(f_occu).astype(np.float64)

    # Sanitize Grids
    xgrid = _extract_varying_vector(xgrid_raw)
    ygrid = _extract_varying_vector(ygrid_raw)

    # Orientation Check (RegularGridInterpolator requires strictly ascending)
    if xgrid[-1] < xgrid[0]:
        xgrid = np.flip(xgrid)
        pmap = np.flip(pmap, axis=1)
        occu = np.flip(occu, axis=1)
    
    if ygrid[-1] < ygrid[0]:
        ygrid = np.flip(ygrid)
        pmap = np.flip(pmap, axis=0)
        occu = np.flip(occu, axis=0)

    # Create Interpolators
    # IDL interp2d(A, x0, y0, ...) behaves like linear interpolation on grid.
    # RegularGridInterpolator matches this behavior.
    # bounds_error=False matches IDL behavior of returning missing/clamped values (we use NaN for missing)
    interp_pmap = RegularGridInterpolator((ygrid, xgrid), pmap, bounds_error=False, fill_value=np.nan)
    interp_occu = RegularGridInterpolator((ygrid, xgrid), occu, bounds_error=False, fill_value=np.nan)

    _CACHE[cache_key] = {
        'pmap': interp_pmap,
        'occu': interp_occu,
        'x_range': (xgrid.min(), xgrid.max()),
        'y_range': (ygrid.min(), ygrid.max())
    }
    return _CACHE[cache_key]

def iracpc_pmap_corr(observed_flux, x, y, channel,
                     pmap_dir='data/pmap_fits',
                     threshold_occ=True,
                     threshold_val=20,
                     combine_gauss=False,
                     full_array=False,
                     missing=np.nan):
    """
    Python equivalent of iracpc_pmap_corr.pro.
    Strict translation: No phase wrapping, no coordinate manipulation beyond xfov/yfov.
    """
    
    #
    if full_array:
        xfov = 8.0
        yfov = 216.0
    else:
        xfov = 0.0
        yfov = 0.0

    f0 = np.atleast_1d(observed_flux).astype(float)
    x0 = np.atleast_1d(x).astype(float)
    y0 = np.atleast_1d(y).astype(float)
    
    if np.isscalar(channel):
        channels = np.full(f0.shape, channel)
    else:
        channels = np.atleast_1d(channel)
        if len(channels) == 1 and len(f0) > 1:
            channels = np.full(f0.shape, channels[0])

    corrected_flux = np.zeros_like(f0) * np.nan
    
    unique_chans = np.unique(channels)
    
    for ch in unique_chans:
        idx = np.where(channels == ch)[0]
        if len(idx) == 0: continue
        
        try:
            data = _load_channel_data(ch, pmap_dir, combine_gauss)
            interp_pmap = data['pmap']
            interp_occu = data['occu']
            
            #
            # IDL: interp2d(pmap, xgrid, ygrid, x0[Index]-xfov, y0[Index]-yfov, /regular)
            # Direct input usage.
            x_query = x0[idx] - xfov
            y_query = y0[idx] - yfov
            
            # Create (Y, X) points for interpolation
            pts = np.column_stack((y_query, x_query))
            
            # Interpolate Map
            gain_map = interp_pmap(pts)
            occu_map = interp_occu(pts)
            
            # Apply Correction
            valid_gain = np.isfinite(gain_map) & (gain_map != 0)
            
            corr_ch = np.full(len(idx), missing)
            corr_ch[valid_gain] = f0[idx][valid_gain] / gain_map[valid_gain]
            
            # Threshold Occupation
            if threshold_occ:
                mask_bad = (occu_map < threshold_val) | np.isnan(occu_map)
                if not combine_gauss:
                    corr_ch[mask_bad] = missing
            
            corrected_flux[idx] = corr_ch
            
        except Exception as e:
            print(f"[PMap] Error processing Channel {ch}: {e}")
            continue

    if np.isscalar(observed_flux):
        return corrected_flux[0]
    return corrected_flux
